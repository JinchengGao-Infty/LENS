"""ICDE main comparison: which interactions does each method hand over?

One cell = (dataset, p, seed, method).
Methods run here: lens, h2_refit, iloco, fast.  DIAMOND runs in its own
environment through icde_diamond_adapter.py on the npz written by
``--dump-data``.
"""

from __future__ import annotations

import os

# XGBoost starts one OpenMP thread per core in part of GPU training regardless of
# n_jobs; the pool size has to be set before any library loads OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import functools
import itertools
import json
import math
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

# NumPy asks for transparent huge pages on large arrays; on a fragmented shared
# machine every such allocation stalls in kernel compaction (50x slower).  The
# package import has already loaded NumPy, so the environment switch is too late.
np._core.multiarray._set_madvise_hugepage(False)

from lens.calibration.multiple_testing import by_rejections
from lens.grids import PairGrid, build_grid
from lens.interaction_vectors import compute_interaction_sample
from lens.model_inference import (
    family_rms_floor_amplitude_thresholds,
    normal_replication_pvalues_at_pair_thresholds,
    normal_replication_pvalues_at_threshold,
    run_replication_inference_streaming,
)
from lens.splits import make_three_way_split

Pair = tuple[int, int]
SYNTHETIC = ("sparse_pairs", "additive", "additive_corr", "pair_corr")
ALPHAS = (0.05, 0.2)
KAPPAS = (0.01, 0.02)
XGB = {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist"}
# CPU threads per process; 16 workers share a 256-core machine with other users.
CPU_THREADS = 8


# ---------------------------------------------------------------- data

def make_synthetic(name: str, *, p: int, seed: int, n: int = 20000, sigma: float = 0.25):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, max(p, 50)))[:, :p].copy()
    noise = rng.standard_normal(n) * sigma
    X = base
    if name in ("additive_corr", "pair_corr"):
        rho = 0.9
        X[:, 2] = rho * X[:, 0] + math.sqrt(1 - rho**2) * base[:, 2]
        X[:, 3] = rho * X[:, 1] + math.sqrt(1 - rho**2) * base[:, 3]
    if name == "sparse_pairs":
        f = (np.sin(X[:, 6]) + 0.5 * X[:, 7] ** 2 + 0.5 * X[:, 8]
             + 1.5 * X[:, 0] * X[:, 1] + X[:, 2] * X[:, 3] + 0.75 * X[:, 4] * X[:, 5])
        truth = [(0, 1), (2, 3), (4, 5)]
    elif name == "additive":
        f = np.sin(X[:, 6]) + 0.5 * X[:, 7] ** 2 + 0.5 * X[:, 8]
        truth = []
    elif name == "additive_corr":
        f = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2 + 0.5 * X[:, 4]
        truth = []
    elif name == "pair_corr":
        f = X[:, 0] * X[:, 1] + 0.5 * np.sin(X[:, 4]) + 0.25 * X[:, 5] ** 2
        truth = [(0, 1)]
    else:
        raise ValueError(name)
    return X, f + noise, f, truth


def load_cell(args):
    if args.dataset.startswith("F"):
        data = np.load(Path(args.data_dir) / f"{args.dataset}_seed{args.seed}.npz")
        X, y = data["X"], data["y"]
        return X, y, y, [tuple(int(v) for v in pair) for pair in data["true_pairs"]]
    return make_synthetic(args.dataset, p=args.p, seed=args.seed)


# ---------------------------------------------------------------- teacher

class Teacher:
    """XGBoost with early stopping; margin predictions on the GPU via CuPy."""

    def __init__(self, X, y, *, seed: int, max_depth: int | None = None):
        from xgboost import XGBRegressor

        rng = np.random.default_rng(seed)
        order = rng.permutation(len(y))
        cut = int(0.9 * len(y))
        params = {**XGB, **({"max_depth": max_depth} if max_depth else {})}
        model = XGBRegressor(device="cuda", random_state=seed, early_stopping_rounds=50,
                             n_jobs=CPU_THREADS, **params)
        model.fit(X[order[:cut]], y[order[:cut]],
                  eval_set=[(X[order[cut:]], y[order[cut:]])], verbose=False)
        self.booster = model.get_booster()[: model.best_iteration + 1]

    def predict_margin(self, X):
        import cupy as cp

        return cp.asnumpy(self.booster.inplace_predict(cp.asarray(X), predict_type="margin"))


# DIAMOND's MLP (sim.py, model_type="mlp") without its knockoff coupling layer.
MLP_HIDDEN = (140, 100, 60, 20)


class MLPTeacher:
    """DIAMOND's MLP training settings: Adam 1e-3, batch 256, MSE, at most 100
    epochs, early stopping with patience 10 on a held-out 10%, best epoch kept."""

    def __init__(self, X, y, *, seed: int):
        import torch

        torch.set_num_threads(CPU_THREADS)
        torch.manual_seed(seed)
        order = np.random.default_rng(seed).permutation(len(y))
        cut = int(0.9 * len(y))
        self.x_mean, self.x_sd = X.mean(axis=0), X.std(axis=0)
        self.y_mean, self.y_sd = float(y.mean()), float(y.std())
        Xs = torch.tensor((X - self.x_mean) / self.x_sd, dtype=torch.float32, device="cuda")
        ys = torch.tensor((y - self.y_mean) / self.y_sd, dtype=torch.float32, device="cuda")
        train = torch.as_tensor(order[:cut], device="cuda")
        val = torch.as_tensor(order[cut:], device="cuda")
        dims = (X.shape[1], *MLP_HIDDEN, 1)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(torch.nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(torch.nn.ELU())
        self.net = torch.nn.Sequential(*layers).cuda()
        optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        best, best_state, wait = math.inf, None, 0
        for epoch in range(100):
            self.net.train()
            shuffled = train[torch.randperm(len(train), device="cuda")]
            for start in range(0, len(shuffled), 256):
                batch = shuffled[start:start + 256]
                optimizer.zero_grad()
                loss = torch.mean((self.net(Xs[batch]).squeeze(1) - ys[batch]) ** 2)
                loss.backward()
                optimizer.step()
            self.net.eval()
            with torch.no_grad():
                val_loss = float(torch.mean((self.net(Xs[val]).squeeze(1) - ys[val]) ** 2))
            if val_loss < best:
                best, wait = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
            else:
                wait += 1
                if wait >= 10:
                    break
        self.net.load_state_dict(best_state)
        self.net.eval()

    def predict_margin(self, X):
        import torch

        mean = torch.as_tensor(self.x_mean, device="cuda")
        sd = torch.as_tensor(self.x_sd, device="cuda")
        out = []
        with torch.no_grad():
            for start in range(0, len(X), 1 << 20):
                chunk = torch.from_numpy(np.ascontiguousarray(X[start:start + (1 << 20)], dtype=np.float64)).cuda()
                out.append(self.net(((chunk - mean) / sd).float()).squeeze(1).double().cpu().numpy())
        return np.concatenate(out) * self.y_sd + self.y_mean


def signal_r2(teacher, X, f) -> float:
    pred = teacher.predict_margin(X)
    return float(1 - np.mean((f - pred) ** 2) / np.var(f))


def by_list(pairs, pvalues, alpha):
    keep = by_rejections(np.asarray([pvalues[pair] for pair in pairs]), alpha=alpha)
    return sorted(pair for pair, k in zip(pairs, keep) if k)


# ---------------------------------------------------------------- methods

def run_lens(X, y, f, seed, *, teacher=Teacher):
    split = make_three_way_split(len(y), seed=seed)
    d, c, e = split.discovery, split.confirmation, split.evaluation
    discovery = teacher(X[d], y[d], seed=2 * seed + 1)
    confirmation = teacher(X[c], y[c], seed=2 * seed + 2)
    p = X.shape[1]
    pairs = list(itertools.combinations(range(p), 2))
    grids = [build_grid(X[d][:, j], "continuous", n_points=12) for j in range(p)]
    pair_grids = {pair: PairGrid(grids[pair[0]], grids[pair[1]]) for pair in pairs}
    contexts = X[d][np.random.default_rng(2 * seed + 4).choice(len(d), min(3000, len(d)), replace=False)]
    means, energy = {}, {}
    for pair in pairs:
        sample = compute_interaction_sample(discovery.predict_margin, contexts, pair, pair_grids[pair],
                                            estimate_covariance="none")
        means[pair] = sample.mean.copy()
        energy[pair] = float(sample.raw_energy)
    ordered = sorted(pairs, key=lambda pair: (-energy[pair], pair))
    X_eval = X[e]
    replication = run_replication_inference_streaming(
        means,
        lambda pair: compute_interaction_sample(confirmation.predict_margin, X_eval, pair,
                                                pair_grids[pair], estimate_covariance="none"),
        ordered, n_eval=len(X_eval), minimum_amplitude=0.25, n_bootstrap=1, seed=2 * seed + 3,
    )
    _, relative_p = normal_replication_pvalues_at_threshold(replication, 0.25)
    rms_norm, thresholds = family_rms_floor_amplitude_thresholds(replication, 0.25)
    _, rms_p = normal_replication_pvalues_at_pair_thresholds(replication, thresholds)
    sd_f = float(np.std(discovery.predict_margin(contexts)))
    lists = {}
    for alpha in ALPHAS:
        lists[f"family_rms_alpha{alpha}"] = by_list(ordered, rms_p, alpha)
        lists[f"pair_relative_alpha{alpha}"] = by_list(ordered, relative_p, alpha)
    # Scale floor: the reproduced strength must also exceed kappa * sd of the predictions.
    for kappa in KAPPAS:
        floored = {pair: max(thresholds[pair], kappa * sd_f / math.sqrt(max(energy[pair], 1e-12)))
                   for pair in ordered}
        _, floor_p = normal_replication_pvalues_at_pair_thresholds(replication, floored)
        for alpha in ALPHAS:
            lists[f"floor{kappa}_alpha{alpha}"] = by_list(ordered, floor_p, alpha)
    # Amplitude units (a_u, mu_u, se of mu_u) so that other floors can be read off later.
    pair_rows = [
        [int(pair[0]), int(pair[1]), math.sqrt(energy[pair]),
         math.sqrt(energy[pair]) * replication.amplitude_ratios[pair],
         math.sqrt(energy[pair]) * replication.standard_errors[pair],
         math.sqrt(replication.confirmation_energies[pair])]
        for pair in ordered
    ]
    return lists, {
        "n_model_fits": 2,
        "teacher_signal_r2": signal_r2(confirmation, X[e], f[e]),
        "ranking": [list(pair) for pair in ordered[:20]],
        "family_rms_norm": rms_norm,
        "discovery_margin_sd": sd_f,
        "pair_columns": ["j", "k", "amplitude", "projection", "projection_se", "confirmation_amplitude"],
        "pairs": pair_rows,
    }


def h2_scores(teacher: Teacher, X_ref, pairs, pair_grids):
    return {
        pair: compute_interaction_sample(teacher.predict_margin, X_ref, pair, pair_grids[pair],
                                         estimate_covariance="none").friedman_h2
        for pair in pairs
    }


def run_h2_refit(X, y, f, seed, *, n_refits: int = 100):
    """H^2 with the Friedman-Popescu additive null, refit B times."""
    p = X.shape[1]
    pairs = list(itertools.combinations(range(p), 2))
    grids = [build_grid(X[:, j], "continuous", n_points=12) for j in range(p)]
    pair_grids = {pair: PairGrid(grids[pair[0]], grids[pair[1]]) for pair in pairs}
    X_ref = X[np.random.default_rng(seed + 7).choice(len(y), 500, replace=False)]
    teacher = Teacher(X, y, seed=seed)
    observed = h2_scores(teacher, X_ref, pairs, pair_grids)
    additive = Teacher(X, y, seed=seed, max_depth=1)
    fitted = additive.predict_margin(X)
    residual = y - fitted
    rng = np.random.default_rng(seed + 11)
    null = {pair: [] for pair in pairs}
    for b in range(n_refits):
        y_null = fitted + rng.permutation(residual)
        refit = Teacher(X, y_null, seed=seed + 1000 + b)
        for pair, value in h2_scores(refit, X_ref, pairs, pair_grids).items():
            null[pair].append(value)
    pvalues = {pair: (1 + sum(v >= observed[pair] for v in null[pair])) / (n_refits + 1) for pair in pairs}
    lists = {f"bootstrap_by_alpha{alpha}": by_list(pairs, pvalues, alpha) for alpha in ALPHAS}
    # The permutation p-value cannot fall below 1/(B+1), above every BY threshold
    # once there are more than a few dozen pairs; a normal fit to the null can.
    z = {pair: (observed[pair] - np.mean(null[pair])) / np.std(null[pair], ddof=1) for pair in pairs}
    normal_p = {pair: 0.5 * math.erfc(z[pair] / math.sqrt(2)) for pair in pairs}
    for alpha in ALPHAS:
        lists[f"normal_by_alpha{alpha}"] = by_list(pairs, normal_p, alpha)
    lists["one_sd_rule_B10"] = sorted(
        pair for pair in pairs
        if observed[pair] > np.mean(null[pair][:10]) + np.std(null[pair][:10], ddof=1)
    )
    return lists, {
        "n_model_fits": 2 + n_refits,
        "teacher_signal_r2": signal_r2(teacher, X, f),
        "min_pvalue": min(pvalues.values()),
        "pair_columns": ["j", "k", "observed", "null_mean", "null_sd"],
        "pairs": [[j, k, float(observed[(j, k)]), float(np.mean(null[(j, k)])), float(np.std(null[(j, k)], ddof=1))]
                  for j, k in pairs],
    }


def run_iloco(X, y, f, seed):
    """iLOCO-Split: half to fit, half to evaluate the leave-out losses."""
    p = X.shape[1]
    order = np.random.default_rng(seed).permutation(len(y))
    tr, te = order[: len(y) // 2], order[len(y) // 2:]
    cache: dict[frozenset, np.ndarray] = {}

    def loss(dropped: frozenset):
        if dropped not in cache:
            keep = [j for j in range(p) if j not in dropped]
            model = Teacher(X[tr][:, keep], y[tr], seed=seed)
            cache[dropped] = (y[te] - model.predict_margin(X[te][:, keep])) ** 2
        return cache[dropped]

    full = loss(frozenset())
    pairs = list(itertools.combinations(range(p), 2))
    pvalues = {}
    for j, k in pairs:
        values = (loss(frozenset({j})) - full) + (loss(frozenset({k})) - full) - (loss(frozenset({j, k})) - full)
        sd = values.std(ddof=1)
        z = values.mean() / (sd / math.sqrt(len(values))) if sd > 0 else (math.inf if values.mean() > 0 else -math.inf)
        pvalues[(j, k)] = 0.5 * math.erfc(z / math.sqrt(2))
    lists = {f"by_alpha{alpha}": by_list(pairs, pvalues, alpha) for alpha in ALPHAS}
    return lists, {"n_model_fits": len(cache), "teacher_signal_r2": None}


def run_iloco_mp(X, y, f, seed, *, n_patches: int = 10000, fraction: float = 0.2):
    """iLOCO-MP (Little, Zheng & Allen 2025, Algorithm 2) with decision-tree minipatches.

    Each minipatch fits a tree on 20% of the rows and 20% of the features.  The
    leave-one-out prediction for row i without features S averages the patches
    that exclude i and all of S; the per-row score is
    Delta_j + Delta_k - Delta_jk on squared error, with a normal p-value.
    """
    import cupy as cp
    from joblib import Parallel, delayed
    from sklearn.tree import DecisionTreeRegressor

    n, p = X.shape
    n_rows, n_feats = round(fraction * n), max(1, round(fraction * p))
    rng = np.random.default_rng(seed)
    pairs = list(itertools.combinations(range(p), 2))
    J, K = np.array([j for j, _ in pairs]), np.array([k for _, k in pairs])
    # Rows of the aggregates: all features, each feature left out, each pair left out.
    numerator = cp.zeros((1 + p + len(pairs), n), dtype=cp.float64)
    denominator = cp.zeros_like(numerator)

    def fit_predict(rows, feats, b):
        tree = DecisionTreeRegressor(random_state=seed + b).fit(X[rows][:, feats], y[rows])
        return tree.predict(X[:, feats]).astype(np.float32)

    for start in range(0, n_patches, 5000):
        size = min(5000, n_patches - start)
        rows = [rng.choice(n, n_rows, replace=False) for _ in range(size)]
        feats = [np.sort(rng.choice(p, n_feats, replace=False)) for _ in range(size)]
        # sklearn trees release the GIL, so threads avoid copying the data to workers.
        preds = np.stack(Parallel(n_jobs=CPU_THREADS, prefer="threads")(
            delayed(fit_predict)(rows[b], feats[b], start + b) for b in range(size)))
        out_row = np.ones((size, n), dtype=np.float32)
        out_feat = np.ones((size, p), dtype=np.float32)
        for b in range(size):
            out_row[b, rows[b]] = 0.0
            out_feat[b, feats[b]] = 0.0
        A, G = cp.asarray(out_row), cp.asarray(out_feat)
        mask = cp.concatenate([cp.ones((size, 1), dtype=cp.float32), G, G[:, J] * G[:, K]], axis=1)
        numerator += mask.T @ (A * cp.asarray(preds))
        denominator += mask.T @ A

    error = (cp.asarray(y) - numerator / denominator) ** 2
    single = error[1:1 + p] - error[0]
    double = error[1 + p:] - error[0]
    scores = single[J] + single[K] - double
    z = cp.asnumpy(scores.mean(axis=1) / (scores.std(axis=1, ddof=1) / math.sqrt(n)))
    pvalues = {pair: 0.5 * math.erfc(float(value) / math.sqrt(2)) for pair, value in zip(pairs, z)}
    lists = {f"by_alpha{alpha}": by_list(pairs, pvalues, alpha) for alpha in ALPHAS}
    return lists, {"n_model_fits": n_patches, "teacher_signal_r2": None}


def run_fast(X, y, f, seed, *, ks=(0, 1, 2, 3, 5, 8, 12, 20)):
    from interpret.glassbox import ExplainableBoostingRegressor

    order = np.random.default_rng(seed).permutation(len(y))
    cut = int(0.8 * len(y))
    tr, va = order[:cut], order[cut:]
    best = None
    for k in ks:
        ebm = ExplainableBoostingRegressor(interactions=k, random_state=seed, n_jobs=CPU_THREADS)
        ebm.fit(X[tr], y[tr])
        mse = float(np.mean((y[va] - ebm.predict(X[va])) ** 2))
        chosen = sorted(tuple(sorted(int(v) for v in term)) for term in ebm.term_features_ if len(term) == 2)
        if best is None or mse < best[0]:
            best = (mse, k, chosen)
    return {"validation_k": best[2]}, {"n_model_fits": len(ks), "chosen_k": best[1], "teacher_signal_r2": None}


METHODS = {
    "lens": run_lens,
    "lens_mlp": functools.partial(run_lens, teacher=MLPTeacher),
    "h2_refit": run_h2_refit,
    "iloco": run_iloco,
    "iloco_mp": run_iloco_mp,
    "fast": run_fast,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--p", type=int, default=30)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--method", choices=sorted(METHODS))
    parser.add_argument("--data-dir", default="data_icde")
    parser.add_argument("--dump-data", help="write the cell's npz here and exit")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    threadpool_limits(CPU_THREADS)  # BLAS/OpenMP pools already loaded with numpy

    X, y, f, truth = load_cell(args)
    if args.dump_data:
        np.savez(args.dump_data, X=X, y=y, f=f, true_pairs=np.asarray(truth, dtype=np.int64).reshape(-1, 2))
        return
    started = time.perf_counter()
    lists, meta = METHODS[args.method](X, y, f, args.seed)
    wall = time.perf_counter() - started
    truth_set = set(truth)
    summary = {}
    for name, chosen in lists.items():
        chosen = [tuple(pair) for pair in chosen]
        tp = sum(pair in truth_set for pair in chosen)
        summary[name] = {
            "size": len(chosen),
            "tp": tp,
            "fdp": (len(chosen) - tp) / len(chosen) if chosen else 0.0,
            "power": tp / len(truth_set) if truth_set else None,
            "pairs": [list(pair) for pair in chosen],
        }
    payload = {
        "dataset": args.dataset, "p": int(X.shape[1]), "seed": args.seed, "method": args.method,
        "n": int(len(y)), "truth": [list(pair) for pair in truth], "lists": summary,
        "wall_seconds": wall, **meta,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=1))
    print(json.dumps({k: (v["size"], v["tp"]) for k, v in summary.items()}), f"{wall:.0f}s", flush=True)


if __name__ == "__main__":
    main()
