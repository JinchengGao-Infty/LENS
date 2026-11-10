"""LENS on the credit datasets (HELOC, UCI default of credit card clients).

The binary counterpart of ``fremtpl2_poisson_allpair``: a logistic XGBoost
Teacher, all pairs, one BY over all pairs.  The Teacher margin is the logit,
the scale on which a logistic GLM is additive.  Per split it records the
replication statistics, the lists of the LENS rule (family-RMS plus the
kappa scale floor), the family-RMS rule alone and the pair-relative rule, the
Bernoulli log-likelihood gain of each frozen discovery table over an additive
baseline, and the outer-test log loss of the additive baseline plus jointly
refitted corrections for several lists.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit

# Large arrays with transparent huge pages stall in kernel compaction on a
# fragmented shared machine; the package import has already loaded NumPy.
np._core.multiarray._set_madvise_hugepage(False)

from experiments.real_data_loader import load_dataset
from lens.experiments._real_helpers import all_pairs, make_outer_inner_split, _by_graph
from lens.experiments.fremtpl2_poisson_allpair import BETA_MAX, KAPPA, TEACHER, grid_index, mean_test
from lens.grids import PairGrid, build_grid
from lens.interaction_vectors import compute_interaction_sample
from lens.model_inference import (
    family_rms_floor_amplitude_thresholds,
    normal_replication_pvalues_at_pair_thresholds,
    normal_replication_pvalues_at_threshold,
    run_replication_inference_streaming,
)

ADDITIVE = {**TEACHER, "n_estimators": 3000, "max_depth": 1}


class LogisticTeacher:
    """Logistic XGBoost, early stopping on 10% of its own rows."""

    def __init__(self, X, y, *, seed: int, params: dict = TEACHER):
        from xgboost import XGBClassifier

        params = {k: v for k, v in params.items() if k != "max_cat_to_onehot"}
        order = np.random.default_rng(seed).permutation(len(y))
        fit, stop = order[: int(0.9 * len(order))], order[int(0.9 * len(order)):]
        model = XGBClassifier(objective="binary:logistic", device="cuda", random_state=seed, **params)
        model.fit(X[fit], y[fit], eval_set=[(X[stop], y[stop])], verbose=False)
        self.n_trees = int(model.best_iteration) + 1
        self.booster = model.get_booster()[: self.n_trees]

    def predict_margin(self, X):
        import cupy as cp

        return cp.asnumpy(self.booster.inplace_predict(cp.asarray(X), predict_type="margin"))


def mean_log_loss(y, eta) -> float:
    return float(np.mean(np.logaddexp(0.0, eta) - y * eta))


def table_lookup(table, pair_grid, X, pair):
    """Frozen residual table as a function of the two features; 0 for missing or unseen values."""
    columns = X[:, pair[0]], X[:, pair[1]]
    a = grid_index(pair_grid.left, columns[0])
    b = grid_index(pair_grid.right, columns[1])
    psi = table[np.clip(a, 0, table.shape[0] - 1), np.clip(b, 0, table.shape[1] - 1)]
    missing = (a < 0) | (b < 0) | np.isnan(columns[0]) | np.isnan(columns[1])
    return np.where(missing, 0.0, psi)


def fit_beta(y, eta0, psi) -> float:
    """argmin over [0, BETA_MAX] of the mean logistic loss of eta0 + beta psi."""

    def derivative(beta):
        return float(np.mean((expit(eta0 + beta * psi) - y) * psi))

    if derivative(0.0) >= 0:
        return 0.0
    if derivative(BETA_MAX) <= 0:
        return BETA_MAX
    return float(brentq(derivative, 0.0, BETA_MAX, xtol=1e-12))


def fit_joint(y, eta0, Psi, *, ridge: float = 1e-8, iterations: int = 50):
    """Logistic GLM on fixed columns Psi with offset eta0 (Newton, step halving)."""
    beta = np.zeros(Psi.shape[1])

    def objective(b):
        return mean_log_loss(y, eta0 + Psi @ b) + 0.5 * ridge * b @ b

    current = objective(beta)
    for _ in range(iterations):
        prob = expit(eta0 + Psi @ beta)
        gradient = Psi.T @ (prob - y) / len(y) + ridge * beta
        hessian = (Psi * (prob * (1 - prob))[:, None]).T @ Psi / len(y) + ridge * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        scale = 1.0
        while objective(beta - scale * step) > current and scale > 1e-6:
            scale /= 2
        beta = beta - scale * step
        new = objective(beta)
        if current - new < 1e-12:
            current = new
            break
        current = new
    return beta


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("heloc", "uci_credit"), required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ranking-max-samples", type=int, default=3000)
    parser.add_argument("--grid-points", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed = args.split_seed
    destination = args.output / f"split{seed}.json"
    started = time.perf_counter()

    data = load_dataset(args.dataset, args.data_dir)
    X, y = np.asarray(data["X"], dtype=np.float64), np.asarray(data["y"], dtype=np.float64)
    names, kinds = list(data["feature_names"]), list(data["feature_kinds"])
    outer_train, outer_test, inner = make_outer_inner_split(y, task="binary", seed=seed)
    rng = np.random.default_rng(5000 + seed)
    d = rng.permutation(outer_train[inner.discovery])
    c = rng.permutation(outer_train[inner.confirmation])
    e = outer_train[inner.evaluation]
    t = outer_test

    timings = {}
    t0 = time.perf_counter()
    discovery = LogisticTeacher(X[d], y[d], seed=2 * seed + 1)
    contexts = X[d][np.random.default_rng(2 * seed + 4).choice(len(d), min(args.ranking_max_samples, len(d)), replace=False)]
    grids = [build_grid(X[d][:, j], kinds[j], n_points=args.grid_points) for j in range(len(names))]
    pairs = all_pairs(len(names))
    pair_grids = {pair: PairGrid(grids[pair[0]], grids[pair[1]]) for pair in pairs}
    means, tables, energy = {}, {}, {}
    for pair in pairs:
        sample = compute_interaction_sample(discovery.predict_margin, contexts, pair, pair_grids[pair],
                                            estimate_covariance="none")
        means[pair] = sample.mean.copy()
        tables[pair] = sample.mean_surface.copy()
        energy[pair] = float(sample.raw_energy)
    ordered = tuple(sorted(pairs, key=lambda pair: (-energy[pair], pair)))
    timings["discovery_seconds"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    confirmation = LogisticTeacher(X[c], y[c], seed=2 * seed + 2)
    X_e = X[e]
    replication = run_replication_inference_streaming(
        means,
        lambda pair: compute_interaction_sample(confirmation.predict_margin, X_e, pair, pair_grids[pair],
                                                estimate_covariance="none"),
        ordered, n_eval=len(X_e), minimum_amplitude=args.eta, n_bootstrap=1, seed=2 * seed + 3,
    )
    _, relative_p = normal_replication_pvalues_at_threshold(replication, args.eta)
    rms_norm, rms_thresholds = family_rms_floor_amplitude_thresholds(replication, args.eta)
    _, rms_p = normal_replication_pvalues_at_pair_thresholds(replication, rms_thresholds)
    sd_f = float(np.std(discovery.predict_margin(contexts)))
    floor_thresholds = {pair: max(rms_thresholds[pair], KAPPA * sd_f / np.sqrt(max(energy[pair], 1e-12)))
                        for pair in pairs}
    _, floor_p = normal_replication_pvalues_at_pair_thresholds(replication, floor_thresholds)
    relative = set(_by_graph(ordered, relative_p, alpha=args.alpha))
    rms = set(_by_graph(ordered, rms_p, alpha=args.alpha))
    floor = set(_by_graph(ordered, floor_p, alpha=args.alpha))
    timings["replication_seconds"] = time.perf_counter() - t0

    # Log-likelihood gain of each frozen discovery table over an additive baseline.
    t0 = time.perf_counter()
    additive = LogisticTeacher(X[d], y[d], seed=2 * seed + 5, params=ADDITIVE)
    eta0 = {name: additive.predict_margin(X[idx]) for name, idx in (("c", c), ("e", e), ("t", t))}
    psi = {pair: {name: table_lookup(tables[pair], pair_grids[pair], X[idx], pair)
                  for name, idx in (("c", c), ("e", e), ("t", t))} for pair in pairs}
    beta, gain = {}, {}
    for pair in pairs:
        beta[pair] = fit_beta(y[c], eta0["c"], psi[pair]["c"])
        g = beta[pair] * psi[pair]["e"]
        gain[pair] = mean_test(y[e] * g - np.logaddexp(0.0, eta0["e"] + g) + np.logaddexp(0.0, eta0["e"]))
    gain_p = {pair: gain[pair][2] for pair in pairs}
    gain_list = set(_by_graph(ordered, gain_p, alpha=args.alpha))
    timings["utility_seconds"] = time.perf_counter() - t0

    prevalence = float(y[c].mean())
    log_loss = {
        "intercept_only": mean_log_loss(y[t], np.full(len(t), np.log(prevalence / (1 - prevalence)))),
        "additive_f0": mean_log_loss(y[t], eta0["t"]),
        "discovery_teacher": mean_log_loss(y[t], discovery.predict_margin(X[t])),
        "confirmation_teacher": mean_log_loss(y[t], confirmation.predict_margin(X[t])),
    }
    lists = {
        "lens": [pair for pair in ordered if pair in floor],
        "family_rms": [pair for pair in ordered if pair in rms],
        "pair_relative": [pair for pair in ordered if pair in relative],
        "gain": [pair for pair in ordered if pair in gain_list],
        "top10": list(ordered[:10]),
        "all": list(ordered),
        # Single-model ablation: the same floors on the discovery strength, without the second model.
        "single_model": [pair for pair in ordered
                         if np.sqrt(energy[pair]) > max(args.eta * rms_norm, KAPPA * sd_f)],
        # Energy ranking cut at the length of the LENS list.
        "top_lens": list(ordered[:len(floor)]),
        # Both models strong on their own, no alignment or test.
        "both_strong": [pair for pair in ordered
                        if min(np.sqrt(energy[pair]), np.sqrt(replication.confirmation_energies[pair]))
                        > max(args.eta * rms_norm, KAPPA * sd_f)],
        # Reproduced strength above the LENS floor, without the test.
        "projection_floor": [pair for pair in ordered
                             if replication.amplitude_ratios[pair] > floor_thresholds[pair]],
    }
    for name, chosen in lists.items():
        if not chosen:
            log_loss[f"f0_plus_{name}"] = log_loss["additive_f0"]
            continue
        coefficients = fit_joint(y[c], eta0["c"], np.column_stack([psi[pair]["c"] for pair in chosen]))
        eta_t = np.column_stack([psi[pair]["t"] for pair in chosen]) @ coefficients
        log_loss[f"f0_plus_{name}"] = mean_log_loss(y[t], eta0["t"] + eta_t)

    rows = []
    for rank, pair in enumerate(ordered, start=1):
        a = float(np.sqrt(energy[pair]))
        rows.append({
            "pair": list(pair),
            "names": [names[pair[0]], names[pair[1]]],
            "rank": rank,
            "amplitude": a,
            "projection": a * replication.amplitude_ratios[pair],
            "projection_se": a * replication.standard_errors[pair],
            "cosine": replication.cosine_similarities[pair],
            "lens_pvalue": floor_p[pair],
            "lens_selected": pair in floor,
            "family_rms_selected": pair in rms,
            "pair_relative_selected": pair in relative,
            "beta": beta[pair],
            "gain_mean": gain[pair][0],
            "gain_se": gain[pair][1],
            "gain_pvalue": gain_p[pair],
            "gain_selected": pair in gain_list,
        })
    payload = {
        "dataset": args.dataset,
        "split_seed": seed,
        "part_sizes": {"discovery": len(d), "confirmation": len(c), "evaluation": len(e), "outer_test": len(t)},
        "eta": args.eta,
        "alpha": args.alpha,
        "kappa": KAPPA,
        "n_trees": {"discovery": discovery.n_trees, "confirmation": confirmation.n_trees, "additive": additive.n_trees},
        "family_rms_norm": rms_norm,
        "discovery_margin_sd": sd_f,
        "lists": {name: [[names[a], names[b]] for a, b in chosen] for name, chosen in lists.items()},
        "outer_test_mean_log_loss": log_loss,
        "pairs": rows,
        "timings": {**timings, "wall_seconds": time.perf_counter() - started},
    }
    tmp = destination.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(destination)
    print(json.dumps({
        "dataset": args.dataset, "split": seed,
        "sizes": {name: len(chosen) for name, chosen in lists.items()},
        "log_loss": {k: round(v, 5) for k, v in log_loss.items()},
        "wall": round(payload["timings"]["wall_seconds"]),
    }), flush=True)


if __name__ == "__main__":
    main()
