"""LENS on freMTPL2 under the actuarial frequency setting.

Poisson XGBoost Teacher on claim counts with a log(Exposure) offset, all 36
pairs, one BY over all pairs.  The Teacher margin without the offset is the
log claim rate per policy-year, the scale on which a Poisson GLM is additive.
VehBrand, VehGas, Area and Region use XGBoost's native categorical splits.

ClaimNb is capped at 4 and Exposure at 1, as in the actuarial literature on
this dataset.

Per split the script records
  * the replication statistics in amplitude units (a_u = ||r_D||,
    mu_u = projection of the confirmation table on the unit discovery
    direction) and the lists of the family-RMS and pair-relative rules;
  * the Poisson loss gain of each pair: an additive baseline f0 is fitted on
    D, the discovery residual table psi_u is frozen as a lookup function, its
    coefficient beta_u >= 0 is fitted on C, and the per-row gain
    y g - mu0 (exp(g) - 1), g = beta_u psi_u, is tested on E;
  * the outer-test deviance of f0 plus jointly refitted corrections for
    several lists of pairs.
``--train-size`` keeps only that many rows of D and C (E is unchanged), for
the training-size study.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

# Large arrays with transparent huge pages stall in kernel compaction on a
# fragmented shared machine; the package import has already loaded NumPy.
np._core.multiarray._set_madvise_hugepage(False)

from lens.experiments._real_helpers import all_pairs, make_outer_inner_split, _by_graph
from lens.grids import PairGrid, build_grid
from lens.interaction_vectors import compute_interaction_sample
from lens.model_inference import (
    family_rms_floor_amplitude_thresholds,
    normal_replication_pvalues_at_pair_thresholds,
    normal_replication_pvalues_at_threshold,
    run_replication_inference_streaming,
)

FEATURES = (
    "VehPower", "VehAge", "DrivAge", "BonusMalus", "VehBrand",
    "VehGas", "Area", "Density", "Region",
)
CATEGORICAL = {"VehBrand", "VehGas", "Area", "Region"}
TEACHER = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "max_depth": 6,
    "tree_method": "hist",
    "early_stopping_rounds": 50,
    "max_cat_to_onehot": 1,
}
ADDITIVE = {**TEACHER, "n_estimators": 3000, "max_depth": 1}
BETA_MAX = 2.0
KAPPA = 0.02  # scale floor: the reproduced strength must exceed KAPPA * sd of the discovery margin


def load_fremtpl2_poisson() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_openml

    frame = fetch_openml(data_id=41214, as_frame=True, parser="auto").frame
    columns = []
    for name in FEATURES:
        column = frame[name]
        if name in CATEGORICAL:
            column = column.astype("category").cat.codes
        columns.append(column.to_numpy(dtype=np.float64))
    X = np.column_stack(columns)
    claims = np.minimum(frame["ClaimNb"].to_numpy(dtype=np.float64), 4.0)
    exposure = np.minimum(frame["Exposure"].to_numpy(dtype=np.float64), 1.0)
    return X, claims, exposure


class PoissonTeacher:
    """Poisson XGBoost with offset, early stopping on 10% of its own rows."""

    def __init__(self, X, claims, exposure, *, seed: int, params: dict = TEACHER):
        from xgboost import XGBRegressor

        order = np.random.default_rng(seed).permutation(len(claims))
        fit, stop = order[: int(0.9 * len(order))], order[int(0.9 * len(order)):]
        model = XGBRegressor(
            objective="count:poisson", device="cuda", random_state=seed,
            enable_categorical=True,
            feature_types=["c" if name in CATEGORICAL else "q" for name in FEATURES],
            **params,
        )
        model.fit(
            X[fit], claims[fit], base_margin=np.log(exposure[fit]),
            eval_set=[(X[stop], claims[stop])],
            base_margin_eval_set=[np.log(exposure[stop])], verbose=False,
        )
        self.n_trees = int(model.best_iteration) + 1
        self.booster = model.get_booster()[: self.n_trees]

    def predict_margin(self, X):
        import cupy as cp

        return cp.asnumpy(self.booster.inplace_predict(cp.asarray(X), predict_type="margin"))

    def log_mu(self, X, exposure):
        import cupy as cp

        return cp.asnumpy(self.booster.inplace_predict(
            cp.asarray(X), predict_type="margin", base_margin=cp.asarray(np.log(exposure))
        ))


def mean_deviance(claims, log_mu) -> float:
    mu = np.exp(log_mu)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(claims > 0, claims * np.log(claims / mu), 0.0)
    return float(2.0 * np.mean(term - (claims - mu)))


def grid_index(grid, values):
    """Map feature values to grid cells: nearest point, or exact level (-1 if unseen)."""
    if grid.kind == "categorical":
        lookup = {float(v): i for i, v in enumerate(grid.values)}
        return np.asarray([lookup.get(float(v), -1) for v in values])
    mids = (grid.values[1:] + grid.values[:-1]) / 2
    return np.searchsorted(mids, values)


def table_lookup(table, pair_grid, X, pair):
    a = grid_index(pair_grid.left, X[:, pair[0]])
    b = grid_index(pair_grid.right, X[:, pair[1]])
    psi = table[np.maximum(a, 0), np.maximum(b, 0)]
    return np.where((a < 0) | (b < 0), 0.0, psi)


def fit_beta(y, log_mu0, psi) -> float:
    """argmin over [0, BETA_MAX] of mean(mu0 exp(beta psi) - y beta psi)."""
    mu0 = np.exp(log_mu0)

    def derivative(beta):
        return float(np.mean((mu0 - y) * psi + mu0 * psi * np.expm1(beta * psi)))

    if derivative(0.0) >= 0:
        return 0.0
    if derivative(BETA_MAX) <= 0:
        return BETA_MAX
    return float(brentq(derivative, 0.0, BETA_MAX, xtol=1e-12))


def mean_test(values) -> tuple[float, float, float]:
    mean = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    if se == 0:
        return mean, se, 0.0 if mean > 0 else 1.0
    return mean, se, float(norm.sf(mean / se))


def fit_joint(y, log_mu0, Psi, *, ridge: float = 1e-8, iterations: int = 50):
    """Poisson GLM on fixed columns Psi with offset log_mu0 (Newton, step halving)."""
    beta = np.zeros(Psi.shape[1])

    def objective(b):
        eta = Psi @ b
        return float(np.mean(np.exp(log_mu0 + eta) - y * eta) + 0.5 * ridge * b @ b)

    current = objective(beta)
    for _ in range(iterations):
        mu = np.exp(log_mu0 + Psi @ beta)
        gradient = Psi.T @ (mu - y) / len(y) + ridge * beta
        hessian = (Psi * mu[:, None]).T @ Psi / len(y) + ridge * np.eye(len(beta))
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
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-size", type=int, help="rows kept in each of D and C")
    parser.add_argument("--eta", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--ranking-max-samples", type=int, default=3000)
    parser.add_argument("--grid-points", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    seed = args.split_seed
    destination = args.output / f"split{seed}.json"
    started = time.perf_counter()

    X, claims, exposure = load_fremtpl2_poisson()
    kinds = ["categorical" if name in CATEGORICAL else "continuous" for name in FEATURES]
    outer_train, outer_test, inner = make_outer_inner_split(claims, task="regression", seed=seed)
    rng = np.random.default_rng(5000 + seed)
    d = rng.permutation(outer_train[inner.discovery])
    c = rng.permutation(outer_train[inner.confirmation])
    if args.train_size:
        d, c = d[: args.train_size], c[: args.train_size]
    e = outer_train[inner.evaluation]
    t = outer_test

    timings = {}
    t0 = time.perf_counter()
    discovery = PoissonTeacher(X[d], claims[d], exposure[d], seed=2 * seed + 1)
    contexts = X[d][np.random.default_rng(2 * seed + 4).choice(len(d), min(args.ranking_max_samples, len(d)), replace=False)]
    grids = [build_grid(X[d][:, j], kinds[j], n_points=args.grid_points) for j in range(len(FEATURES))]
    pairs = all_pairs(len(FEATURES))
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
    confirmation = PoissonTeacher(X[c], claims[c], exposure[c], seed=2 * seed + 2)
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
    relative = set(_by_graph(ordered, relative_p, alpha=args.alpha))
    rms = set(_by_graph(ordered, rms_p, alpha=args.alpha))
    sd_f = float(np.std(discovery.predict_margin(contexts)))
    floor_thresholds = {pair: max(rms_thresholds[pair], KAPPA * sd_f / np.sqrt(max(energy[pair], 1e-12)))
                        for pair in pairs}
    _, floor_p = normal_replication_pvalues_at_pair_thresholds(replication, floor_thresholds)
    floor = set(_by_graph(ordered, floor_p, alpha=args.alpha))
    timings["replication_seconds"] = time.perf_counter() - t0

    # Poisson loss gain of each frozen discovery table, relative to an additive baseline.
    t0 = time.perf_counter()
    additive = PoissonTeacher(X[d], claims[d], exposure[d], seed=2 * seed + 5, params=ADDITIVE)
    log_mu0 = {name: additive.log_mu(X[idx], exposure[idx]) for name, idx in (("c", c), ("e", e), ("t", t))}
    psi = {pair: {name: table_lookup(tables[pair], pair_grids[pair], X[idx], pair)
                  for name, idx in (("c", c), ("e", e), ("t", t))} for pair in pairs}
    beta, gain = {}, {}
    for pair in pairs:
        beta[pair] = fit_beta(claims[c], log_mu0["c"], psi[pair]["c"])
        g = beta[pair] * psi[pair]["e"]
        gain[pair] = mean_test(claims[e] * g - np.exp(log_mu0["e"]) * np.expm1(g))
    gain_p = {pair: gain[pair][2] for pair in pairs}
    gain_list = set(_by_graph(ordered, gain_p, alpha=args.alpha))
    joint_p = {pair: max(rms_p[pair], gain_p[pair]) for pair in pairs}
    joint_list = set(_by_graph(ordered, joint_p, alpha=args.alpha))
    timings["utility_seconds"] = time.perf_counter() - t0

    # Outer-test deviance of f0 plus jointly refitted corrections for several lists.
    deviance = {
        "intercept_only": mean_deviance(
            claims[t], np.log(claims[c].sum() / exposure[c].sum() * exposure[t])),
        "additive_f0": mean_deviance(claims[t], log_mu0["t"]),
        "discovery_teacher": mean_deviance(claims[t], discovery.log_mu(X[t], exposure[t])),
        "confirmation_teacher": mean_deviance(claims[t], confirmation.log_mu(X[t], exposure[t])),
    }
    lists = {
        "lens": [pair for pair in ordered if pair in floor],
        "family_rms": [pair for pair in ordered if pair in rms],
        "pair_relative": [pair for pair in ordered if pair in relative],
        "gain": [pair for pair in ordered if pair in gain_list],
        "rms_and_gain": [pair for pair in ordered if pair in joint_list],
        "top7": list(ordered[:7]),
        "all36": list(ordered),
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
            deviance[f"f0_plus_{name}"] = deviance["additive_f0"]
            continue
        coefficients = fit_joint(claims[c], log_mu0["c"], np.column_stack([psi[pair]["c"] for pair in chosen]))
        eta_t = np.column_stack([psi[pair]["t"] for pair in chosen]) @ coefficients
        deviance[f"f0_plus_{name}"] = mean_deviance(claims[t], log_mu0["t"] + eta_t)

    rows = []
    for rank, pair in enumerate(ordered, start=1):
        a = float(np.sqrt(energy[pair]))
        ratio = replication.amplitude_ratios[pair]
        rows.append({
            "pair": list(pair),
            "names": [FEATURES[pair[0]], FEATURES[pair[1]]],
            "rank": rank,
            "discovery_energy": energy[pair],
            "amplitude": a,
            "projection": a * ratio,
            "projection_se": a * replication.standard_errors[pair],
            "replication_ratio": ratio,
            "cosine": replication.cosine_similarities[pair],
            "confirmation_energy": replication.confirmation_energies[pair],
            "pair_relative_pvalue": relative_p[pair],
            "pair_relative_selected": pair in relative,
            "family_rms_threshold": rms_thresholds[pair],
            "family_rms_pvalue": rms_p[pair],
            "family_rms_selected": pair in rms,
            "lens_pvalue": floor_p[pair],
            "lens_selected": pair in floor,
            "beta": beta[pair],
            "gain_mean": gain[pair][0],
            "gain_se": gain[pair][1],
            "gain_pvalue": gain_p[pair],
            "gain_selected": pair in gain_list,
            "rms_and_gain_selected": pair in joint_list,
        })
    payload = {
        "dataset": "fremtpl2_poisson_offset_native_categorical",
        "split_seed": seed,
        "train_size": args.train_size,
        "part_sizes": {"discovery": len(d), "confirmation": len(c), "evaluation": len(e), "outer_test": len(t)},
        "eta": args.eta,
        "alpha": args.alpha,
        "teacher_params": TEACHER,
        "additive_params": ADDITIVE,
        "n_trees": {"discovery": discovery.n_trees, "confirmation": confirmation.n_trees, "additive": additive.n_trees},
        "family_rms_norm": rms_norm,
        "discovery_margin_sd": sd_f,
        "kappa": KAPPA,
        "beta_max": BETA_MAX,
        "lists": {name: [[FEATURES[a], FEATURES[b]] for a, b in chosen] for name, chosen in lists.items()},
        "outer_test_mean_poisson_deviance": deviance,
        "pairs": rows,
        "timings": {**timings, "wall_seconds": time.perf_counter() - started},
    }
    tmp = destination.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(destination)
    print(json.dumps({
        "split": seed, "train_size": args.train_size,
        "sizes": {name: len(chosen) for name, chosen in lists.items()},
        "deviance": {k: round(v, 5) for k, v in deviance.items()},
        "wall": round(payload["timings"]["wall_seconds"]),
    }), flush=True)


if __name__ == "__main__":
    main()
