"""Tables of every freMTPL2 pair for the overview figure.

Same split, models and grids as ``fremtpl2_poisson_allpair``.  Per pair: the
discovery model's partial-dependence surface F_D and residual table r_D on its
contexts, and the confirmation model's residual table averaged over evaluation
contexts (c_u, the mean of the tables h_C the test compares with r_D).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

np._core.multiarray._set_madvise_hugepage(False)

from lens.experiments._real_helpers import all_pairs, make_outer_inner_split
from lens.experiments.fremtpl2_poisson_allpair import (
    CATEGORICAL, FEATURES, PoissonTeacher, load_fremtpl2_poisson,
)
from lens.grids import PairGrid, build_grid
from lens.interaction_vectors import compute_interaction_sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-seed", type=int, default=30)
    parser.add_argument("--eval-rows", type=int, default=20000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed = args.split_seed

    X, claims, exposure = load_fremtpl2_poisson()
    kinds = ["categorical" if name in CATEGORICAL else "continuous" for name in FEATURES]
    outer_train, _, inner = make_outer_inner_split(claims, task="regression", seed=seed)
    rng = np.random.default_rng(5000 + seed)
    d = rng.permutation(outer_train[inner.discovery])
    c = rng.permutation(outer_train[inner.confirmation])
    e = outer_train[inner.evaluation][: args.eval_rows]

    discovery = PoissonTeacher(X[d], claims[d], exposure[d], seed=2 * seed + 1)
    confirmation = PoissonTeacher(X[c], claims[c], exposure[c], seed=2 * seed + 2)
    contexts = X[d][np.random.default_rng(2 * seed + 4).choice(len(d), 3000, replace=False)]
    grids = [build_grid(X[d][:, j], kinds[j], n_points=12) for j in range(len(FEATURES))]

    rows = []
    for pair in all_pairs(len(FEATURES)):
        grid = PairGrid(grids[pair[0]], grids[pair[1]])
        disc = compute_interaction_sample(discovery.predict_margin, contexts, pair, grid,
                                          estimate_covariance="none")
        conf = compute_interaction_sample(confirmation.predict_margin, X[e], pair, grid,
                                          estimate_covariance="none")
        rows.append({
            "pair": list(pair),
            "names": [FEATURES[pair[0]], FEATURES[pair[1]]],
            "levels": [grids[pair[0]].values.tolist(), grids[pair[1]].values.tolist()],
            "weights": [grids[pair[0]].weights.tolist(), grids[pair[1]].weights.tolist()],
            "amplitude": float(np.sqrt(disc.raw_energy)),
            "theta": float(disc.mean @ conf.mean / disc.raw_energy),
            "surface_D": disc.raw_mean_surface.tolist(),
            "residual_D": disc.mean_surface.tolist(),
            "residual_C": conf.mean_surface.tolist(),
        })
    payload = {
        "split_seed": seed,
        "discovery_margin_sd": float(np.std(discovery.predict_margin(contexts))),
        "pairs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload))


if __name__ == "__main__":
    main()
