"""Prepare one discovery-only cell for the official legacy iRF runtime."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.dgps import generate
from lens.grids import build_grid
from lens.splits import make_three_way_split
from lens.synthetic_oracle import compute_run_matched_oracle


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dgp = generate(args.dgp, args.seed)
    split = make_three_way_split(len(dgp.y), seed=args.seed)
    discovery = np.asarray(split.discovery, dtype=np.int64)
    rng = np.random.default_rng(args.seed * 2 + 211)
    order = rng.permutation(discovery)
    cut = max(1, int(0.8 * len(order)))
    train, test = order[:cut], order[cut:]
    grids = {
        index: build_grid(dgp.X[discovery, index], "continuous", n_points=12)
        for index in range(dgp.X.shape[1])
    }
    truth = tuple(tuple(pair) for pair in dgp.true_pairs)
    oracle = compute_run_matched_oracle(
        args.dgp,
        dgp.X[split.evaluation],
        grids,
        truth,
        grid_batch_size=144,
    )
    masses = np.asarray([oracle.masses[pair] for pair in truth], dtype=np.float64)
    temporary = args.output / ".input.tmp.npz"
    np.savez_compressed(
        temporary,
        X_train=np.asarray(dgp.X[train], dtype=np.float64),
        y_train=np.asarray(dgp.y[train], dtype=np.float64),
        X_test=np.asarray(dgp.X[test], dtype=np.float64),
        y_test=np.asarray(dgp.y[test], dtype=np.float64),
        truth_pairs=np.asarray(truth, dtype=np.int64).reshape(-1, 2),
        truth_masses=masses,
        dgp=np.asarray(args.dgp),
        seed=np.asarray(args.seed, dtype=np.int64),
        discovery_sha256=np.asarray(
            hashlib.sha256(discovery.tobytes()).hexdigest()
        ),
    )
    temporary.replace(args.output / "input.npz")
    _atomic_json(
        args.output / "prepare_config.json",
        {
            "stage": 1,
            "method": "iRF-official",
            "dgp": args.dgp,
            "seed": args.seed,
            "discovery_only": True,
            "native_train_fraction_within_discovery": 0.8,
            "native_test_fraction_within_discovery": 0.2,
            "oracle_uses_evaluation_features_without_labels": True,
        },
    )


if __name__ == "__main__":
    main()
