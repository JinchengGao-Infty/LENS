"""Prepare auditable CSV inputs for official CRAN hstats and IADT cells."""

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


def _write_csv(path: Path, X: np.ndarray, y: np.ndarray | None = None) -> None:
    matrix = np.asarray(X, dtype=np.float64)
    names = [f"x{index}" for index in range(matrix.shape[1])]
    if y is not None:
        matrix = np.column_stack([matrix, np.asarray(y, dtype=np.float64)])
        names.append("y")
    temporary = path.with_name(f".{path.name}.tmp")
    np.savetxt(temporary, matrix, delimiter=",", header=",".join(names), comments="")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=("hstats", "IADT"))
    parser.add_argument("--dgp", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--stage2-root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    dgp = generate(args.dgp, args.seed)
    split = make_three_way_split(len(dgp.y), seed=args.seed)
    truth = tuple(tuple(pair) for pair in dgp.true_pairs)
    grids = {
        index: build_grid(dgp.X[split.discovery, index], "continuous", n_points=12)
        for index in range(dgp.X.shape[1])
    }
    oracle = compute_run_matched_oracle(
        args.dgp,
        dgp.X[split.evaluation],
        grids,
        truth,
        grid_batch_size=144,
    )
    meta: dict[str, object] = {
        "method": args.method,
        "dgp": args.dgp,
        "seed": args.seed,
        "n": len(dgp.y),
        "p": dgp.X.shape[1],
        "truth_pairs": [list(pair) for pair in truth],
        "truth_masses": [oracle.masses[pair] for pair in truth],
        "top_k": 15,
        "split_hashes": {
            "discovery": hashlib.sha256(
                np.asarray(split.discovery, dtype=np.int64).tobytes()
            ).hexdigest(),
            "confirmation": hashlib.sha256(
                np.asarray(split.confirmation, dtype=np.int64).tobytes()
            ).hexdigest(),
            "evaluation": hashlib.sha256(
                np.asarray(split.evaluation, dtype=np.int64).tobytes()
            ).hexdigest(),
        },
    }
    if args.method == "hstats":
        _write_csv(
            args.output / "train.csv",
            dgp.X[split.discovery],
            dgp.y[split.discovery],
        )
        _write_csv(args.output / "reference.csv", dgp.X[split.discovery])
        meta.update(
            {
                "stage": 1,
                "discovery_only": True,
                "hstats_n_max": 500,
                "pairwise_m": dgp.X.shape[1],
            }
        )
    else:
        if args.stage2_root is None:
            raise ValueError("IADT requires --stage2-root")
        stage2_summary = (
            args.stage2_root
            / f"dgp={args.dgp}"
            / f"seed={args.seed}"
            / "summary_metrics.json"
        )
        summary = json.loads(stage2_summary.read_text())
        candidates = [tuple(pair) for pair in summary["candidate_pairs"]]
        rng = np.random.default_rng(args.seed * 2 + 223)
        reference_indices = np.sort(
            rng.choice(split.confirmation, min(100, len(split.confirmation)), replace=False)
        )
        evaluation_indices = np.sort(
            rng.choice(split.evaluation, min(100, len(split.evaluation)), replace=False)
        )
        _write_csv(
            args.output / "train.csv",
            dgp.X[split.confirmation],
            dgp.y[split.confirmation],
        )
        _write_csv(args.output / "reference.csv", dgp.X[reference_indices])
        _write_csv(args.output / "evaluation.csv", dgp.X[evaluation_indices])
        meta.update(
            {
                "stage": 2,
                "candidate_pairs": [list(pair) for pair in candidates],
                "candidate_source": str(stage2_summary),
                "candidate_definition": summary["candidate_definition"],
                "candidate_digest": summary["candidates_sha256"],
                "reference_n": len(reference_indices),
                "evaluation_n": len(evaluation_indices),
                "native_null_scope": (
                    "set-level: the pair's covariates do not contribute interactions "
                    "with each other or with other covariates"
                ),
            }
        )
    _atomic_json(args.output / "input_meta.json", meta)


if __name__ == "__main__":
    main()
