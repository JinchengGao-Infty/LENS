#!/usr/bin/env python3
"""Aggregate the complete XGBoost/Softplus Stage-1 teacher-family submatrix.

The parent formal contract registered three teacher families.  Its XGBoost and
Softplus MLP tracks completed as a deliberately labelled two-family partial
matrix after the original RBF track was blocked.  This script does not promote
that parent matrix to a three-family ALL_DONE result; it validates and
aggregates only the two complete Stage-1 ranking tracks used by the paper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np


SCHEMA_VERSION = "lens-teacher-family-stage1-subset-aggregate-1"
FORMAL_COMMIT = "0edf45f500500a6043021de66cf698488d675cca"
TEACHERS = ("softplus_mlp", "xgboost")
DGPS = ("Additive", "Friedman1", "Smooth", *(f"F{i}" for i in range(1, 11)))
SEEDS = tuple(range(32))
EXPECTED_RANKERS = {
    "softplus_mlp": (
        "LENS-raw",
        "FAST",
        "NID-weight-same-MLP",
        "Integrated-Hessians-same-MLP",
    ),
    "xgboost": ("LENS-raw", "FAST", "TreeSHAP"),
}
STAGE1_METRICS = ("recall_at_k", "weighted_recall_at_k", "average_precision")


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("aggregate values are empty or non-finite")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(root: Path) -> tuple[dict, list[str]]:
    marker = root.parent / "PARTIAL_DONE_softplus_mlp_xgboost"
    tasks = root.parent / "tasks_softplus_mlp_xgboost.tsv"
    if not marker.is_file() or not tasks.is_file():
        raise FileNotFoundError("two-family completion marker or task ledger is missing")

    rows: dict[str, dict[str, list[dict]]] = {
        teacher: {dgp: [] for dgp in DGPS} for teacher in TEACHERS
    }
    manifest: list[str] = []
    for teacher in TEACHERS:
        for dgp in DGPS:
            for seed in SEEDS:
                shard = root / f"teacher={teacher}" / f"dgp={dgp}" / f"seed={seed}"
                done = shard / "DONE"
                config_path = shard / "run_config.json"
                summary_path = shard / "summary_metrics.json"
                failed = shard / "FAILED_RUN.json"
                if failed.exists() or not all(
                    path.is_file() for path in (done, config_path, summary_path)
                ):
                    raise FileNotFoundError(f"incomplete teacher-family shard: {shard}")

                config = json.loads(config_path.read_text())
                summary = json.loads(summary_path.read_text())
                expected_config = {
                    "teacher": teacher,
                    "dgp": dgp,
                    "seed": seed,
                    "n": 10000,
                    "top_k": 15,
                    "grid_points": 12,
                    "ranking_max_samples": 512,
                    "n_bootstrap": 999,
                    "alpha": 0.05,
                    "device": "cuda",
                    "mlp_max_epochs": 200,
                    "mlp_min_epochs": 30,
                    "mlp_patience": 15,
                    "ih_n_explain": 64,
                    "skip_fast": False,
                    "skip_family_baselines": False,
                }
                for key, expected in expected_config.items():
                    if config.get(key) != expected:
                        raise ValueError(
                            f"config mismatch {key} at {shard}: "
                            f"{config.get(key)!r} != {expected!r}"
                        )
                if config.get("git_commit") != FORMAL_COMMIT:
                    raise ValueError(f"unexpected formal commit at {shard}")
                if (
                    summary.get("schema_version")
                    != "lens-teacher-family-sensitivity-cell-1"
                    or summary.get("status") != "DONE"
                    or summary.get("teacher") != teacher
                    or summary.get("dgp") != dgp
                    or summary.get("seed") != seed
                    or summary.get("candidate_budget") != 15
                    or summary.get("ranking_sample_size") != 512
                    or tuple(summary.get("rankers", ())) != EXPECTED_RANKERS[teacher]
                    or set(summary.get("methods", {}))
                    != set(EXPECTED_RANKERS[teacher])
                ):
                    raise ValueError(f"cell identity or Stage-1 contract mismatch at {shard}")
                if teacher == "softplus_mlp":
                    for role in ("discovery", "confirmation"):
                        metadata = summary["teacher_fit_metadata"][role]
                        if (
                            metadata.get("activation") != "Softplus"
                            or metadata.get("device") != "cuda"
                            or metadata.get("silent_fallback") is not False
                        ):
                            raise ValueError(f"MLP provenance mismatch at {shard}")
                for method in EXPECTED_RANKERS[teacher]:
                    stage1 = summary["methods"][method]["stage1"]
                    for metric in STAGE1_METRICS:
                        raw_value = stage1[metric]
                        if dgp == "Additive" and raw_value is None:
                            continue
                        value = float(raw_value)
                        if not np.isfinite(value):
                            raise ValueError(
                                f"non-finite {method} {metric} at {shard}"
                            )
                rows[teacher][dgp].append(summary)
                for path in (done, config_path, summary_path):
                    relative = path.relative_to(root.parent)
                    manifest.append(f"{_sha256(path)}  {relative.as_posix()}")

    per_teacher: dict[str, object] = {}
    for teacher in TEACHERS:
        nonnull = [
            row
            for dgp in DGPS
            if dgp != "Additive"
            for row in rows[teacher][dgp]
        ]
        methods: dict[str, object] = {}
        for method in EXPECTED_RANKERS[teacher]:
            methods[method] = {
                metric: _summary(
                    row["methods"][method]["stage1"][metric] for row in nonnull
                )
                for metric in STAGE1_METRICS
            }
        per_dgp: dict[str, object] = {}
        for dgp in DGPS:
            if dgp == "Additive":
                continue
            per_dgp[dgp] = {
                method: {
                    metric: _summary(
                        row["methods"][method]["stage1"][metric]
                        for row in rows[teacher][dgp]
                    )
                    for metric in STAGE1_METRICS
                }
                for method in EXPECTED_RANKERS[teacher]
            }
        per_teacher[teacher] = {
            "completed_cells": len(DGPS) * len(SEEDS),
            "rankers": list(EXPECTED_RANKERS[teacher]),
            "teacher_fidelity_nonnull": {
                role: {
                    metric: _summary(
                        row["teacher_fidelity"][role][metric] for row in nonnull
                    )
                    for metric in ("observed_y_r2", "signal_r2", "signal_rmse")
                }
                for role in ("discovery", "confirmation")
            },
            "macro_non_null": methods,
            "per_dgp": per_dgp,
        }

    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "DONE",
        "git_commit": FORMAL_COMMIT,
        "parent_matrix_status": "PARTIAL_DONE_softplus_mlp_xgboost",
        "parent_contract_expected_teachers": [
            "softplus_mlp",
            "rbf_nystroem",
            "xgboost",
        ],
        "subset_teachers": list(TEACHERS),
        "expected_cells": len(TEACHERS) * len(DGPS) * len(SEEDS),
        "completed_cells": len(TEACHERS) * len(DGPS) * len(SEEDS),
        "dgps": list(DGPS),
        "n_non_null_dgps": len(DGPS) - 1,
        "seeds": list(SEEDS),
        "candidate_budget": 15,
        "ranking_sample_size": 512,
        "source_task_ledger_sha256": _sha256(tasks),
        "source_completion_marker_sha256": _sha256(marker),
        "per_teacher": per_teacher,
        "claim_boundary": (
            "Complete two-family Stage-1 ranking sensitivity only. The parent "
            "three-family matrix remains intentionally without ALL_DONE; no "
            "Stage-2 multiplier result is promoted here, and this sensitivity "
            "must not be pooled with the B=500 LENS-Stable confirmation table."
        ),
    }
    return payload, sorted(manifest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    payload, manifest = aggregate(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.manifest.write_text("\n".join(manifest) + "\n")
    print(
        "TEACHER_FAMILY_STAGE1_SUBSET_DONE "
        f"cells={payload['completed_cells']} output={args.output}"
    )


if __name__ == "__main__":
    main()
