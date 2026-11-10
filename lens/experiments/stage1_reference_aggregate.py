"""Derive the common-protocol Stage-1 reference table from accepted v2 shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


RANKERS = (
    "LENS-raw",
    "LENS-debiased",
    "Friedman-H2",
    "TreeSHAP",
    "FAST",
    "NID",
    "Random",
)
METRIC_MAP = {
    "recall_at_k": "power",
    "weighted_recall_at_k": "weighted_power",
    "false_discovery_proportion_at_k": "fdp",
    "true_positives_at_k": "true_positives",
    "false_positives_at_k": "false_positives",
}


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires finite values")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def aggregate(root: Path, dgps: tuple[str, ...], seeds: tuple[str, ...]) -> dict:
    rows: list[dict] = []
    for dgp in dgps:
        for seed in seeds:
            cell = root / f"dgp={dgp}" / f"seed={seed}"
            if not (cell / "DONE").is_file() or not (cell / "summary_metrics.json").is_file():
                raise FileNotFoundError(f"incomplete accepted Stage-1 reference cell: {cell}")
            payload = json.loads((cell / "summary_metrics.json").read_text())
            if payload.get("dgp") != dgp or payload.get("seed") != int(seed):
                raise ValueError(f"identity mismatch at {cell}")
            if payload.get("candidate_budget_per_ranker") != 15:
                raise ValueError(f"candidate budget mismatch at {cell}")
            if set(payload.get("methods", {})) != set(RANKERS):
                raise ValueError(f"ranker set mismatch at {cell}")
            rows.append(payload)

    per_dgp: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for dgp in dgps:
        selected = [row for row in rows if row["dgp"] == dgp]
        per_dgp[dgp] = {}
        for ranker in RANKERS:
            per_dgp[dgp][ranker] = {}
            for output_metric, source_metric in METRIC_MAP.items():
                values = [
                    row["methods"][ranker]["candidate_metrics"].get(source_metric)
                    for row in selected
                ]
                finite = [float(value) for value in values if value is not None]
                if finite:
                    per_dgp[dgp][ranker][output_metric] = _summary(finite)

    nonnull = [dgp for dgp in dgps if dgp != "Additive"]
    macro: dict[str, dict[str, dict[str, float | int]]] = {}
    for ranker in RANKERS:
        macro[ranker] = {}
        for metric in METRIC_MAP:
            values = [
                float(per_dgp[dgp][ranker][metric]["mean"])
                for dgp in nonnull
                if metric in per_dgp[dgp][ranker]
            ]
            if values:
                macro[ranker][metric] = _summary(values)

    return {
        "schema_version": "lens-v2-stage1-reference-aggregate-1",
        "status": "DONE",
        "source_root": str(root),
        "expected_cells": len(dgps) * len(seeds),
        "completed_cells": len(rows),
        "candidate_budget": 15,
        "rankers": list(RANKERS),
        "dgps": list(dgps),
        "seeds": [int(seed) for seed in seeds],
        "per_dgp": per_dgp,
        "macro_over_nonnull_dgps": macro,
        "claim_boundary": (
            "Stage-1 candidate ranking derived from accepted v2 shards; "
            "no confirmation decisions enter these metrics."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dgps", required=True, type=_parse_csv)
    parser.add_argument("--seeds", required=True, type=_parse_csv)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    payload = aggregate(args.root, args.dgps, args.seeds)
    _atomic_json(args.output, payload)
    print(
        f"STAGE1_REFERENCE_AGGREGATE_DONE cells={payload['completed_cells']} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
