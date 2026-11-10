"""Fail-closed aggregation for official iRF, R, and DIAMOND baselines."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


STAGE1_METRICS = ("recall_at_k", "weighted_recall_at_k", "average_precision")
DIAMOND_METRICS = (
    "uncalibrated_auc",
    "uncalibrated_fdr",
    "uncalibrated_power",
    "calibrated_auc",
    "calibrated_fdr",
    "calibrated_power",
)


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires at least one finite value")
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def _metric_summary(rows: list[dict], metric: str) -> dict[str, float | int] | None:
    values = [
        row.get(metric) if metric == "wall_seconds" else row["metrics"].get(metric)
        for row in rows
    ]
    # jsonlite serializes a NULL list element as an empty object under the
    # official R cell writer.  It is the on-disk representation of a missing
    # metric (for example, Stage-1 recall on the Additive DGP), not a numeric
    # observation.  Preserve fail-closed behavior for every non-empty object.
    normalized = [None if value == {} else value for value in values]
    invalid = [value for value in normalized if isinstance(value, (dict, list))]
    if invalid:
        raise TypeError(f"non-scalar metric {metric}: {invalid[0]!r}")
    finite = [float(value) for value in normalized if value is not None]
    return _summary(finite) if finite else None


def _require_cell(path: Path, *, stage: int, method: str) -> dict:
    if not (path / "DONE").is_file() or not (path / "result.json").is_file():
        raise FileNotFoundError(f"incomplete official baseline cell: {path}")
    if (path / "FAILED_RUN.json").exists():
        raise RuntimeError(f"failed official baseline cell: {path}")
    row = json.loads((path / "result.json").read_text())
    if row.get("status") != "DONE" or row.get("stage") != stage:
        raise ValueError(f"invalid status/stage in {path}")
    if row.get("method") != method:
        raise ValueError(f"method mismatch in {path}: {row.get('method')} != {method}")
    return row


def aggregate_irf(root: Path, dgps: tuple[str, ...], seeds: tuple[str, ...]) -> dict:
    rows = [
        _require_cell(
            root / f"dgp={dgp}" / f"seed={seed}", stage=1, method="iRF-official"
        )
        for dgp in dgps
        for seed in seeds
    ]
    per_dgp: dict[str, dict[str, dict[str, float | int]]] = {}
    for dgp in dgps:
        selected = [row for row in rows if row["dgp"] == dgp]
        per_dgp[dgp] = {
            metric: summary
            for metric in (*STAGE1_METRICS, "wall_seconds")
            if (summary := _metric_summary(selected, metric)) is not None
        }
    nonnull = [dgp for dgp in dgps if dgp != "Additive"]
    macro = {
        metric: _summary(
            per_dgp[dgp][metric]["mean"]
            for dgp in nonnull
            if metric in per_dgp[dgp]
        )
        for metric in STAGE1_METRICS
    }
    shifts = [
        float(row["method_config"]["constant_target_shift_for_nonnegative_RIT_weights"])
        for row in rows
    ]
    return {
        "schema_version": "lens-official-irf-aggregate-1",
        "status": "DONE",
        "stage": 1,
        "method": "iRF-official",
        "expected_cells": len(dgps) * len(seeds),
        "completed_cells": len(rows),
        "dgps": list(dgps),
        "seeds": [int(seed) for seed in seeds],
        "per_dgp": per_dgp,
        "macro_over_nonnull_dgps": macro,
        "compatibility_adapter_audit": {
            "cells_with_constant_target_shift": int(sum(shift > 0 for shift in shifts)),
            "target_shift": _summary(shifts),
            "regression_signature_shim": (
                "drop unused sample_weight_test=None passed by official run_iRF"
            ),
        },
        "claim_boundary": "Stage-1 native-model discovery ranking only.",
    }


def aggregate_r(root: Path, dgps: tuple[str, ...], seeds: tuple[str, ...]) -> dict:
    hstats: list[dict] = []
    for dgp in dgps:
        for seed in seeds:
            hstats.append(
                _require_cell(
                    root / "method=hstats" / f"dgp={dgp}" / f"seed={seed}",
                    stage=1,
                    method="hstats-official",
                )
            )

    def grouped(rows: list[dict], metrics: tuple[str, ...]) -> dict:
        output: dict[str, dict[str, dict[str, float | int]]] = {}
        for dgp in dgps:
            selected = [row for row in rows if row["dgp"] == dgp]
            output[dgp] = {
                metric: summary
                for metric in (*metrics, "wall_seconds")
                if (summary := _metric_summary(selected, metric)) is not None
            }
        return output

    hstats_per_dgp = grouped(hstats, STAGE1_METRICS)
    nonnull = [dgp for dgp in dgps if dgp != "Additive"]

    def macro(per_dgp: dict, metrics: tuple[str, ...]) -> dict:
        result = {}
        for metric in metrics:
            values = [
                per_dgp[dgp][metric]["mean"]
                for dgp in nonnull
                if metric in per_dgp[dgp]
            ]
            if values:
                result[metric] = _summary(values)
        return result

    return {
        "schema_version": "lens-official-r-baselines-aggregate-1",
        "status": "DONE",
        "expected_cells": len(dgps) * len(seeds),
        "completed_cells": len(hstats),
        "dgps": list(dgps),
        "seeds": [int(seed) for seed in seeds],
        "stage1": {
            "method": "hstats-official",
            "per_dgp": hstats_per_dgp,
            "macro_over_nonnull_dgps": macro(hstats_per_dgp, STAGE1_METRICS),
        },
    }


def aggregate_diamond(
    official_output: Path, dgps: tuple[str, ...], seeds: tuple[str, ...]
) -> dict:
    for dgp in dgps:
        func = int(dgp.removeprefix("F"))
        for seed in seeds:
            path = official_output / f"func{func}_seed{seed}.json"
            row = json.loads(path.read_text())
            if row.get("func_num") != func or row.get("seed") != int(seed):
                raise ValueError(f"DIAMOND identity mismatch in {path}")
            interactions = np.asarray(row["interactions"], dtype=np.float64)
            if interactions.shape != (60, 60) or not np.all(np.isfinite(interactions)):
                raise ValueError(f"invalid DIAMOND interactions in {path}")

    csv_path = official_output / "fdr_control" / "fdr_power.csv"
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = Counter(row["dataset"] for row in rows)
    if set(counts) != set(dgps) or any(counts[dgp] != 20 for dgp in dgps):
        raise ValueError(f"unexpected official DIAMOND FDR rows: {dict(counts)}")
    per_dgp: dict[str, dict[str, dict[str, float | int]]] = {}
    for dgp in dgps:
        selected = [row for row in rows if row["dataset"] == dgp]
        per_dgp[dgp] = {
            metric: _summary(float(row[metric]) for row in selected)
            for metric in DIAMOND_METRICS
        }
    macro = {
        metric: _summary(per_dgp[dgp][metric]["mean"] for dgp in dgps)
        for metric in DIAMOND_METRICS
    }
    if any(not math.isfinite(float(macro[m]["mean"])) for m in DIAMOND_METRICS):
        raise ValueError("non-finite DIAMOND macro metric")
    return {
        "schema_version": "lens-official-diamond-aggregate-1",
        "status": "DONE",
        "method": "DIAMOND-official",
        "source_cells": len(dgps) * len(seeds),
        "official_fdr_evaluation_rows": len(rows),
        "dgps": list(dgps),
        "seeds": [int(seed) for seed in seeds],
        "target_fdr": 0.2,
        "per_dgp": per_dgp,
        "macro_over_dgps": macro,
        "claim_boundary": (
            "End-to-end native DIAMOND simulation track; not a same-data Stage-2 test."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("irf", "r", "diamond"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--dgps", required=True, type=_parse_csv)
    parser.add_argument("--seeds", required=True, type=_parse_csv)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.kind == "irf":
        payload = aggregate_irf(args.root, args.dgps, args.seeds)
    elif args.kind == "r":
        payload = aggregate_r(args.root, args.dgps, args.seeds)
    else:
        payload = aggregate_diamond(args.root, args.dgps, args.seeds)
    _atomic_json(args.output, payload)
    print(
        f"OFFICIAL_SUPPLEMENTAL_AGGREGATE_DONE kind={args.kind} "
        f"output={args.output}"
    )


if __name__ == "__main__":
    main()
