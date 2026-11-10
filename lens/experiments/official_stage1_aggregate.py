"""Fail-closed aggregation for the official Stage-1 baseline matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _parse_csv(value: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _summary(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        "n": int(len(array)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--methods", required=True, type=_parse_csv)
    parser.add_argument("--dgps", required=True, type=_parse_csv)
    parser.add_argument("--seeds", required=True, type=_parse_csv)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rows: list[dict] = []
    missing: list[str] = []
    failed: list[str] = []
    for method in args.methods:
        for dgp in args.dgps:
            for seed_text in args.seeds:
                cell = args.root / f"method={method}" / f"dgp={dgp}" / f"seed={seed_text}"
                if (cell / "FAILED_RUN.json").exists():
                    failed.append(str(cell))
                    continue
                if not (cell / "DONE").exists() or not (cell / "result.json").exists():
                    missing.append(str(cell))
                    continue
                row = json.loads((cell / "result.json").read_text())
                if row.get("status") != "DONE" or row.get("stage") != 1:
                    failed.append(str(cell))
                    continue
                rows.append(row)
    expected = len(args.methods) * len(args.dgps) * len(args.seeds)
    if missing or failed or len(rows) != expected:
        report = {
            "status": "FAILED",
            "expected": expected,
            "completed": len(rows),
            "missing": missing,
            "failed": failed,
        }
        _atomic_json(args.output.with_suffix(".FAILED.json"), report)
        raise SystemExit(
            f"official Stage-1 matrix incomplete: completed={len(rows)}/{expected}, "
            f"missing={len(missing)}, failed={len(failed)}"
        )

    per_dgp: dict[str, dict[str, dict[str, dict[str, float | int]]]] = {}
    for dgp in args.dgps:
        per_dgp[dgp] = {}
        for method in args.methods:
            selected = [row for row in rows if row["dgp"] == dgp and row["method"] == method]
            per_dgp[dgp][method] = {}
            for metric in ("recall_at_k", "weighted_recall_at_k", "average_precision"):
                values = [float(row["metrics"][metric]) for row in selected if row["metrics"][metric] is not None]
                if values:
                    per_dgp[dgp][method][metric] = _summary(values)
            per_dgp[dgp][method]["wall_seconds"] = _summary(
                [float(row["wall_seconds"]) for row in selected]
            )

    nonnull = [dgp for dgp in args.dgps if dgp != "Additive"]
    macro: dict[str, dict[str, dict[str, float | int]]] = {}
    for method in args.methods:
        macro[method] = {}
        for metric in ("recall_at_k", "weighted_recall_at_k", "average_precision"):
            values = [
                float(per_dgp[dgp][method][metric]["mean"])
                for dgp in nonnull
                if metric in per_dgp[dgp][method]
            ]
            if values:
                macro[method][metric] = _summary(values)

    payload = {
        "schema_version": "lens-official-stage1-aggregate-1",
        "status": "DONE",
        "expected_cells": expected,
        "completed_cells": len(rows),
        "methods": list(args.methods),
        "dgps": list(args.dgps),
        "seeds": [int(seed) for seed in args.seeds],
        "per_dgp": per_dgp,
        "macro_over_nonnull_dgps": macro,
        "claim_boundary": "Stage-1 discovery ranking only; no Stage-2 inference claim.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.output, payload)
    print(f"OFFICIAL_STAGE1_AGGREGATE_DONE cells={len(rows)} output={args.output}")


if __name__ == "__main__":
    main()
