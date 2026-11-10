"""Replay the frozen Stage-1 Friedman H2 ranking to recover complete-ranking AP.

This replay reuses the original DGPs, split seeds, Teacher factory, finite grid,
and ranking contexts.  It computes only discovery-time Friedman H2 rankings,
and verifies every reproduced Top-15 list against the completed formal run
before accepting the derived AP.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

from experiments.dgps import generate
from lens.splits import make_three_way_split
from lens.synthetic_rankers import discover_synthetic_rankers
from lens.teachers import XGBoostTeacherFactory


FORMAL_DGPS = (
    "Friedman1",
    "Smooth",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
)
FORMAL_SEEDS = tuple(range(32))
SCHEMA = "lens-friedman-h2-full-ap-replay-1"


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _average_precision(ranking: tuple[tuple[int, int], ...], truth: set[tuple[int, int]]) -> float:
    if not truth:
        raise ValueError("AP is undefined for an empty truth set")
    hits = 0
    total = 0.0
    for rank, pair in enumerate(ranking, start=1):
        if pair in truth:
            hits += 1
            total += hits / rank
    if hits != len(truth):
        raise RuntimeError(f"complete ranking contains {hits}/{len(truth)} true pairs")
    return total / len(truth)


def _cell_output(root: Path, dgp: str, seed: int) -> Path:
    return root / f"dgp={dgp}" / f"seed={seed}" / "result.json"


def run_worker(args: argparse.Namespace) -> None:
    output = Path(args.output)
    source = Path(args.source)
    cells = [
        (dgp, seed)
        for dgp in FORMAL_DGPS
        for seed in FORMAL_SEEDS
    ]
    assigned = [
        cell for index, cell in enumerate(cells) if index % args.n_shards == args.shard_index
    ]
    for dgp, seed in assigned:
        destination = _cell_output(output, dgp, seed)
        if destination.exists():
            raise FileExistsError(f"refusing to overwrite {destination}")
        started = time.perf_counter()
        generated = generate(dgp, seed=seed, n=None)
        split = make_three_way_split(len(generated.y), seed=seed)
        discovery = discover_synthetic_rankers(
            generated.X,
            generated.y,
            split=split,
            feature_kinds=("continuous",) * generated.X.shape[1],
            teacher_factory=XGBoostTeacherFactory(
                "regression",
                params={"device": args.device},
            ),
            seed=seed,
            top_k=15,
            grid_points=12,
            grid_batch_size=144,
            ranking_max_samples=3000,
            rankers=("Friedman-H2",),
        )
        ranking = tuple(discovery.rankings["Friedman-H2"])
        archived_summary = json.loads(
            (
                source
                / f"dgp={dgp}"
                / f"seed={seed}"
                / "summary_metrics.json"
            ).read_text()
        )
        archived_top15 = tuple(
            tuple(pair)
            for pair in archived_summary["methods"]["Friedman-H2"]["candidate_pairs"]
        )
        if ranking[:15] != archived_top15:
            raise RuntimeError(f"Top-15 replay mismatch for {dgp} seed {seed}")
        archived_ranking_hash = archived_summary["diagnostics"]["discovery"][
            "ranking_indices_sha256"
        ]
        replay_ranking_hash = discovery.diagnostics["ranking_indices_sha256"]
        if replay_ranking_hash != archived_ranking_hash:
            raise RuntimeError(f"ranking-context replay mismatch for {dgp} seed {seed}")
        truth = {tuple(pair) for pair in generated.true_pairs}
        payload = {
            "schema_version": SCHEMA,
            "status": "DONE",
            "dgp": dgp,
            "seed": seed,
            "n_pairs": len(ranking),
            "n_true_pairs": len(truth),
            "average_precision": _average_precision(ranking, truth),
            "top15_verified": True,
            "ranking_indices_sha256": replay_ranking_hash,
            "elapsed_seconds": time.perf_counter() - started,
        }
        _atomic_json(destination, payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    _atomic_json(
        output / f"SHARD_{args.shard_index}_DONE.json",
        {
            "schema_version": SCHEMA,
            "status": "DONE",
            "shard_index": args.shard_index,
            "n_shards": args.n_shards,
            "completed_cells": len(assigned),
        },
    )


def aggregate(args: argparse.Namespace) -> None:
    output = Path(args.output)
    rows = []
    for dgp in FORMAL_DGPS:
        for seed in FORMAL_SEEDS:
            path = _cell_output(output, dgp, seed)
            if not path.exists():
                raise FileNotFoundError(path)
            row = json.loads(path.read_text())
            if (
                row.get("schema_version") != SCHEMA
                or row.get("status") != "DONE"
                or not row.get("top15_verified")
            ):
                raise ValueError(f"invalid replay row {path}")
            rows.append(row)
    per_dgp = {}
    for dgp in FORMAL_DGPS:
        values = [
            float(row["average_precision"])
            for row in rows
            if row["dgp"] == dgp
        ]
        per_dgp[dgp] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values),
            "n": len(values),
        }
    dgp_means = [per_dgp[dgp]["mean"] for dgp in FORMAL_DGPS]
    aggregate_payload = {
        "schema_version": SCHEMA,
        "status": "DONE",
        "source_root": str(Path(args.source)),
        "source_git_commit": args.source_git_commit,
        "source_script_sha256": _sha256(Path(__file__)),
        "completed_cells": len(rows),
        "expected_cells": len(FORMAL_DGPS) * len(FORMAL_SEEDS),
        "dgps": list(FORMAL_DGPS),
        "seeds": list(FORMAL_SEEDS),
        "top15_replay_verified_cells": sum(bool(row["top15_verified"]) for row in rows),
        "macro_over_dgp_means": {
            "mean": statistics.mean(dgp_means),
            "std": statistics.pstdev(dgp_means),
            "n": len(dgp_means),
        },
        "micro_over_dgp_seed_cells": {
            "mean": statistics.mean(float(row["average_precision"]) for row in rows),
            "std": statistics.pstdev(float(row["average_precision"]) for row in rows),
            "n": len(rows),
        },
        "per_dgp": per_dgp,
        "total_replay_seconds": sum(float(row["elapsed_seconds"]) for row in rows),
    }
    _atomic_json(output / "aggregate.json", aggregate_payload)
    (output / "ALL_DONE").touch()
    print(json.dumps(aggregate_payload, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--source-git-commit", default="982762ac12a83286c7410cf4e79fee12051b838b")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--n-shards", type=int, default=8)
    parser.add_argument("--aggregate", action="store_true")
    args = parser.parse_args()
    if args.aggregate:
        aggregate(args)
        return
    if args.shard_index is None or not 0 <= args.shard_index < args.n_shards:
        raise SystemExit("--shard-index must be in [0, n-shards)")
    run_worker(args)


if __name__ == "__main__":
    main()
