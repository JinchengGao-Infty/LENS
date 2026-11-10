"""V2 synthetic ranker-union benchmark with one shared exact refit stream."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np

from experiments.dgps import generate
from lens.experiments._artifacts import ArtifactRun
from lens.pipeline import StableLensConfig
from lens.shared_confirmation import candidate_sets_digest, run_shared_exact_confirmation
from lens.splits import make_three_way_split
from lens.synthetic_oracle import compute_run_matched_oracle
from lens.synthetic_rankers import FORMAL_SYNTHETIC_RANKERS, discover_synthetic_rankers
from lens.teachers import XGBoostTeacherFactory

Pair = tuple[int, int]


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("expected a nonempty comma-separated list")
    return parsed


def _parse_int_csv(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    if not parsed or any(width < 1 for width in parsed):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return parsed


def _indices_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _graph_metrics(
    graph: set[Pair],
    truth: set[Pair],
    masses: dict[Pair, float],
) -> dict[str, float | int | None]:
    true_positives = len(graph & truth)
    false_positives = len(graph - truth)
    total_mass = float(sum(masses.get(pair, 0.0) for pair in truth))
    return {
        "power": true_positives / len(truth) if truth else None,
        "fdp": false_positives / len(graph) if graph else 0.0,
        "weighted_power": (
            sum(masses.get(pair, 0.0) for pair in graph & truth) / total_mass
            if total_mass > 0
            else None
        ),
        "n_discoveries": len(graph),
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def _write_npz_atomic(path: Path, **payload: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _write_grids(path: Path, grids: dict) -> None:
    payload: dict[str, np.ndarray] = {
        "schema": np.asarray("lens-v2-frozen-grids-1"),
        "feature_indices": np.asarray(sorted(grids), dtype=np.int64),
    }
    for feature in sorted(grids):
        grid = grids[feature]
        values = np.asarray(grid.values)
        if values.dtype == object:
            values = values.astype(str)
        payload[f"feature_{feature}_values"] = values
        payload[f"feature_{feature}_weights"] = np.asarray(
            grid.weights, dtype=np.float64
        )
        payload[f"feature_{feature}_kind"] = np.asarray(grid.kind)
    _write_npz_atomic(path, **payload)


def _tree_bytes(path: Path) -> int:
    return int(sum(item.stat().st_size for item in path.rglob("*") if item.is_file()))


def _max_rss_kib() -> int:
    raw = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Linux reports KiB; macOS reports bytes.
    return raw // 1024 if sys.platform == "darwin" else raw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dgp", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--n-permutations", type=int, default=500)
    parser.add_argument("--grid-points", type=int, default=12)
    parser.add_argument("--grid-batch-size", type=int, default=144)
    parser.add_argument("--ranking-max-samples", type=int, default=3000)
    parser.add_argument("--rankers", type=_parse_csv, default=FORMAL_SYNTHETIC_RANKERS)
    parser.add_argument("--nid-hidden", type=_parse_int_csv, default=(128, 64))
    parser.add_argument("--nid-epochs", type=int, default=200)
    parser.add_argument("--nid-lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--cache-mode",
        choices=["exact_statistics", "full_summary"],
        default="exact_statistics",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rankers = tuple(args.rankers)
    unknown = sorted(set(rankers) - set(FORMAL_SYNTHETIC_RANKERS))
    if unknown:
        raise SystemExit(f"unknown rankers: {unknown}")
    if len(rankers) != len(set(rankers)):
        raise SystemExit("rankers must be unique")
    config_dict = {
        **vars(args),
        "rankers": list(rankers),
        "nid_hidden": list(args.nid_hidden),
        "protocol": "discovery_measurable_union_shared_exact_confirmation",
    }
    run = ArtifactRun.start(args.output, config_dict)
    wall_started = time.perf_counter()
    try:
        dgp = generate(args.dgp, seed=args.seed, n=args.n)
        feature_kinds = ("continuous",) * dgp.X.shape[1]
        config = StableLensConfig(
            top_k=args.top_k,
            n_permutations=args.n_permutations,
            grid_points=args.grid_points,
            grid_batch_size=args.grid_batch_size,
            ranking_score="raw_energy",
            alpha=args.alpha,
        )
        config.validate()
        split = make_three_way_split(
            len(dgp.y), seed=args.seed, fractions=config.split_fractions
        )
        split.validate(len(dgp.y))
        _write_npz_atomic(
            run.root / "split_indices.npz",
            discovery=np.asarray(split.discovery, dtype=np.int64),
            confirmation=np.asarray(split.confirmation, dtype=np.int64),
            evaluation=np.asarray(split.evaluation, dtype=np.int64),
            schema=np.asarray("lens-v2-three-way-split-1"),
        )
        factory = XGBoostTeacherFactory(
            "regression",
            params={"device": args.device},
        )

        run.log("discovery_started", rankers=list(rankers))
        discovery = discover_synthetic_rankers(
            dgp.X,
            dgp.y,
            split=split,
            feature_kinds=feature_kinds,
            teacher_factory=factory,
            seed=args.seed,
            top_k=args.top_k,
            grid_points=args.grid_points,
            grid_batch_size=args.grid_batch_size,
            ranking_max_samples=args.ranking_max_samples,
            rankers=rankers,
            nid_hidden=args.nid_hidden,
            nid_epochs=args.nid_epochs,
            nid_lr=args.nid_lr,
        )
        run.log(
            "discovery_finished",
            candidate_union_count=len(discovery.union_candidates),
            diagnostics=discovery.diagnostics,
            timings=discovery.timings,
        )
        _write_grids(run.root / "grids.npz", discovery.grids)
        grids_sha256 = _file_hash(run.root / "grids.npz")

        def progress(event: dict) -> None:
            if event["stage"] != "permutation_refit" or (
                event["completed"] % args.progress_every == 0
                or event["completed"] == event["total"]
            ):
                run.log("progress", **event)
                print(json.dumps(event, sort_keys=True), flush=True)

        confirmation = run_shared_exact_confirmation(
            dgp.X,
            dgp.y,
            split=split,
            grids=discovery.grids,
            candidate_sets=discovery.candidate_sets,
            teacher_factory=factory,
            seed=args.seed,
            config=config,
            cache_root=(
                (run.root / "cache") if args.cache_mode == "full_summary" else None
            ),
            cache_mode=args.cache_mode,
            dgp_name=args.dgp,
            progress_callback=progress,
        )
        if confirmation.union_candidates != discovery.union_candidates:
            raise RuntimeError("discovery and confirmation candidate unions disagree")

        oracle_started = time.perf_counter()
        truth = set(tuple(pair) for pair in dgp.true_pairs)
        oracle = compute_run_matched_oracle(
            args.dgp,
            dgp.X[split.evaluation],
            discovery.grids,
            sorted(truth),
            grid_batch_size=args.grid_batch_size,
        )
        oracle_seconds = time.perf_counter() - oracle_started
        masses = dict(oracle.masses)

        observed_statistics = np.asarray(
            [confirmation.observed_statistics[pair] for pair in confirmation.union_candidates],
            dtype=np.float64,
        )
        split_hashes = {
            "discovery": _indices_hash(split.discovery),
            "confirmation": _indices_hash(split.confirmation),
            "evaluation": _indices_hash(split.evaluation),
        }
        confirmation_seed = int(confirmation.diagnostics["confirmation_seed"])
        permutation_seed = int(confirmation.diagnostics["permutation_seed"])
        _write_npz_atomic(
            run.root / "null_statistics.npz",
            pairs=np.asarray(confirmation.union_candidates, dtype=np.int64),
            observed_statistics=observed_statistics,
            null_statistics=np.asarray(confirmation.null_statistics, dtype=np.float64),
            candidate_sets_sha256=np.asarray(
                candidate_sets_digest(discovery.candidate_sets)
            ),
            grids_sha256=np.asarray(grids_sha256),
            split_discovery_sha256=np.asarray(split_hashes["discovery"]),
            split_confirmation_sha256=np.asarray(split_hashes["confirmation"]),
            split_evaluation_sha256=np.asarray(split_hashes["evaluation"]),
            dgp=np.asarray(args.dgp),
            seed=np.asarray(args.seed, dtype=np.int64),
            teacher_name=np.asarray(factory.name),
            confirmation_seed=np.asarray(confirmation_seed, dtype=np.int64),
            permutation_seed=np.asarray(permutation_seed, dtype=np.int64),
            refit_indices=np.arange(args.n_permutations, dtype=np.int64),
            refit_teacher_seeds=np.full(
                args.n_permutations, confirmation_seed, dtype=np.int64
            ),
            refit_teacher_seed_policy=np.asarray(
                "same_as_observed_confirmation"
            ),
            cache_mode=np.asarray(args.cache_mode),
            schema=np.asarray("lens-v2-synthetic-union-null-statistics-2"),
        )

        rank_positions = {
            method: {pair: index + 1 for index, pair in enumerate(ranking)}
            for method, ranking in discovery.rankings.items()
        }
        candidate_rows = []
        for pair in confirmation.union_candidates:
            member_of = [
                method for method in rankers if pair in discovery.candidate_sets[method]
            ]
            selected_by = [
                method for method in rankers if pair in confirmation.graphs[method]
            ]
            candidate_rows.append(
                {
                    "seed": args.seed,
                    "dgp": args.dgp,
                    "pair": list(pair),
                    "is_true": pair in truth,
                    "oracle_mass": masses.get(pair),
                    "member_of_ranker_top_k": member_of,
                    "selected_by_ranker_bh": selected_by,
                    "discovery_rank": {
                        method: rank_positions[method][pair] for method in rankers
                    },
                    "discovery_score": {
                        method: discovery.scores[method][pair] for method in rankers
                    },
                    "T_E_observed": confirmation.observed_statistics[pair],
                    "exact_rank_pvalue": confirmation.exact_pvalues[pair],
                }
            )
        run.write_candidate_rows(candidate_rows)

        ranker_summaries = {}
        for method in rankers:
            candidates = set(discovery.candidate_sets[method])
            graph = set(confirmation.graphs[method])
            ranker_summaries[method] = {
                "candidate_pairs": [list(pair) for pair in discovery.candidate_sets[method]],
                "confirmed_pairs": [list(pair) for pair in confirmation.graphs[method]],
                "candidate_metrics": _graph_metrics(candidates, truth, masses),
                "confirmed_metrics": _graph_metrics(graph, truth, masses),
                "selector_seconds": discovery.timings.get(method),
            }

        full_summary_cache_bytes = _tree_bytes(run.root / "cache")
        exact_statistics_bytes = (run.root / "null_statistics.npz").stat().st_size
        summary = {
            "schema_version": "lens-v2-synthetic-union-e2e-2",
            "seed": args.seed,
            "dgp": args.dgp,
            "n": len(dgp.y),
            "p": dgp.X.shape[1],
            "rankers": list(rankers),
            "true_pairs": [list(pair) for pair in sorted(truth)],
            "candidate_budget_per_ranker": args.top_k,
            "candidate_union_count": len(confirmation.union_candidates),
            "candidate_union_pairs": [
                list(pair) for pair in confirmation.union_candidates
            ],
            "candidate_sets_sha256": candidate_sets_digest(discovery.candidate_sets),
            "split_sizes": {
                "discovery": len(split.discovery),
                "confirmation": len(split.confirmation),
                "evaluation": len(split.evaluation),
            },
            "split_indices_sha256": split_hashes,
            "grids_sha256": grids_sha256,
            "oracle": {
                "definition": oracle.definition,
                "background": "this_run_X_eval",
                "grids": "this_run_D_disc_frozen",
                "n_eval": oracle.n_eval,
                "total_true_mass": oracle.total_true_mass,
                "masses": [
                    {"pair": list(pair), "mass": masses[pair]}
                    for pair in sorted(masses)
                ],
                "candidate_independent": True,
            },
            "methods": ranker_summaries,
            "timings": {
                "discovery": discovery.timings,
                "confirmation": confirmation.timings,
                "oracle_seconds": oracle_seconds,
                "wall_seconds": time.perf_counter() - wall_started,
            },
            "diagnostics": {
                **confirmation.diagnostics,
                "discovery": discovery.diagnostics,
                "cache_mode": args.cache_mode,
                "exact_statistics_bytes": exact_statistics_bytes,
                "full_summary_cache_bytes": full_summary_cache_bytes,
                "full_summary_cache_refit_count": (
                    args.n_permutations + 1
                    if args.cache_mode == "full_summary"
                    else 0
                ),
                "full_summary_cache_omitted_reason": (
                    "not_required_by_exact_bh_main_table"
                    if args.cache_mode == "exact_statistics"
                    else None
                ),
                "max_rss_kib": _max_rss_kib(),
                "gpu_requested": args.device == "cuda",
                "device_requested": args.device,
                "oracle_never_enters_candidates": True,
            },
        }
        run.finish(
            summary,
            extra_required_files=[
                "split_indices.npz",
                "grids.npz",
                "null_statistics.npz",
            ],
        )
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
