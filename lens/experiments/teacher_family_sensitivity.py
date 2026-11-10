"""One fail-closed LENS teacher-family sensitivity cell."""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, mean_squared_error, r2_score

from experiments.dgps import generate, noise_free_response
from lens.calibration.maxT import maxT_pvalues
from lens.calibration.multiple_testing import bh_rejections
from lens.experiments._artifacts import ArtifactRun
from lens.experiments import official_stage1_baseline as official
from lens.grids import Grid1D, PairGrid, build_grid
from lens.interaction_vectors import InteractionSample, compute_interaction_sample
from lens.model_inference import run_model_inference
from lens.ranking import rank_interactions
from lens.splits import make_three_way_split
from lens.synthetic_oracle import compute_run_matched_oracle
from lens.synthetic_rankers import _fast_scores, _ranking_from_scores, _tree_shap_scores
from lens.teachers import (
    FTTransformerTeacherFactory,
    RBFNystroemTeacherFactory,
    SoftplusMLPTeacherFactory,
    TabPFNTeacherFactory,
    XGBoostTeacherFactory,
)

Pair = tuple[int, int]

EXPECTED_EXTERNAL_COMMITS = {
    "nid": "02afd5b75b758e179f39c182a27de786b18be416",
    "path_explain": "93c75a5e9b3caeb17408297ea564e2f579ea15e3",
}


def _all_pairs(n_features: int) -> tuple[Pair, ...]:
    return tuple(itertools.combinations(range(n_features), 2))


def _sha256_ints(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _write_npz_atomic(path: Path, **payload: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _fidelity(
    observed_y: np.ndarray, signal_y: np.ndarray, prediction: np.ndarray
) -> dict[str, float | str]:
    values = np.asarray(prediction, dtype=np.float64)
    if values.shape != observed_y.shape or not np.all(np.isfinite(values)):
        raise ValueError("teacher fidelity predictions are invalid")
    signal_r2 = float(r2_score(signal_y, values))
    if signal_r2 >= 0.90:
        band = "strong"
    elif signal_r2 >= 0.70:
        band = "interpretable"
    else:
        band = "weak"
    return {
        "observed_y_r2": float(r2_score(observed_y, values)),
        "observed_y_rmse": float(mean_squared_error(observed_y, values) ** 0.5),
        "signal_r2": signal_r2,
        "signal_rmse": float(mean_squared_error(signal_y, values) ** 0.5),
        "signal_r2_band": band,
    }


def _ranking_metrics(
    ranking: Sequence[Pair],
    scores: Mapping[Pair, float],
    truth: set[Pair],
    masses: Mapping[Pair, float],
    *,
    top_k: int,
) -> dict[str, float | int | None]:
    selected = set(ranking[:top_k])
    labels = np.asarray([pair in truth for pair in ranking], dtype=np.int8)
    ranked_scores = np.asarray([scores[pair] for pair in ranking], dtype=np.float64)
    total_mass = float(sum(masses.get(pair, 0.0) for pair in truth))
    return {
        "recall_at_k": len(selected & truth) / len(truth) if truth else None,
        "weighted_recall_at_k": (
            sum(masses.get(pair, 0.0) for pair in selected & truth) / total_mass
            if total_mass > 0
            else None
        ),
        "average_precision": (
            float(average_precision_score(labels, ranked_scores)) if truth else None
        ),
        "true_positives_at_k": len(selected & truth),
        "false_positives_at_k": len(selected - truth),
    }


def _graph_metrics(
    graph: set[Pair], truth: set[Pair], masses: Mapping[Pair, float]
) -> dict[str, float | int | None]:
    total_mass = float(sum(masses.get(pair, 0.0) for pair in truth))
    true_positives = len(graph & truth)
    false_positives = len(graph - truth)
    return {
        "conditional_power": true_positives / len(truth) if truth else None,
        "weighted_conditional_power": (
            sum(masses.get(pair, 0.0) for pair in graph & truth) / total_mass
            if total_mass > 0
            else None
        ),
        "candidate_null_rejection_rate": None,
        "descriptive_fdp": false_positives / len(graph) if graph else 0.0,
        "n_discoveries": len(graph),
        "true_positives": true_positives,
        "false_positives": false_positives,
    }


def _teacher_factory(args: argparse.Namespace):
    if args.teacher == "xgboost":
        return XGBoostTeacherFactory("regression", params={"device": args.device})
    if args.teacher == "softplus_mlp":
        return SoftplusMLPTeacherFactory(
            device=args.device,
            max_epochs=args.mlp_max_epochs,
            min_epochs=min(args.mlp_min_epochs, args.mlp_max_epochs),
            patience=args.mlp_patience,
        )
    if args.teacher == "rbf_nystroem":
        return RBFNystroemTeacherFactory(
            n_components=args.rbf_components,
            alpha=args.rbf_alpha,
            device=args.device,
        )
    if args.teacher == "ft_transformer":
        return FTTransformerTeacherFactory(
            device=args.device,
            max_epochs=args.ft_max_epochs,
            min_epochs=min(args.ft_min_epochs, args.ft_max_epochs),
            patience=args.ft_patience,
        )
    if args.teacher == "tabpfn":
        if args.tabpfn_model_path is None:
            raise ValueError("TabPFN requires an explicit local checkpoint path")
        return TabPFNTeacherFactory(
            model_path=args.tabpfn_model_path,
            n_estimators=args.tabpfn_n_estimators,
            prediction_batch_size=args.tabpfn_prediction_batch_size,
            device=args.device,
        )
    raise ValueError(f"unknown teacher {args.teacher!r}")


def _same_mlp_nid_scores(
    teacher,
    pairs: Sequence[Pair],
    external_root: Path,
) -> tuple[dict[Pair, float], dict[str, object]]:
    import torch

    repo = external_root / "nid" / "1. madex"
    if official._repo_commit(external_root / "nid") != EXPECTED_EXTERNAL_COMMITS["nid"]:
        raise ValueError("official NID source commit mismatch")
    official._prepare_nid_imports(repo)
    try:
        module = official._load_module(
            "lens_same_mlp_nid_detection", repo / "neural_interaction_detection.py"
        )
        weights = [
            layer.weight.detach().cpu().numpy()
            for layer in teacher.core_model
            if isinstance(layer, torch.nn.Linear)
        ]
        input_weights, later_weights = module.preprocess_weights(weights)
        rows = module.interpret_interactions_from_weights(
            input_weights, later_weights, get_main_effects=False
        )
    finally:
        sys.path.pop(0)
    sparse = {
        tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
        for pair, value in rows
        if len(pair) == 2
    }
    scores = {pair: abs(float(sparse.get(pair, 0.0))) for pair in pairs}
    return scores, {
        "official_api": "preprocess_weights + interpret_interactions_from_weights",
        "official_source_commit": EXPECTED_EXTERNAL_COMMITS["nid"],
        "same_fitted_teacher": True,
        "activation_of_consumed_teacher": "Softplus",
    }


def _same_mlp_integrated_hessian_scores(
    teacher,
    X: np.ndarray,
    pairs: Sequence[Pair],
    external_root: Path,
    *,
    seed: int,
    n_explain: int,
) -> tuple[dict[Pair, float], dict[str, object]]:
    import torch

    repo = external_root / "path_explain"
    if official._repo_commit(repo) != EXPECTED_EXTERNAL_COMMITS["path_explain"]:
        raise ValueError("official path_explain source commit mismatch")
    module = official._load_module(
        "lens_same_mlp_path_explainer_torch",
        repo / "path_explain" / "explainers" / "path_explainer_torch.py",
    )
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), min(n_explain, len(X)), replace=False))
    background_indices = rng.choice(len(X), min(128, len(X)), replace=False)
    model = copy.deepcopy(teacher.torch_model).cpu().eval()
    inputs = torch.tensor(X[chosen], dtype=torch.float32, requires_grad=True)
    background = torch.tensor(X[background_indices], dtype=torch.float32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    matrix = module.PathExplainerTorch(model).interactions(
        inputs,
        baseline=background,
        num_samples=50,
        use_expectation=True,
        verbose=False,
    )
    values = np.asarray(matrix.detach().cpu().numpy(), dtype=np.float64)
    expected_shape = (len(chosen), X.shape[1], X.shape[1])
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise ValueError(f"unexpected Integrated Hessians output {values.shape}")
    scores = {
        pair: float(np.mean(np.abs(values[:, pair[0], pair[1]]))) for pair in pairs
    }
    return scores, {
        "official_api": "PathExplainerTorch.interactions",
        "official_source_commit": EXPECTED_EXTERNAL_COMMITS["path_explain"],
        "same_fitted_teacher": True,
        "n_explain": int(len(chosen)),
        "n_background": int(len(background_indices)),
        "num_samples": 50,
    }


def _teacher_metadata(teacher) -> dict[str, object]:
    metadata = getattr(teacher, "fit_metadata", None)
    if metadata is None:
        return {"implementation": type(teacher).__name__}
    return {"implementation": type(teacher).__name__, **dict(metadata)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--teacher",
        choices=(
            "xgboost",
            "softplus_mlp",
            "rbf_nystroem",
            "ft_transformer",
            "tabpfn",
        ),
        required=True,
    )
    parser.add_argument("--dgp", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n", type=int, default=10000)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--grid-points", type=int, default=12)
    parser.add_argument("--grid-batch-size", type=int, default=144)
    parser.add_argument("--ranking-max-samples", type=int, default=512)
    parser.add_argument("--n-bootstrap", type=int, default=999)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--mlp-max-epochs", type=int, default=200)
    parser.add_argument("--mlp-min-epochs", type=int, default=30)
    parser.add_argument("--mlp-patience", type=int, default=15)
    parser.add_argument("--rbf-components", type=int, default=512)
    parser.add_argument("--rbf-alpha", type=float, default=1e-3)
    parser.add_argument("--ft-max-epochs", type=int, default=200)
    parser.add_argument("--ft-min-epochs", type=int, default=30)
    parser.add_argument("--ft-patience", type=int, default=15)
    parser.add_argument("--tabpfn-model-path", type=Path)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=8)
    parser.add_argument("--tabpfn-prediction-batch-size", type=int, default=4096)
    parser.add_argument("--ih-n-explain", type=int, default=64)
    parser.add_argument("--external-root", type=Path)
    parser.add_argument("--skip-fast", action="store_true")
    parser.add_argument("--skip-family-baselines", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.teacher == "softplus_mlp" and not args.skip_family_baselines and args.external_root is None:
        raise SystemExit("Softplus MLP family baselines require --external-root")
    if args.teacher == "tabpfn" and args.tabpfn_model_path is None:
        raise SystemExit("TabPFN requires --tabpfn-model-path")
    if args.top_k < 1 or args.ranking_max_samples < 2 or args.n_bootstrap < 1:
        raise SystemExit("invalid ranking, candidate, or bootstrap budget")

    run = ArtifactRun.start(args.output, vars(args))
    wall_started = time.perf_counter()
    try:
        dgp = generate(args.dgp, seed=args.seed, n=args.n)
        features = np.asarray(dgp.X, dtype=np.float64)
        target = np.asarray(dgp.y, dtype=np.float64)
        truth = set(tuple(pair) for pair in dgp.true_pairs)
        feature_kinds = ("continuous",) * features.shape[1]
        split = make_three_way_split(len(target), seed=args.seed)
        split.validate(len(target))
        X_discovery, y_discovery = features[split.discovery], target[split.discovery]
        X_confirmation, y_confirmation = features[split.confirmation], target[split.confirmation]
        X_evaluation, y_evaluation = features[split.evaluation], target[split.evaluation]
        signal_evaluation = noise_free_response(args.dgp, X_evaluation)
        grids: dict[int, Grid1D] = {
            index: build_grid(
                X_discovery[:, index], feature_kinds[index], n_points=args.grid_points
            )
            for index in range(features.shape[1])
        }
        pairs = _all_pairs(features.shape[1])
        if args.top_k > len(pairs):
            raise ValueError("top-k exceeds the number of feature pairs")
        ranking_rng = np.random.default_rng(args.seed * 2 + 4)
        ranking_local = np.sort(
            ranking_rng.choice(
                len(X_discovery),
                min(args.ranking_max_samples, len(X_discovery)),
                replace=False,
            )
        )
        X_ranking = X_discovery[ranking_local]
        ranking_global = np.asarray(split.discovery)[ranking_local]
        factory = _teacher_factory(args)

        fit_started = time.perf_counter()
        discovery_teacher = factory.fit(
            X_discovery, y_discovery, seed=args.seed * 2 + 1
        )
        discovery_fit_seconds = time.perf_counter() - fit_started
        discovery_fidelity = _fidelity(
            y_evaluation,
            signal_evaluation,
            discovery_teacher.predict_margin(X_evaluation),
        )
        run.log(
            "discovery_teacher_finished",
            seconds=discovery_fit_seconds,
            fidelity=discovery_fidelity,
            metadata=_teacher_metadata(discovery_teacher),
        )

        scores: dict[str, dict[Pair, float]] = {}
        rankings: dict[str, tuple[Pair, ...]] = {}
        selector_seconds: dict[str, float] = {}
        lens_samples: dict[Pair, InteractionSample] = {}
        lens_started = time.perf_counter()
        for pair_index, pair in enumerate(pairs):
            lens_samples[pair] = compute_interaction_sample(
                discovery_teacher.predict_margin,
                X_ranking,
                pair,
                PairGrid(grids[pair[0]], grids[pair[1]]),
                grid_batch_size=args.grid_batch_size,
                estimate_covariance="none",
            )
            if (pair_index + 1) % args.progress_every == 0 or pair_index + 1 == len(pairs):
                run.log("stage1_lens_progress", completed=pair_index + 1, total=len(pairs))
        lens_rows = rank_interactions(lens_samples, score="raw_energy")
        scores["LENS-raw"] = dict(lens_rows)
        rankings["LENS-raw"] = tuple(pair for pair, _ in lens_rows)
        selector_seconds["LENS-raw"] = time.perf_counter() - lens_started

        if not args.skip_fast:
            started = time.perf_counter()
            scores["FAST"] = _fast_scores(
                X_ranking,
                discovery_teacher.predict_margin(X_ranking),
                feature_kinds,
                pairs,
            )
            rankings["FAST"] = _ranking_from_scores(scores["FAST"], pairs)
            selector_seconds["FAST"] = time.perf_counter() - started
        family_baseline_metadata: dict[str, object] = {}
        if args.teacher == "xgboost" and not args.skip_family_baselines:
            started = time.perf_counter()
            scores["TreeSHAP"] = _tree_shap_scores(
                discovery_teacher.model, X_ranking, pairs
            )
            rankings["TreeSHAP"] = _ranking_from_scores(scores["TreeSHAP"], pairs)
            selector_seconds["TreeSHAP"] = time.perf_counter() - started
        if args.teacher == "softplus_mlp" and not args.skip_family_baselines:
            assert args.external_root is not None
            started = time.perf_counter()
            scores["NID-weight-same-MLP"], nid_metadata = _same_mlp_nid_scores(
                discovery_teacher, pairs, args.external_root
            )
            rankings["NID-weight-same-MLP"] = _ranking_from_scores(
                scores["NID-weight-same-MLP"], pairs
            )
            selector_seconds["NID-weight-same-MLP"] = time.perf_counter() - started
            family_baseline_metadata["NID-weight-same-MLP"] = nid_metadata
            started = time.perf_counter()
            (
                scores["Integrated-Hessians-same-MLP"],
                ih_metadata,
            ) = _same_mlp_integrated_hessian_scores(
                discovery_teacher,
                X_ranking,
                pairs,
                args.external_root,
                seed=args.seed * 2 + 5,
                n_explain=args.ih_n_explain,
            )
            rankings["Integrated-Hessians-same-MLP"] = _ranking_from_scores(
                scores["Integrated-Hessians-same-MLP"], pairs
            )
            selector_seconds["Integrated-Hessians-same-MLP"] = (
                time.perf_counter() - started
            )
            family_baseline_metadata["Integrated-Hessians-same-MLP"] = ih_metadata

        expected_pairs = set(pairs)
        for method in rankings:
            if len(rankings[method]) != len(pairs) or set(rankings[method]) != expected_pairs:
                raise ValueError(f"{method} did not emit a complete ranking")
            if set(scores[method]) != expected_pairs or not np.all(
                np.isfinite(list(scores[method].values()))
            ):
                raise ValueError(f"{method} emitted invalid scores")
        candidate_sets = {
            method: tuple(ranking[: args.top_k]) for method, ranking in rankings.items()
        }
        union_candidates = tuple(
            sorted({pair for candidates in candidate_sets.values() for pair in candidates})
        )
        run.log(
            "stage1_finished",
            rankers=list(rankings),
            candidate_union_count=len(union_candidates),
            selector_seconds=selector_seconds,
        )
        discovery_teacher_metadata = _teacher_metadata(discovery_teacher)
        del discovery_teacher
        gc.collect()
        if args.device == "cuda":
            import torch

            torch.cuda.empty_cache()

        fit_started = time.perf_counter()
        confirmation_teacher = factory.fit(
            X_confirmation, y_confirmation, seed=args.seed * 2 + 2
        )
        confirmation_fit_seconds = time.perf_counter() - fit_started
        confirmation_fidelity = _fidelity(
            y_evaluation,
            signal_evaluation,
            confirmation_teacher.predict_margin(X_evaluation),
        )
        pair_grids = {
            pair: PairGrid(grids[pair[0]], grids[pair[1]]) for pair in union_candidates
        }
        inference_started = time.perf_counter()
        inference = run_model_inference(
            confirmation_teacher.predict_margin,
            X_evaluation,
            pair_grids,
            n_bootstrap=args.n_bootstrap,
            seed=args.seed * 2 + 3,
            grid_batch_size=args.grid_batch_size,
        )
        inference_seconds = time.perf_counter() - inference_started
        oracle = compute_run_matched_oracle(
            args.dgp,
            X_evaluation,
            grids,
            sorted(truth),
            grid_batch_size=args.grid_batch_size,
        )
        masses = dict(oracle.masses)
        pair_index = {pair: index for index, pair in enumerate(inference.pairs)}
        method_summaries: dict[str, object] = {}
        method_graphs: dict[str, dict[str, tuple[Pair, ...]]] = {}
        for method, candidates in candidate_sets.items():
            indices = np.asarray([pair_index[pair] for pair in candidates], dtype=np.int64)
            marginal = np.asarray([inference.pvalues[pair] for pair in candidates])
            bh_mask = bh_rejections(marginal, alpha=args.alpha)
            method_max_t = maxT_pvalues(
                inference.observed_statistics[indices],
                inference.bootstrap_statistics[:, indices],
            )
            max_t_mask = method_max_t <= args.alpha
            bh_graph = tuple(pair for pair, keep in zip(candidates, bh_mask) if keep)
            max_t_graph = tuple(pair for pair, keep in zip(candidates, max_t_mask) if keep)
            method_graphs[method] = {"BH": bh_graph, "maxT": max_t_graph}
            null_candidates = set(candidates) - truth
            summary = {
                "candidate_pairs": [list(pair) for pair in candidates],
                "stage1": _ranking_metrics(
                    rankings[method], scores[method], truth, masses, top_k=args.top_k
                ),
                "stage2_bh": _graph_metrics(set(bh_graph), truth, masses),
                "stage2_maxT": _graph_metrics(set(max_t_graph), truth, masses),
                "selector_seconds": selector_seconds[method],
            }
            for key, graph in (("stage2_bh", set(bh_graph)), ("stage2_maxT", set(max_t_graph))):
                summary[key]["candidate_null_rejection_rate"] = (
                    len(graph & null_candidates) / len(null_candidates)
                    if null_candidates
                    else None
                )
            method_summaries[method] = summary

        candidate_rows: list[dict[str, object]] = []
        rank_positions = {
            method: {pair: index + 1 for index, pair in enumerate(ranking)}
            for method, ranking in rankings.items()
        }
        for pair in union_candidates:
            candidate_rows.append(
                {
                    "teacher": args.teacher,
                    "dgp": args.dgp,
                    "seed": args.seed,
                    "pair": list(pair),
                    "is_true": pair in truth,
                    "oracle_mass": masses.get(pair),
                    "member_of_ranker_top_k": [
                        method for method, candidates in candidate_sets.items() if pair in candidates
                    ],
                    "discovery_rank": {
                        method: rank_positions[method][pair] for method in rankings
                    },
                    "discovery_score": {method: scores[method][pair] for method in rankings},
                    "T_E_observed": float(
                        inference.observed_statistics[pair_index[pair]]
                    ),
                    "multiplier_pvalue": inference.pvalues[pair],
                    "selected_by_bh": [
                        method for method, graphs in method_graphs.items() if pair in graphs["BH"]
                    ],
                    "selected_by_maxT": [
                        method for method, graphs in method_graphs.items() if pair in graphs["maxT"]
                    ],
                }
            )
        run.write_candidate_rows(candidate_rows)
        _write_npz_atomic(
            run.root / "multiplier_statistics.npz",
            pairs=np.asarray(inference.pairs, dtype=np.int64),
            observed_statistics=np.asarray(inference.observed_statistics, dtype=np.float64),
            bootstrap_statistics=np.asarray(inference.bootstrap_statistics, dtype=np.float64),
            discovery_indices_sha256=np.asarray(_sha256_ints(split.discovery)),
            confirmation_indices_sha256=np.asarray(_sha256_ints(split.confirmation)),
            evaluation_indices_sha256=np.asarray(_sha256_ints(split.evaluation)),
            ranking_indices_sha256=np.asarray(_sha256_ints(ranking_global)),
            schema=np.asarray("lens-teacher-family-multiplier-statistics-1"),
        )
        summary = {
            "schema_version": "lens-teacher-family-sensitivity-cell-1",
            "status": "DONE",
            "teacher": args.teacher,
            "dgp": args.dgp,
            "seed": args.seed,
            "n": len(target),
            "p": features.shape[1],
            "split_sizes": {
                "discovery": len(split.discovery),
                "confirmation": len(split.confirmation),
                "evaluation": len(split.evaluation),
            },
            "split_indices_sha256": {
                "discovery": _sha256_ints(split.discovery),
                "confirmation": _sha256_ints(split.confirmation),
                "evaluation": _sha256_ints(split.evaluation),
            },
            "ranking_sample_size": len(X_ranking),
            "ranking_indices_sha256": _sha256_ints(ranking_global),
            "candidate_budget": args.top_k,
            "candidate_union_count": len(union_candidates),
            "n_bootstrap": args.n_bootstrap,
            "calibration": "LENS-Model shared Rademacher multiplier",
            "rankers": list(rankings),
            "methods": method_summaries,
            "teacher_fidelity": {
                "discovery": discovery_fidelity,
                "confirmation": confirmation_fidelity,
            },
            "teacher_fit_metadata": {
                "discovery": discovery_teacher_metadata,
                "confirmation": _teacher_metadata(confirmation_teacher),
            },
            "family_baseline_metadata": family_baseline_metadata,
            "oracle": {
                "definition": oracle.definition,
                "total_true_mass": oracle.total_true_mass,
                "candidate_independent": True,
            },
            "timings": {
                "discovery_teacher": discovery_fit_seconds,
                "confirmation_teacher": confirmation_fit_seconds,
                "stage1_selectors": selector_seconds,
                "stage2_multiplier": inference_seconds,
                "wall_seconds": time.perf_counter() - wall_started,
            },
            "claim_boundary": (
                "Teacher-family sensitivity with fixed-teacher LENS-Model multiplier; "
                "not the accepted LENS-Stable exact-permutation track."
            ),
        }
        run.finish(summary, extra_required_files=["multiplier_statistics.npz"])
    except BaseException as error:
        run.fail(error)
        raise


if __name__ == "__main__":
    main()
