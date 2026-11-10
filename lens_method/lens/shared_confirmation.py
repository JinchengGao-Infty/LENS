"""Shared exact confirmation for discovery-measurable candidate sets.

Every registered ranker freezes its candidates before this module is called.
One observed confirmation teacher and one stream of permuted confirmation
teachers are then shared across the sorted candidate union.  Multiplicity is
applied separately inside each ranker's fixed candidate set.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .cache import InteractionSummary, summarize_sample, write_refit_cache
from .calibration.exact_rank import exact_rank_pvalue
from .calibration.multiple_testing import bh_rejections
from .grids import Grid1D, PairGrid
from .interaction_vectors import compute_interaction_sample
from .pipeline import StableLensConfig
from .splits import ThreeWaySplit
from .teachers import FittedTeacher, TeacherFactory

Pair = tuple[int, int]
CacheMode = Literal["exact_statistics", "full_summary"]


@dataclass(frozen=True)
class SharedExactConfirmationResult:
    split: ThreeWaySplit
    confirmation_teacher: FittedTeacher
    candidate_sets: Mapping[str, tuple[Pair, ...]]
    union_candidates: tuple[Pair, ...]
    observed_statistics: Mapping[Pair, float]
    exact_pvalues: Mapping[Pair, float]
    graphs: Mapping[str, tuple[Pair, ...]]
    null_statistics: np.ndarray
    timings: Mapping[str, float]
    diagnostics: Mapping[str, object]


def candidate_sets_digest(candidate_sets: Mapping[str, Sequence[Pair]]) -> str:
    canonical = {
        str(method): [list(pair) for pair in pairs]
        for method, pairs in sorted(candidate_sets.items())
    }
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _validate_candidate_sets(
    candidate_sets: Mapping[str, Sequence[Pair]],
    *,
    n_features: int,
    candidate_budget: int,
) -> dict[str, tuple[Pair, ...]]:
    if not candidate_sets:
        raise ValueError("candidate_sets must not be empty")
    normalized: dict[str, tuple[Pair, ...]] = {}
    for method, raw_pairs in candidate_sets.items():
        name = str(method)
        if not name or name in normalized:
            raise ValueError(f"invalid or duplicate method name {name!r}")
        pairs = tuple((int(pair[0]), int(pair[1])) for pair in raw_pairs)
        if len(pairs) != candidate_budget:
            raise ValueError(
                f"{name} emitted {len(pairs)} candidates; expected {candidate_budget}"
            )
        if len(set(pairs)) != len(pairs):
            raise ValueError(f"{name} candidate set contains duplicates")
        for pair in pairs:
            if not (0 <= pair[0] < pair[1] < n_features):
                raise ValueError(f"{name} emitted invalid pair {pair}")
        normalized[name] = pairs
    return normalized


def run_shared_exact_confirmation(
    X: np.ndarray,
    y: np.ndarray,
    *,
    split: ThreeWaySplit,
    grids: Mapping[int, Grid1D],
    candidate_sets: Mapping[str, Sequence[Pair]],
    teacher_factory: TeacherFactory,
    seed: int,
    config: StableLensConfig,
    cache_root: str | Path | None = None,
    cache_mode: CacheMode = "exact_statistics",
    dgp_name: str = "unknown",
    progress_callback: Callable[[dict], None] | None = None,
) -> SharedExactConfirmationResult:
    """Confirm multiple fixed ranker candidate sets with one exact refit stream."""
    started_total = time.perf_counter()
    config.validate()
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y)
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("X and y must have aligned shapes (n,p) and (n,)")
    if np.any(np.isinf(features)) or not np.all(np.isfinite(target)):
        raise ValueError("X must not contain inf and y must be finite")
    split.validate(len(target))
    if set(grids) != set(range(features.shape[1])):
        raise ValueError("grids must contain exactly one entry per feature")
    if cache_mode not in {"exact_statistics", "full_summary"}:
        raise ValueError(f"unknown cache_mode {cache_mode!r}")
    if cache_mode == "exact_statistics" and cache_root is not None:
        raise ValueError("exact_statistics mode must not receive cache_root")
    if cache_mode == "full_summary" and cache_root is None:
        raise ValueError("full_summary mode requires cache_root")

    fixed_sets = _validate_candidate_sets(
        candidate_sets,
        n_features=features.shape[1],
        candidate_budget=config.top_k,
    )
    union_candidates = tuple(sorted({pair for pairs in fixed_sets.values() for pair in pairs}))
    if not union_candidates:
        raise ValueError("candidate union must not be empty")
    sets_digest = candidate_sets_digest(fixed_sets)

    X_confirmation = features[split.confirmation]
    y_confirmation = target[split.confirmation]
    X_evaluation = features[split.evaluation]
    if X_evaluation.shape[0] < 2 * config.n_crossfit_folds:
        raise ValueError("evaluation split is too small for cross-fitted summaries")

    timings = {
        "confirmation_teacher": 0.0,
        "observed_interactions": 0.0,
        "null_teacher_refits": 0.0,
        "null_interactions": 0.0,
        "covariance_summaries": 0.0,
        "cache_writes": 0.0,
        "tail_evaluation": 0.0,
    }
    confirmation_seed = int(seed * 2 + 2)
    fit_started = time.perf_counter()
    confirmation_teacher = teacher_factory.fit(
        X_confirmation,
        y_confirmation,
        seed=confirmation_seed,
    )
    timings["confirmation_teacher"] += time.perf_counter() - fit_started

    interaction_started = time.perf_counter()
    observed_samples = {
        pair: compute_interaction_sample(
            confirmation_teacher.predict_margin,
            X_evaluation,
            pair,
            PairGrid(grids[pair[0]], grids[pair[1]]),
            grid_batch_size=config.grid_batch_size,
            estimate_covariance="none",
        )
        for pair in union_candidates
    }
    timings["observed_interactions"] += time.perf_counter() - interaction_started
    # The exact statistic is read directly from the common interaction sample in
    # both cache modes.  Full summaries are optional audit material and must not
    # alter T_E, p-values, or graph decisions.
    observed_statistics = {
        pair: float(sample.statistic) for pair, sample in observed_samples.items()
    }
    observed_summaries: dict[Pair, InteractionSummary] | None = None
    if cache_mode == "full_summary":
        summary_started = time.perf_counter()
        observed_summaries = {
            pair: summarize_sample(sample, n_folds=config.n_crossfit_folds)
            for pair, sample in observed_samples.items()
        }
        timings["covariance_summaries"] += time.perf_counter() - summary_started
    if progress_callback is not None:
        progress_callback(
            {
                "stage": "observed_confirmation",
                "n_union_candidates": len(union_candidates),
            }
        )

    cache_metadata = {
        "confirmation_seed": confirmation_seed,
        "vector_order": "F",
        "interaction_coordinates": "weighted_ANOVA_orthonormal_contrasts",
        "candidate_sets_sha256": sets_digest,
        "candidate_union_count": len(union_candidates),
        "shared_refit_across_rankers": True,
        "cache_mode": cache_mode,
    }
    if cache_mode == "full_summary":
        assert cache_root is not None and observed_summaries is not None
        cache_started = time.perf_counter()
        write_refit_cache(
            cache_root,
            seed=seed,
            dgp=dgp_name,
            teacher=teacher_factory.name,
            refit="observed",
            summaries=observed_summaries,
            metadata={**cache_metadata, "calibration_role": "observed_confirmation"},
        )
        timings["cache_writes"] += time.perf_counter() - cache_started

    null_statistics = np.empty(
        (config.n_permutations, len(union_candidates)), dtype=np.float64
    )
    permutation_rng = np.random.default_rng(seed * 2 + 3)
    for refit_index in range(config.n_permutations):
        permuted_labels = permutation_rng.permutation(y_confirmation)
        fit_started = time.perf_counter()
        null_teacher = teacher_factory.fit(
            X_confirmation,
            permuted_labels,
            seed=confirmation_seed,
        )
        timings["null_teacher_refits"] += time.perf_counter() - fit_started
        interaction_started = time.perf_counter()
        null_samples = {
            pair: compute_interaction_sample(
                null_teacher.predict_margin,
                X_evaluation,
                pair,
                PairGrid(grids[pair[0]], grids[pair[1]]),
                grid_batch_size=config.grid_batch_size,
                estimate_covariance="none",
            )
            for pair in union_candidates
        }
        timings["null_interactions"] += time.perf_counter() - interaction_started
        for candidate_index, pair in enumerate(union_candidates):
            null_statistics[refit_index, candidate_index] = null_samples[pair].statistic
        summaries: dict[Pair, InteractionSummary] | None = None
        if cache_mode == "full_summary":
            summary_started = time.perf_counter()
            summaries = {
                pair: summarize_sample(sample, n_folds=config.n_crossfit_folds)
                for pair, sample in null_samples.items()
            }
            timings["covariance_summaries"] += time.perf_counter() - summary_started
        if cache_mode == "full_summary":
            assert cache_root is not None and summaries is not None
            cache_started = time.perf_counter()
            write_refit_cache(
                cache_root,
                seed=seed,
                dgp=dgp_name,
                teacher=teacher_factory.name,
                refit=refit_index,
                summaries=summaries,
                metadata={
                    **cache_metadata,
                    "calibration_role": "permuted_confirmation",
                },
            )
            timings["cache_writes"] += time.perf_counter() - cache_started
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "permutation_refit",
                    "completed": refit_index + 1,
                    "total": config.n_permutations,
                    "n_union_candidates": len(union_candidates),
                }
            )

    tail_started = time.perf_counter()
    observed_statistics_array = np.asarray(
        [observed_statistics[pair] for pair in union_candidates], dtype=np.float64
    )
    exact_pvalues = {
        pair: exact_rank_pvalue(
            observed_statistics_array[index], null_statistics[:, index]
        )
        for index, pair in enumerate(union_candidates)
    }
    graphs: dict[str, tuple[Pair, ...]] = {}
    for method, candidates in fixed_sets.items():
        pvalues = np.asarray([exact_pvalues[pair] for pair in candidates], dtype=np.float64)
        keep = bh_rejections(pvalues, alpha=config.alpha)
        graphs[method] = tuple(pair for pair, selected in zip(candidates, keep) if selected)
    timings["tail_evaluation"] += time.perf_counter() - tail_started
    timings["total"] = time.perf_counter() - started_total

    minimum_permutations = int(np.ceil(config.top_k / config.alpha) - 1)
    diagnostics = {
        "candidate_budget_per_ranker": config.top_k,
        "candidate_union_count": len(union_candidates),
        "candidate_sets_sha256": sets_digest,
        "minimum_attainable_exact_p": 1.0 / (config.n_permutations + 1),
        "minimum_exact_permutations_for_first_bh_rejection": minimum_permutations,
        "exact_bh_resolution_sufficient": config.n_permutations >= minimum_permutations,
        "exactness_scope": "global_null_after_independent_screening",
        "bh_scope": "power_oriented_heuristic_per_ranker_fixed_candidates",
        "shared_refit_across_rankers": True,
        "cache_mode": cache_mode,
        "full_summary_cache_retained": cache_mode == "full_summary",
        "confirmation_seed": confirmation_seed,
        "permutation_seed": int(seed * 2 + 3),
        "refit_teacher_seed_policy": "same_as_observed_confirmation",
    }
    return SharedExactConfirmationResult(
        split=split,
        confirmation_teacher=confirmation_teacher,
        candidate_sets=fixed_sets,
        union_candidates=union_candidates,
        observed_statistics=observed_statistics,
        exact_pvalues=exact_pvalues,
        graphs=graphs,
        null_statistics=null_statistics,
        timings={key: float(value) for key, value in timings.items()},
        diagnostics=diagnostics,
    )
