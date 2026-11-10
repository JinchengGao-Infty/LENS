"""Independent discovery/confirmation/evaluation LENS-Stable pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence

import numpy as np

from .cache import InteractionSummary, summarize_sample, write_refit_cache
from .calibration.exact_rank import exact_rank_pvalue
from .calibration.maxT import maxT_pvalues, stepdown_maxT_pvalues
from .calibration.multiple_testing import bh_rejections, by_rejections
from .grids import FeatureKind, Grid1D, PairGrid, build_grid
from .interaction_vectors import InteractionSample, compute_interaction_sample
from .ranking import RankingScore, rank_interactions
from .splits import ThreeWaySplit, make_three_way_split
from .teachers import TeacherFactory


@dataclass(frozen=True)
class StableLensConfig:
    top_k: int = 15
    n_permutations: int = 20
    grid_points: int = 12
    grid_batch_size: int = 144
    n_crossfit_folds: int = 3
    ranking_score: RankingScore = "raw_energy"
    alpha: float = 0.05
    split_fractions: tuple[float, float, float] = (0.34, 0.33, 0.33)
    include_experimental_calibrators: bool = False

    def validate(self) -> None:
        if self.top_k < 1 or self.n_permutations < 2:
            raise ValueError("top_k must be positive and n_permutations must be >= 2")
        if self.grid_points < 2 or self.grid_batch_size < 1:
            raise ValueError("invalid grid configuration")
        if self.n_crossfit_folds < 3:
            raise ValueError("CF calibration needs at least three folds")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must lie in (0,1)")


@dataclass(frozen=True)
class StableLensResult:
    split: ThreeWaySplit
    ranking: list[tuple[tuple[int, int], float]]
    candidates: tuple[tuple[int, int], ...]
    observed: Mapping[tuple[int, int], InteractionSummary]
    refits: Mapping[tuple[int, int], tuple[InteractionSummary, ...]]
    pvalues: Mapping[str, Mapping[tuple[int, int], float]]
    graphs: Mapping[str, tuple[tuple[int, int], ...]]
    null_statistics: np.ndarray
    timings: Mapping[str, float]
    diagnostics: Mapping[str, object]


def _all_pairs(n_features: int) -> list[tuple[int, int]]:
    return [(left, right) for left in range(n_features) for right in range(left + 1, n_features)]


def _feature_grids(
    X_reference: np.ndarray,
    feature_kinds: Sequence[FeatureKind],
    grid_points: int,
) -> dict[int, Grid1D]:
    if len(feature_kinds) != X_reference.shape[1]:
        raise ValueError("feature_kinds length must equal n_features")
    return {
        index: build_grid(X_reference[:, index], kind, n_points=grid_points)
        for index, kind in enumerate(feature_kinds)
    }


def _samples_for_pairs(
    teacher,
    X_eval: np.ndarray,
    pairs: Sequence[tuple[int, int]],
    grids: Mapping[int, Grid1D],
    *,
    grid_batch_size: int,
    estimate_covariance: bool | str,
) -> dict[tuple[int, int], InteractionSample]:
    samples = {}
    for pair in pairs:
        samples[pair] = compute_interaction_sample(
            teacher.predict_margin,
            X_eval,
            pair,
            PairGrid(grids[pair[0]], grids[pair[1]]),
            grid_batch_size=grid_batch_size,
            estimate_covariance=estimate_covariance,
        )
    return samples


def run_stable_lens(
    X: np.ndarray,
    y: np.ndarray,
    *,
    feature_kinds: Sequence[FeatureKind],
    teacher_factory: TeacherFactory,
    seed: int,
    config: StableLensConfig = StableLensConfig(),
    split: ThreeWaySplit | None = None,
    cache_root: str | Path | None = None,
    dgp_name: str = "unknown",
    progress_callback: Callable[[dict], None] | None = None,
) -> StableLensResult:
    """Run LENS-Stable with an independent candidate set and shared refits.

    The confirmation teacher and every permutation teacher use the same training
    seed. Thus, conditional on the data, only the exchangeable label vector
    changes across the observed/permuted training maps.

    Non-promoted Gamma calibrators are excluded from the default result.  Set
    ``config.include_experimental_calibrators`` explicitly to reproduce their
    historical research outputs.
    """
    t_total = time.perf_counter()
    timings = {
        "discovery_teacher": 0.0,
        "discovery_interactions": 0.0,
        "confirmation_teacher": 0.0,
        "observed_interactions": 0.0,
        "null_teacher_refits": 0.0,
        "null_interactions": 0.0,
        "covariance_summaries": 0.0,
        "tail_evaluation": 0.0,
    }
    config.validate()
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y)
    if features.ndim != 2 or target.ndim != 1 or len(features) != len(target):
        raise ValueError("X and y must have aligned shapes (n,p) and (n,)")
    if np.any(np.isinf(features)) or not np.all(np.isfinite(target)):
        raise ValueError("X must not contain inf and y must be finite")
    if split is None:
        split = make_three_way_split(
            len(target),
            seed=seed,
            fractions=config.split_fractions,
        )
    split.validate(len(target))

    X_discovery = features[split.discovery]
    y_discovery = target[split.discovery]
    X_confirmation = features[split.confirmation]
    y_confirmation = target[split.confirmation]
    X_evaluation = features[split.evaluation]
    if X_evaluation.shape[0] < 2 * config.n_crossfit_folds:
        raise ValueError("evaluation split is too small for cross-fitted covariance summaries")

    grids = _feature_grids(X_discovery, feature_kinds, config.grid_points)
    pairs = _all_pairs(features.shape[1])

    discovery_seed = int(seed * 2 + 1)
    confirmation_seed = int(seed * 2 + 2)
    started = time.perf_counter()
    discovery_teacher = teacher_factory.fit(X_discovery, y_discovery, seed=discovery_seed)
    timings["discovery_teacher"] += time.perf_counter() - started
    started = time.perf_counter()
    discovery_samples = _samples_for_pairs(
        discovery_teacher,
        X_discovery,
        pairs,
        grids,
        grid_batch_size=config.grid_batch_size,
        estimate_covariance=(
            "trace" if config.ranking_score == "debiased_energy" else "none"
        ),
    )
    timings["discovery_interactions"] += time.perf_counter() - started
    ranking = rank_interactions(discovery_samples, score=config.ranking_score)
    candidates = tuple(pair for pair, _ in ranking[: min(config.top_k, len(ranking))])
    if progress_callback is not None:
        progress_callback({"stage": "discovery", "n_candidates": len(candidates)})

    started = time.perf_counter()
    confirmation_teacher = teacher_factory.fit(
        X_confirmation,
        y_confirmation,
        seed=confirmation_seed,
    )
    timings["confirmation_teacher"] += time.perf_counter() - started
    started = time.perf_counter()
    observed_samples = _samples_for_pairs(
        confirmation_teacher,
        X_evaluation,
        candidates,
        grids,
        grid_batch_size=config.grid_batch_size,
        estimate_covariance="none",
    )
    timings["observed_interactions"] += time.perf_counter() - started
    started = time.perf_counter()
    observed = {
        pair: summarize_sample(sample, n_folds=config.n_crossfit_folds)
        for pair, sample in observed_samples.items()
    }
    timings["covariance_summaries"] += time.perf_counter() - started
    if progress_callback is not None:
        progress_callback({"stage": "observed_confirmation", "n_candidates": len(candidates)})

    refits: dict[tuple[int, int], list[InteractionSummary]] = {
        pair: [] for pair in candidates
    }
    null_statistics = np.empty((config.n_permutations, len(candidates)), dtype=np.float64)
    permutation_rng = np.random.default_rng(seed * 2 + 3)
    for refit_index in range(config.n_permutations):
        permuted_labels = permutation_rng.permutation(y_confirmation)
        started = time.perf_counter()
        null_teacher = teacher_factory.fit(
            X_confirmation,
            permuted_labels,
            seed=confirmation_seed,
        )
        timings["null_teacher_refits"] += time.perf_counter() - started
        started = time.perf_counter()
        null_samples = _samples_for_pairs(
            null_teacher,
            X_evaluation,
            candidates,
            grids,
            grid_batch_size=config.grid_batch_size,
            estimate_covariance="none",
        )
        timings["null_interactions"] += time.perf_counter() - started
        started = time.perf_counter()
        for candidate_index, pair in enumerate(candidates):
            summary = summarize_sample(
                null_samples[pair],
                n_folds=config.n_crossfit_folds,
            )
            refits[pair].append(summary)
            null_statistics[refit_index, candidate_index] = summary.statistic
        if cache_root is not None:
            write_refit_cache(
                cache_root,
                seed=seed,
                dgp=dgp_name,
                teacher=teacher_factory.name,
                refit=refit_index,
                summaries={pair: refits[pair][-1] for pair in candidates},
                metadata={
                    "confirmation_seed": confirmation_seed,
                    "vector_order": "F",
                    "interaction_coordinates": "weighted_ANOVA_orthonormal_contrasts",
                    "calibration_role": "permuted_confirmation",
                },
            )
        timings["covariance_summaries"] += time.perf_counter() - started
        if progress_callback is not None:
            progress_callback(
                {
                    "stage": "permutation_refit",
                    "completed": refit_index + 1,
                    "total": config.n_permutations,
                }
            )

    if cache_root is not None:
        write_refit_cache(
            cache_root,
            seed=seed,
            dgp=dgp_name,
            teacher=teacher_factory.name,
            refit="observed",
            summaries=observed,
            metadata={
                "confirmation_seed": confirmation_seed,
                "vector_order": "F",
                "interaction_coordinates": "weighted_ANOVA_orthonormal_contrasts",
                "calibration_role": "observed_confirmation",
            },
        )

    tail_started = time.perf_counter()
    observed_statistics = np.asarray([observed[pair].statistic for pair in candidates])
    exact = {
        pair: exact_rank_pvalue(observed_statistics[index], null_statistics[:, index])
        for index, pair in enumerate(candidates)
    }
    max_t_values = maxT_pvalues(observed_statistics, null_statistics)
    stepdown_values = stepdown_maxT_pvalues(observed_statistics, null_statistics)
    max_t = {pair: float(max_t_values[index]) for index, pair in enumerate(candidates)}
    stepdown = {pair: float(stepdown_values[index]) for index, pair in enumerate(candidates)}
    pvalues = {
        "Exact-Perm": exact,
        "maxT": max_t,
        "stepdown-maxT": stepdown,
    }

    exact_array = np.asarray([exact[pair] for pair in candidates])
    bh_mask = bh_rejections(exact_array, alpha=config.alpha)
    by_mask = by_rejections(exact_array, alpha=config.alpha)
    max_t_mask = max_t_values <= config.alpha
    stepdown_mask = stepdown_values <= config.alpha
    graphs = {
        "LENS-BH": tuple(pair for pair, keep in zip(candidates, bh_mask) if keep),
        "LENS-BH-Exact": tuple(pair for pair, keep in zip(candidates, bh_mask) if keep),
        "LENS-BY": tuple(pair for pair, keep in zip(candidates, by_mask) if keep),
        "LENS-maxT": tuple(pair for pair, keep in zip(candidates, max_t_mask) if keep),
        "LENS-stepdown-maxT": tuple(
            pair for pair, keep in zip(candidates, stepdown_mask) if keep
        ),
    }
    if config.include_experimental_calibrators:
        from .experimental.calibration.cf_mixture_gamma import (
            cf_mixture_gamma_pvalue,
        )
        from .experimental.calibration.cf_pooled_gamma import (
            cf_pooled_gamma_pvalue,
        )
        from .experimental.calibration.scalar_gamma import scalar_gamma_pvalue

        experimental_pvalues = {
            "Scalar-Gamma": {
                pair: scalar_gamma_pvalue(
                    observed_statistics[index], null_statistics[:, index]
                )
                for index, pair in enumerate(candidates)
            },
            "CF-Pooled-Gamma": {
                pair: cf_pooled_gamma_pvalue(observed_statistics[index], refits[pair])
                for index, pair in enumerate(candidates)
            },
            "CF-Mixture-Gamma": {
                pair: cf_mixture_gamma_pvalue(observed_statistics[index], refits[pair])
                for index, pair in enumerate(candidates)
            },
        }
        pvalues.update(experimental_pvalues)
        for method, method_pvalues in experimental_pvalues.items():
            values = np.asarray([method_pvalues[pair] for pair in candidates])
            mask = bh_rejections(values, alpha=config.alpha)
            graphs[f"LENS-BH-{method}"] = tuple(
                pair for pair, keep in zip(candidates, mask) if keep
            )
    timings["tail_evaluation"] += time.perf_counter() - tail_started
    timings["total"] = time.perf_counter() - t_total
    minimum_exact_permutations_for_bh = int(np.ceil(len(candidates) / config.alpha) - 1)
    diagnostics = {
        "minimum_attainable_exact_p": 1.0 / (config.n_permutations + 1),
        "minimum_exact_permutations_for_first_bh_rejection": (
            minimum_exact_permutations_for_bh
        ),
        "exact_bh_resolution_sufficient": (
            config.n_permutations >= minimum_exact_permutations_for_bh
        ),
        "exactness_scope": "global_null_after_independent_screening",
        "bh_scope": "power_oriented_heuristic",
        "experimental_calibrators_included": (
            config.include_experimental_calibrators
        ),
    }

    return StableLensResult(
        split=split,
        ranking=ranking,
        candidates=candidates,
        observed=observed,
        refits={pair: tuple(summaries) for pair, summaries in refits.items()},
        pvalues=pvalues,
        graphs=graphs,
        null_statistics=null_statistics,
        timings={key: float(value) for key, value in timings.items()},
        diagnostics=diagnostics,
    )
