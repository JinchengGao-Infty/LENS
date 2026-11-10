"""Fixed-teacher LENS-Model inference on an independent sample."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import norm

from .calibration.maxT import maxT_pvalues
from .grids import PairGrid
from .interaction_vectors import InteractionSample, compute_interaction_sample


@dataclass(frozen=True)
class ModelInferenceResult:
    pairs: tuple[tuple[int, int], ...]
    samples: Mapping[tuple[int, int], InteractionSample]
    pvalues: Mapping[tuple[int, int], float]
    maxT_pvalues: Mapping[tuple[int, int], float]
    observed_statistics: np.ndarray
    bootstrap_statistics: np.ndarray


@dataclass(frozen=True)
class ReplicationInferenceResult:
    """One-sided inference for cross-teacher interaction replication.

    The discovery interaction vector is frozen before the confirmation and
    inference rows are used.  For each pair ``u``, the estimand is

    ``theta_u = <r_disc,u, r_conf,u> / ||r_disc,u||^2``.

    Thus ``theta_u`` is the confirmation-teacher amplitude in the frozen
    discovery interaction direction, relative to the discovery amplitude.
    The tested null is ``theta_u <= minimum_amplitude``.  P-values use one
    shared multiplier matrix across pairs and are conditional/asymptotic, not
    finite-sample exact.
    """

    pairs: tuple[tuple[int, int], ...]
    minimum_amplitude: float
    amplitude_ratios: Mapping[tuple[int, int], float]
    standard_errors: Mapping[tuple[int, int], float]
    pvalues: Mapping[tuple[int, int], float]
    discovery_energies: Mapping[tuple[int, int], float]
    confirmation_energies: Mapping[tuple[int, int], float]
    cosine_similarities: Mapping[tuple[int, int], float]
    observed_statistics: np.ndarray
    bootstrap_statistics: np.ndarray


def replication_pvalues_at_threshold(
    result: ReplicationInferenceResult,
    minimum_amplitude: float,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Re-evaluate a replication threshold without new teacher queries.

    The centered multiplier matrix is threshold-invariant.  Changing the null
    boundary only shifts the observed studentized statistic, so a development
    sweep over predeclared practical-amplitude thresholds does not require
    recomputing interaction surfaces or multiplier draws.
    """
    if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0:
        raise ValueError("minimum_amplitude must be finite and non-negative")
    observed = np.empty(len(result.pairs), dtype=np.float64)
    pvalues: dict[tuple[int, int], float] = {}
    n_bootstrap = result.bootstrap_statistics.shape[0]
    if result.bootstrap_statistics.shape[1] != len(result.pairs):
        raise ValueError("replication bootstrap matrix disagrees with pair count")
    for index, pair in enumerate(result.pairs):
        standard_error = float(result.standard_errors[pair])
        amplitude = float(result.amplitude_ratios[pair])
        if standard_error <= 0:
            observed[index] = -1.0 if amplitude <= minimum_amplitude else 1.0
            pvalues[pair] = 1.0 if amplitude <= minimum_amplitude else 1.0 / (
                n_bootstrap + 1
            )
            continue
        observed[index] = (amplitude - minimum_amplitude) / standard_error
        pvalues[pair] = float(
            (
                1
                + np.count_nonzero(
                    result.bootstrap_statistics[:, index] >= observed[index]
                )
            )
            / (n_bootstrap + 1)
        )
    return observed, pvalues


def normal_replication_pvalues_at_threshold(
    result: ReplicationInferenceResult,
    minimum_amplitude: float,
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Studentized-normal p-values for a frozen replication threshold.

    This is the no-resampling deployment calibration.  Its validity is
    conditional and asymptotic via a scalar self-normalized CLT for each fixed
    discovery-measurable candidate; it is not finite-sample exact.  BY can be
    applied afterward when arbitrary dependence across candidate p-values is
    desired.
    """
    if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0:
        raise ValueError("minimum_amplitude must be finite and non-negative")
    thresholds = {pair: float(minimum_amplitude) for pair in result.pairs}
    return normal_replication_pvalues_at_pair_thresholds(result, thresholds)


def normal_replication_pvalues_at_pair_thresholds(
    result: ReplicationInferenceResult,
    minimum_amplitudes: Mapping[tuple[int, int], float],
) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    """Normal p-values for discovery-measurable pair-specific null boundaries.

    Conditional on discovery, each boundary is fixed before confirmation and
    evaluation rows are used.  Therefore replacing one common boundary by a
    complete discovery-measurable mapping preserves the same pointwise scalar
    self-normalized CLT; a multiplicity procedure is still required afterward.

    A zero empirical standard error is handled without division: the p-value
    is 1 when the observed amplitude is at or below its boundary and 0 when it
    is above.  Under the positive-population-variance CLT contract, this sample
    degeneracy event has probability tending to zero; population-degenerate
    nulls take the p=1 branch.
    """

    if set(minimum_amplitudes) != set(result.pairs):
        raise ValueError("pair thresholds must match the complete replication family")
    observed = np.empty(len(result.pairs), dtype=np.float64)
    pvalues: dict[tuple[int, int], float] = {}
    for index, pair in enumerate(result.pairs):
        minimum_amplitude = float(minimum_amplitudes[pair])
        if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0:
            raise ValueError("pair thresholds must be finite and non-negative")
        standard_error = float(result.standard_errors[pair])
        amplitude = float(result.amplitude_ratios[pair])
        if standard_error <= 0:
            observed[index] = -1.0 if amplitude <= minimum_amplitude else 1.0
            pvalues[pair] = 1.0 if amplitude <= minimum_amplitude else 0.0
            continue
        observed[index] = (amplitude - minimum_amplitude) / standard_error
        pvalues[pair] = float(norm.sf(observed[index]))
    return observed, pvalues


def family_rms_floor_amplitude_thresholds(
    result: ReplicationInferenceResult,
    minimum_amplitude: float,
    *,
    energy_tolerance: float = 1e-12,
) -> tuple[float, dict[tuple[int, int], float]]:
    """Lift pair-relative practical nulls to a common family RMS scale.

    Let ``r_u = ||d_u||`` be pair ``u``'s frozen discovery-direction norm and
    ``s_C = sqrt(mean_u r_u^2)`` the RMS norm over the declared family.  The
    returned ratio-scale boundary is

    ``b_u = eta * max(1, s_C / max(r_u, sqrt(energy_tolerance)))``.

    Equivalently, the confirmation projection along the discovery unit
    direction must exceed ``eta * max(r_u, s_C)``.  Strong discovery pairs keep
    the original relative-effect null; weak pairs cannot use their own nearly
    zero magnitude as an arbitrarily small practical-significance threshold.
    """

    if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0:
        raise ValueError("minimum_amplitude must be finite and non-negative")
    if not np.isfinite(energy_tolerance) or energy_tolerance <= 0:
        raise ValueError("energy_tolerance must be finite and positive")
    energies = np.asarray(
        [result.discovery_energies[pair] for pair in result.pairs],
        dtype=np.float64,
    )
    if energies.shape != (len(result.pairs),) or not np.all(np.isfinite(energies)):
        raise ValueError("discovery energies must be complete and finite")
    if np.any(energies < 0):
        raise ValueError("discovery energies must be non-negative")
    family_rms_norm = float(np.sqrt(np.mean(energies)))
    norm_floor = float(np.sqrt(energy_tolerance))
    thresholds = {
        pair: float(
            minimum_amplitude
            * max(
                1.0,
                family_rms_norm
                / max(float(np.sqrt(result.discovery_energies[pair])), norm_floor),
            )
        )
        for pair in result.pairs
    }
    return family_rms_norm, thresholds


def _shared_rademacher_matrix(
    n_bootstrap: int,
    n_eval: int,
    *,
    seed: int,
) -> np.ndarray:
    if n_bootstrap < 1 or n_eval < 2:
        raise ValueError("n_bootstrap must be positive and n_eval must be at least 2")
    rng = np.random.default_rng(seed)
    multipliers = rng.integers(0, 2, size=(n_bootstrap, n_eval), dtype=np.int8)
    return multipliers.astype(np.float64) * 2.0 - 1.0


def run_model_inference(
    predict,
    X_inference: np.ndarray,
    pair_grids: Mapping[tuple[int, int], PairGrid],
    *,
    n_bootstrap: int = 999,
    seed: int = 0,
    grid_batch_size: int = 144,
) -> ModelInferenceResult:
    """Multiplier inference conditional on a frozen teacher and candidate set."""
    pairs: Sequence[tuple[int, int]] = tuple(sorted(pair_grids))
    samples = {
        pair: compute_interaction_sample(
            predict,
            X_inference,
            pair,
            pair_grids[pair],
            grid_batch_size=grid_batch_size,
        )
        for pair in pairs
    }
    n_eval = len(X_inference)
    multipliers = _shared_rademacher_matrix(n_bootstrap, n_eval, seed=seed)
    observed = np.asarray([samples[pair].statistic for pair in pairs])
    bootstrap = np.empty((n_bootstrap, len(pairs)), dtype=np.float64)
    pvalues = {}
    for index, pair in enumerate(pairs):
        centered = samples[pair].vectors - samples[pair].mean
        z_star = multipliers @ centered / np.sqrt(n_eval)
        statistics = np.einsum("bi,bi->b", z_star, z_star)
        bootstrap[:, index] = statistics
        pvalues[pair] = float(
            (1 + np.count_nonzero(statistics >= observed[index])) / (n_bootstrap + 1)
        )
    adjusted = maxT_pvalues(observed, bootstrap)
    return ModelInferenceResult(
        pairs=tuple(pairs),
        samples=samples,
        pvalues=pvalues,
        maxT_pvalues={pair: float(adjusted[index]) for index, pair in enumerate(pairs)},
        observed_statistics=observed,
        bootstrap_statistics=bootstrap,
    )


def run_replication_inference(
    discovery_means: Mapping[tuple[int, int], np.ndarray],
    confirmation_samples: Mapping[tuple[int, int], InteractionSample],
    pairs: Sequence[tuple[int, int]],
    *,
    minimum_amplitude: float = 0.0,
    n_bootstrap: int = 999,
    seed: int = 0,
    numeric_tolerance: float = 1e-12,
) -> ReplicationInferenceResult:
    """Test whether pair interactions replicate in a frozen discovery direction.

    Candidate order, discovery means, grids, and both fitted teachers must be
    fixed before the independent inference rows are inspected.  For pair
    ``u``, each inference-row scalar is

    ``a_i = <r_disc,u, h_conf,u,i> / ||r_disc,u||^2``.

    A shared Rademacher multiplier bootstrap calibrates the one-sided statistic
    ``sqrt(m) * (mean(a) - minimum_amplitude) / sd(a)``.  Since the multiplier
    approximation is applied to centered inference rows, this is an
    asymptotic model/replication test rather than an exact label-permutation
    test.
    """
    ordered_pairs = tuple((int(pair[0]), int(pair[1])) for pair in pairs)
    if not ordered_pairs or len(ordered_pairs) != len(set(ordered_pairs)):
        raise ValueError("pairs must be a non-empty unique sequence")
    if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0:
        raise ValueError("minimum_amplitude must be finite and non-negative")
    if numeric_tolerance <= 0 or not np.isfinite(numeric_tolerance):
        raise ValueError("numeric_tolerance must be finite and positive")
    missing_discovery = [pair for pair in ordered_pairs if pair not in discovery_means]
    missing_confirmation = [
        pair for pair in ordered_pairs if pair not in confirmation_samples
    ]
    if missing_discovery or missing_confirmation:
        raise ValueError(
            "replication inputs omit requested pairs: "
            f"discovery={missing_discovery}, confirmation={missing_confirmation}"
        )

    n_eval_values = {confirmation_samples[pair].n_eval for pair in ordered_pairs}
    if len(n_eval_values) != 1:
        raise ValueError("confirmation samples must share one inference-row count")
    n_eval = n_eval_values.pop()
    return _run_replication_inference_from_getter(
        discovery_means,
        lambda pair: confirmation_samples[pair],
        ordered_pairs,
        n_eval=n_eval,
        minimum_amplitude=minimum_amplitude,
        n_bootstrap=n_bootstrap,
        seed=seed,
        numeric_tolerance=numeric_tolerance,
    )


def run_replication_inference_streaming(
    discovery_means: Mapping[tuple[int, int], np.ndarray],
    confirmation_sample_factory: Callable[[tuple[int, int]], InteractionSample],
    pairs: Sequence[tuple[int, int]],
    *,
    n_eval: int,
    minimum_amplitude: float = 0.0,
    n_bootstrap: int = 999,
    seed: int = 0,
    numeric_tolerance: float = 1e-12,
) -> ReplicationInferenceResult:
    """Streaming equivalent of :func:`run_replication_inference`.

    Only one confirmation ``InteractionSample`` is materialized at a time.
    This keeps all-pair inference memory proportional to one pair while using
    the same shared multiplier matrix and returning the same statistics as the
    mapping-based implementation.
    """
    ordered_pairs = tuple((int(pair[0]), int(pair[1])) for pair in pairs)
    if not ordered_pairs or len(ordered_pairs) != len(set(ordered_pairs)):
        raise ValueError("pairs must be a non-empty unique sequence")
    if n_eval < 2:
        raise ValueError("n_eval must be at least 2")
    if not callable(confirmation_sample_factory):
        raise TypeError("confirmation_sample_factory must be callable")
    if not np.isfinite(minimum_amplitude) or minimum_amplitude < 0:
        raise ValueError("minimum_amplitude must be finite and non-negative")
    if numeric_tolerance <= 0 or not np.isfinite(numeric_tolerance):
        raise ValueError("numeric_tolerance must be finite and positive")
    missing_discovery = [pair for pair in ordered_pairs if pair not in discovery_means]
    if missing_discovery:
        raise ValueError(
            f"replication discovery inputs omit requested pairs: {missing_discovery}"
        )
    return _run_replication_inference_from_getter(
        discovery_means,
        confirmation_sample_factory,
        ordered_pairs,
        n_eval=n_eval,
        minimum_amplitude=minimum_amplitude,
        n_bootstrap=n_bootstrap,
        seed=seed,
        numeric_tolerance=numeric_tolerance,
    )


def _run_replication_inference_from_getter(
    discovery_means: Mapping[tuple[int, int], np.ndarray],
    sample_getter: Callable[[tuple[int, int]], InteractionSample],
    ordered_pairs: tuple[tuple[int, int], ...],
    *,
    n_eval: int,
    minimum_amplitude: float,
    n_bootstrap: int,
    seed: int,
    numeric_tolerance: float,
) -> ReplicationInferenceResult:
    multipliers = _shared_rademacher_matrix(n_bootstrap, n_eval, seed=seed)
    observed = np.empty(len(ordered_pairs), dtype=np.float64)
    bootstrap = np.empty((n_bootstrap, len(ordered_pairs)), dtype=np.float64)
    amplitude_ratios: dict[tuple[int, int], float] = {}
    standard_errors: dict[tuple[int, int], float] = {}
    pvalues: dict[tuple[int, int], float] = {}
    discovery_energies: dict[tuple[int, int], float] = {}
    confirmation_energies: dict[tuple[int, int], float] = {}
    cosine_similarities: dict[tuple[int, int], float] = {}

    for index, pair in enumerate(ordered_pairs):
        discovery = np.asarray(discovery_means[pair], dtype=np.float64)
        sample = sample_getter(pair)
        if sample.pair != pair:
            raise ValueError(
                f"confirmation sample factory returned {sample.pair} for requested {pair}"
            )
        if sample.n_eval != n_eval:
            raise ValueError(
                f"confirmation sample for {pair} has n_eval={sample.n_eval}, "
                f"expected {n_eval}"
            )
        vectors = np.asarray(sample.vectors, dtype=np.float64)
        if discovery.ndim != 1 or vectors.shape != (n_eval, discovery.size):
            raise ValueError(f"discovery/confirmation coordinates disagree for {pair}")
        if not np.all(np.isfinite(discovery)) or not np.all(np.isfinite(vectors)):
            raise ValueError(f"non-finite replication input for {pair}")

        discovery_energy = float(discovery @ discovery)
        confirmation_energy = float(sample.mean @ sample.mean)
        discovery_energies[pair] = discovery_energy
        confirmation_energies[pair] = confirmation_energy
        if discovery_energy <= numeric_tolerance:
            # A vanishing discovery direction cannot support a replication
            # claim.  Treat it as an automatic non-rejection rather than
            # dividing by a noise-scale denominator.
            amplitude_ratios[pair] = 0.0
            standard_errors[pair] = 0.0
            cosine_similarities[pair] = 0.0
            observed[index] = -1.0
            bootstrap[:, index] = 0.0
            pvalues[pair] = 1.0
            continue

        ratios = vectors @ discovery / discovery_energy
        ratio_mean = float(np.mean(ratios))
        residual = ratios - ratio_mean
        sample_sd = float(np.sqrt((residual @ residual) / (n_eval - 1)))
        denominator = max(sample_sd, numeric_tolerance)
        observed[index] = np.sqrt(n_eval) * (
            ratio_mean - minimum_amplitude
        ) / denominator
        bootstrap[:, index] = (
            multipliers @ residual / np.sqrt(n_eval) / denominator
        )
        pvalues[pair] = float(
            (1 + np.count_nonzero(bootstrap[:, index] >= observed[index]))
            / (n_bootstrap + 1)
        )
        amplitude_ratios[pair] = ratio_mean
        standard_errors[pair] = sample_sd / np.sqrt(n_eval)
        cosine_denominator = np.sqrt(discovery_energy * confirmation_energy)
        cosine_similarities[pair] = (
            float(discovery @ sample.mean) / cosine_denominator
            if cosine_denominator > numeric_tolerance
            else 0.0
        )

    return ReplicationInferenceResult(
        pairs=ordered_pairs,
        minimum_amplitude=float(minimum_amplitude),
        amplitude_ratios=amplitude_ratios,
        standard_errors=standard_errors,
        pvalues=pvalues,
        discovery_energies=discovery_energies,
        confirmation_energies=confirmation_energies,
        cosine_similarities=cosine_similarities,
        observed_statistics=observed,
        bootstrap_statistics=bootstrap,
    )
