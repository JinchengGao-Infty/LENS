"""Cross-fitted moment estimators shared by CF Gamma calibrators."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from ...cache import InteractionSummary


@dataclass(frozen=True)
class GammaComponent:
    mean: float
    variance: float
    q0: float
    q1: float
    shape: float
    scale: float


def cf_gamma_component(summary: InteractionSummary, *, eps: float = 1e-12) -> GammaComponent:
    means = np.asarray(summary.fold_means, dtype=np.float64)
    covariances = np.asarray(summary.fold_covariances, dtype=np.float64)
    k_folds, dimension = means.shape
    if k_folds < 3:
        raise ValueError("CF moments require at least three folds")
    if covariances.shape != (k_folds, dimension, dimension):
        raise ValueError("fold covariance shape does not match fold means")
    if not np.all(np.isfinite(means)) or not np.all(np.isfinite(covariances)):
        raise ValueError("cross-fit summaries contain NaN or inf")

    q0_terms = [means[a] @ means[c] for a, c in permutations(range(k_folds), 2)]
    q0 = float(np.mean(q0_terms))

    q1_terms = [
        means[a] @ covariances[d] @ means[c]
        for a, c, d in permutations(range(k_folds), 3)
    ]
    q1 = float(np.mean(q1_terms))

    pooled_covariance = covariances.mean(axis=0)
    trace_covariance = float(np.trace(pooled_covariance))
    trace_covariance_squared = float(np.einsum("ij,ji->", pooled_covariance, pooled_covariance))
    m = summary.n_eval
    mean = trace_covariance + m * max(q0, 0.0)
    variance = 2.0 * trace_covariance_squared + 4.0 * m * max(q1, 0.0)
    if mean <= eps or variance <= eps:
        shape = float("inf")
        scale = 0.0
    else:
        shape = mean * mean / variance
        scale = variance / mean
    return GammaComponent(
        mean=float(mean),
        variance=float(variance),
        q0=q0,
        q1=q1,
        shape=float(shape),
        scale=float(scale),
    )
