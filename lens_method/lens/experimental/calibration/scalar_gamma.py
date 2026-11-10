"""Experimental two-moment Gamma fit to scalar refit statistics."""

from __future__ import annotations

import numpy as np
from scipy.stats import gamma


def gamma_tail_from_moments(
    observed: float,
    mean: float,
    variance: float,
    *,
    eps: float = 1e-12,
) -> float:
    if not all(np.isfinite(value) for value in (observed, mean, variance)):
        raise ValueError("observed, mean, and variance must be finite")
    if mean <= eps or variance <= eps:
        return 1.0 if observed <= mean + eps else 0.0
    shape = mean * mean / variance
    scale = variance / mean
    return float(gamma.sf(observed, a=shape, scale=scale))


def scalar_gamma_pvalue(observed: float, null_statistics: np.ndarray) -> float:
    null = np.asarray(null_statistics, dtype=np.float64)
    if null.ndim != 1 or null.size < 2 or not np.all(np.isfinite(null)):
        raise ValueError("Scalar-Gamma requires at least two finite null statistics")
    return gamma_tail_from_moments(
        observed,
        float(null.mean()),
        float(null.var(ddof=1)),
    )
