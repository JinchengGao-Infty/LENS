"""Rank-based Monte Carlo permutation p-values."""

from __future__ import annotations

import numpy as np


def exact_rank_pvalue(observed: float, null_statistics: np.ndarray) -> float:
    null = np.asarray(null_statistics, dtype=np.float64)
    if null.ndim != 1 or null.size < 1 or not np.all(np.isfinite(null)):
        raise ValueError("null_statistics must be a non-empty finite 1-D array")
    if not np.isfinite(observed):
        raise ValueError("observed statistic must be finite")
    return float((1 + np.count_nonzero(null >= observed)) / (null.size + 1))
