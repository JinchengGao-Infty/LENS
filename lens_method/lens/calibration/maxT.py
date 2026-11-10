"""Permutation maxT procedures for global-null graph calibration."""

from __future__ import annotations

import numpy as np


def _validate(observed: np.ndarray, null_statistics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observed, dtype=np.float64)
    null = np.asarray(null_statistics, dtype=np.float64)
    if obs.ndim != 1 or obs.size == 0:
        raise ValueError("observed must be a non-empty 1-D array")
    if null.ndim != 2 or null.shape[1] != obs.size or null.shape[0] < 1:
        raise ValueError("null_statistics must have shape (B, n_candidates)")
    if not np.all(np.isfinite(obs)) or not np.all(np.isfinite(null)):
        raise ValueError("maxT inputs must be finite")
    return obs, null


def maxT_pvalues(observed: np.ndarray, null_statistics: np.ndarray) -> np.ndarray:
    """Single-step maxT adjusted p-values, valid for the global null."""
    obs, null = _validate(observed, null_statistics)
    null_max = null.max(axis=1)
    return (1 + np.count_nonzero(null_max[:, None] >= obs[None, :], axis=0)) / (
        null.shape[0] + 1
    )


def stepdown_maxT_pvalues(observed: np.ndarray, null_statistics: np.ndarray) -> np.ndarray:
    """Westfall-Young style step-down maxT values.

    Strong FWER control beyond the global null additionally needs subset
    pivotality; callers must not infer that assumption from this implementation.
    """
    obs, null = _validate(observed, null_statistics)
    order = np.argsort(-obs, kind="stable")
    ordered_observed = obs[order]
    ordered_null = null[:, order]
    raw = np.empty(obs.size, dtype=np.float64)
    for rank in range(obs.size):
        remaining_max = ordered_null[:, rank:].max(axis=1)
        raw[rank] = (1 + np.count_nonzero(remaining_max >= ordered_observed[rank])) / (
            null.shape[0] + 1
        )
    adjusted_ordered = np.maximum.accumulate(raw)
    adjusted = np.empty_like(adjusted_ordered)
    adjusted[order] = adjusted_ordered
    return adjusted
