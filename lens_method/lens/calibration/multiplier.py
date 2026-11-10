"""Fixed-teacher multiplier bootstrap for the LENS-Model contract."""

from __future__ import annotations

import numpy as np


def multiplier_bootstrap(
    vectors: np.ndarray,
    *,
    n_bootstrap: int = 999,
    seed: int = 0,
    multipliers: str = "rademacher",
) -> tuple[float, np.ndarray, float]:
    h = np.asarray(vectors, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 2 or not np.all(np.isfinite(h)):
        raise ValueError("vectors must have finite shape (m>=2, q)")
    if n_bootstrap < 1:
        raise ValueError("n_bootstrap must be positive")
    mean = h.mean(axis=0)
    centered = h - mean
    observed = float(h.shape[0] * (mean @ mean))
    rng = np.random.default_rng(seed)
    if multipliers == "rademacher":
        xi = rng.integers(0, 2, size=(n_bootstrap, h.shape[0]), dtype=np.int8)
        xi = xi.astype(np.float64) * 2.0 - 1.0
    elif multipliers == "gaussian":
        xi = rng.standard_normal((n_bootstrap, h.shape[0]))
    else:
        raise ValueError("multipliers must be 'rademacher' or 'gaussian'")
    bootstrap_vectors = xi @ centered / np.sqrt(h.shape[0])
    bootstrap_statistics = np.einsum("bi,bi->b", bootstrap_vectors, bootstrap_vectors)
    pvalue = float(
        (1 + np.count_nonzero(bootstrap_statistics >= observed)) / (n_bootstrap + 1)
    )
    return observed, bootstrap_statistics, pvalue
