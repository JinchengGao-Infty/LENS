"""Multiple-testing rules with explicit claim boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _step_up(pvalues: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or not np.all(np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must be a finite 1-D array in [0,1]")
    order = np.argsort(p, kind="stable")
    passing = np.flatnonzero(p[order] <= thresholds)
    if passing.size == 0:
        return np.zeros(p.size, dtype=bool)
    cutoff_rank = int(passing[-1])
    rejected = np.zeros(p.size, dtype=bool)
    rejected[order[: cutoff_rank + 1]] = True
    return rejected


def bh_rejections(pvalues: np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg; LENS labels this a power-oriented heuristic."""
    p = np.asarray(pvalues)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    thresholds = alpha * np.arange(1, p.size + 1) / p.size
    return _step_up(p, thresholds)


def by_rejections(pvalues: np.ndarray, *, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Yekutieli step-up rule for arbitrary dependence."""
    p = np.asarray(pvalues)
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    harmonic = np.sum(1.0 / np.arange(1, p.size + 1))
    thresholds = alpha * np.arange(1, p.size + 1) / (p.size * harmonic)
    return _step_up(p, thresholds)


@dataclass(frozen=True)
class OrderedFDRResult:
    """Result of a fixed-order FDR procedure stopping after k acceptances."""

    rejected: np.ndarray
    tested: np.ndarray
    critical_values: np.ndarray
    acceptances: int


def ordered_fdr_critical_values(
    n_hypotheses: int,
    *,
    alpha: float = 0.05,
    stop_after_acceptances: int = 1,
) -> np.ndarray:
    """Lynch et al. (2017), Theorem 4.1 arbitrary-dependence constants.

    The hypothesis order and ``stop_after_acceptances`` must be fixed without
    using the tested p-values.  Strong FDR control additionally requires every
    true elementary-null p-value to be marginally super-uniform under its own
    partial configuration.
    """
    if n_hypotheses < 1:
        raise ValueError("n_hypotheses must be positive")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    k = int(stop_after_acceptances)
    if not 1 <= k <= n_hypotheses:
        raise ValueError("stop_after_acceptances must lie in [1,n_hypotheses]")

    ranks = np.arange(1, n_hypotheses + 1, dtype=np.float64)
    critical = np.full(n_hypotheses, alpha / k, dtype=np.float64)
    later = ranks > k
    critical[later] = (
        (n_hypotheses - k + 1)
        * alpha
        / ((n_hypotheses - ranks[later] + 1) * k)
    )
    return np.minimum(critical, 1.0)


def ordered_fdr_rejections(
    pvalues: np.ndarray,
    *,
    alpha: float = 0.05,
    stop_after_acceptances: int = 1,
) -> OrderedFDRResult:
    """Test in a frozen order and stop immediately after the kth acceptance."""
    p = np.asarray(pvalues, dtype=np.float64)
    if p.ndim != 1 or p.size == 0 or not np.all(np.isfinite(p)):
        raise ValueError("pvalues must be a non-empty finite 1-D array")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("pvalues must lie in [0,1]")
    critical = ordered_fdr_critical_values(
        p.size,
        alpha=alpha,
        stop_after_acceptances=stop_after_acceptances,
    )
    rejected = np.zeros(p.size, dtype=bool)
    tested = np.zeros(p.size, dtype=bool)
    acceptances = 0
    for index, (pvalue, threshold) in enumerate(zip(p, critical, strict=True)):
        if acceptances >= stop_after_acceptances:
            break
        tested[index] = True
        if pvalue <= threshold:
            rejected[index] = True
        else:
            acceptances += 1
    return OrderedFDRResult(
        rejected=rejected,
        tested=tested,
        critical_values=critical,
        acceptances=acceptances,
    )
