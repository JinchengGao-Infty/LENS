"""Analytic pooled Gamma using total-variance mixture moments, Eq. (G9)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ...cache import InteractionSummary

from ._cf import cf_gamma_component
from .scalar_gamma import gamma_tail_from_moments


def pooled_moments(refit_summaries: Sequence[InteractionSummary]) -> tuple[float, float]:
    if len(refit_summaries) == 0:
        raise ValueError("CF-Pooled-Gamma requires at least one refit summary")
    components = [cf_gamma_component(summary) for summary in refit_summaries]
    means = np.asarray([component.mean for component in components])
    variances = np.asarray([component.variance for component in components])
    pooled_mean = float(means.mean())
    pooled_variance = float(np.mean(variances + (means - pooled_mean) ** 2))
    return pooled_mean, pooled_variance


def cf_pooled_gamma_pvalue(
    observed: float,
    refit_summaries: Sequence[InteractionSummary],
) -> float:
    mean, variance = pooled_moments(refit_summaries)
    return gamma_tail_from_moments(observed, mean, variance)
