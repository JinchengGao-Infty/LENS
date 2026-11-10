"""Cross-Fitted Mixture Gamma (CF-MG), Eq. (G8)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.stats import gamma

from ...cache import InteractionSummary

from ._cf import GammaComponent, cf_gamma_component


def component_tail(observed: float, component: GammaComponent, *, eps: float = 1e-12) -> float:
    if component.mean <= eps or component.variance <= eps:
        return 1.0 if observed <= component.mean + eps else 0.0
    return float(gamma.sf(observed, a=component.shape, scale=component.scale))


def cf_mixture_gamma_pvalue(
    observed: float,
    refit_summaries: Sequence[InteractionSummary],
) -> float:
    if not np.isfinite(observed):
        raise ValueError("observed statistic must be finite")
    if len(refit_summaries) == 0:
        raise ValueError("CF-Mixture-Gamma requires at least one refit summary")
    tails = [component_tail(observed, cf_gamma_component(summary)) for summary in refit_summaries]
    return float(np.mean(tails))
