"""Experimental Refit-Spectral calibration via Imhof/Gil-Pelaez inversion.

This is a numerical reference path, not the default fast path. Every returned
tail includes the quadrature error estimate so experiments can reject unreliable
inversions rather than silently reporting them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.integrate import quad
from scipy.stats import ncx2

from ...cache import InteractionSummary

from ._cf import cf_gamma_component


@dataclass(frozen=True)
class SpectralTail:
    probability: float
    absolute_error: float
    backend: str = "unknown"
    status: int = 0


def generalized_chi_square_tail(
    observed: float,
    eigenvalues: np.ndarray,
    noncentralities: np.ndarray,
    *,
    deterministic_shift: float = 0.0,
    epsabs: float = 1e-8,
    epsrel: float = 1e-7,
    limit: int = 500,
    backend: Literal["auto", "davies", "imhof"] = "auto",
) -> SpectralTail:
    weights = np.asarray(eigenvalues, dtype=np.float64)
    deltas = np.asarray(noncentralities, dtype=np.float64)
    if weights.ndim != 1 or deltas.shape != weights.shape or weights.size == 0:
        raise ValueError("eigenvalues/noncentralities must be aligned non-empty vectors")
    if np.any(weights <= 0) or np.any(deltas < 0):
        raise ValueError("this implementation requires positive weights and nonnegative nc")
    threshold = float(observed - deterministic_shift)
    if threshold <= 0:
        return SpectralTail(1.0, 0.0, backend="deterministic")

    # Exact shortcut prevents unnecessary quadrature in equal-eigenvalue cases.
    if np.allclose(weights, weights[0], rtol=1e-12, atol=1e-14):
        probability = ncx2.sf(
            threshold / weights[0],
            df=weights.size,
            nc=float(deltas.sum()),
        )
        return SpectralTail(float(probability), 0.0, backend="ncx2")

    if backend in {"auto", "davies"}:
        try:
            from chi2comb import ChiSquared, chi2comb_cdf
        except Exception as error:
            if backend == "davies":
                raise RuntimeError(
                    "Refit-Spectral requires the pinned chi2comb Davies backend; "
                    "run scripts/install_chi2comb.sh"
                ) from error
        else:
            components = [
                ChiSquared(float(weight), float(delta), 1)
                for weight, delta in zip(weights, deltas)
            ]
            cdf, status, _ = chi2comb_cdf(
                threshold,
                components,
                0.0,
                lim=max(limit, 10_000),
                atol=epsabs,
            )
            if status != 0 or not np.isfinite(cdf) or not 0.0 <= cdf <= 1.0:
                raise RuntimeError(
                    "Davies generalized-chi-square inversion failed: "
                    f"status={status}, cdf={cdf}"
                )
            return SpectralTail(
                probability=float(1.0 - cdf),
                absolute_error=float(epsabs),
                backend="davies",
                status=int(status),
            )

    if backend not in {"auto", "imhof"}:
        raise ValueError(f"unknown generalized-chi-square backend: {backend}")

    distribution_mean = float(np.sum(weights * (1.0 + deltas)))

    def integrand(t: float) -> float:
        if t == 0.0:
            return distribution_mean - threshold
        denominator = 1.0 - 2.0j * t * weights
        log_phi = np.sum(-0.5 * np.log(denominator) + (1.0j * t * weights * deltas) / denominator)
        value = np.exp(-1.0j * t * threshold + log_phi)
        return float(np.imag(value) / t)

    integral, error = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=epsabs,
        epsrel=epsrel,
        limit=limit,
    )
    probability = float(np.clip(0.5 + integral / np.pi, 0.0, 1.0))
    return SpectralTail(
        probability,
        float(error / np.pi),
        backend="imhof-diagnostic",
    )


def refit_spectral_component(
    observed: float,
    summary: InteractionSummary,
    *,
    eig_tol: float = 1e-10,
    backend: Literal["davies", "imhof", "auto"] = "davies",
) -> SpectralTail:
    component = cf_gamma_component(summary)
    mean = np.asarray(summary.mean, dtype=np.float64)
    covariance = np.asarray(summary.covariance, dtype=np.float64)
    norm_squared = float(mean @ mean)
    shrunk_mean = np.sqrt(max(component.q0, 0.0) / (norm_squared + 1e-15)) * mean

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    max_eigenvalue = float(np.max(eigenvalues)) if eigenvalues.size else 0.0
    if max_eigenvalue <= 0:
        shift = summary.n_eval * float(shrunk_mean @ shrunk_mean)
        return SpectralTail(
            1.0 if observed <= shift else 0.0,
            0.0,
            backend="deterministic",
        )
    keep = eigenvalues > eig_tol * max_eigenvalue
    positive = eigenvalues[keep]
    projected = eigenvectors[:, keep].T @ shrunk_mean
    noncentralities = summary.n_eval * projected**2 / positive

    discarded = eigenvectors[:, ~keep].T @ shrunk_mean
    deterministic_shift = summary.n_eval * float(discarded @ discarded)
    return generalized_chi_square_tail(
        observed,
        positive,
        noncentralities,
        deterministic_shift=deterministic_shift,
        backend=backend,
    )


def refit_spectral_pvalue(
    observed: float,
    refit_summaries: Sequence[InteractionSummary],
) -> SpectralTail:
    if len(refit_summaries) == 0:
        raise ValueError("Refit-Spectral requires at least one refit summary")
    tails = [refit_spectral_component(observed, summary) for summary in refit_summaries]
    return SpectralTail(
        probability=float(np.mean([tail.probability for tail in tails])),
        absolute_error=float(np.mean([tail.absolute_error for tail in tails])),
    )
