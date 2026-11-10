"""Finite product-marginal grids and weights.

The vectorization convention is deliberately explicit.  Mathematical ``vec``
is column-major, so a ``(G_j, G_k)`` surface is flattened with ``order='F'``.
Under that convention the weight matrix in the paper is

    diag(w_k) \u2297 diag(w_j).

Using NumPy's default row-major flattening here would silently permute those
weights for non-square or non-uniform grids.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

FeatureKind = Literal["continuous", "categorical"]
VEC_ORDER = "F"


def _as_finite_1d(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape={arr.shape}")
    if arr.size == 0:
        raise ValueError(f"{name} must not be empty")
    if np.issubdtype(arr.dtype, np.number) and not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


@dataclass(frozen=True)
class Grid1D:
    """One feature's support grid and product-marginal integration weights."""

    values: np.ndarray
    weights: np.ndarray
    kind: FeatureKind

    def __post_init__(self) -> None:
        values = _as_finite_1d(self.values, "grid values")
        weights = np.asarray(self.weights, dtype=np.float64)
        if weights.ndim != 1 or weights.shape != values.shape:
            raise ValueError(
                "grid weights must be one-dimensional and match values; "
                f"got values={values.shape}, weights={weights.shape}"
            )
        if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("grid weights must be finite and strictly positive")
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            raise ValueError("grid weights must have positive finite mass")
        if self.kind not in {"continuous", "categorical"}:
            raise ValueError(f"unknown feature kind: {self.kind!r}")

        object.__setattr__(self, "values", values.copy())
        object.__setattr__(self, "weights", weights / total)

    @property
    def size(self) -> int:
        return int(self.values.size)


@dataclass(frozen=True)
class PairGrid:
    """Tensor-product grid for a feature pair."""

    left: Grid1D
    right: Grid1D

    @property
    def shape(self) -> tuple[int, int]:
        return (self.left.size, self.right.size)

    @property
    def dimension(self) -> int:
        return self.left.size * self.right.size

    @property
    def contrast_dimension(self) -> int:
        return max(1, (self.left.size - 1) * (self.right.size - 1))

    @property
    def mass(self) -> np.ndarray:
        """Return ``w_j w_k^T`` in surface layout."""
        return np.multiply.outer(self.left.weights, self.right.weights)

    @property
    def sqrt_mass(self) -> np.ndarray:
        return np.sqrt(self.mass)

    @property
    def vector_weights(self) -> np.ndarray:
        """Diagonal of ``diag(w_k) \u2297 diag(w_j)`` for column-major vec."""
        return self.mass.reshape(-1, order=VEC_ORDER)


def build_grid(
    values: np.ndarray,
    kind: FeatureKind,
    *,
    n_points: int = 12,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> Grid1D:
    """Build a target-aligned marginal grid using training/reference data only.

    Continuous features use an equal-weight empirical quantile grid. Duplicate
    quantiles are deduplicated and the remaining support points retain equal
    mass. Categorical features use all observed levels with empirical frequency
    weights.
    """
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.size == 0:
        raise ValueError(f"feature values must be a non-empty 1-D array, got {raw.shape}")
    if np.issubdtype(raw.dtype, np.number):
        if np.any(np.isinf(raw)):
            raise ValueError("feature values contain inf")
        # Missingness must be represented as a separate feature if it is part of
        # the scientific target. The numeric pair grid itself is conditional on
        # an observed value and therefore excludes NaNs.
        x = raw[np.isfinite(raw)]
        if x.size == 0:
            raise ValueError("feature has no finite values from which to build a grid")
    else:
        x = _as_finite_1d(raw, "feature values")
    if kind == "continuous":
        if n_points < 2:
            raise ValueError("continuous grids require n_points >= 2")
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError("quantile bounds must satisfy 0 <= lower < upper <= 1")
        probs = np.linspace(lower_quantile, upper_quantile, n_points)
        try:
            grid_values = np.quantile(x.astype(np.float64), probs, method="linear")
        except TypeError:  # NumPy < 1.22 compatibility
            grid_values = np.quantile(x.astype(np.float64), probs, interpolation="linear")
        grid_values = np.unique(grid_values)
        weights = np.full(grid_values.size, 1.0 / grid_values.size, dtype=np.float64)
        return Grid1D(grid_values, weights, kind)

    if kind == "categorical":
        grid_values, counts = np.unique(x, return_counts=True)
        weights = counts.astype(np.float64) / counts.sum()
        return Grid1D(grid_values, weights, kind)

    raise ValueError(f"unknown feature kind: {kind!r}")
