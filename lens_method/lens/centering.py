"""Weighted finite-grid ANOVA centering and vectorization."""

from __future__ import annotations

import numpy as np

from .grids import PairGrid, VEC_ORDER


def _normalized_weights(weights: np.ndarray, expected: int, name: str) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64)
    if w.ndim != 1 or w.size != expected:
        raise ValueError(f"{name} must have shape ({expected},), got {w.shape}")
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError(f"{name} must be finite and strictly positive")
    return w / w.sum()


def weighted_anova_center(
    surfaces: np.ndarray,
    weights_left: np.ndarray,
    weights_right: np.ndarray,
) -> np.ndarray:
    """Apply weighted two-way ANOVA centering to one or many surfaces.

    ``surfaces`` has shape ``(..., G_j, G_k)``.  The implementation is fully
    vectorized over leading dimensions and implements Eq. (A1).
    """
    matrix = np.asarray(surfaces, dtype=np.float64)
    if matrix.ndim < 2:
        raise ValueError("surfaces must have at least two dimensions")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("surfaces contain NaN or inf")

    g_left, g_right = matrix.shape[-2:]
    w_left = _normalized_weights(weights_left, g_left, "weights_left")
    w_right = _normalized_weights(weights_right, g_right, "weights_right")

    row_mean = np.einsum("...ab,b->...a", matrix, w_right)[..., :, None]
    col_mean = np.einsum("a,...ab->...b", w_left, matrix)[..., None, :]
    grand_mean = np.einsum("a,...ab,b->...", w_left, matrix, w_right)[..., None, None]
    centered = matrix - row_mean - col_mean + grand_mean
    return centered


def weighted_interaction_vectors(
    centered_surfaces: np.ndarray,
    pair_grid: PairGrid,
) -> np.ndarray:
    """Return ``W^(1/2) vec(H)`` using the declared column-major convention."""
    centered = np.asarray(centered_surfaces, dtype=np.float64)
    if centered.ndim < 2 or centered.shape[-2:] != pair_grid.shape:
        raise ValueError(
            f"surface shape must end in {pair_grid.shape}, got {centered.shape}"
        )
    weighted = centered * pair_grid.sqrt_mass
    if weighted.ndim == 2:
        return weighted.reshape(-1, order=VEC_ORDER)

    # F-flattening each final 2-D surface equals swapping the two grid axes and
    # C-flattening. This keeps all leading sample/batch dimensions untouched.
    return weighted.swapaxes(-1, -2).reshape(*weighted.shape[:-2], -1)


def orthonormal_weight_contrasts(weights: np.ndarray) -> np.ndarray:
    """Deterministic orthonormal basis perpendicular to ``sqrt(weights)``.

    A Householder reflection maps the first Euclidean basis vector to the
    normalized square-root weight vector. Its remaining columns therefore span
    the weighted-centering contrast space.
    """
    w = _normalized_weights(weights, np.asarray(weights).size, "weights")
    root = np.sqrt(w)
    first = np.zeros_like(root)
    first[0] = 1.0
    difference = first - root
    norm = float(np.linalg.norm(difference))
    if norm <= 1e-15:
        return np.eye(root.size, dtype=np.float64)[:, 1:]
    direction = difference / norm
    reflection = np.eye(root.size, dtype=np.float64) - 2.0 * np.outer(
        direction, direction
    )
    return reflection[:, 1:]


def weighted_interaction_contrasts(
    centered_surfaces: np.ndarray,
    pair_grid: PairGrid,
) -> np.ndarray:
    """Return isometric coordinates of ``W^(1/2) vec(H)``.

    For a weighted-centered ``H``, the matrix
    ``diag(sqrt(w_j)) H diag(sqrt(w_k))`` lies in the tensor product of two
    contrast spaces. Projecting onto their orthonormal bases removes only exact
    zero directions, reducing ``G_j G_k`` coordinates to
    ``(G_j - 1)(G_k - 1)`` while preserving every inner product.
    """
    centered = np.asarray(centered_surfaces, dtype=np.float64)
    if centered.ndim < 2 or centered.shape[-2:] != pair_grid.shape:
        raise ValueError(
            f"surface shape must end in {pair_grid.shape}, got {centered.shape}"
        )
    if min(pair_grid.shape) < 2:
        # A constant feature has no interaction contrast. Keep one explicit zero
        # coordinate so downstream covariance code retains a valid dimension.
        return np.zeros((*centered.shape[:-2], 1), dtype=np.float64)
    weighted = centered * pair_grid.sqrt_mass
    left = orthonormal_weight_contrasts(pair_grid.left.weights)
    right = orthonormal_weight_contrasts(pair_grid.right.weights)
    contrasts = np.einsum(
        "ai,...ab,bj->...ij", left, weighted, right, optimize=True
    )
    if contrasts.ndim == 2:
        return contrasts.reshape(-1, order=VEC_ORDER)
    return contrasts.swapaxes(-1, -2).reshape(*contrasts.shape[:-2], -1)


def weighted_mass(centered_surface: np.ndarray, pair_grid: PairGrid) -> float:
    vector = weighted_interaction_vectors(centered_surface, pair_grid)
    if vector.ndim != 1:
        raise ValueError("weighted_mass expects one surface")
    return float(vector @ vector)
