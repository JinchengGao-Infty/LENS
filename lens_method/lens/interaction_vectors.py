"""Efficient evaluation of weighted finite-grid interaction vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .centering import (
    weighted_anova_center,
    weighted_interaction_contrasts,
    weighted_interaction_vectors,
)
from .grids import PairGrid

PredictFn = Callable[[np.ndarray], np.ndarray]
CoordinateMode = Literal["contrast", "full"]
DEFAULT_GRID_BATCH_SIZE = 144
DEFAULT_MAX_BATCH_BYTES = 256 * 1024**2


def _predict_1d(predict: PredictFn, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(predict(matrix), dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1 or values.shape[0] != matrix.shape[0]:
        raise ValueError(
            "predict must return shape (n,) or (n,1); "
            f"got {values.shape} for n={matrix.shape[0]}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("teacher predictions contain NaN or inf")
    return values


def evaluate_pair_surfaces(
    predict: PredictFn,
    X_eval: np.ndarray,
    pair: tuple[int, int],
    pair_grid: PairGrid,
    *,
    grid_batch_size: int = DEFAULT_GRID_BATCH_SIZE,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
) -> np.ndarray:
    """Evaluate ``M^(i)`` for every inference row and grid cell.

    Grid cells are predicted in blocks. This avoids one predictor call per cell
    while bounding the temporary matrix at roughly
    ``grid_batch_size * n_eval * n_features`` entries. The requested grid
    batch is also capped by ``max_batch_bytes`` so the same fast default remains
    safe in the p=500 scaling experiment.
    """
    X = np.asarray(X_eval, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"X_eval must be a non-empty 2-D array, got {X.shape}")
    if np.any(np.isinf(X)):
        raise ValueError("X_eval contains inf")
    left, right = pair
    if not (0 <= left < right < X.shape[1]):
        raise ValueError(f"pair must satisfy 0 <= j < k < p, got {pair}")
    if grid_batch_size < 1:
        raise ValueError("grid_batch_size must be positive")
    if max_batch_bytes < X.nbytes:
        raise ValueError("max_batch_bytes must fit at least one copy of X_eval")

    g_left, g_right = pair_grid.shape
    n_eval = X.shape[0]
    cells = [(a, b) for a in range(g_left) for b in range(g_right)]
    predictions = np.empty((g_left, g_right, n_eval), dtype=np.float64)
    effective_batch_size = min(grid_batch_size, max(1, max_batch_bytes // X.nbytes))

    for start in range(0, len(cells), effective_batch_size):
        block = cells[start : start + effective_batch_size]
        batch = np.tile(X, (len(block), 1))
        for offset, (a, b) in enumerate(block):
            row_slice = slice(offset * n_eval, (offset + 1) * n_eval)
            batch[row_slice, left] = pair_grid.left.values[a]
            batch[row_slice, right] = pair_grid.right.values[b]
        block_predictions = _predict_1d(predict, batch).reshape(len(block), n_eval)
        for offset, (a, b) in enumerate(block):
            predictions[a, b] = block_predictions[offset]

    return predictions.transpose(2, 0, 1)


@dataclass(frozen=True)
class InteractionSample:
    """Per-row interaction vectors and their empirical quadratic statistic."""

    pair: tuple[int, int]
    pair_grid: PairGrid
    vectors: np.ndarray
    mean: np.ndarray
    covariance: np.ndarray | None
    covariance_trace: float | None
    statistic: float
    mean_surface: np.ndarray
    raw_mean_surface: np.ndarray

    @property
    def n_eval(self) -> int:
        return int(self.vectors.shape[0])

    @property
    def raw_energy(self) -> float:
        return float(self.mean @ self.mean)

    @property
    def debiased_energy(self) -> float:
        if self.covariance_trace is None:
            raise ValueError("debiased energy requires covariance-trace estimation")
        correction = self.covariance_trace / self.n_eval
        return max(self.raw_energy - correction, 0.0)

    @property
    def friedman_h2(self) -> float:
        mean_level = float(np.sum(self.pair_grid.mass * self.raw_mean_surface))
        denominator = float(
            np.sum(self.pair_grid.mass * (self.raw_mean_surface - mean_level) ** 2)
        )
        return self.raw_energy / denominator if denominator > 1e-15 else 0.0


def compute_interaction_sample(
    predict: PredictFn,
    X_eval: np.ndarray,
    pair: tuple[int, int],
    pair_grid: PairGrid,
    *,
    grid_batch_size: int = DEFAULT_GRID_BATCH_SIZE,
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES,
    estimate_covariance: bool | str = True,
    coordinate_mode: CoordinateMode = "contrast",
) -> InteractionSample:
    surfaces = evaluate_pair_surfaces(
        predict,
        X_eval,
        pair,
        pair_grid,
        grid_batch_size=grid_batch_size,
        max_batch_bytes=max_batch_bytes,
    )
    centered = weighted_anova_center(
        surfaces,
        pair_grid.left.weights,
        pair_grid.right.weights,
    )
    if coordinate_mode == "contrast":
        vectors = weighted_interaction_contrasts(centered, pair_grid)
    elif coordinate_mode == "full":
        vectors = weighted_interaction_vectors(centered, pair_grid)
    else:
        raise ValueError("coordinate_mode must be 'contrast' or 'full'")
    mean = vectors.mean(axis=0)
    n_eval = vectors.shape[0]
    if n_eval < 2:
        raise ValueError("at least two evaluation rows are required")
    covariance_mode = (
        "full" if estimate_covariance is True else "none" if estimate_covariance is False else estimate_covariance
    )
    if covariance_mode not in {"none", "trace", "full"}:
        raise ValueError("estimate_covariance must be bool or one of 'none', 'trace', 'full'")
    covariance = None
    covariance_trace = None
    if covariance_mode in {"trace", "full"}:
        centered_vectors = vectors - mean
        covariance_trace = float(np.einsum("ij,ij->", centered_vectors, centered_vectors) / (n_eval - 1))
    if covariance_mode == "full":
        covariance = centered_vectors.T @ centered_vectors / (n_eval - 1)
    statistic = float(n_eval * (mean @ mean))
    mean_surface = centered.mean(axis=0)
    raw_mean_surface = surfaces.mean(axis=0)
    return InteractionSample(
        pair=pair,
        pair_grid=pair_grid,
        vectors=vectors,
        mean=mean,
        covariance=covariance,
        covariance_trace=covariance_trace,
        statistic=statistic,
        mean_surface=mean_surface,
        raw_mean_surface=raw_mean_surface,
    )
