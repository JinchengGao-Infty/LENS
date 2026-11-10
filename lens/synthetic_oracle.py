"""Run-matched oracle masses for v2 synthetic weighted power."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from experiments.dgps import noise_free_response
from .grids import Grid1D, PairGrid
from .interaction_vectors import compute_interaction_sample

Pair = tuple[int, int]


@dataclass(frozen=True)
class RunMatchedOracle:
    dgp: str
    masses: Mapping[Pair, float]
    total_true_mass: float
    n_eval: int
    definition: str = "v2_weighted_finite_grid_raw_energy"


def compute_run_matched_oracle(
    dgp_name: str,
    X_eval: np.ndarray,
    grids: Mapping[int, Grid1D],
    true_pairs: Sequence[Pair],
    *,
    grid_batch_size: int,
) -> RunMatchedOracle:
    """Evaluate known ``F*`` on the run's frozen grids and evaluation background."""
    features = np.asarray(X_eval, dtype=np.float64)
    if features.ndim != 2 or len(features) == 0:
        raise ValueError("X_eval must be a nonempty two-dimensional array")
    if set(grids) != set(range(features.shape[1])):
        raise ValueError("grids must contain exactly one entry per feature")
    pairs = tuple((int(pair[0]), int(pair[1])) for pair in true_pairs)
    if len(set(pairs)) != len(pairs):
        raise ValueError("true_pairs contains duplicates")
    for pair in pairs:
        if not (0 <= pair[0] < pair[1] < features.shape[1]):
            raise ValueError(f"invalid true pair {pair}")

    def predict(matrix: np.ndarray) -> np.ndarray:
        return noise_free_response(dgp_name, matrix)

    masses: dict[Pair, float] = {}
    for pair in pairs:
        sample = compute_interaction_sample(
            predict,
            features,
            pair,
            PairGrid(grids[pair[0]], grids[pair[1]]),
            grid_batch_size=grid_batch_size,
            estimate_covariance="none",
        )
        mass = float(sample.raw_energy)
        if not np.isfinite(mass) or mass < 0:
            raise ValueError(f"invalid oracle mass for {dgp_name} pair {pair}: {mass}")
        masses[pair] = mass
    return RunMatchedOracle(
        dgp=dgp_name,
        masses=masses,
        total_true_mass=float(sum(masses.values())),
        n_eval=len(features),
    )
