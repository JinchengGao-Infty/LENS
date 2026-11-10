"""Auditable independent discovery/confirmation/evaluation splits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ThreeWaySplit:
    discovery: np.ndarray
    confirmation: np.ndarray
    evaluation: np.ndarray

    def validate(self, n_samples: int) -> None:
        groups = [
            np.asarray(self.discovery, dtype=np.int64),
            np.asarray(self.confirmation, dtype=np.int64),
            np.asarray(self.evaluation, dtype=np.int64),
        ]
        if any(group.ndim != 1 or group.size == 0 for group in groups):
            raise ValueError("all three split components must be non-empty 1-D arrays")
        joined = np.concatenate(groups)
        if np.any(joined < 0) or np.any(joined >= n_samples):
            raise ValueError("split contains out-of-range indices")
        if np.unique(joined).size != joined.size:
            raise ValueError("discovery, confirmation, and evaluation must be disjoint")
        if joined.size != n_samples or set(joined.tolist()) != set(range(n_samples)):
            raise ValueError("three-way split must partition every sample exactly once")


def make_three_way_split(
    n_samples: int,
    *,
    seed: int,
    fractions: tuple[float, float, float] = (0.34, 0.33, 0.33),
) -> ThreeWaySplit:
    if n_samples < 6:
        raise ValueError("at least six samples are required for a three-way split")
    weights = np.asarray(fractions, dtype=np.float64)
    if weights.shape != (3,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("fractions must contain three positive finite values")
    weights = weights / weights.sum()

    raw_counts = weights * n_samples
    counts = np.floor(raw_counts).astype(int)
    remainder = n_samples - int(counts.sum())
    for position in np.argsort(-(raw_counts - counts))[:remainder]:
        counts[position] += 1
    if np.any(counts == 0):
        raise ValueError(f"fractions produce an empty split: counts={counts.tolist()}")

    permutation = np.random.default_rng(seed).permutation(n_samples)
    first, second = counts[0], counts[0] + counts[1]
    split = ThreeWaySplit(
        discovery=permutation[:first],
        confirmation=permutation[first:second],
        evaluation=permutation[second:],
    )
    split.validate(n_samples)
    return split
