"""Small helpers shared by the real-data runners: the pair family, the outer and
inner splits, and the Benjamini--Yekutieli list."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import train_test_split

from lens.calibration.multiple_testing import by_rejections
from lens.splits import ThreeWaySplit, make_three_way_split

Pair = tuple[int, int]


def all_pairs(n_features: int) -> tuple[Pair, ...]:
    if n_features < 2:
        raise ValueError("at least two features are required")
    return tuple(
        (left, right)
        for left in range(n_features)
        for right in range(left + 1, n_features)
    )


def make_outer_inner_split(
    y: np.ndarray,
    *,
    task: str,
    seed: int,
    outer_test_size: float = 0.2,
    inner_fractions: tuple[float, float, float] = (0.34, 0.33, 0.33),
) -> tuple[np.ndarray, np.ndarray, ThreeWaySplit]:
    """Return outer train/test indices and a local three-way outer-train split."""
    target = np.asarray(y)
    if target.ndim != 1 or len(target) < 12:
        raise ValueError("y must be a 1-D array with at least 12 rows")
    if not 0 < outer_test_size < 1:
        raise ValueError("outer_test_size must lie in (0,1)")
    indices = np.arange(len(target), dtype=np.int64)
    outer_train, outer_test = train_test_split(
        indices,
        test_size=outer_test_size,
        random_state=seed,
        stratify=target if task == "binary" else None,
    )
    outer_train = np.asarray(outer_train, dtype=np.int64)
    outer_test = np.asarray(outer_test, dtype=np.int64)
    inner = make_three_way_split(
        len(outer_train),
        seed=seed,
        fractions=inner_fractions,
    )
    if set(outer_train) & set(outer_test):
        raise RuntimeError("outer train and test splits overlap")
    if set(np.concatenate((outer_train, outer_test))) != set(indices):
        raise RuntimeError("outer split does not partition the dataset")
    inner.validate(len(outer_train))
    return outer_train, outer_test, inner


def _by_graph(
    pairs: Sequence[Pair],
    pvalues: Mapping[Pair, float],
    *,
    alpha: float,
) -> tuple[Pair, ...]:
    family = tuple(pairs)
    values = np.asarray([pvalues[pair] for pair in family], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("BY family contains a non-finite p-value")
    selected = by_rejections(values, alpha=alpha)
    return tuple(pair for pair, keep in zip(family, selected, strict=True) if keep)
