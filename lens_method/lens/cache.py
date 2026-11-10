"""Cache schema for observed and refitted interaction-vector summaries."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .interaction_vectors import InteractionSample


@dataclass(frozen=True)
class InteractionSummary:
    """Sufficient cached state for the requested fast calibration family."""

    mean: np.ndarray
    covariance: np.ndarray
    statistic: float
    fold_means: np.ndarray
    fold_covariances: np.ndarray
    fold_sizes: np.ndarray

    @property
    def n_eval(self) -> int:
        return int(self.fold_sizes.sum())

    @property
    def n_folds(self) -> int:
        return int(self.fold_sizes.size)


def _pack_symmetric(matrix: np.ndarray) -> np.ndarray:
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be square")
    return array[np.triu_indices(array.shape[0])]


def _unpack_symmetric(packed: np.ndarray, dimension: int) -> np.ndarray:
    values = np.asarray(packed, dtype=np.float64)
    expected = dimension * (dimension + 1) // 2
    if values.shape != (expected,):
        raise ValueError(f"packed covariance has shape {values.shape}, expected ({expected},)")
    matrix = np.empty((dimension, dimension), dtype=np.float64)
    rows, columns = np.triu_indices(dimension)
    matrix[rows, columns] = values
    matrix[columns, rows] = values
    return matrix


def summarize_vectors(vectors: np.ndarray, *, n_folds: int = 3) -> InteractionSummary:
    h = np.asarray(vectors, dtype=np.float64)
    if h.ndim != 2 or h.shape[0] < 2 or h.shape[1] < 1:
        raise ValueError(f"vectors must have shape (m>=2, q>=1), got {h.shape}")
    if not np.all(np.isfinite(h)):
        raise ValueError("vectors contain NaN or inf")
    if n_folds < 3:
        raise ValueError("cross-fitted moments require n_folds >= 3")
    if h.shape[0] < 2 * n_folds:
        raise ValueError("each cross-fit fold needs at least two rows")

    folds = np.array_split(np.arange(h.shape[0]), n_folds)
    fold_means = []
    fold_covariances = []
    fold_sizes = []
    for indices in folds:
        fold = h[indices]
        fold_mean = fold.mean(axis=0)
        fold_residual = fold - fold_mean
        fold_cov = fold_residual.T @ fold_residual / (fold.shape[0] - 1)
        fold_means.append(fold_mean)
        fold_covariances.append(fold_cov)
        fold_sizes.append(fold.shape[0])

    fold_covariance_array = np.stack(fold_covariances)
    mean = h.mean(axis=0)
    return InteractionSummary(
        mean=mean,
        # Equations (G3)--(G10) define Omega_hat as the arithmetic mean of
        # the fold covariances, not as the full-sample empirical covariance.
        covariance=fold_covariance_array.mean(axis=0),
        statistic=float(h.shape[0] * (mean @ mean)),
        fold_means=np.stack(fold_means),
        fold_covariances=fold_covariance_array,
        fold_sizes=np.asarray(fold_sizes, dtype=np.int64),
    )


def summarize_sample(sample: InteractionSample, *, n_folds: int = 3) -> InteractionSummary:
    """Summarize vectors once using the cross-fitted covariance contract.

    ``InteractionSample.covariance`` is intentionally ignored. Computing a
    full-sample covariance before the fold covariances is both redundant and
    inconsistent with the pooled covariance in (G3)--(G10).
    """
    return summarize_vectors(sample.vectors, n_folds=n_folds)


def write_summary_cache(
    root: str | Path,
    *,
    seed: int,
    dgp: str,
    teacher: str,
    pair: tuple[int, int],
    refit: int | str,
    summary: InteractionSummary,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one atomic-ish compressed cache record plus JSON provenance.

    The directory layout directly encodes ``(seed, DGP, teacher, pair, refit)``.
    Callers should write only completed records into artifact manifests.
    """
    safe_dgp = dgp.replace("/", "_")
    safe_teacher = teacher.replace("/", "_")
    directory = Path(root) / f"seed={seed}" / f"dgp={safe_dgp}" / f"teacher={safe_teacher}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"pair={pair[0]}-{pair[1]}__refit={refit}"
    destination = directory / f"{stem}.npz"
    temporary = directory / f".{stem}.tmp.npz"
    np.savez_compressed(
        temporary,
        mean=summary.mean,
        statistic=np.asarray(summary.statistic),
        fold_means=summary.fold_means,
        fold_covariances_upper=np.stack(
            [_pack_symmetric(covariance) for covariance in summary.fold_covariances]
        ),
        fold_sizes=summary.fold_sizes,
        dimension=np.asarray(summary.mean.size, dtype=np.int64),
        cache_schema=np.asarray("lens-interaction-summary-v2"),
    )
    temporary.replace(destination)

    provenance = {
        "seed": seed,
        "dgp": dgp,
        "teacher": teacher,
        "pair": list(pair),
        "refit": refit,
        "n_eval": summary.n_eval,
        "n_folds": summary.n_folds,
        "covariance_storage": "mean_of_packed_fold_covariances",
        **(metadata or {}),
    }
    json_destination = directory / f"{stem}.json"
    json_temporary = directory / f".{stem}.tmp.json"
    json_temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    json_temporary.replace(json_destination)
    return destination


def write_refit_cache(
    root: str | Path,
    *,
    seed: int,
    dgp: str,
    teacher: str,
    refit: int | str,
    summaries: Mapping[tuple[int, int], InteractionSummary],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically cache every candidate summary from one shared teacher refit.

    The scientific key remains ``(seed, DGP, teacher, pair, refit)``; batching
    only moves the pair component into an explicit in-file index. This avoids
    millions of tiny files in the formal benchmark without coarsening or
    dropping any sufficient statistic.
    """
    if not summaries:
        raise ValueError("summaries must contain at least one pair")
    ordered = sorted(summaries.items())
    safe_dgp = dgp.replace("/", "_")
    safe_teacher = teacher.replace("/", "_")
    directory = Path(root) / f"seed={seed}" / f"dgp={safe_dgp}" / f"teacher={safe_teacher}"
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"refit={refit}"
    destination = directory / f"{stem}.npz"
    temporary = directory / f".{stem}.tmp.npz"
    payload: dict[str, np.ndarray] = {
        "pairs": np.asarray([pair for pair, _ in ordered], dtype=np.int64),
        "cache_schema": np.asarray("lens-refit-summary-v1"),
    }
    records = []
    for index, (pair, summary) in enumerate(ordered):
        prefix = f"record_{index}"
        payload[f"{prefix}_mean"] = summary.mean
        payload[f"{prefix}_statistic"] = np.asarray(summary.statistic)
        payload[f"{prefix}_fold_means"] = summary.fold_means
        payload[f"{prefix}_fold_covariances_upper"] = np.stack(
            [_pack_symmetric(covariance) for covariance in summary.fold_covariances]
        )
        payload[f"{prefix}_fold_sizes"] = summary.fold_sizes
        payload[f"{prefix}_dimension"] = np.asarray(summary.mean.size, dtype=np.int64)
        records.append(
            {
                "record": index,
                "pair": list(pair),
                "dimension": int(summary.mean.size),
                "n_eval": summary.n_eval,
                "n_folds": summary.n_folds,
            }
        )
    np.savez_compressed(temporary, **payload)
    temporary.replace(destination)

    provenance = {
        "cache_schema": "lens-refit-summary-v1",
        "seed": seed,
        "dgp": dgp,
        "teacher": teacher,
        "refit": refit,
        "covariance_storage": "mean_of_packed_fold_covariances",
        "records": records,
        **(metadata or {}),
    }
    json_destination = directory / f"{stem}.json"
    json_temporary = directory / f".{stem}.tmp.json"
    json_temporary.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    json_temporary.replace(json_destination)
    return destination


def load_refit_cache(path: str | Path) -> dict[tuple[int, int], InteractionSummary]:
    """Load a batched shared-refit cache into pair-indexed summaries."""
    output: dict[tuple[int, int], InteractionSummary] = {}
    with np.load(Path(path), allow_pickle=False) as payload:
        if str(payload["cache_schema"]) != "lens-refit-summary-v1":
            raise ValueError(f"unsupported refit cache schema in {path}")
        pairs = np.asarray(payload["pairs"], dtype=np.int64)
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("refit cache pairs must have shape (n_pairs, 2)")
        for index, row in enumerate(pairs):
            prefix = f"record_{index}"
            dimension = int(payload[f"{prefix}_dimension"])
            fold_covariances = np.stack(
                [
                    _unpack_symmetric(packed, dimension)
                    for packed in payload[f"{prefix}_fold_covariances_upper"]
                ]
            )
            summary = InteractionSummary(
                mean=payload[f"{prefix}_mean"],
                covariance=fold_covariances.mean(axis=0),
                statistic=float(payload[f"{prefix}_statistic"]),
                fold_means=payload[f"{prefix}_fold_means"],
                fold_covariances=fold_covariances,
                fold_sizes=payload[f"{prefix}_fold_sizes"],
            )
            output[(int(row[0]), int(row[1]))] = summary
    return output


def load_refit_summary(
    path: str | Path,
    pair: tuple[int, int],
) -> InteractionSummary:
    """Load one pair without inflating every record in the batched NPZ."""
    with np.load(Path(path), allow_pickle=False) as payload:
        if str(payload["cache_schema"]) != "lens-refit-summary-v1":
            raise ValueError(f"unsupported refit cache schema in {path}")
        pairs = np.asarray(payload["pairs"], dtype=np.int64)
        matches = np.flatnonzero(np.all(pairs == np.asarray(pair), axis=1))
        if matches.size != 1:
            raise KeyError(f"expected pair {pair} exactly once in {path}")
        prefix = f"record_{int(matches[0])}"
        dimension = int(payload[f"{prefix}_dimension"])
        fold_covariances = np.stack(
            [
                _unpack_symmetric(packed, dimension)
                for packed in payload[f"{prefix}_fold_covariances_upper"]
            ]
        )
        return InteractionSummary(
            mean=payload[f"{prefix}_mean"],
            covariance=fold_covariances.mean(axis=0),
            statistic=float(payload[f"{prefix}_statistic"]),
            fold_means=payload[f"{prefix}_fold_means"],
            fold_covariances=fold_covariances,
            fold_sizes=payload[f"{prefix}_fold_sizes"],
        )


def load_summary_cache(path: str | Path) -> InteractionSummary:
    with np.load(Path(path), allow_pickle=False) as payload:
        dimension = int(payload["dimension"]) if "dimension" in payload else int(payload["mean"].size)
        if "covariance_upper" in payload:
            covariance = _unpack_symmetric(payload["covariance_upper"], dimension)
            fold_covariances = np.stack(
                [
                    _unpack_symmetric(packed, dimension)
                    for packed in payload["fold_covariances_upper"]
                ]
            )
        elif "fold_covariances_upper" in payload:
            # V2 avoids storing Omega_hat twice: by contract it is exactly the
            # arithmetic mean of these fold covariances.
            fold_covariances = np.stack(
                [
                    _unpack_symmetric(packed, dimension)
                    for packed in payload["fold_covariances_upper"]
                ]
            )
            covariance = fold_covariances.mean(axis=0)
        else:  # Backward-compatible reader for early development records.
            covariance = payload["covariance"]
            fold_covariances = payload["fold_covariances"]
        return InteractionSummary(
            mean=payload["mean"],
            covariance=covariance,
            statistic=float(payload["statistic"]),
            fold_means=payload["fold_means"],
            fold_covariances=fold_covariances,
            fold_sizes=payload["fold_sizes"],
        )
