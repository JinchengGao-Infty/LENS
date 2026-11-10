"""Discovery-only rankers for the v2 synthetic union benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from typing import Mapping, Sequence

import numpy as np

from .grids import FeatureKind, Grid1D, PairGrid, build_grid
from .interaction_vectors import InteractionSample, compute_interaction_sample
from .ranking import rank_interactions
from .splits import ThreeWaySplit
from .teachers import TeacherFactory

Pair = tuple[int, int]

FORMAL_SYNTHETIC_RANKERS: tuple[str, ...] = (
    "LENS-raw",
    "LENS-debiased",
    "Friedman-H2",
    "TreeSHAP",
    "FAST",
    "NID",
    "Random",
)


@dataclass(frozen=True)
class SyntheticDiscoveryResult:
    grids: Mapping[int, Grid1D]
    rankings: Mapping[str, tuple[Pair, ...]]
    scores: Mapping[str, Mapping[Pair, float]]
    candidate_sets: Mapping[str, tuple[Pair, ...]]
    union_candidates: tuple[Pair, ...]
    timings: Mapping[str, float]
    diagnostics: Mapping[str, object]


class _NIDMLP:
    """Small deterministic ReLU network used only by the registered NID ranker."""

    def __init__(self, input_dim: int, hidden: Sequence[int], seed: int):
        rng = np.random.RandomState(seed)
        dims = [input_dim, *hidden, 1]
        self.weights: list[np.ndarray] = []
        self.biases: list[np.ndarray] = []
        for fan_in, fan_out in zip(dims[:-1], dims[1:]):
            self.weights.append(rng.randn(fan_out, fan_in) * np.sqrt(2.0 / fan_in))
            self.biases.append(np.zeros(fan_out, dtype=np.float64))

    def forward(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        activations = [X]
        pre_activations = []
        current = X
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            pre_activation = current @ weight.T + bias
            pre_activations.append(pre_activation)
            current = (
                pre_activation
                if index == len(self.weights) - 1
                else np.maximum(pre_activation, 0.0)
            )
            activations.append(current)
        return current[:, 0], activations, pre_activations

    def gradients(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        activations: list[np.ndarray],
        pre_activations: list[np.ndarray],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        delta = (2.0 / len(target)) * (prediction - target)[:, None]
        weight_gradients = [np.zeros_like(weight) for weight in self.weights]
        bias_gradients = [np.zeros_like(bias) for bias in self.biases]
        for layer in reversed(range(len(self.weights))):
            weight_gradients[layer] = delta.T @ activations[layer]
            bias_gradients[layer] = delta.sum(axis=0)
            if layer > 0:
                delta = delta @ self.weights[layer]
                delta *= pre_activations[layer - 1] > 0.0
        return weight_gradients, bias_gradients

    def adam_step(
        self,
        weight_gradients: list[np.ndarray],
        bias_gradients: list[np.ndarray],
        state: dict[str, object],
        *,
        lr: float,
    ) -> None:
        beta1, beta2, epsilon = 0.9, 0.999, 1e-8
        state["step"] = int(state["step"]) + 1
        step = int(state["step"])
        for index in range(len(self.weights)):
            for prefix, gradient in (
                ("weight", weight_gradients[index]),
                ("bias", bias_gradients[index]),
            ):
                first = state[f"first_{prefix}"]
                second = state[f"second_{prefix}"]
                assert isinstance(first, list) and isinstance(second, list)
                first[index] = beta1 * first[index] + (1.0 - beta1) * gradient
                second[index] = beta2 * second[index] + (1.0 - beta2) * gradient**2
                first_hat = first[index] / (1.0 - beta1**step)
                second_hat = second[index] / (1.0 - beta2**step)
                update = lr * first_hat / (np.sqrt(second_hat) + epsilon)
                if prefix == "weight":
                    self.weights[index] -= update
                else:
                    self.biases[index] -= update


def _nid_scores(
    X: np.ndarray,
    y: np.ndarray,
    pairs: Sequence[Pair],
    *,
    seed: int,
    hidden: Sequence[int],
    epochs: int,
    lr: float,
) -> tuple[dict[Pair, float], float]:
    if epochs < 1 or lr <= 0 or not hidden:
        raise ValueError("NID requires positive epochs/lr and nonempty hidden widths")
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / np.where(X_std == 0.0, 1.0, X_std)
    y_mean = float(y.mean())
    y_std = float(y.std())
    y_scaled = (y - y_mean) / (y_std if y_std > 0 else 1.0)
    model = _NIDMLP(X.shape[1], hidden, seed)
    state: dict[str, object] = {
        "step": 0,
        "first_weight": [np.zeros_like(weight) for weight in model.weights],
        "second_weight": [np.zeros_like(weight) for weight in model.weights],
        "first_bias": [np.zeros_like(bias) for bias in model.biases],
        "second_bias": [np.zeros_like(bias) for bias in model.biases],
    }
    final_loss = float("nan")
    for _ in range(epochs):
        prediction, activations, pre_activations = model.forward(X_scaled)
        final_loss = float(np.mean((prediction - y_scaled) ** 2))
        gradients = model.gradients(
            prediction, y_scaled, activations, pre_activations
        )
        model.adam_step(*gradients, state, lr=lr)
    downstream = np.abs(model.weights[-1]).reshape(-1)
    for weight in reversed(model.weights[1:-1]):
        downstream = np.abs(weight).T @ downstream
    first_layer = np.abs(model.weights[0])
    scores = {
        pair: float(
            np.sum(
                np.minimum(first_layer[:, pair[0]], first_layer[:, pair[1]])
                * downstream
            )
        )
        for pair in pairs
    }
    if not np.isfinite(final_loss) or not np.all(np.isfinite(list(scores.values()))):
        raise ValueError("NID training produced non-finite loss or scores")
    return scores, final_loss


def _all_pairs(n_features: int) -> tuple[Pair, ...]:
    return tuple(
        (left, right)
        for left in range(n_features)
        for right in range(left + 1, n_features)
    )


def _ranking_from_scores(scores: Mapping[Pair, float], pairs: Sequence[Pair]) -> tuple[Pair, ...]:
    values = np.asarray([scores[pair] for pair in pairs], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("ranker emitted NaN or inf scores")
    return tuple(sorted(pairs, key=lambda pair: (-float(scores[pair]), pair)))


def _tree_shap_scores(model, X: np.ndarray, pairs: Sequence[Pair]) -> dict[Pair, float]:
    import xgboost as xgb

    values = model.get_booster().predict(xgb.DMatrix(X), pred_interactions=True)
    if values.ndim != 3 or values.shape[0] != len(X):
        raise ValueError(f"unexpected TreeSHAP interaction shape {values.shape}")
    return {
        pair: float(np.mean(np.abs(values[:, pair[0], pair[1]])))
        for pair in pairs
    }


def _fast_scores(
    X: np.ndarray,
    teacher_values: np.ndarray,
    feature_kinds: Sequence[FeatureKind],
    pairs: Sequence[Pair],
) -> dict[Pair, float]:
    from interpret.utils import measure_interactions

    type_map = {"continuous": "continuous", "categorical": "nominal"}
    scored = measure_interactions(
        X,
        teacher_values,
        interactions=list(pairs),
        feature_types=[type_map[kind] for kind in feature_kinds],
        objective="rmse",
    )
    scores = {(int(pair[0]), int(pair[1])): float(value) for pair, value in scored}
    if set(scores) != set(pairs):
        raise ValueError("FAST did not emit a complete pair score map")
    return scores


def discover_synthetic_rankers(
    X: np.ndarray,
    y: np.ndarray,
    *,
    split: ThreeWaySplit,
    feature_kinds: Sequence[FeatureKind],
    teacher_factory: TeacherFactory,
    seed: int,
    top_k: int,
    grid_points: int,
    grid_batch_size: int,
    ranking_max_samples: int = 3000,
    rankers: Sequence[str] = FORMAL_SYNTHETIC_RANKERS,
    nid_hidden: Sequence[int] = (128, 64),
    nid_epochs: int = 200,
    nid_lr: float = 1e-3,
) -> SyntheticDiscoveryResult:
    """Freeze complete rankings and top-k sets using discovery data only."""
    started_total = time.perf_counter()
    features = np.asarray(X, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("X and y must have aligned shapes (n,p) and (n,)")
    if len(feature_kinds) != features.shape[1]:
        raise ValueError("feature_kinds length must equal n_features")
    split.validate(len(target))
    requested = tuple(str(method) for method in rankers)
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("rankers must be a nonempty unique sequence")
    unknown = sorted(set(requested) - set(FORMAL_SYNTHETIC_RANKERS))
    if unknown:
        raise ValueError(f"unknown synthetic rankers: {unknown}")
    if top_k < 1 or ranking_max_samples < 2:
        raise ValueError("top_k and ranking_max_samples must be positive")

    X_discovery = features[split.discovery]
    y_discovery = target[split.discovery]
    pairs = _all_pairs(features.shape[1])
    if top_k > len(pairs):
        raise ValueError(f"top_k={top_k} exceeds {len(pairs)} available pairs")
    grids = {
        index: build_grid(
            X_discovery[:, index], feature_kinds[index], n_points=grid_points
        )
        for index in range(features.shape[1])
    }

    sample_rng = np.random.default_rng(seed * 2 + 4)
    if len(X_discovery) > ranking_max_samples:
        ranking_local_indices = np.sort(
            sample_rng.choice(len(X_discovery), ranking_max_samples, replace=False)
        )
    else:
        ranking_local_indices = np.arange(len(X_discovery), dtype=np.int64)
    X_ranking = X_discovery[ranking_local_indices]
    global_ranking_indices = np.asarray(split.discovery)[ranking_local_indices]
    ranking_index_hash = hashlib.sha256(
        np.asarray(global_ranking_indices, dtype=np.int64).tobytes()
    ).hexdigest()

    timings: dict[str, float] = {}
    discovery_seed = int(seed * 2 + 1)
    teacher_started = time.perf_counter()
    discovery_teacher = teacher_factory.fit(
        X_discovery,
        y_discovery,
        seed=discovery_seed,
    )
    timings["discovery_teacher"] = time.perf_counter() - teacher_started

    samples: dict[Pair, InteractionSample] = {}
    if {"LENS-raw", "LENS-debiased", "Friedman-H2"} & set(requested):
        lens_started = time.perf_counter()
        samples = {
            pair: compute_interaction_sample(
                discovery_teacher.predict_margin,
                X_ranking,
                pair,
                PairGrid(grids[pair[0]], grids[pair[1]]),
                grid_batch_size=grid_batch_size,
                estimate_covariance="trace",
            )
            for pair in pairs
        }
        timings["LENS-family"] = time.perf_counter() - lens_started

    scores: dict[str, dict[Pair, float]] = {}
    rankings: dict[str, tuple[Pair, ...]] = {}
    if "LENS-raw" in requested:
        ranking_rows = rank_interactions(samples, score="raw_energy")
        scores["LENS-raw"] = dict(ranking_rows)
        rankings["LENS-raw"] = tuple(pair for pair, _ in ranking_rows)
    if "LENS-debiased" in requested:
        ranking_rows = rank_interactions(samples, score="debiased_energy")
        scores["LENS-debiased"] = dict(ranking_rows)
        rankings["LENS-debiased"] = tuple(pair for pair, _ in ranking_rows)
    if "Friedman-H2" in requested:
        scores["Friedman-H2"] = {pair: samples[pair].friedman_h2 for pair in pairs}
        rankings["Friedman-H2"] = _ranking_from_scores(scores["Friedman-H2"], pairs)
    if "TreeSHAP" in requested:
        tree_started = time.perf_counter()
        if not hasattr(discovery_teacher, "model"):
            raise TypeError("TreeSHAP requires an XGBoost discovery teacher")
        scores["TreeSHAP"] = _tree_shap_scores(discovery_teacher.model, X_ranking, pairs)
        rankings["TreeSHAP"] = _ranking_from_scores(scores["TreeSHAP"], pairs)
        timings["TreeSHAP"] = time.perf_counter() - tree_started
    if "FAST" in requested:
        fast_started = time.perf_counter()
        teacher_ranking = discovery_teacher.predict_margin(X_ranking)
        scores["FAST"] = _fast_scores(X_ranking, teacher_ranking, feature_kinds, pairs)
        rankings["FAST"] = _ranking_from_scores(scores["FAST"], pairs)
        timings["FAST"] = time.perf_counter() - fast_started
    if "NID" in requested:
        nid_started = time.perf_counter()
        scores["NID"], nid_final_loss = _nid_scores(
            X_discovery,
            y_discovery,
            pairs,
            seed=seed * 2 + 5,
            hidden=nid_hidden,
            epochs=nid_epochs,
            lr=nid_lr,
        )
        rankings["NID"] = _ranking_from_scores(scores["NID"], pairs)
        timings["NID"] = time.perf_counter() - nid_started
        timings["NID_final_standardized_loss"] = nid_final_loss
    if "Random" in requested:
        random_started = time.perf_counter()
        random_rng = np.random.default_rng(seed * 2 + 6)
        random_values = random_rng.random(len(pairs))
        scores["Random"] = {
            pair: float(random_values[index]) for index, pair in enumerate(pairs)
        }
        rankings["Random"] = _ranking_from_scores(scores["Random"], pairs)
        timings["Random"] = time.perf_counter() - random_started

    expected_pairs = set(pairs)
    for method in requested:
        ranking = rankings.get(method)
        method_scores = scores.get(method)
        if ranking is None or method_scores is None:
            raise RuntimeError(f"ranker {method} did not run")
        if len(ranking) != len(pairs) or set(ranking) != expected_pairs:
            raise ValueError(f"{method} did not emit one complete pair ranking")
        if set(method_scores) != expected_pairs:
            raise ValueError(f"{method} score map is incomplete")
        if not np.all(np.isfinite([method_scores[pair] for pair in pairs])):
            raise ValueError(f"{method} score map contains NaN or inf")

    candidate_sets = {
        method: tuple(rankings[method][:top_k]) for method in requested
    }
    union_candidates = tuple(
        sorted({pair for method_pairs in candidate_sets.values() for pair in method_pairs})
    )
    timings["total"] = time.perf_counter() - started_total
    diagnostics = {
        "rankers": list(requested),
        "n_all_pairs": len(pairs),
        "candidate_budget_per_ranker": top_k,
        "candidate_union_count": len(union_candidates),
        "discovery_seed": discovery_seed,
        "ranking_sample_size": len(X_ranking),
        "ranking_indices_sha256": ranking_index_hash,
        "ranking_max_samples": ranking_max_samples,
        "grids_frozen_from_full_discovery": True,
        "all_rankers_discovery_measurable": True,
        "nid_hidden": list(nid_hidden),
        "nid_epochs": int(nid_epochs),
        "nid_lr": float(nid_lr),
    }
    return SyntheticDiscoveryResult(
        grids=grids,
        rankings=rankings,
        scores=scores,
        candidate_sets=candidate_sets,
        union_candidates=union_candidates,
        timings={key: float(value) for key, value in timings.items()},
        diagnostics=diagnostics,
    )
