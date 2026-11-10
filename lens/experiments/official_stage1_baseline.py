"""Run one official Stage-1 interaction-ranking baseline cell.

The external checkout is the executable source of truth.  This module only
adapts LENS arrays/prediction callables to the public APIs and converts the
returned pair scores into one common ranking artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import types
from typing import Callable, Mapping

import numpy as np

from experiments.dgps import generate
from lens.grids import build_grid
from lens.splits import make_three_way_split
from lens.synthetic_oracle import compute_run_matched_oracle
from lens.teachers import XGBoostTeacherFactory

Pair = tuple[int, int]

METHODS = (
    "Archipelago",
    "Faith-Shap",
    "NID-official",
    "Integrated-Hessians",
    "TreeSHAP-IQ",
    "SHAP-IQ",
)

PROVENANCE = {
    "Archipelago": ("archipelago", "8ff437e5672809827d7daa6a5656aeedbc0e1094"),
    "Faith-Shap": ("faith_shap", "a5226d600a8ea5761a95da9e23bd58eee5aa85a5"),
    "NID-official": ("nid", "02afd5b75b758e179f39c182a27de786b18be416"),
    "Integrated-Hessians": ("path_explain", "93c75a5e9b3caeb17408297ea564e2f579ea15e3"),
    "TreeSHAP-IQ": ("shapiq", "a063be3056a480333281b13998d36107a4527329"),
    "SHAP-IQ": ("shapiq", "a063be3056a480333281b13998d36107a4527329"),
}


def _all_pairs(p: int) -> tuple[Pair, ...]:
    return tuple(itertools.combinations(range(p), 2))


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _sha256_ints(values: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes()).hexdigest()


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _repo_commit(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _prepare_nid_imports(repo: Path):
    """Expose the official ``utils`` namespace despite third-party name clashes."""
    for name in tuple(sys.modules):
        if name == "utils" or name.startswith("utils."):
            del sys.modules[name]
    namespace = types.ModuleType("utils")
    namespace.__path__ = [str(repo / "utils")]
    sys.modules["utils"] = namespace
    sys.path.insert(0, str(repo))
    return __import__("utils.general_utils", fromlist=["proprocess_data"])


def _complete_scores(
    sparse: Mapping[Pair, float], pairs: tuple[Pair, ...]
) -> dict[Pair, float]:
    scores = {pair: float(abs(sparse.get(pair, 0.0))) for pair in pairs}
    values = np.asarray(list(scores.values()), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("official baseline emitted NaN or inf scores")
    return scores


def _archipelago_scores(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    pairs: tuple[Pair, ...],
    external_root: Path,
    *,
    seed: int,
    n_explain: int,
) -> tuple[dict[Pair, float], dict[str, object]]:
    module = _load_module(
        "lens_official_archipelago_explainer",
        external_root / "archipelago" / "src" / "explainer.py",
    )
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), min(n_explain, len(X)), replace=False))
    baseline = np.mean(X, axis=0)
    totals = {pair: 0.0 for pair in pairs}

    def model_wrapper(matrix: np.ndarray) -> np.ndarray:
        return np.asarray(predict(matrix), dtype=np.float64).reshape(-1, 1)

    for index in chosen:
        explainer = module.Archipelago(
            model_wrapper,
            input=X[index],
            baseline=baseline,
            output_indices=0,
            batch_size=512,
        )
        rows = explainer.archdetect(
            get_main_effects=False,
            get_pairwise_effects=False,
        )["interactions"]
        for pair, score in rows:
            normalized = tuple(sorted((int(pair[0]), int(pair[1]))))
            totals[normalized] += float(score)
    scores = {pair: totals[pair] / len(chosen) for pair in pairs}
    return scores, {
        "official_api": "Archipelago.archdetect",
        "n_explain": int(len(chosen)),
        "baseline": "discovery_mean",
        "aggregation": "mean official squared two-context strength",
    }


def _faith_shap_scores(
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    pairs: tuple[Pair, ...],
    external_root: Path,
    *,
    seed: int,
    n_explain: int,
    n_perturbations: int,
    n_tabular_average: int,
) -> tuple[dict[Pair, float], dict[str, object]]:
    repo = external_root / "faith_shap"
    sys.path.insert(0, str(repo))
    try:
        module = _load_module("lens_official_faith_shap_explainer", repo / "explainer.py")
    finally:
        sys.path.pop(0)
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), min(n_explain, len(X)), replace=False))
    totals = {pair: 0.0 for pair in pairs}
    feature_names = [f"x{index}" for index in range(X.shape[1])]
    explainer = module.TabularExplainer(
        2,
        "faith_shap",
        feature_names,
        lasso=True,
        lasso_alpha=0.02,
    )

    for offset, index in enumerate(chosen):
        instance = np.asarray(X[index], dtype=np.float64)
        mapping_rng = np.random.default_rng(seed * 100003 + offset)

        def mapping(mask: np.ndarray, raw_instance: np.ndarray) -> np.ndarray:
            mask_array = np.asarray(mask, dtype=np.float64)
            background_indices = mapping_rng.integers(0, len(X), size=len(mask_array))
            background = np.asarray(X[background_indices], dtype=np.float64)
            return mask_array * raw_instance[None, :] + (1.0 - mask_array) * background

        generator = module.SamplePerturbation(predict, mapping)
        np.random.seed(seed * 100003 + offset)
        perturbation = generator.sample_perturbations(
            n_perturbations,
            X.shape[1],
            instance,
            method="faith_shap",
            n_tabular_average=n_tabular_average,
            interaction_order=2,
        )
        _, _, _, raw = explainer.explain_instance(
            {name: float(instance[i]) for i, name in enumerate(feature_names)},
            perturbation,
        )
        for key, value in raw.items():
            if isinstance(key, tuple) and len(key) == 2:
                pair = tuple(sorted((int(key[0]), int(key[1]))))
                totals[pair] += abs(float(value))
    scores = {pair: totals[pair] / len(chosen) for pair in pairs}
    return scores, {
        "official_api": "SamplePerturbation + TabularExplainer",
        "n_explain": int(len(chosen)),
        "n_perturbations": int(n_perturbations),
        "n_tabular_average": int(n_tabular_average),
        "interaction_order": 2,
        "lasso_alpha": 0.02,
        "aggregation": "mean absolute official pair coefficient",
    }


def _nid_scores(
    X: np.ndarray,
    y: np.ndarray,
    pairs: tuple[Pair, ...],
    external_root: Path,
    *,
    seed: int,
    device: str,
) -> tuple[dict[Pair, float], dict[str, object]]:
    import torch

    repo = external_root / "nid" / "1. madex"
    general = _prepare_nid_imports(repo)
    try:
        module = _load_module(
            "lens_official_nid_detection", repo / "neural_interaction_detection.py"
        )
        permutation = np.random.default_rng(seed).permutation(len(X))
        X_ordered = np.asarray(X[permutation], dtype=np.float64)
        y_ordered = np.asarray(y[permutation], dtype=np.float64)
        X_mean, X_std = X_ordered.mean(0), X_ordered.std(0)
        X_scaled = (X_ordered - X_mean) / np.where(X_std > 0, X_std, 1.0)
        y_mean, y_std = float(y_ordered.mean()), float(y_ordered.std())
        y_scaled = (y_ordered - y_mean) / (y_std if y_std > 0 else 1.0)
        Xs, Ys = general.proprocess_data(
            X_scaled, y_scaled, valid_size=500, test_size=500
        )
        torch_device = torch.device(device)
        interactions, final_loss = module.detect_interactions(
            Xs,
            Ys,
            detector="NID",
            arch=[256, 128, 64],
            batch_size=100,
            device=torch_device,
            seed=seed,
            nepochs=100,
            learning_rate=0.01,
            early_stopping=True,
            patience=5,
            l1_const=1e-4,
        )
    finally:
        sys.path.pop(0)
    sparse = {
        tuple(sorted((int(key[0]), int(key[1])))): float(value)
        for key, value in interactions
        if len(key) == 2
    }
    return _complete_scores(sparse, pairs), {
        "official_api": "neural_interaction_detection.detect_interactions",
        "architecture": [256, 128, 64],
        "nepochs": 100,
        "learning_rate": 0.01,
        "l1_const": 1e-4,
        "official_test_loss": float(final_loss),
    }


def _train_official_nid_mlp(
    X: np.ndarray,
    y: np.ndarray,
    external_root: Path,
    *,
    seed: int,
    device: str,
):
    import torch
    from torch.utils import data

    repo = external_root / "nid" / "1. madex"
    general = _prepare_nid_imports(repo)
    general.set_seed(seed)
    permutation = np.random.default_rng(seed).permutation(len(X))
    X_ordered = np.asarray(X[permutation], dtype=np.float64)
    y_ordered = np.asarray(y[permutation], dtype=np.float64)
    mean, std = X_ordered.mean(0), X_ordered.std(0)
    X_scaled = (X_ordered - mean) / np.where(std > 0, std, 1.0)
    y_mean, y_std = float(y_ordered.mean()), float(y_ordered.std())
    y_scaled = (y_ordered - y_mean) / (y_std if y_std > 0 else 1.0)
    Xs, Ys = general.proprocess_data(X_scaled, y_scaled, valid_size=500, test_size=500)
    loaders = {}
    for key in Xs:
        feats = general.force_float(Xs[key])
        targets = general.force_float(Ys[key])
        weights = torch.ones(len(feats), dtype=torch.float32).unsqueeze(1)
        loaders[key] = data.DataLoader(data.TensorDataset(feats, targets, weights), 100)
    torch_device = torch.device(device)
    model = general.MLP(X.shape[1], [256, 128, 64], act_func=torch.nn.Softplus()).to(
        torch_device
    )
    model, loss = general.train(
        model,
        loaders,
        device=torch_device,
        nepochs=100,
        learning_rate=0.01,
        early_stopping=True,
        patience=5,
        l1_const=0.0,
    )
    sys.path.pop(0)
    return model, mean, np.where(std > 0, std, 1.0), float(loss)


def _integrated_hessian_scores(
    X: np.ndarray,
    y: np.ndarray,
    pairs: tuple[Pair, ...],
    external_root: Path,
    *,
    seed: int,
    device: str,
    n_explain: int,
) -> tuple[dict[Pair, float], dict[str, object]]:
    import torch

    path_module = _load_module(
        "lens_official_path_explainer_torch",
        external_root
        / "path_explain"
        / "path_explain"
        / "explainers"
        / "path_explainer_torch.py",
    )
    PathExplainerTorch = path_module.PathExplainerTorch

    model, mean, std, loss = _train_official_nid_mlp(
        X, y, external_root, seed=seed, device=device
    )
    # The official implementation allocates its interaction accumulator on CPU.
    # Moving the already-trained native model to CPU preserves the official math.
    model = model.cpu().eval()
    X_scaled = (np.asarray(X, dtype=np.float64) - mean) / std
    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X_scaled), min(n_explain, len(X_scaled)), replace=False))
    background = rng.choice(len(X_scaled), min(128, len(X_scaled)), replace=False)
    inputs = torch.tensor(X_scaled[chosen], dtype=torch.float32, requires_grad=True)
    baselines = torch.tensor(X_scaled[background], dtype=torch.float32)
    np.random.seed(seed)
    torch.manual_seed(seed)
    matrix = PathExplainerTorch(model).interactions(
        inputs,
        baseline=baselines,
        num_samples=50,
        use_expectation=True,
        verbose=False,
    )
    values = np.asarray(matrix.detach().cpu().numpy(), dtype=np.float64)
    if values.shape != (len(chosen), X.shape[1], X.shape[1]):
        raise ValueError(f"unexpected Integrated Hessians shape: {values.shape}")
    scores = {
        pair: float(np.mean(np.abs(values[:, pair[0], pair[1]]))) for pair in pairs
    }
    return scores, {
        "official_api": "PathExplainerTorch.interactions",
        "native_model": "official NID MLP with Softplus",
        "architecture": [256, 128, 64],
        "n_explain": int(len(chosen)),
        "n_background": int(len(background)),
        "num_samples": 50,
        "official_test_loss": loss,
    }


def _shapiq_scores(
    method: str,
    model,
    predict: Callable[[np.ndarray], np.ndarray],
    X: np.ndarray,
    pairs: tuple[Pair, ...],
    *,
    seed: int,
    n_explain: int,
    budget: int,
) -> tuple[dict[Pair, float], dict[str, object]]:
    import shapiq

    rng = np.random.default_rng(seed)
    chosen = np.sort(rng.choice(len(X), min(n_explain, len(X)), replace=False))
    totals = {pair: 0.0 for pair in pairs}
    if method == "TreeSHAP-IQ":
        explainer = shapiq.TreeExplainer(model=model, index="k-SII", max_order=2)
        for index in chosen:
            values = explainer.explain(X[index])
            for pair in pairs:
                totals[pair] += abs(float(values[pair]))
        meta = {
            "official_api": "shapiq.TreeExplainer",
            "index": "k-SII",
            "mode": "pathdependent",
            "estimated": False,
        }
    else:
        background_indices = rng.choice(len(X), min(256, len(X)), replace=False)
        explainer = shapiq.TabularExplainer(
            model=predict,
            data=X[background_indices],
            index="k-SII",
            max_order=2,
            imputer="marginal",
            random_state=seed,
        )
        for offset, index in enumerate(chosen):
            values = explainer.explain(X[index], budget=budget, random_state=seed + offset)
            for pair in pairs:
                totals[pair] += abs(float(values[pair]))
        meta = {
            "official_api": "shapiq.TabularExplainer",
            "index": "k-SII",
            "imputer": "marginal",
            "budget": int(budget),
            "n_background": int(len(background_indices)),
            "estimated": True,
        }
    scores = {pair: totals[pair] / len(chosen) for pair in pairs}
    meta["n_explain"] = int(len(chosen))
    meta["aggregation"] = "mean absolute official pair interaction"
    return scores, meta


def _metrics(
    scores: Mapping[Pair, float],
    pairs: tuple[Pair, ...],
    truth: set[Pair],
    masses: Mapping[Pair, float],
    top_k: int,
) -> tuple[list[Pair], dict[str, float | int | None]]:
    from sklearn.metrics import average_precision_score

    ranking = sorted(pairs, key=lambda pair: (-float(scores[pair]), pair))
    selected = ranking[:top_k]
    hits = set(selected) & truth
    total_mass = float(sum(masses.values()))
    labels = np.asarray([pair in truth for pair in pairs], dtype=np.int64)
    values = np.asarray([scores[pair] for pair in pairs], dtype=np.float64)
    return selected, {
        "recall_at_k": len(hits) / len(truth) if truth else None,
        "weighted_recall_at_k": (
            sum(float(masses[pair]) for pair in hits) / total_mass
            if total_mass > 0
            else None
        ),
        "average_precision": (
            float(average_precision_score(labels, values)) if truth else None
        ),
        "true_positives_at_k": len(hits),
        "false_positives_at_k": len(selected) - len(hits),
        "k": int(top_k),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--dgp", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ranking-max-samples", type=int, default=3000)
    parser.add_argument("--n-explain", type=int, default=None)
    parser.add_argument("--shapiq-budget", type=int, default=1024)
    parser.add_argument("--faith-perturbations", type=int, default=2500)
    parser.add_argument("--faith-tabular-average", type=int, default=5)
    parser.add_argument(
        "--external-root", default="external/official_baselines", type=Path
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "DONE").exists():
        print(f"already complete: {output}")
        return
    started = time.perf_counter()
    config = {
        **vars(args),
        "external_root": str(args.external_root),
        "output": str(output),
        "stage": 1,
        "protocol": "official_source_truth_discovery_only_pair_ranking",
    }
    _atomic_json(output / "run_config.json", config)
    try:
        repo_name, expected_commit = PROVENANCE[args.method]
        repo_path = args.external_root / repo_name
        actual_commit = _repo_commit(repo_path)
        if actual_commit != expected_commit:
            raise RuntimeError(
                f"official provenance mismatch for {repo_name}: "
                f"expected {expected_commit}, got {actual_commit}"
            )
        dgp = generate(args.dgp, seed=args.seed, n=args.n)
        split = make_three_way_split(len(dgp.y), seed=args.seed)
        X_discovery = np.asarray(dgp.X[split.discovery], dtype=np.float64)
        y_discovery = np.asarray(dgp.y[split.discovery], dtype=np.float64)
        rng = np.random.default_rng(args.seed * 2 + 4)
        if len(X_discovery) > args.ranking_max_samples:
            local = np.sort(
                rng.choice(
                    len(X_discovery), args.ranking_max_samples, replace=False
                )
            )
        else:
            local = np.arange(len(X_discovery), dtype=np.int64)
        X_ranking = X_discovery[local]
        pairs = _all_pairs(dgp.X.shape[1])
        teacher = None
        if args.method in {"Archipelago", "Faith-Shap", "TreeSHAP-IQ", "SHAP-IQ"}:
            teacher = XGBoostTeacherFactory(
                "regression", params={"device": args.device}
            ).fit(X_discovery, y_discovery, seed=args.seed * 2 + 1)

        if args.method == "Archipelago":
            scores, method_config = _archipelago_scores(
                teacher.predict_margin,
                X_ranking,
                pairs,
                args.external_root,
                seed=args.seed * 2 + 101,
                n_explain=args.n_explain or 64,
            )
        elif args.method == "Faith-Shap":
            scores, method_config = _faith_shap_scores(
                teacher.predict_margin,
                X_ranking,
                pairs,
                args.external_root,
                seed=args.seed * 2 + 103,
                n_explain=args.n_explain or 50,
                n_perturbations=args.faith_perturbations,
                n_tabular_average=args.faith_tabular_average,
            )
        elif args.method == "NID-official":
            scores, method_config = _nid_scores(
                X_discovery,
                y_discovery,
                pairs,
                args.external_root,
                seed=args.seed * 2 + 105,
                device=args.device,
            )
        elif args.method == "Integrated-Hessians":
            scores, method_config = _integrated_hessian_scores(
                X_discovery,
                y_discovery,
                pairs,
                args.external_root,
                seed=args.seed * 2 + 107,
                device=args.device,
                n_explain=args.n_explain or 32,
            )
        else:
            shapiq_predict = lambda matrix: np.asarray(
                teacher.model.predict(np.asarray(matrix)), dtype=np.float64
            ).reshape(-1)
            scores, method_config = _shapiq_scores(
                args.method,
                teacher.model,
                shapiq_predict,
                X_ranking,
                pairs,
                seed=args.seed * 2 + 109,
                n_explain=args.n_explain or (64 if args.method == "TreeSHAP-IQ" else 50),
                budget=args.shapiq_budget,
            )

        scores = _complete_scores(scores, pairs)
        grids = {
            index: build_grid(X_discovery[:, index], "continuous", n_points=12)
            for index in range(dgp.X.shape[1])
        }
        truth = {tuple(pair) for pair in dgp.true_pairs}
        oracle = compute_run_matched_oracle(
            args.dgp,
            dgp.X[split.evaluation],
            grids,
            sorted(truth),
            grid_batch_size=144,
        )
        selected, metrics = _metrics(scores, pairs, truth, oracle.masses, args.top_k)
        ranking = sorted(pairs, key=lambda pair: (-scores[pair], pair))
        result = {
            "schema_version": "lens-official-stage1-cell-1",
            "status": "DONE",
            "stage": 1,
            "method": args.method,
            "dgp": args.dgp,
            "seed": args.seed,
            "n": len(dgp.y),
            "p": dgp.X.shape[1],
            "truth_pairs": [list(pair) for pair in sorted(truth)],
            "top_k_pairs": [list(pair) for pair in selected],
            "complete_ranking": [
                {"pair": list(pair), "score": scores[pair]} for pair in ranking
            ],
            "metrics": metrics,
            "method_config": method_config,
            "provenance": {
                "repo": repo_name,
                "commit": actual_commit,
                "official_code_used": True,
                "adapter_only": True,
            },
            "split_hashes": {
                "discovery": _sha256_ints(split.discovery),
                "confirmation": _sha256_ints(split.confirmation),
                "evaluation": _sha256_ints(split.evaluation),
                "ranking_discovery_indices": _sha256_ints(split.discovery[local]),
            },
            "discovery_only": True,
            "wall_seconds": time.perf_counter() - started,
        }
        _atomic_json(output / "result.json", result)
        (output / "DONE").touch()
        print(
            f"OFFICIAL_STAGE1_CELL_DONE method={args.method} dgp={args.dgp} "
            f"seed={args.seed} seconds={result['wall_seconds']:.3f}",
            flush=True,
        )
    except Exception as error:
        _atomic_json(
            output / "FAILED_RUN.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.perf_counter() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
