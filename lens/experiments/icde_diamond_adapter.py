"""Run the official DIAMOND knockoffsdiag pipeline on a given dataset.

Runs inside the DIAMOND environment with ``--diamond-src`` pointing at the
official ``src`` directory.  The steps are those of ``sim.py`` followed by the
per-file calibration and knockoff filter of ``fdr_control.py``; only the data
source changes.  ``--model``: ``xgboost`` (sim.py's branch), ``xgboost_matched``
(the LENS Teacher's XGBoost settings on DIAMOND's split), ``mlp`` (sim.py's
DeepPINK MLP with the default expected-Hessian explainer).

Modes:
  dump-f   write DIAMOND's own F-series data (UniformSampler, gen_Y) to npz.
  run      read an npz (X, y, true_pairs) and write the selected pairs.
"""

from __future__ import annotations

import os

# XGBoost starts one OpenMP thread per core in part of GPU training regardless of
# n_jobs; the pool size has to be set before any library loads OpenMP.
os.environ.setdefault("OMP_NUM_THREADS", "8")
# NumPy asks for transparent huge pages on large arrays; on a fragmented shared
# machine every such allocation stalls in kernel compaction (50x slower).
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import argparse
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np


def _import_diamond(src: str):
    for path in (src, f"{src}/knockoffs"):
        if path not in sys.path:
            sys.path.insert(0, path)
    from knockoffs import gen_Knockoffsdiag
    from simulation import gen_X, gen_Y
    from models import model_utils
    import fdr_control

    return gen_Knockoffsdiag, gen_X, gen_Y, model_utils, fdr_control


def dump_f(args) -> None:
    _, gen_X, gen_Y, _, _ = _import_diamond(args.diamond_src)
    X = gen_X.UniformSampler(
        p=30, low_list=[0] * 30, high_list=[1] * 30, seed=args.seed
    ).sample(n=20000)
    Y, inter_gt, _ = gen_Y.generate_interaction_response(X, func_num=args.func)
    true_pairs = sorted(
        {pair for group in inter_gt for pair in itertools.combinations(sorted(group), 2)}
    )
    np.savez(args.data, X=X, y=np.asarray(Y, dtype=np.float64).reshape(-1),
             true_pairs=np.asarray(true_pairs, dtype=np.int64).reshape(-1, 2))


def _fit_explain_mlp(src, seed, p, X_train, X_val, X_test, Y_train, Y_val, Y_test):
    """sim.py's mlp branch with its default explainer "eg" (expected Hessians)."""
    import tempfile

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
    from path_explain import PathExplainerTorch

    import sim
    from models import DeepPink

    seed_everything(seed)
    device = torch.device("cuda")
    data_module = sim.DataModule(X_train, X_val, X_test, Y_train, Y_val, Y_test)
    model = sim.DeepPinkModel(
        DeepPink.DeepPINK(p=p, model_type="mlp", hidden_dims=[140, 100, 60, 20]),
        criterion=torch.nn.MSELoss(),
    )
    early_stop = EarlyStopping(monitor="val_loss", patience=10, mode="min")
    checkpoint = ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min")
    with tempfile.TemporaryDirectory() as root:
        trainer = Trainer(max_epochs=100, accelerator="gpu", callbacks=[early_stop, checkpoint],
                          enable_checkpointing=True, default_root_dir=root, enable_progress_bar=False)
        trainer.fit(model, data_module)
        model = sim.DeepPinkModel.load_from_checkpoint(
            checkpoint.best_model_path, model=model.model, criterion=model.criterion
        )
    net = model.model.to(device)
    test = torch.tensor(X_test, dtype=torch.float32, device=device)
    test.requires_grad = True
    baseline = torch.tensor(X_train, dtype=torch.float32, device=device)
    explainer = PathExplainerTorch(net)
    attributions = explainer.attributions(test, baseline=baseline, use_expectation=True).detach().cpu().numpy()
    interactions = explainer.interactions(test, baseline=baseline, use_expectation=True).detach().cpu().numpy()

    def predict(values):
        with torch.no_grad():
            return net(torch.tensor(values, dtype=torch.float32, device=device)).cpu().numpy()

    return predict, np.mean(np.abs(attributions), axis=0), np.mean(np.abs(interactions), axis=0)


def run(args) -> None:
    import pandas as pd
    import torch
    import xgboost as xgb
    from pygam import LinearGAM
    from shap import TreeExplainer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    gen_Knockoffsdiag, _, _, model_utils, fdr_control = _import_diamond(args.diamond_src)
    data = np.load(args.data)
    X, Y = data["X"], data["y"].reshape(-1, 1)
    signal = data["f"] if "f" in data.files else data["y"]
    true_pairs = [tuple(int(v) for v in pair) for pair in data["true_pairs"]]
    seed = args.seed
    p = X.shape[1]
    started = time.perf_counter()

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(8)
    X_scaled = StandardScaler().fit_transform(X)
    y_scaler = StandardScaler()
    Y_scaled = y_scaler.fit_transform(Y)
    X_knockoff = gen_Knockoffsdiag.conditional_sequential_gen_ko(
        X_scaled, np.zeros(p), n_jobs=20, seed=seed
    )
    knockoff_seconds = time.perf_counter() - started
    X_concat = np.concatenate((X_scaled, X_knockoff), axis=1)
    index = np.arange(len(Y))
    X_train, X_val_test, Y_train, Y_val_test, _, index_val_test = train_test_split(
        X_concat, Y_scaled, index, test_size=0.3, random_state=seed
    )
    X_val, X_test, Y_val, Y_test, _, index_test = train_test_split(
        X_val_test, Y_val_test, index_val_test, test_size=0.5, random_state=seed
    )
    if args.model == "mlp":
        predict, attributions, interactions = _fit_explain_mlp(
            args.diamond_src, seed, p, X_train, X_val, X_test, Y_train, Y_val, Y_test
        )
    else:
        # "xgboost" is sim.py's branch; "xgboost_matched" uses the LENS Teacher's settings.
        params = (
            {"callbacks": [xgb.callback.EarlyStopping(rounds=10, metric_name="rmse", save_best=True)]}
            if args.model == "xgboost"
            else {"n_estimators": 1000, "learning_rate": 0.05, "max_depth": 6, "tree_method": "hist",
                  "early_stopping_rounds": 50}
        )
        model = xgb.XGBRegressor(device="cuda" if torch.cuda.is_available() else "cpu", n_jobs=8,
                                 random_state=seed, **params)
        model.fit(X_train, Y_train, eval_set=[(X_val, Y_val)], verbose=False)
        explainer = TreeExplainer(model)
        attributions = np.mean(np.abs(explainer.shap_values(X=X_test)), axis=0)
        interactions = np.mean(np.abs(explainer.shap_interaction_values(X=X_test)), axis=0)
        predict = model.predict
    fit_seconds = time.perf_counter() - started - knockoff_seconds
    target = y_scaler.transform(signal[index_test].reshape(-1, 1)).ravel()
    signal_r2 = float(1 - np.mean((target - np.ravel(predict(X_test))) ** 2) / np.var(target))

    true_set = set(true_pairs)
    record = {
        "seed": seed,
        "func_num": 0,
        "true_indices": sorted({i for pair in true_pairs for i in pair}),
        "true_pairs": true_pairs,
        "false_pairs": [(i, j) for i, j in itertools.combinations(range(p), 2) if (i, j) not in true_set],
        "original_knockoff_pairs": [(i, j) for i in range(p) for j in range(p, 2 * p) if i != j - p],
        "knockoff_knockoff_pairs": list(itertools.combinations(range(p, 2 * p), 2)),
        "interactions": interactions,
        "attributions": attributions,
    }
    frame = fdr_control.prepare_interaction_dataframe(record)

    # Calibration exactly as in fdr_control.fdr_control for one simulation file.
    np.random.seed(seed)
    num_features = 2 * p
    attr_onehot = np.array(frame["attr_onehot"].values.tolist())
    pair_onehot = np.array(frame["pair_onehot"].values.tolist())
    projection = np.random.normal(size=(num_features, num_features))
    labels = frame["type"].apply(lambda value: 1 if value == "Original" else 0).values
    propensity = LogisticRegression().fit(attr_onehot, labels).predict_proba(attr_onehot)[:, 1]
    stabilized = np.mean(labels)
    iptw = np.where(labels == 1, stabilized / propensity, (1 - stabilized) / (1 - propensity))
    design = np.concatenate([attr_onehot, pair_onehot @ projection], axis=1)
    observed = frame["interaction"].values
    gam = LinearGAM(constraints=None, penalties=None)
    gam.fit(design, observed, weights=1 / iptw)
    calibrated = np.abs(observed - gam.predict(design))

    pairs = [tuple(int(v) for v in pair) for pair in frame["pair"]]
    results = {}
    for q in args.q:
        selected = {}
        for name, scores in (("calibrated", calibrated), ("uncalibrated", observed)):
            chosen, _ = model_utils.get_selected_interactions(
                list(zip(pairs, scores.tolist())), p, q
            )
            selected[name] = sorted(
                [list(pair) for pair, _ in chosen if pair[0] < p and pair[1] < p]
            )
        results[str(q)] = selected
    payload = {
        "method": "DIAMOND-official",
        "model": args.model,
        "teacher_signal_r2": signal_r2,
        "seed": seed,
        "p": p,
        "n": int(X.shape[0]),
        "selected": results,
        "original_pair_scores": {
            f"{i},{j}": float(calibrated[k]) for k, (i, j) in enumerate(pairs) if j < p
        },
        "n_model_fits": 1,
        "wall_seconds": time.perf_counter() - started,
        "knockoff_seconds": knockoff_seconds,
        "fit_and_explain_seconds": fit_seconds,
    }
    Path(args.output).write_text(json.dumps(payload))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("dump-f", "run"))
    parser.add_argument("--diamond-src", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--func", type=int)
    parser.add_argument("--output")
    parser.add_argument("--q", type=float, nargs="+", default=[0.05, 0.2])
    parser.add_argument("--model", choices=("xgboost", "xgboost_matched", "mlp"), default="xgboost")
    args = parser.parse_args()
    dump_f(args) if args.mode == "dump-f" else run(args)


if __name__ == "__main__":
    main()
