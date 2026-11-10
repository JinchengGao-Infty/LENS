#!/usr/bin/env python
"""Python-3.7-compatible adapter for the author-maintained iRF code."""

from __future__ import print_function

import argparse
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
from sklearn.metrics import average_precision_score


EXPECTED_COMMIT = "7fa1724cf817d24c079a590acd2bf35fc4f95c91"


def atomic_json(path, payload):
    temporary = path.with_name("." + path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def parse_key(key):
    if isinstance(key, (tuple, list, set, frozenset, np.ndarray)):
        values = [int(value) for value in key]
    else:
        text = str(key).strip().replace("(", "").replace(")", "")
        for separator in ("_", ",", " "):
            if separator in text:
                values = [int(value) for value in text.split(separator) if value != ""]
                break
        else:
            values = [int(text)]
    return tuple(sorted(values))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--official-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=15)
    args = parser.parse_args()
    if (args.output / "DONE").exists():
        return
    started = time.time()
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(args.official_root), "rev-parse", "HEAD"]
        ).decode().strip()
        if commit != EXPECTED_COMMIT:
            raise RuntimeError("iRF provenance mismatch: " + commit)
        sys.path.insert(0, str(args.official_root))
        from irf import irf_utils
        from irf.ensemble import RandomForestRegressorWithWeights

        # The pinned official regression path passes ``sample_weight_test=None``
        # to ``wrf_reg.fit``, while that same checkout's regression signature
        # omits the keyword.  Drop only this unused validation-weight keyword;
        # all fitting, RIT, and stability calculations remain official code.
        official_wrf_reg_fit = irf_utils.wrf_reg.fit

        def regression_signature_compatibility(self, *fit_args, **fit_kwargs):
            fit_kwargs.pop("sample_weight_test", None)
            return official_wrf_reg_fit(self, *fit_args, **fit_kwargs)

        irf_utils.wrf_reg.fit = regression_signature_compatibility

        data = np.load(str(args.input), allow_pickle=False)
        X_train = data["X_train"]
        y_train = data["y_train"]
        X_test = data["X_test"]
        y_test = data["y_test"]
        # The official regression RIT samples paths with fitted leaf values as
        # non-negative weights.  A constant response shift leaves squared-error
        # tree splits and pair structure unchanged while satisfying that native
        # domain requirement for centered synthetic responses.
        minimum_target = float(min(np.min(y_train), np.min(y_test)))
        target_shift = max(0.0, -minimum_target + 1e-6)
        if target_shift:
            y_train = y_train + target_shift
            y_test = y_test + target_shift
        truth_rows = data["truth_pairs"]
        truth = set(tuple(map(int, row)) for row in truth_rows)
        masses = dict(
            (tuple(map(int, pair)), float(mass))
            for pair, mass in zip(truth_rows, data["truth_masses"])
        )
        seed = int(data["seed"])
        dgp = str(data["dgp"])
        rf = RandomForestRegressorWithWeights(n_estimators=20, random_state=seed)
        outputs = irf_utils.run_iRF(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            K=5,
            rf=rf,
            B=30,
            random_state_classifier=seed,
            propn_n_samples=0.2,
            bin_class_type=None,
            M=20,
            max_depth=5,
            noisy_split=False,
            num_splits=2,
            n_estimators_bootstrap=5,
        )
        stability = outputs[-1]
        pairs = tuple(itertools.combinations(range(X_train.shape[1]), 2))
        scores = dict((pair, 0.0) for pair in pairs)
        raw_interactions = []
        for key, value in stability.items():
            parsed = parse_key(key)
            raw_interactions.append({"features": list(parsed), "stability": float(value)})
            if len(parsed) == 2 and parsed in scores:
                scores[parsed] = max(scores[parsed], float(value))
        ranking = sorted(pairs, key=lambda pair: (-scores[pair], pair))
        selected = ranking[: args.top_k]
        hits = set(selected) & truth
        total_mass = sum(masses.values())
        labels = np.asarray([pair in truth for pair in pairs], dtype=np.int64)
        values = np.asarray([scores[pair] for pair in pairs], dtype=np.float64)
        metrics = {
            "recall_at_k": len(hits) / float(len(truth)) if truth else None,
            "weighted_recall_at_k": (
                sum(masses[pair] for pair in hits) / total_mass if total_mass > 0 else None
            ),
            "average_precision": (
                float(average_precision_score(labels, values)) if truth else None
            ),
            "true_positives_at_k": len(hits),
            "false_positives_at_k": len(selected) - len(hits),
            "k": args.top_k,
        }
        result = {
            "schema_version": "lens-official-stage1-irf-cell-1",
            "status": "DONE",
            "stage": 1,
            "method": "iRF-official",
            "dgp": dgp,
            "seed": seed,
            "top_k_pairs": [list(pair) for pair in selected],
            "complete_ranking": [
                {"pair": list(pair), "score": scores[pair]} for pair in ranking
            ],
            "raw_official_interactions": raw_interactions,
            "metrics": metrics,
            "method_config": {
                "K": 5,
                "B": 30,
                "n_estimators": 20,
                "proportion_bootstrap": 0.2,
                "M": 20,
                "max_depth_RIT": 5,
                "num_splits": 2,
                "n_estimators_bootstrap": 5,
                "official_signature_compatibility_shim": (
                    "drop unused sample_weight_test=None passed by run_iRF to wrf_reg.fit"
                ),
                "constant_target_shift_for_nonnegative_RIT_weights": target_shift,
            },
            "provenance": {
                "repo": "Yu-Group/iterative-Random-Forest",
                "commit": commit,
                "official_code_used": True,
                "adapter_only": True,
            },
            "discovery_only": True,
            "wall_seconds": time.time() - started,
        }
        atomic_json(args.output / "result.json", result)
        (args.output / "DONE").touch()
        print(
            "OFFICIAL_IRF_CELL_DONE dgp={} seed={} seconds={:.3f}".format(
                dgp, seed, result["wall_seconds"]
            )
        )
    except Exception as error:
        atomic_json(
            args.output / "FAILED_RUN.json",
            {
                "status": "FAILED",
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
                "wall_seconds": time.time() - started,
            },
        )
        raise


if __name__ == "__main__":
    main()
