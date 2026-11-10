"""Summarize the ICDE main comparison cells (mean over seeds per dataset x p)."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np

# Decision rules shown in the summary; each method's native defaults first.
RULES = {
    "lens": ["family_rms_alpha0.05", "floor0.01_alpha0.05", "floor0.02_alpha0.05", "pair_relative_alpha0.05"],
    "lens_mlp": ["family_rms_alpha0.05", "floor0.01_alpha0.05", "floor0.02_alpha0.05"],
    "diamond": ["calibrated_q0.05", "calibrated_q0.2"],
    "diamond_matched": ["calibrated_q0.05", "calibrated_q0.2"],
    "diamond_mlp": ["calibrated_q0.05", "calibrated_q0.2"],
    "h2_refit": ["normal_by_alpha0.05", "bootstrap_by_alpha0.05", "one_sd_rule_B10"],
    "iloco": ["by_alpha0.05", "by_alpha0.2"],
    "iloco_mp": ["by_alpha0.05", "by_alpha0.2"],
    "fast": ["validation_k"],
}


def diamond_lists(payload: dict, truth: set) -> dict:
    lists = {}
    for q, selected in payload["selected"].items():
        chosen = [tuple(pair) for pair in selected["calibrated"]]
        tp = sum(pair in truth for pair in chosen)
        lists[f"calibrated_q{q}"] = {
            "size": len(chosen), "tp": tp,
            "fdp": (len(chosen) - tp) / len(chosen) if chosen else 0.0,
            "power": tp / len(truth) if truth else None,
        }
    return lists


def parse(stem: str) -> tuple[str, str, str, str]:
    """``{dataset}_p{p}_seed{seed}_{method}``; method names may contain underscores."""
    method = next(m for m in sorted(RULES, key=len, reverse=True) if stem.endswith("_" + m))
    dataset, p, seed = stem[: -len(method) - 1].rsplit("_", 2)
    return dataset, p, seed, method


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data_icde"))
    args = parser.parse_args()
    table = defaultdict(lambda: defaultdict(list))
    for path in sorted(args.run.glob("*.json")):
        dataset, p, seed, method = parse(path.stem)
        payload = json.loads(path.read_text())
        if method.startswith("diamond"):
            data = args.data_dir / (f"{dataset}_{seed}.npz" if dataset.startswith("F") else f"{dataset}_{p}_{seed}.npz")
            truth = {tuple(int(v) for v in pair) for pair in np.load(data)["true_pairs"]}
            lists = diamond_lists(payload, truth)
        else:
            lists = payload["lists"]
        for rule in RULES[method]:
            entry = lists[rule]
            table[(dataset, p)][(method, rule)].append((
                entry["size"], entry["tp"], entry["fdp"], entry["power"],
                payload["wall_seconds"], payload["n_model_fits"], payload.get("teacher_signal_r2"),
            ))

    for key in sorted(table):
        print(f"\n== {key[0]} {key[1]}")
        print(f"{'method/rule':38s} {'n':>2s} {'size':>6s} {'FDP':>6s} {'power':>6s} {'any':>5s} "
              f"{'R2':>5s} {'sec':>7s} {'fits':>6s}")
        for (method, rule), rows in sorted(table[key].items()):
            size = statistics.mean(r[0] for r in rows)
            fdp = statistics.mean(r[2] for r in rows)
            powers = [r[3] for r in rows if r[3] is not None]
            power = f"{statistics.mean(powers):6.2f}" if powers else "     -"
            anyrej = statistics.mean(r[0] > 0 for r in rows)
            r2s = [r[6] for r in rows if r[6] is not None]
            r2 = f"{statistics.mean(r2s):5.2f}" if r2s else "    -"
            sec = statistics.mean(r[4] for r in rows)
            fits = statistics.mean(r[5] for r in rows)
            print(f"{method + '/' + rule:38s} {len(rows):2d} {size:6.1f} {fdp:6.2f} {power} {anyrej:5.2f} "
                  f"{r2} {sec:7.0f} {fits:6.0f}")


if __name__ == "__main__":
    main()
