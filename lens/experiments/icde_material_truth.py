"""Population strength of every true pair in DIAMOND's F1-F10, and power against
the pairs that clear a scale floor.

The strength of pair u is the amplitude a_u of the true function's pair residual
table (the quantity LENS estimates; with the true function as both models the
reproduced strength mu_u equals a_u), in units of sd(f) on the same contexts.
A true pair is *material* at kappa when a_u >= kappa * sd(f).  The second part
re-scores the formal-run lists: power counts only material true pairs, FDP is
unchanged (false = outside the full truth).
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import importlib.util
import itertools
import json
import math
from pathlib import Path
import statistics

import numpy as np

from lens.grids import PairGrid, build_grid
from lens.interaction_vectors import compute_interaction_sample

KAPPAS = (0.01, 0.02)
RULES = {
    "lens": ["family_rms_alpha0.05", "floor0.01_alpha0.05", "floor0.02_alpha0.05"],
    "lens_mlp": ["family_rms_alpha0.05", "floor0.01_alpha0.05", "floor0.02_alpha0.05"],
    "diamond": ["calibrated_q0.2"],
    "diamond_matched": ["calibrated_q0.2"],
    "diamond_mlp": ["calibrated_q0.2"],
    "h2_refit": ["normal_by_alpha0.05"],
    "iloco": ["by_alpha0.05"],
    "iloco_mp": ["by_alpha0.05"],
    "fast": ["validation_k"],
}


def strengths(gen_y, func: int, X: np.ndarray, seed: int = 0) -> dict:
    def f(Z):
        return gen_y.generate_interaction_response(Z, func)[0].ravel().astype(np.float64)

    p = X.shape[1]
    contexts = X[np.random.default_rng(seed).choice(len(X), 3000, replace=False)]
    grids = [build_grid(X[:, j], "continuous", n_points=12) for j in range(p)]
    sd_f = float(np.std(f(contexts)))
    out = {}
    for j, k in itertools.combinations(range(p), 2):
        sample = compute_interaction_sample(f, contexts, (j, k), PairGrid(grids[j], grids[k]),
                                            estimate_covariance="none")
        out[(j, k)] = math.sqrt(sample.raw_energy) / sd_f
    return out


def selected(payload: dict, method: str, rule: str) -> set:
    if method.startswith("diamond"):
        return {tuple(pair) for pair in payload["selected"][rule.split("_q")[1]]["calibrated"]}
    return {tuple(pair) for pair in payload["lists"][rule]["pairs"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--gen-y", type=Path, required=True, help="DIAMOND's simulation/gen_Y.py")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("gen_Y", args.gen_y)
    gen_y = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_y)

    report = {}
    for func in range(1, 11):
        data = np.load(args.data_dir / f"F{func}_seed3.npz")
        truth = {tuple(int(v) for v in pair) for pair in data["true_pairs"]}
        strength = strengths(gen_y, func, data["X"])
        report[f"F{func}"] = {
            "true": {f"{j},{k}": strength[(j, k)] for j, k in sorted(truth)},
            "false_max": max(v for pair, v in strength.items() if pair not in truth),
        }
        material = {kappa: {pair for pair in truth if strength[pair] >= kappa} for kappa in KAPPAS}
        print(f"\n== F{func}: {len(truth)} true pairs, material at 0.01/0.02: "
              f"{len(material[0.01])}/{len(material[0.02])}, false max a/sd = {report[f'F{func}']['false_max']:.4f}")
        print("   true a/sd: " + ", ".join(f"{j}-{k}:{strength[(j, k)]:.3f}" for j, k in sorted(truth)))
        rows = defaultdict(list)
        for path in sorted(args.run.glob(f"F{func}_p30_seed*_*.json")):
            method = next(m for m in sorted(RULES, key=len, reverse=True) if path.stem.endswith("_" + m))
            payload = json.loads(path.read_text())
            for rule in RULES[method]:
                chosen = selected(payload, method, rule)
                fdp = len(chosen - truth) / len(chosen) if chosen else 0.0
                powers = [len(chosen & material[kappa]) / len(material[kappa]) if material[kappa] else math.nan
                          for kappa in KAPPAS]
                rows[f"{method}/{rule}"].append((fdp, len(chosen & truth) / len(truth), *powers))
        print(f"   {'method/rule':36s} {'FDP':>5s} {'power':>5s} {'mat.01':>6s} {'mat.02':>6s}")
        for name, values in sorted(rows.items()):
            means = [statistics.mean(column) for column in zip(*values)]
            print(f"   {name:36s} " + " ".join(f"{v:5.2f} " for v in means))
        report[f"F{func}"]["methods"] = {name: [statistics.mean(c) for c in zip(*values)]
                                         for name, values in rows.items()}
    args.output.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
