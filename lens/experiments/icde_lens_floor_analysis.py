"""How a scale floor changes the LENS lists on the ICDE synthetic cells.

Reads LENS cells that carry per-pair amplitudes (a_u, mu_u, se) and re-decides
each list with the threshold on mu_u set to max(eta*a_u, eta*s_D, kappa*sd_F),
where sd_F is the spread of the discovery model's predictions on its contexts.
kappa=0 is the family-RMS rule.  Also prints, per dataset, the largest
reproduced strength among false pairs and the smallest among true pairs, both
in units of sd_F.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

import numpy as np
from scipy.stats import norm

from lens.calibration.multiple_testing import by_rejections

ETA = 0.25
KAPPAS = (0.0, 0.01, 0.02, 0.05, 0.1)


def decide(rows, s_d, sd_f, kappa, alpha=0.05):
    a, mu, se = rows[:, 2], rows[:, 3], rows[:, 4]
    threshold = np.maximum(ETA * np.maximum(a, s_d), kappa * sd_f)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (mu - threshold) / se, np.where(mu > threshold, np.inf, -np.inf))
    keep = by_rejections(norm.sf(z), alpha=alpha)
    return {(int(j), int(k)) for j, k in rows[keep, :2]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--method", default="lens", choices=("lens", "lens_mlp"))
    args = parser.parse_args()
    table = defaultdict(lambda: defaultdict(list))
    extremes = defaultdict(lambda: {"false_max": [], "true_min": [], "true_all": []})
    for path in sorted(args.run.glob(f"*_{args.method}.json")):
        cell = json.loads(path.read_text())
        rows = np.asarray(cell["pairs"], dtype=np.float64)
        truth = {tuple(pair) for pair in cell["truth"]}
        key = (cell["dataset"], cell["p"])
        sd_f, s_d = cell["discovery_margin_sd"], cell["family_rms_norm"]
        is_true = np.array([(int(j), int(k)) in truth for j, k in rows[:, :2]])
        scaled = rows[:, 3] / sd_f
        extremes[key]["false_max"].append(float(scaled[~is_true].max()))
        if is_true.any():
            extremes[key]["true_min"].append(float(scaled[is_true].min()))
            extremes[key]["true_all"].extend(scaled[is_true].tolist())
        for kappa in KAPPAS:
            chosen = decide(rows, s_d, sd_f, kappa)
            tp = len(chosen & truth)
            table[key][kappa].append((len(chosen), (len(chosen) - tp) / len(chosen) if chosen else 0.0,
                                      tp / len(truth) if truth else None))

    print(f"{'dataset':14s} {'p':>3s} {'n':>2s} {'false max mu/sd_F':>18s} {'true min mu/sd_F':>17s} "
          f"{'true<.01':>8s} {'true<.02':>8s}  "
          + "  ".join(f"k={kappa:<4} size/FDP/pow" for kappa in KAPPAS))
    for key in sorted(table):
        ext = extremes[key]
        true_min = f"{min(ext['true_min']):17.3f}" if ext["true_min"] else f"{'-':>17s}"
        cells = []
        for kappa in KAPPAS:
            rows = table[key][kappa]
            powers = [r[2] for r in rows if r[2] is not None]
            power = f"{statistics.mean(powers):.2f}" if powers else "  - "
            cells.append(f"{statistics.mean(r[0] for r in rows):5.1f}/{statistics.mean(r[1] for r in rows):.2f}/{power}")
        below = [f"{np.mean(np.asarray(ext['true_all']) < cut):8.2f}" if ext["true_all"] else f"{'-':>8s}"
                 for cut in (0.01, 0.02)]
        print(f"{key[0]:14s} {key[1]:3d} {len(table[key][0.0]):2d} {max(ext['false_max']):18.3f} {true_min} "
              f"{below[0]} {below[1]}  "
              + "  ".join(f"{c:>20s}" for c in cells))


if __name__ == "__main__":
    main()
