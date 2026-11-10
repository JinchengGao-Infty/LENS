"""Summarize the per-pair H2 denominator diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _quantiles(values: list[float]) -> list[float]:
    ordered = sorted(values)
    return [ordered[int(q * (len(ordered) - 1))] for q in (0.1, 0.5, 0.9)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cells = [json.loads(path.read_text()) for path in sorted(Path(args.output).glob("*_seed*.json"))]

    recall = {"LENS": [], "H2": []}
    false_top = {"LENS": {0: 0, 1: 0, 2: 0}, "H2": {0: 0, 1: 0, 2: 0}}
    true_h2, true_main, null00_h2 = [], [], []
    per_dgp: dict[str, list[tuple[float, float]]] = {}
    for cell in cells:
        rows = {tuple(row["pair"]): row for row in cell["pairs"]}
        n_true = sum(row["true"] for row in rows.values())
        cell_recall = {}
        for name, key in (("LENS", "lens_ranking"), ("H2", "h2_ranking")):
            top = [rows[tuple(pair)] for pair in cell[key][:15]]
            cell_recall[name] = sum(row["true"] for row in top) / min(15, n_true)
            recall[name].append(cell_recall[name])
            for row in top:
                if not row["true"]:
                    false_top[name][row["n_active"]] += 1
        per_dgp.setdefault(cell["dgp"], []).append((cell_recall["LENS"], cell_recall["H2"]))
        for row in rows.values():
            if row["true"]:
                true_h2.append(row["h2"])
                if row["main"] is not None:
                    true_main.append(row["main"])
            elif row["n_active"] == 0:
                null00_h2.append(row["h2"])

    print(f"cells: {len(cells)}")
    print(f"Recall@15  LENS {statistics.mean(recall['LENS']):.3f}  H2 {statistics.mean(recall['H2']):.3f}")
    for dgp, values in per_dgp.items():
        lens = statistics.mean(v[0] for v in values)
        h2 = statistics.mean(v[1] for v in values)
        print(f"  {dgp:10s} n={len(values)}  LENS {lens:.3f}  H2 {h2:.3f}")
    for name, counts in false_top.items():
        total = sum(counts.values())
        print(
            f"{name} false pairs in top-15: {total}; both inactive {counts[0]}, "
            f"one active {counts[1]}, both active {counts[2]}"
        )
    print(f"true pairs: H2 quantiles(10/50/90) {[round(v, 4) for v in _quantiles(true_h2)]}")
    print(f"true pairs: M quantiles {[round(v, 4) for v in _quantiles(true_main)]}")
    print(f"both-inactive null pairs: H2 quantiles {[round(v, 4) for v in _quantiles(null00_h2)]}")
    print(
        "share of both-inactive null pairs whose H2 exceeds the median true-pair H2: "
        f"{sum(v > statistics.median(true_h2) for v in null00_h2) / len(null00_h2):.3f}"
    )

    # Every (true u, false v) comparison inside a cell.  Since H2 = Delta / (M + Delta),
    # H2 ranks v above u exactly when Delta_v * M_u > Delta_u * M_v.  Pairs whose
    # denominator fell below the implementation threshold have H2 = 0 and M = None.
    energy_right = energy_right_h2_wrong = energy_wrong = 0
    ratio_true, ratio_inverting_false = [], []
    for cell in cells:
        true_rows = [row for row in cell["pairs"] if row["true"]]
        false_rows = [row for row in cell["pairs"] if not row["true"]]
        for u in true_rows:
            for v in false_rows:
                if u["delta"] > v["delta"]:
                    energy_right += 1
                    if v["h2"] > u["h2"]:
                        energy_right_h2_wrong += 1
                        if u["main"] is not None and v["main"] is not None and v["delta"] > 0:
                            ratio_true.append(u["main"] / u["delta"])
                            ratio_inverting_false.append(v["main"] / v["delta"])
                else:
                    energy_wrong += 1
    total = energy_right + energy_wrong
    print(f"true-vs-false comparisons: {total}")
    print(f"  energy orders correctly: {energy_right / total:.3f}")
    print(f"  of those, H2 inverts the order: {energy_right_h2_wrong / energy_right:.3f}")
    print(f"  M/Delta of the true pair in inverted comparisons: {[round(v, 1) for v in _quantiles(ratio_true)]}")
    print(f"  M/Delta of the false pair in inverted comparisons: {[round(v, 1) for v in _quantiles(ratio_inverting_false)]}")
    below = sum(row["main"] is None for cell in cells for row in cell["pairs"])
    print(f"pairs whose H2 denominator fell below the threshold: {below}")


if __name__ == "__main__":
    main()
