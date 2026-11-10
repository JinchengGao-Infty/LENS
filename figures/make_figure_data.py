"""Collect the numbers behind the ICDE figures into one small JSON (stdlib only).

Reads the synthetic runs, the material-truth strengths of F1-F10 and the
freMTPL2 runs; ``make_figures.py`` draws from the output.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1] / "results"
LENS_RULE = "floor0.02_alpha0.05"
NO_FLOOR = "family_rms_alpha0.05"
KAPPA = 0.02
METHODS = {  # label -> (method file suffix, rule)
    "LENS (XGBoost)": ("lens", LENS_RULE),
    "LENS (MLP)": ("lens_mlp", LENS_RULE),
    "LENS, no floor (XGBoost)": ("lens", NO_FLOOR),
    "LENS, no floor (MLP)": ("lens_mlp", NO_FLOOR),
    "DIAMOND (XGBoost)": ("diamond", "0.2"),
    "DIAMOND (our XGBoost)": ("diamond_matched", "0.2"),
    "DIAMOND (MLP)": ("diamond_mlp", "0.2"),
    "H2, refit null": ("h2_refit", "normal_by_alpha0.05"),
    "iLOCO-Split": ("iloco", "by_alpha0.05"),
    "iLOCO-MP": ("iloco_mp", "by_alpha0.05"),
    "FAST": ("fast", "validation_k"),
}
PROXIES = {(0, 3), (1, 2), (2, 3)}  # pair_corr: copies x2 ~ x0, x3 ~ x1


def chosen(payload: dict, method: str, rule: str) -> set:
    if method.startswith("diamond"):
        return {tuple(p) for p in payload["selected"][rule]["calibrated"]}
    return {tuple(p) for p in payload["lists"][rule]["pairs"]}


def load(run: Path, stem: str) -> dict:
    return json.loads((run / f"{stem}.json").read_text())


def scaling(run: Path) -> dict:
    out = {}
    for dataset in ("sparse_pairs", "additive", "additive_corr"):
        for label, (method, rule) in METHODS.items():
            series = []
            for p in (10, 30, 50):
                sizes, fdps, anys = [], [], []
                for seed in range(3, 23):
                    truth = {tuple(t) for t in load(run, f"{dataset}_p{p}_seed{seed}_lens")["truth"]}
                    picked = chosen(load(run, f"{dataset}_p{p}_seed{seed}_{method}"), method, rule)
                    sizes.append(len(picked))
                    fdps.append(len(picked - truth) / len(picked) if picked else 0.0)
                    anys.append(bool(picked))
                series.append({"pairs": p * (p - 1) // 2, "size": statistics.mean(sizes),
                               "fdp": statistics.mean(fdps), "any": statistics.mean(anys)})
            out[f"{dataset}|{label}"] = series
    return out


def detection(run: Path, truth_file: Path) -> list:
    strengths = json.loads(truth_file.read_text())
    rows = []
    for func in range(1, 11):
        true = {tuple(int(v) for v in key.split(",")): s for key, s in strengths[f"F{func}"]["true"].items()}
        hits = defaultdict(lambda: defaultdict(int))
        for label in ("LENS (XGBoost)", "LENS (MLP)", "DIAMOND (XGBoost)", "DIAMOND (our XGBoost)", "DIAMOND (MLP)"):
            method, rule = METHODS[label]
            for seed in range(3, 23):
                picked = chosen(load(run, f"F{func}_p30_seed{seed}_{method}"), method, rule)
                for pair in true:
                    hits[pair][label] += pair in picked
        for pair, strength in true.items():
            rows.append({"function": func, "pair": list(pair), "strength": strength,
                         "rate": {label: count / 20 for label, count in hits[pair].items()}})
    return rows


def floor_groups(run: Path) -> dict:
    """Reproduced strength mu/sd_D of the pairs the rule without floor keeps."""
    groups = defaultdict(list)
    for method, model in (("lens", "XGBoost"), ("lens_mlp", "MLP")):
        for path in sorted(run.glob(f"*_{method}.json")):
            cell = json.loads(path.read_text())
            truth = {tuple(t) for t in cell["truth"]}
            kept = {tuple(p) for p in cell["lists"][NO_FLOOR]["pairs"]}
            sd = cell["discovery_margin_sd"]
            dataset = cell["dataset"]
            for j, k, a, mu, se in cell["pairs"]:
                pair = (int(j), int(k))
                if pair not in kept:
                    continue
                if pair in truth:
                    kind = "true, F1-F10" if dataset.startswith("F") else f"true, {dataset}"
                elif dataset == "pair_corr" and pair in PROXIES:
                    kind = "proxy"
                elif dataset in ("additive", "additive_corr"):
                    kind = "false, additive"
                else:
                    kind = "false, other"
                groups[f"{kind}|{model}"].append(mu / sd)
    return groups


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def by_keep(pvalues: list, alpha: float) -> set:
    m = len(pvalues)
    harmonic = sum(1.0 / i for i in range(1, m + 1))
    order = sorted(range(m), key=pvalues.__getitem__)
    k = 0
    for rank, i in enumerate(order, 1):
        if pvalues[i] <= rank * alpha / (m * harmonic):
            k = rank
    return set(order[:k])


def decide(cell: dict, kappa: float, eta: float, alpha: float) -> set:
    """Re-decide a LENS list from the stored a_u, mu_u and standard errors."""
    s_d, sd = cell["family_rms_norm"], cell["discovery_margin_sd"]
    pvalues = []
    for _, _, a, mu, se in cell["pairs"]:
        threshold = max(eta * a, eta * s_d, kappa * sd)
        z = (mu - threshold) / se if se > 0 else (math.inf if mu > threshold else -math.inf)
        pvalues.append(normal_sf(z))
    return {(int(cell["pairs"][i][0]), int(cell["pairs"][i][1])) for i in by_keep(pvalues, alpha)}


SETTINGS = ([("kappa", k, 0.25, 0.05) for k in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1)]
            + [("eta", 0.02, e, 0.05) for e in (0.0, 0.1, 0.5)]
            + [("alpha", 0.02, 0.25, a) for a in (0.01, 0.2)])


def sensitivity(run: Path, truth_file: Path) -> list:
    """Synthetic metrics of LENS when kappa, eta or alpha moves away from the default."""
    strengths = json.loads(truth_file.read_text())
    material = {f: {tuple(int(v) for v in key.split(",")) for key, s in strengths[f]["true"].items() if s >= KAPPA}
                for f in strengths}
    cells = {method: [json.loads(p.read_text()) for p in sorted(run.glob(f"*_{method}.json"))]
             for method in ("lens", "lens_mlp")}
    stored = sum(decide(c, KAPPA, 0.25, 0.05) == {tuple(p) for p in c["lists"][LENS_RULE]["pairs"]}
                 for c in cells["lens"])
    print(f"sensitivity: recomputed default lists match {stored}/{len(cells['lens'])} stored XGBoost lists")
    out = []
    for vary, kappa, eta, alpha in SETTINGS:
        for method, cell_list in cells.items():
            fdp, power, material_power, anys, sparse_fdp, corr_fdp = [], [], [], [], [], []
            for c in cell_list:
                truth = {tuple(t) for t in c["truth"]}
                picked = decide(c, kappa, eta, alpha)
                false = len(picked - truth) / len(picked) if picked else 0.0
                if c["dataset"].startswith("F"):
                    fdp.append(false)
                    power.append(len(picked & truth) / len(truth))
                    material_power.append(len(picked & material[c["dataset"]]) / len(material[c["dataset"]]))
                elif c["dataset"] in ("additive", "additive_corr"):
                    anys.append(bool(picked))
                elif c["dataset"] == "sparse_pairs" and c["p"] == 50:
                    sparse_fdp.append(false)
                elif c["dataset"] == "pair_corr" and c["p"] == 50:
                    corr_fdp.append(false)
            out.append({"vary": vary, "kappa": kappa, "eta": eta, "alpha": alpha, "model": method,
                        "f_fdp": statistics.mean(fdp), "f_power": statistics.mean(power),
                        "f_material": statistics.mean(material_power), "additive_any": statistics.mean(anys),
                        "sparse_fdp": statistics.mean(sparse_fdp), "paircorr_fdp": statistics.mean(corr_fdp)})
    return out


def real_sensitivity(fremtpl2_run: Path, credit_run: Path) -> list:
    """Mean LENS list size on the real datasets under each setting of SETTINGS."""
    sources = {"freMTPL2": fremtpl2_run / "full", "HELOC": credit_run / "heloc", "UCI credit": credit_run / "uci_credit"}
    out = []
    for vary, kappa, eta, alpha in SETTINGS:
        row = {"vary": vary, "kappa": kappa, "eta": eta, "alpha": alpha}
        for name, folder in sources.items():
            sizes = []
            for path in sorted(folder.glob("split*.json")):
                r = json.loads(path.read_text())
                cell = {"family_rms_norm": r["family_rms_norm"], "discovery_margin_sd": r["discovery_margin_sd"],
                        "pairs": [(*q["pair"], q["amplitude"], q["projection"], q["projection_se"]) for q in r["pairs"]]}
                sizes.append(len(decide(cell, kappa, eta, alpha)))
            row[name] = statistics.mean(sizes)
        out.append(row)
    return out


def h2_mechanism(run: Path) -> dict:
    """Energy, main-effect energy and H^2 of every pair in the diagnostic runs."""
    cells = [json.loads(p.read_text()) for p in sorted(run.glob("*.json"))]

    def top(cell, key, k=15):
        return sum(r["true"] for r in sorted(cell["pairs"], key=lambda r: -r[key])[:k])

    recalls = []
    for c in cells:
        n_true = sum(r["true"] for r in c["pairs"])
        recalls.append((min(15, n_true) and top(c, "delta") / min(15, n_true),
                        min(15, n_true) and top(c, "h2") / min(15, n_true), c["dgp"], c["seed"]))
    # The example run: the F-function cell whose recall gap is the median gap.
    gaps = sorted((e - h, dgp, seed) for e, h, dgp, seed in recalls if dgp.startswith("F"))
    _, dgp, seed = gaps[len(gaps) // 2]
    example = next(c for c in cells if c["dgp"] == dgp and c["seed"] == seed)
    return {
        "example": {"dgp": dgp, "seed": seed, "p": example["p"],
                    "pairs": [{"delta": r["delta"], "main": r["main"], "h2": r["h2"], "true": r["true"],
                               "n_active": r["n_active"]} for r in example["pairs"]]},
        "recall": {"energy": statistics.mean(r[0] for r in recalls), "h2": statistics.mean(r[1] for r in recalls)},
    }


def fremtpl2_sizes(run: Path) -> list:
    rows = []
    for size in ("train20000", "train50000", "full"):
        runs = [json.loads(p.read_text()) for p in sorted((run / size).glob("split*.json"))]
        dev = lambda key: statistics.mean(r["outer_test_mean_poisson_deviance"][key] for r in runs)
        rows.append({
            "size": size, "policies": statistics.mean(r["part_sizes"]["discovery"] for r in runs),
            "additive": dev("additive_f0"),
            "black_box": statistics.mean((r["outer_test_mean_poisson_deviance"]["discovery_teacher"]
                                          + r["outer_test_mean_poisson_deviance"]["confirmation_teacher"]) / 2
                                         for r in runs),
            "lens": dev("f0_plus_lens"), "relative": dev("f0_plus_pair_relative"),
            "top7": dev("f0_plus_top7"), "all36": dev("f0_plus_all36"),
            "lens_size": statistics.mean(len(r["lists"]["lens"]) for r in runs),
        })
    return rows


def timing(run: Path) -> dict:
    out = defaultdict(list)
    for path in sorted(run.glob("sparse_pairs_p*_seed3_*.json")):
        cell = json.loads(path.read_text())
        method = path.stem.split("_seed3_")[1]
        out[method].append({"pairs": cell["p"] * (cell["p"] - 1) // 2, "seconds": cell["wall_seconds"],
                            "fits": cell["n_model_fits"]})
    return {m: sorted(v, key=lambda r: r["pairs"]) for m, v in out.items()}


LEVEL_NAMES = {  # category codes follow the sorted category names of the OpenML frame
    "VehBrand": ["B1", "B10", "B11", "B12", "B13", "B14", "B2", "B3", "B4", "B5", "B6"],
    "Region": ["R11", "R21", "R22", "R23", "R24", "R25", "R26", "R31", "R41", "R42", "R43", "R52", "R53",
               "R54", "R72", "R73", "R74", "R82", "R83", "R91", "R93", "R94"],
    "VehGas": ["Diesel", "Regular"],
}
GALLERY = ("VehAge x VehBrand", "DrivAge x BonusMalus", "VehAge x BonusMalus", "VehAge x Region")


def gallery(tables: Path) -> dict:
    """Discovery and averaged confirmation tables of a few freMTPL2 pairs (split 30)."""
    payload = json.loads(tables.read_text())
    pairs = {" x ".join(r["names"]): r for r in payload["pairs"]}
    out = []
    for name in GALLERY:
        r = pairs[name]
        ticks = [LEVEL_NAMES.get(f) or [f"{v:g}" for v in levels] for f, levels in zip(r["names"], r["levels"])]
        out.append({"name": name, "features": r["names"], "ticks": ticks, "weights": r["weights"],
                    "theta": r["theta"], "amplitude": r["amplitude"], "D": r["residual_D"], "C": r["residual_C"]})
    return {"pairs": out, "sd": payload["discovery_margin_sd"]}


def fremtpl2(run: Path) -> dict:
    runs = [json.loads(p.read_text()) for p in sorted((run / "full").glob("split*.json"))]
    pairs = defaultdict(lambda: {"a": [], "theta": [], "kept": 0})
    for r in runs:
        sd = r["discovery_margin_sd"]
        for row in r["pairs"]:
            key = " x ".join(row["names"])
            pairs[key]["a"].append(row["amplitude"] / sd)
            pairs[key]["theta"].append(row["replication_ratio"])
            pairs[key]["kept"] += row["lens_selected"]
    return {
        "pairs": {k: {"a": statistics.mean(v["a"]), "theta": statistics.mean(v["theta"]), "kept": v["kept"]}
                  for k, v in pairs.items()},
        "s_over_sd": statistics.mean(r["family_rms_norm"] / r["discovery_margin_sd"] for r in runs),
    }


SW = {frozenset(p) for p in (("VehPower", "VehAge"), ("VehPower", "VehBrand"), ("VehAge", "VehBrand"),
                              ("VehAge", "VehGas"), ("DrivAge", "BonusMalus"))}  # Schelldorfer and Wuthrich 2019
HH = {frozenset(p) for p in (("VehAge", "VehGas"), ("DrivAge", "Region"), ("DrivAge", "BonusMalus"),
                              ("BonusMalus", "VehBrand"), ("VehAge", "BonusMalus"))}  # nonzero H, Havrylenko-Heger
FREMTPL2_ROWS = 15  # pairs kept in some split, then the strongest pairs never kept


def write_tables(data: dict, run: Path, out: Path) -> None:
    """LaTeX rows of the sensitivity table and of the freMTPL2 pair table."""
    syn = {(r["vary"], r["kappa"], r["eta"], r["alpha"], r["model"]): r for r in data["sensitivity"]}
    rows = []
    for vary, kappa, eta, alpha in SETTINGS:
        if vary == "kappa":
            label = rf"$\kappa={kappa:g}$"
        elif vary == "eta":
            label = rf"$\eta={eta:g}$"
        else:
            label = rf"$\alpha={alpha:g}$"
        if (vary, kappa) == ("kappa", KAPPA):
            label += " (LENS)"
        cells = []
        for key in ("f_fdp", "f_material", "additive_any", "sparse_fdp", "paircorr_fdp"):
            cells += [f"{syn[(vary, kappa, eta, alpha, m)][key]:.2f}" for m in ("lens", "lens_mlp")]
        real = next(r for r in data["real_sensitivity"]
                    if (r["vary"], r["kappa"], r["eta"], r["alpha"]) == (vary, kappa, eta, alpha))
        cells += [f"{real[name]:.1f}" for name in ("freMTPL2", "HELOC", "UCI credit")]
        rows.append(label + " & " + " & ".join(cells) + r" \\")
        if vary == "kappa" and kappa == 0.1:
            rows.append(r"\midrule")
    rows.append(r"\bottomrule")  # inside the file: a rule after \input in a tabular fails
    (out / "tab_sensitivity.tex").write_text("\n".join(rows) + "\n")

    runs = [json.loads(p.read_text()) for p in sorted((run / "full").glob("split*.json"))]
    stats = defaultdict(lambda: defaultdict(list))
    for r in runs:
        for q in r["pairs"]:
            k = tuple(q["names"])
            stats[k]["kept"].append(q["lens_selected"])
            stats[k]["rank"].append(q["rank"])
            stats[k]["a"].append(q["amplitude"] / r["discovery_margin_sd"])
            stats[k]["theta"].append(q["replication_ratio"])
    order = sorted(stats, key=lambda k: (-sum(stats[k]["kept"]), statistics.mean(stats[k]["rank"])))
    shown = [k for k in order if sum(stats[k]["kept"])][:FREMTPL2_ROWS]
    shown += sorted((k for k in order if not sum(stats[k]["kept"])), key=lambda k: statistics.mean(stats[k]["rank"]))[:2]
    lines = []
    for k in shown:
        v = stats[k]
        mark = lambda ref: r"\ding{51}" if frozenset(k) in ref else ""
        num = lambda x: f"{abs(x):.2f}" if abs(x) < 0.005 else f"{x:.2f}" if x > 0 else f"$-${-x:.2f}"
        lines.append(f"{k[0]}$\\times${k[1]} & {sum(v['kept'])} & {statistics.mean(v['rank']):.1f} & "
                     f"{statistics.mean(v['a']):.3f} & {statistics.mean(v['theta']):.2f} & "
                     f"{num(min(v['theta']))} to {num(max(v['theta']))} & {mark(SW)} & {mark(HH)} \\\\")
    lines.append(r"\bottomrule")
    (out / "tab_fremtpl2.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=ROOT / "synthetic")
    parser.add_argument("--truth", type=Path, default=ROOT / "synthetic/material_truth.json")
    parser.add_argument("--fremtpl2", type=Path, default=ROOT / "real/fremtpl2")
    parser.add_argument("--credit", type=Path, default=ROOT / "real/credit")
    parser.add_argument("--h2", type=Path, default=ROOT / "h2_diagnostic")
    parser.add_argument("--timing", type=Path, default=ROOT / "timing")
    parser.add_argument("--tables", type=Path, default=ROOT / "real/overview_tables/split30.json")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "figure_data.json")
    args = parser.parse_args()
    data = {
        "scaling": scaling(args.run),
        "detection": detection(args.run, args.truth),
        "floor": floor_groups(args.run),
        "fremtpl2": fremtpl2(args.fremtpl2),
        "fremtpl2_sizes": fremtpl2_sizes(args.fremtpl2),
        "sensitivity": sensitivity(args.run, args.truth),
        "real_sensitivity": real_sensitivity(args.fremtpl2, args.credit),
        "h2": h2_mechanism(args.h2),
        "timing": timing(args.timing),
        "gallery": gallery(args.tables),
    }
    args.output.write_text(json.dumps(data))
    write_tables(data, args.fremtpl2, args.output.parent)
    for key, values in sorted(data["floor"].items()):
        values = sorted(values)
        print(f"{key:28s} n={len(values):5d} median={values[len(values) // 2]:.4f} "
              f"share<0.02={sum(v < 0.02 for v in values) / len(values):.2f}")
    for row in data["sensitivity"]:
        print(f"{row['vary']:5s} k={row['kappa']:<5} eta={row['eta']:<4} a={row['alpha']:<4} {row['model']:8s} "
              f"F: fdp {row['f_fdp']:.2f} pow {row['f_power']:.2f} mat {row['f_material']:.2f} | "
              f"add any {row['additive_any']:.2f} | sparse fdp {row['sparse_fdp']:.2f} | "
              f"paircorr fdp {row['paircorr_fdp']:.2f}")
    for row in data["real_sensitivity"]:
        print(row)
    print("h2 example", data["h2"]["example"]["dgp"], data["h2"]["example"]["seed"], data["h2"]["recall"])
    for row in data["fremtpl2_sizes"]:
        print(row)


if __name__ == "__main__":
    main()
