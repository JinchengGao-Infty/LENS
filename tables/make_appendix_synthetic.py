"""Appendix tables of the synthetic study, read from results/synthetic/summary.txt."""
import re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUMMARY = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results/synthetic/summary.txt"
OUT = HERE / "appendix_synthetic_tables.tex"

blocks = {}
for chunk in SUMMARY.read_text().split("\n== ")[1:]:
    head, *lines = chunk.strip().splitlines()
    rows = {}
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 9:
            continue
        num = lambda x: float("nan") if x == "-" else float(x)
        rows[parts[0]] = dict(n=int(parts[1]), size=num(parts[2]), fdp=num(parts[3]),
                              power=num(parts[4]), any=num(parts[5]), sec=parts[7])
    blocks[head.replace("== ", "").strip()] = rows

RULES = [
    ("lens/floor0.02_alpha0.05", r"LENS (default, $\kappa=0.02$)", "XGBoost"),
    ("lens/floor0.01_alpha0.05", r"LENS, $\kappa=0.01$", "XGBoost"),
    ("lens/family_rms_alpha0.05", r"LENS, $\kappa=0$", "XGBoost"),
    ("lens/pair_relative_alpha0.05", r"LENS, $b_u=\eta$", "XGBoost"),
    ("lens_mlp/floor0.02_alpha0.05", r"LENS (default, $\kappa=0.02$)", "MLP"),
    ("lens_mlp/floor0.01_alpha0.05", r"LENS, $\kappa=0.01$", "MLP"),
    ("lens_mlp/family_rms_alpha0.05", r"LENS, $\kappa=0$", "MLP"),
    ("diamond/calibrated_q0.2", r"DIAMOND, $q=0.2$", "XGBoost"),
    ("diamond/calibrated_q0.05", r"DIAMOND, $q=0.05$", "XGBoost"),
    ("diamond_matched/calibrated_q0.2", r"DIAMOND, $q=0.2$", r"XGBoost$^\dagger$"),
    ("diamond_mlp/calibrated_q0.2", r"DIAMOND, $q=0.2$", "MLP"),
    ("diamond_mlp/calibrated_q0.05", r"DIAMOND, $q=0.05$", "MLP"),
    ("h2_refit/normal_by_alpha0.05", r"$H^2$, normal refit null", "XGBoost"),
    ("h2_refit/bootstrap_by_alpha0.05", r"$H^2$, rank p-values", "XGBoost"),
    ("iloco/by_alpha0.05", r"iLOCO-Split, $\alpha=0.05$", "XGBoost"),
    ("iloco/by_alpha0.2", r"iLOCO-Split, $\alpha=0.2$", "XGBoost"),
    ("iloco_mp/by_alpha0.05", r"iLOCO-MP, $\alpha=0.05$", "trees"),
    ("fast/validation_k", r"FAST", "EBM"),
]

def f(x, d=2):
    return "--" if x != x else f"{x:.{d}f}"

out = []
designs = [("sparse_pairs", "Sparse", True), ("pair_corr", "Pair-corr", True),
           ("additive", "Additive", False), ("additive_corr", "Additive-corr", False)]
for key, name, has_truth in designs:
    cols = "size & FDP & power" if has_truth else "size & any"
    ncol = 3 if has_truth else 2
    out.append(r"\begin{table}[H]\centering")
    what = "list size, FDP, and power" if has_truth else \
        "list size and the share of runs that return any pair (every pair is false)"
    out.append(rf"\caption{{{name} design: {what}, means over 20 seeds, for $p=10$, 30, and 50 "
               rf"(45, 435, and 1,225 candidate pairs).}}")
    out.append(rf"\label{{tab:app-{key}}}\footnotesize\setlength{{\tabcolsep}}{{4pt}}")
    out.append(r"\begin{tabular}{ll" + ("c" * ncol) * 3 + "}")
    out.append(r"\toprule")
    out.append(" & & " + " & ".join(rf"\multicolumn{{{ncol}}}{{c}}{{$p={p}$}}" for p in (10, 30, 50)) + r" \\")
    cm = "".join(rf"\cmidrule(lr){{{3 + i * ncol}-{2 + (i + 1) * ncol}}}" for i in range(3))
    out.append(cm)
    out.append("Method & Model & " + " & ".join([cols] * 3) + r" \\")
    out.append(r"\midrule")
    for rule, label, model in RULES:
        cells = []
        for p in (10, 30, 50):
            r = blocks.get(f"{key} p{p}", {}).get(rule)
            if r is None:
                cells += ["--"] * ncol
            elif has_truth:
                cells += [f(r["size"], 1), f(r["fdp"]), f(r["power"])]
            else:
                cells += [f(r["size"], 1), f(r["any"])]
        out.append(f"{label} & {model} & " + " & ".join(cells) + r" \\")
    out.append(r"\bottomrule\end{tabular}")
    out.append(r"\tabnote{$^\dagger$ DIAMOND with the XGBoost settings of LENS.}")
    out.append(r"\end{table}")
    out.append("")

# F1--F10: FDP / power per function for the main rules.
main_rules = [r for r in RULES if r[0] in (
    "lens/floor0.02_alpha0.05", "lens_mlp/floor0.02_alpha0.05", "diamond/calibrated_q0.2",
    "diamond_mlp/calibrated_q0.2", "h2_refit/normal_by_alpha0.05", "iloco/by_alpha0.05",
    "iloco_mp/by_alpha0.05", "fast/validation_k")]
for metric, mname in (("fdp", "FDP"), ("power", "power"), ("size", "list size")):
    out.append(r"\begin{table}[H]\centering")
    out.append(rf"\caption{{F1--F10 ($p=30$): {mname} per function, means over 20 seeds.}}")
    out.append(rf"\label{{tab:app-f-{metric}}}\footnotesize\setlength{{\tabcolsep}}{{3.5pt}}")
    out.append(r"\begin{tabular}{ll" + "c" * 11 + "}")
    out.append(r"\toprule")
    out.append("Method & Model & " + " & ".join(f"F{i}" for i in range(1, 11)) + r" & mean \\")
    out.append(r"\midrule")
    for rule, label, model in main_rules:
        vals = [blocks[f"F{i} p30"][rule][metric] for i in range(1, 11)]
        d = 1 if metric == "size" else 2
        out.append(f"{label} & {model} & " + " & ".join(f(v, d) for v in vals)
                   + f" & {f(sum(vals) / 10, d)}" + r" \\")
    out.append(r"\bottomrule\end{tabular}\end{table}")
    out.append("")

OUT.write_text("\n".join(out) + "\n")
print("wrote", OUT)
