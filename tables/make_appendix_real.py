"""Appendix table of selection rules on real data (insurance and credit), from ablation_v5 runs."""
import json, glob, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results/real"
OUT = HERE / "appendix_real_table.tex"
cols = [("fremtpl2/train20000", "outer_test_mean_poisson_deviance", "all36"),
        ("fremtpl2/train50000", "outer_test_mean_poisson_deviance", "all36"),
        ("fremtpl2/full", "outer_test_mean_poisson_deviance", "all36"),
        ("credit/heloc", "outer_test_mean_log_loss", "all"),
        ("credit/uci_credit", "outer_test_mean_log_loss", "all")]
rules = [("lens", "LENS (default)"), ("family_rms", r"LENS, $\kappa=0$"),
         ("pair_relative", r"LENS, $b_u=\eta$"),
         ("projection_floor", r"$\hat\mu_u$ above the boundary, no test"),
         ("top_lens", r"Top-$|\widehat G|$ by energy"),
         ("single_model", r"$\hat f_D$ only: $a_u$ above the floor"),
         ("both_strong", r"Both fits above the floor, no comparison"),
         ("ALL", "All pairs")]
data = {}
for d, key, allname in cols:
    rows = {}
    for f in sorted(glob.glob(str(ROOT / d / "split*.json"))):
        j = json.load(open(f))
        L = j[key]
        rows.setdefault("f0", []).append((L["additive_f0"], 0))
        rows.setdefault("teacher", []).append((L["discovery_teacher"], 0))
        for r, _ in rules:
            name = allname if r == "ALL" else r
            rows.setdefault(r, []).append((L[f"f0_plus_{name}"], len(j["lists"][name])))
    data[d] = rows
out = [r"\begin{table}[H]\centering",
       r"\caption{Selection rules on real data: mean list size and test loss over five splits "
       r"(Poisson deviance for freMTPL2 at three training sizes per model; log loss for credit). "
       r"Each list is refitted on top of the same additive model $f_0$.}",
       r"\label{tab:app-real}\footnotesize\setlength{\tabcolsep}{3.5pt}",
       r"\begin{tabular}{l" + "rc" * 5 + "}", r"\toprule",
       r" & \multicolumn{6}{c}{freMTPL2 (rows per model)} & \multicolumn{4}{c}{Credit} \\",
       r"\cmidrule(lr){2-7}\cmidrule(lr){8-11}",
       r" & \multicolumn{2}{c}{20,000} & \multicolumn{2}{c}{50,000} & \multicolumn{2}{c}{184,420} "
       r"& \multicolumn{2}{c}{HELOC} & \multicolumn{2}{c}{UCI} \\",
       r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}\cmidrule(lr){10-11}",
       r"Rule" + r" & size & loss" * 5 + r" \\", r"\midrule"]
for tag, label in (("f0", r"Additive $f_0$"), ("teacher", r"Black box $\hat f_D$")):
    out.append(label + "".join(f" & -- & {np.mean([v for v, _ in data[d][tag]]):.5f}" for d, _, _ in cols) + r" \\")
out.append(r"\midrule")
for r, label in rules:
    cells = []
    for d, _, _ in cols:
        v = data[d][r]
        cells.append(f" & {np.mean([s for _, s in v]):.1f} & {np.mean([x for x, _ in v]):.5f}")
    out.append(label + "".join(cells) + r" \\")
out += [r"\bottomrule\end{tabular}\end{table}"]
OUT.write_text("\n".join(out) + "\n")
print(OUT.read_text())
