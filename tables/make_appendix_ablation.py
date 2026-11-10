"""Synthetic ablation table: rules computed from the rerun LENS XGBoost runs (results/synthetic_ablation)."""
import json, glob, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
ROOT = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / "results/synthetic_ablation"
OUT = HERE / "appendix_ablation_synthetic.tex"

def by(ordered, p, alpha=0.05):
    m = len(ordered); c = sum(1 / i for i in range(1, m + 1))
    ps = sorted(p.values()); k = 0
    for i, v in enumerate(ps, 1):
        if v <= alpha * i / (m * c): k = i
    return [] if k == 0 else [u for u in ordered if p[u] <= ps[k - 1]]

res = defaultdict(lambda: defaultdict(list)); bad = 0
for f in sorted(glob.glob(str(ROOT / "*_lens.json"))):
    d = json.load(open(f))
    name = re.sub(r"_seed\d+_lens.json", "", Path(f).name)
    group = "F" if re.match(r"F\d+_", name) else name
    truth = {tuple(sorted(t)) for t in d["truth"]}
    s, sd = d["family_rms_norm"], d["discovery_margin_sd"]
    floor = max(.25 * s, .02 * sd)
    P = [(int(j), int(k), a, mu, se, ac) for j, k, a, mu, se, ac in d["pairs"]]
    ordered = [(j, k) for j, k, *_ in P]
    b = {(j, k): max(.25 * a, floor) for j, k, a, *_ in P}
    pv = {(j, k): (1.0 if a == 0 else (float(mu <= b[(j, k)]) if se == 0 else norm.sf((mu - b[(j, k)]) / se)))
          for j, k, a, mu, se, ac in P}
    lens = by(ordered, pv)
    bad += sorted(map(list, lens)) != sorted(d["lists"]["floor0.02_alpha0.05"]["pairs"])
    lists = {"lens": lens, "top": ordered[:len(lens)],
             "single": [(j, k) for j, k, a, *_ in P if a > floor],
             "both": [(j, k) for j, k, a, mu, se, ac in P if min(a, ac) > floor],
             "nohtest": [(j, k) for j, k, a, mu, *_ in P if mu > b[(j, k)]]}
    for key, L in lists.items():
        L = {tuple(sorted(u)) for u in L}
        r = res[group]
        r[key, "size"].append(len(L))
        r[key, "fdp"].append(len(L - truth) / max(len(L), 1))
        r[key, "power"].append(len(L & truth) / max(len(truth), 1))
        r[key, "any"].append(float(len(L) > 0))
print(f"runs read; recomputed LENS lists differing from stored: {bad}", file=sys.stderr)

labels = [("lens", "LENS (default)"), ("nohtest", r"$\hat\mu_u$ above the boundary, no test"),
          ("top", r"Top-$|\widehat G|$ by energy"), ("both", "Both fits above the floor, no comparison"),
          ("single", r"$\hat f_D$ only: $a_u$ above the floor")]
designs = [("additive", "Additive", False), ("additive_corr", "Additive-corr", False),
           ("sparse_pairs", "Sparse", True), ("pair_corr", "Pair-corr", True)]
m = lambda g, k, x: np.mean(res[g][k, x])
out = [r"\begin{table}[H]\centering",
       r"\caption{Ablations on synthetic data with XGBoost, means over 20 seeds. Each rule removes one "
       r"ingredient of LENS and keeps the boundary of Eq.~\eqref{eq:null}. For the additive designs, "
       r"the share of runs that return any pair (every pair is false); otherwise list size, FDP, and power.}",
       r"\label{tab:app-ablation}\footnotesize\setlength{\tabcolsep}{3.2pt}"]
for p in (10, 30, 50):
    if p == 10:
        out += [r"\begin{tabular}{lcc ccc ccc ccc}", r"\toprule",
                r" & Additive & Additive-corr & \multicolumn{3}{c}{Sparse} & \multicolumn{3}{c}{Pair-corr} & \multicolumn{3}{c}{F1--F10 ($p=30$)} \\",
                r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}\cmidrule(lr){10-12}",
                r"Rule & any & any & size & FDP & power & size & FDP & power & size & FDP & power \\"]
    out.append(r"\midrule")
    out.append(rf"\multicolumn{{12}}{{l}}{{\emph{{$p={p}$}}}} \\")
    for key, label in labels:
        row = [label]
        for g, _, truth in designs:
            g = f"{g}_p{p}"
            if not res[g][key, "size"]:
                row += ["--"] * (3 if truth else 1); continue
            row += [f"{m(g, key, 'size'):.1f}", f"{m(g, key, 'fdp'):.2f}", f"{m(g, key, 'power'):.2f}"] if truth \
                else [f"{m(g, key, 'any'):.2f}"]
        if p == 30:
            row += [f"{m('F', key, 'size'):.1f}", f"{m('F', key, 'fdp'):.2f}", f"{m('F', key, 'power'):.2f}"]
        else:
            row += [""] * 3
        out.append(" & ".join(row) + r" \\")
out += [r"\bottomrule\end{tabular}\end{table}"]
OUT.write_text("\n".join(out) + "\n")
print("\n".join(out))
