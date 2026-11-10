"""Draw the ICDE data figures from figure_data.json (see make_figure_data.py)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Nimbus Roman", "Nimbus Roman No9 L", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.1,
    "lines.markersize": 3.5,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
WIDTH = 3.45  # IEEE column width in inches

STYLE = {  # label -> (color, marker, linestyle)
    "LENS (XGBoost)": ("#1A6FA8", "o", "-"),
    "LENS (MLP)": ("#177A5A", "s", "-"),
    "LENS, no floor (XGBoost)": ("#1A6FA8", "o", ":"),
    "DIAMOND (XGBoost)": ("#C85A15", "^", "-"),
    "DIAMOND (MLP)": ("#C85A15", "v", "--"),
    "H2, refit null": ("#7B3F99", "D", "-"),
    "iLOCO-Split": ("#5A6470", "P", "-"),
    "iLOCO-MP": ("#9AA3AD", "X", "-"),
    "FAST": ("#B8860B", "*", "-"),
}
DISPLAY = {"H2, refit null": "$H^2$, refit null"}
KAPPA = 0.02


def save(fig, name):
    fig.savefig(HERE / f"{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def scaling(data):
    labels = ["H2, refit null", "iLOCO-Split", "iLOCO-MP", "FAST", "DIAMOND (MLP)", "LENS (MLP)"]
    names = {"LENS (MLP)": "LENS", "DIAMOND (MLP)": "DIAMOND"}
    fig, axes = plt.subplots(1, 2, figsize=(3.32, 1.85))
    for ax, (dataset, key, title) in zip(axes, (("sparse_pairs", "size", "(a) Sparse: pairs on the list"),
                                                ("additive", "any", "(b) Additive: FDP"))):
        for label in labels:
            series = data["scaling"][f"{dataset}|{label}"]
            color, marker, ls = STYLE[label]
            lens = label.startswith("LENS")
            ax.plot([s["pairs"] for s in series], [max(s[key], 0.08) if key == "size" else s[key] for s in series],
                    color=STYLE["LENS (XGBoost)"][0] if lens else color, marker=marker, ls="-", label=names.get(label, DISPLAY.get(label, label)), clip_on=False,
                    lw=2.0 if lens else 0.8, ms=4.5 if lens else 2.6, alpha=1.0 if lens else 0.55,
                    zorder=5 if lens else 2)
        ax.set_xscale("log")
        ax.set_xticks([45, 435, 1225], ["45", "435", "1,225"])
        ax.minorticks_off()
        ax.set_xlabel("candidate pairs")
        ax.set_title(title, loc="left", fontsize=7.5)
    axes[0].set_yscale("log")
    axes[0].axhline(3, color="black", lw=0.6, ls="--", zorder=0)
    axes[0].set_ylim(1, 1000)
    axes[1].set_ylim(-0.03, 1.03)
    handles, names = axes[0].get_legend_handles_labels()
    order = [5, 4, 0, 1, 2, 3]
    fig.legend([handles[i] for i in order], [names[i] for i in order], loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.17), columnspacing=0.9, handlelength=1.8, handletextpad=0.4)
    fig.tight_layout(w_pad=1.0)
    save(fig, "fig_scaling")


def strength(data):
    edges = [0.003, 0.01, 0.02, 0.05, 0.1, 0.4]
    labels = ["LENS (XGBoost)", "LENS (MLP)", "DIAMOND (XGBoost)", "DIAMOND (MLP)"]
    rows = [r for r in data["detection"] if r["strength"] > 0]
    fig, ax = plt.subplots(figsize=(3.9, 1.85))
    centers = [math.sqrt(a * b) for a, b in zip(edges[:-1], edges[1:])]
    counts = [sum(a < r["strength"] <= b for r in rows) for a, b in zip(edges[:-1], edges[1:])]
    for label in labels:
        rates = [np.mean([r["rate"][label] for r in rows if a < r["strength"] <= b])
                 for a, b in zip(edges[:-1], edges[1:])]
        color, marker, ls = STYLE[label]
        ax.plot(centers, rates, color=color, marker=marker, ls=ls, label=label.replace("XGBoost", "XGB"),
                clip_on=False)
    ax.axvline(KAPPA, color="black", lw=0.6, ls="--", zorder=0)
    ax.text(KAPPA * 1.07, 0.05, r"$\kappa=0.02$", fontsize=6.5)
    for c, n in zip(centers, counts):
        ax.text(c, 1.06, f"{n}", ha="center", fontsize=6, color="#5A6470")
    ax.text(edges[0], 1.06, "pairs:", ha="right", fontsize=6, color="#5A6470")
    ax.set_xscale("log")
    ax.set_xlim(edges[0], edges[-1])
    ax.set_ylim(0, 1.0)
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2], ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2"])
    ax.minorticks_off()
    ax.set_xlabel(r"strength of the true pair, $\|r_u\|/\mathrm{sd}(f)$")
    ax.set_ylabel("share of runs detected")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.93), frameon=False, handlelength=1.6, borderaxespad=0.2)
    save(fig, "fig_strength")


def floor(data):
    groups = [
        ("false, additive|MLP", "false, Additive (MLP)"),
        ("false, additive|XGBoost", "false, Additive (XGB)"),
        ("proxy|MLP", "proxies, Pair-corr (MLP)"),
        ("proxy|XGBoost", "proxies, Pair-corr (XGB)"),
        ("true, F1-F10|XGBoost", "true, F1--F10 (XGB)"),
        ("true, sparse_pairs|XGBoost", "true, Sparse (XGB)"),
    ]
    colors = ["#9AA3AD", "#9AA3AD", "#C85A15", "#C85A15", "#1A6FA8", "#1A6FA8"]
    values = [np.asarray(data["floor"][key]) for key, _ in groups]
    fig, ax = plt.subplots(figsize=(3.05, 1.75))
    positions = np.arange(len(groups))[::-1]
    parts = ax.boxplot([np.clip(v, 1e-5, None) for v in values], positions=positions, vert=False, widths=0.55,
                       whis=(5, 95), showfliers=False, patch_artist=True,
                       medianprops={"color": "black", "lw": 0.9}, whiskerprops={"lw": 0.6},
                       capprops={"lw": 0.6}, boxprops={"lw": 0.6})
    for patch, color in zip(parts["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axvline(KAPPA, color="black", lw=0.6, ls="--", zorder=0)
    ax.text(KAPPA * 1.08, len(groups) - 0.55, r"$\kappa=0.02$", fontsize=6.5)
    ax.set_yticks(positions, [name.replace("--", "–") for _, name in groups])
    for pos, v in zip(positions, values):
        ax.text(1.3, pos, f"{np.mean(v < KAPPA):.0%}", va="center", fontsize=6, color="#5A6470")
    ax.set_xscale("log")
    ax.set_xlim(1e-4, 1.2)
    ax.text(1.3, len(groups) - 0.45, "below", fontsize=6, color="#5A6470")
    ax.set_xlabel(r"reproduced strength $\mu_u/\sigma_D$")
    ax.tick_params(axis="y", length=0)
    save(fig, "fig_floor")


def fremtpl2(data):
    pairs = data["fremtpl2"]["pairs"]
    fig, ax = plt.subplots(figsize=(3.57, 2.0))
    x = np.geomspace(0.002, 0.4, 200)
    ax.plot(x, np.maximum(0.25, KAPPA / x), color="black", lw=0.7, ls="--", zorder=1)
    ax.fill_between(x, np.maximum(0.25, KAPPA / x), 1.6, color="#E8F3FB", zorder=0)
    cmap = plt.get_cmap("Blues")
    for name, v in pairs.items():
        ax.scatter(v["a"], v["theta"], s=16, color=cmap(0.15 + 0.17 * v["kept"]), edgecolor="#1E2D40",
                   linewidth=0.4, zorder=3)
    marks = {  # label position in data coordinates
        "VehAge x VehBrand": (0.36, 1.13, "right"), "DrivAge x BonusMalus": (0.36, 0.40, "right"),
        "VehAge x Region": (0.36, -0.05, "right"),
        "VehBrand x VehGas": (0.012, 1.17, "center"),
    }
    for name, (tx, ty, ha) in marks.items():
        v = pairs[name]
        ax.annotate(name.replace(" x ", "×"), (v["a"], v["theta"]), xytext=(tx, ty), textcoords="data",
                    fontsize=6, ha=ha, va="center",
                    arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "#5A6470", "shrinkA": 1, "shrinkB": 2.5})
    ax.set_xscale("log")
    ax.set_xlim(0.002, 0.4)
    ax.set_ylim(-0.1, 1.45)
    ax.set_xticks([0.005, 0.01, 0.02, 0.05, 0.1, 0.2], ["0.005", "0.01", "0.02", "0.05", "0.1", "0.2"])
    ax.minorticks_off()
    ax.set_xlabel(r"discovery strength $a_u/\sigma_D$ (ranking)")
    ax.set_ylabel(r"replication ratio $\theta_u$ (test)")
    ax.text(0.37, 1.32, "kept if significantly inside", fontsize=6, color="#1A6FA8", ha="right")
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap([cmap(0.15 + 0.17 * k) for k in range(6)]),
                               norm=matplotlib.colors.Normalize(-0.5, 5.5))
    bar = fig.colorbar(sm, ax=ax, ticks=range(6), fraction=0.04, pad=0.02)
    bar.set_label("splits kept", fontsize=7)
    bar.ax.tick_params(labelsize=6, length=2)
    bar.outline.set_linewidth(0.5)
    save(fig, "fig_fremtpl2")


def h2(data):
    """Energy against main-effect dilution M/Delta in one run; H^2 = 1/(1+M/Delta)."""
    ex = data["h2"]["example"]
    pairs = [r for r in ex["pairs"] if r["delta"] > 1e-11]  # the rest are additive to rounding error
    print(f"h2 example: {len(ex['pairs']) - len(pairs)} pairs with energy below 1e-11 omitted, "
          f"{sum(r['true'] for r in ex['pairs'] if r['delta'] <= 1e-11)} of them true")
    delta = np.array([r["delta"] for r in pairs])
    ratio = np.array([max(r["main"], 1e-12) / r["delta"] for r in pairs])
    true = np.array([r["true"] for r in pairs])
    idle = np.array([r["n_active"] == 0 for r in pairs])
    e_cut = np.sort(delta)[-15]
    h_cut = np.sort(ratio)[14]
    fig, ax = plt.subplots(figsize=(3.4, 2.15))
    groups = ((~true & idle, "#9AA3AD", ".", "false, both irrelevant", 6),
              (~true & ~idle, "#C85A15", "^", "false, other", 9),
              (true, "#1A6FA8", "o", "true interaction", 14))
    for mask, color, marker, label, size in groups:
        ax.scatter(delta[mask], ratio[mask], s=size, color=color, marker=marker, linewidth=0.3,
                   edgecolor="white" if marker == "o" else "none", label=label, zorder=3 if marker == "o" else 2)
    ax.axvline(e_cut, color="#1A6FA8", lw=0.7, ls="--", zorder=1)
    ax.axhline(h_cut, color="#7B3F99", lw=0.7, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_yscale("log")
    xlo, xhi = delta.min() / 2, delta.max() * 3
    ylo, yhi = ratio.min() / 3, ratio.max() * 3
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.text(e_cut * 1.25, yhi / 1.6, "top 15 by energy $\\rightarrow$", fontsize=6, color="#1A6FA8", va="top")
    ax.text(xlo * 1.3, h_cut / 1.5, "top 15 by $H^2$ $\\downarrow$", fontsize=6, color="#7B3F99", va="top")
    ax.set_xlabel(r"interaction energy $\Delta_u$ (LENS ranking)")
    ax.set_ylabel(r"dilution $M_u/\Delta_u$")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, handletextpad=0.1,
              borderaxespad=0.1, columnspacing=1.0, fontsize=6.5)
    save(fig, "fig_h2")
    top_e = sorted(range(len(pairs)), key=lambda i: -delta[i])[:15]
    top_h = sorted(range(len(pairs)), key=lambda i: ratio[i])[:15]
    print(f"h2 example {ex['dgp']} seed {ex['seed']}: {true.sum()} true of {len(pairs)}; "
          f"top-15 true by energy {true[top_e].sum()}, by H2 {true[top_h].sum()}, "
          f"H2 top-15 idle false {(idle & ~true)[top_h].sum()}")


def kappa(data):
    from matplotlib.lines import Line2D
    rows = data["sensitivity"]
    kappas = [r["kappa"] for r in rows if r["vary"] == "kappa" and r["model"] == "lens"]
    xs = np.arange(len(kappas))
    fig, axes = plt.subplots(1, 2, figsize=(3.32, 1.6))
    models = (("lens", "LENS (XGB)", "#1A6FA8", "o"), ("lens_mlp", "LENS (MLP)", "#177A5A", "s"))
    for model, label, color, marker in models:
        series = [r for r in rows if r["vary"] == "kappa" and r["model"] == model]
        axes[0].plot(xs, [r["f_material"] for r in series], color=color, marker=marker)
        axes[0].plot(xs, [r["f_fdp"] for r in series], color=color, marker=marker, ls=":", mfc="white")
        axes[1].plot(xs, [r["additive_any"] for r in series], color=color, marker=marker)
        axes[1].plot(xs, [r["sparse_fdp"] for r in series], color=color, marker=marker, ls=":", mfc="white")
    styles = (("(a) F1–F10", "power on material $\\uparrow$", "FDP $\\downarrow$"),
              ("(b) Additive and Sparse", "Additive FDP $\\downarrow$", "Sparse FDP $\\downarrow$"))
    for ax, (title, solid, dotted), loc in zip(axes, styles, (("center left", (0.0, 0.55)),
                                                             ("upper right", (1.03, 0.97)))):
        ax.set_xticks(xs, [f"{k:g}".replace("0.", ".") if k else "0" for k in kappas], fontsize=6)
        ax.axvline(kappas.index(KAPPA), color="black", lw=0.6, ls="--", zorder=0,
                   ymax=1.0 if ax is axes[0] else 0.68)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel(r"$\kappa$")
        ax.set_title(title, loc="left", fontsize=7.5)
        ax.legend([Line2D([], [], color="#5A6470", marker="o", ms=3),
                   Line2D([], [], color="#5A6470", marker="o", ms=3, ls=":", mfc="white")],
                  [solid, dotted], loc=loc[0], bbox_to_anchor=loc[1], frameon=False, fontsize=5.8, handlelength=1.6,
                  borderaxespad=0.1, labelspacing=0.3, handletextpad=0.4)
    handles = [Line2D([], [], color=c, marker=m, ms=3.5) for _, _, c, m in models]
    fig.legend(handles, [label for _, label, _, _ in models], loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.06), columnspacing=1.2, handlelength=1.8, fontsize=6.5)
    fig.tight_layout(w_pad=1.0, rect=(0, 0.06, 1, 1))
    save(fig, "fig_kappa")


def sizes(data):
    rows = data["fremtpl2_sizes"]
    x = [r["policies"] for r in rows]
    fig, ax = plt.subplots(figsize=(3.3, 1.85))
    lines = (("black_box", "black box", "#5A6470", "D", "--"), ("all36", "all 36 pairs", "#C85A15", "^", "-"),
             ("top7", "top 7 by energy", "#B8860B", "*", "-"), ("lens", "LENS list ({} pairs)".format(", ".join(f"{r['lens_size']:g}" for r in rows)), "#1A6FA8", "o", "-"))
    for key, label, color, marker, ls in lines:
        ax.plot(x, [1000 * (r[key] - r["additive"]) for r in rows], color=color, marker=marker, ls=ls, label=label,
                clip_on=False)
    ax.axhline(0, color="black", lw=0.6, zorder=0)
    ax.text(x[-1], 0.25, "additive model", fontsize=6, ha="right", va="bottom", color="#5A6470")
    ax.set_xscale("log")
    ax.set_xticks(x, ["20k", "50k", "184k"])
    ax.minorticks_off()
    ax.set_xlabel("policies per model")
    ax.set_ylabel(r"test deviance $-$ additive ($\times10^{-3}$)")
    ax.legend(loc="lower left", frameon=False, fontsize=6.5, handlelength=1.8)
    save(fig, "fig_sizes")


def cost(data):
    show = (("lens", "LENS"), ("diamond", "DIAMOND (XGBoost)"), ("h2_refit", "H2, refit null"),
            ("iloco", "iLOCO-Split"), ("iloco_mp", "iLOCO-MP"), ("fast", "FAST"))
    style = {"lens": "LENS (XGBoost)", "diamond": "DIAMOND (XGBoost)", "h2_refit": "H2, refit null",
             "iloco": "iLOCO-Split", "iloco_mp": "iLOCO-MP", "fast": "FAST"}
    fig, ax = plt.subplots(figsize=(3.3, 1.6))
    for method, label in show:
        series = data["timing"][method]
        color, marker, ls = STYLE[style[method]]
        ax.plot([s["pairs"] for s in series], [s["seconds"] for s in series], color=color, marker=marker, ls=ls,
                label=DISPLAY.get(label, label), clip_on=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks([45, 435, 1225], ["45", "435", "1,225"])
    ax.minorticks_off()
    ax.set_xlabel("candidate pairs")
    ax.set_ylabel("seconds per run")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2, frameon=False, fontsize=6,
              handlelength=1.8, columnspacing=1.0)
    save(fig, "fig_cost")


def gallery(data):
    pairs = data["gallery"]["pairs"]
    ratios = [min(len(p["ticks"][1]), 16) for p in pairs]
    fig, axes = plt.subplots(2, len(pairs), figsize=(7.1, 2.85), width_ratios=ratios, layout="constrained")
    vmax = 0.25  # one log-rate scale for all tables
    for col, p in enumerate(pairs):
        D, C = np.asarray(p["D"]), np.asarray(p["C"])
        ticks_x = list(p["ticks"][1])
        if p["features"][1] == "VehBrand":  # natural order B1, B2, ..., B14
            order = sorted(range(len(ticks_x)), key=lambda i: int(ticks_x[i][1:]))
            D, C, ticks_x = D[:, order], C[:, order], [ticks_x[i] for i in order]
        for row, (table, name) in enumerate(((D, r"discovery $r_{D,u}$"), (C, r"confirmation $\bar h_{C,u}$"))):
            ax = axes[row, col]
            im = ax.imshow(table, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto", interpolation="nearest")
            ax.set_yticks(range(len(p["ticks"][0]))[::2], p["ticks"][0][::2], fontsize=5)
            step = 1 if len(ticks_x) <= 12 else 1
            ax.set_xticks(range(len(ticks_x))[::step], ticks_x[::step], fontsize=4.6 if len(ticks_x) > 12 else 5,
                          rotation=90)
            ax.tick_params(length=1.5, pad=1)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.4)
            if col == 0:
                ax.set_ylabel(f"{name}\n{p['features'][0]}", fontsize=6.5)
            else:
                ax.set_ylabel(p["features"][0], fontsize=6.5)
            if row == 1:
                ax.set_xlabel(p["features"][1], fontsize=6.5, labelpad=1)
        verdict = "kept" if p["theta"] > 0.25 else "dropped"
        axes[0, col].set_title(f"{p['name'].replace(' x ', '×')}\n"
                               f"$a_u={p['amplitude']:.3f}$, $\\theta_u={p['theta']:.2f}$, {verdict}", fontsize=6.5)
    bar = fig.colorbar(im, ax=axes, orientation="vertical", fraction=0.02, pad=0.01, aspect=30, extend="both")
    bar.set_label("log claim rate", fontsize=6)
    bar.ax.tick_params(labelsize=5, length=1.5, pad=1)
    bar.outline.set_linewidth(0.4)
    save(fig, "fig_gallery")

# F1--F10 means from Table V (results/synthetic/material_truth.txt):
# FDP against all true pairs, power on the material pairs (strength >= 2% of sd(f)).
PARETO = {
    "LENS (XGBoost)": (0.00, 0.91),
    "LENS (MLP)": (0.00, 1.00),
    "DIAMOND (XGBoost)": (0.07, 0.45),
    "DIAMOND (MLP)": (0.10, 0.47),
    "H2, refit null": (0.72, 1.00),
    "iLOCO-Split": (0.67, 0.72),
    "iLOCO-MP": (0.99, 0.28),
    "FAST": (0.51, 1.00),
}


def pareto(data):
    fig, ax = plt.subplots(figsize=(WIDTH, 2.0))
    rivals = sorted((v for k, v in PARETO.items() if not k.startswith("LENS")))
    front, best = [], -1.0
    for x, y in rivals:
        if y > best:
            front.append((x, y))
            best = y
    xs, ys = [front[0][0]], [0.0]
    for (x, y), nxt in zip(front, front[1:] + [(1.05, None)]):
        xs += [x, nxt[0]]
        ys += [y, y]
    ax.fill_between(xs, ys, 0, step=None, color="#9AA3AD", alpha=0.15, lw=0, zorder=0)
    ax.plot(xs, ys, color="#9AA3AD", ls="--", lw=0.8, zorder=1)
    ax.text(0.98, 0.06, "dominated by\nthe baselines", ha="right", fontsize=6.5, color="#5A6470")
    ax.text(0.525, 0.80, "Pareto front of\nthe baselines", fontsize=6, color="#5A6470", va="center")
    offsets = {
        "LENS (XGBoost)": (6, -2, "left"), "LENS (MLP)": (6, 2, "left"),
        "DIAMOND (XGBoost)": (6, -8, "left"), "DIAMOND (MLP)": (6, 3, "left"),
        "H2, refit null": (0, 6, "center"), "iLOCO-Split": (6, -2, "left"),
        "iLOCO-MP": (-6, 0, "right"), "FAST": (0, 6, "center"),
    }
    for label, (x, y) in PARETO.items():
        color, marker, _ = STYLE[label]
        lens = label.startswith("LENS")
        ax.scatter(x, y, color=color, marker=marker, s=34 if lens else (40 if marker == "*" else 22), zorder=3, clip_on=False,
                   edgecolor="white", linewidth=0.4)
        dx, dy, ha = offsets[label]
        ax.annotate(DISPLAY.get(label, label), (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=6.5, color=color,
                    fontweight="bold" if lens else "normal")
    ax.annotate("", xy=(0.24, 0.74), xytext=(0.40, 0.56),
                arrowprops=dict(arrowstyle="-|>", color="#5A6470", lw=0.7, mutation_scale=7))
    ax.text(0.335, 0.70, "better", fontsize=6.5, color="#5A6470", rotation=-42)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("FDP on F1–F10 $\\downarrow$")
    ax.set_ylabel(r"power on material pairs $\uparrow$")
    save(fig, "fig_pareto")


def main():
    data = json.loads((HERE / "figure_data.json").read_text())
    scaling(data)
    strength(data)
    pareto(data)
    floor(data)
    fremtpl2(data)
    h2(data)
    kappa(data)
    sizes(data)
    cost(data)
    gallery(data)


if __name__ == "__main__":
    main()
