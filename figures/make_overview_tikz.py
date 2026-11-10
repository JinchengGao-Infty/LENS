"""Write fig_overview.tex, the TikZ overview figure, with illustrative tables.

The tables are synthetic 2x2 tables, on which every residual is a checker pattern
of some size.  The second model reproduces the kept pattern and almost none of
the dropped one.  The reproduced shares come from the replication ratio, so the
numbers printed in the figure are those of the tables drawn.  Stdlib only.
"""

from __future__ import annotations

import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
G = 2
GRID = [-1 + 2 * i / (G - 1) for i in range(G)]


def project(t):
    """The projection R with equal weights: remove row and column means."""
    rows = [sum(r) / G for r in t]
    cols = [sum(t[i][j] for i in range(G)) / G for j in range(G)]
    mean = sum(rows) / G
    return [[t[i][j] - rows[i] - cols[j] + mean for j in range(G)] for i in range(G)]


def inner(a, b):
    return sum(a[i][j] * b[i][j] for i in range(G) for j in range(G)) / G**2


def norm(a):
    return math.sqrt(inner(a, a))


def scaled(a, s):
    return [[v * s for v in r] for r in a]


def heatmap(t, x0, y0, size, scale, frame="black!55", width="0.35pt", grid="white"):
    """Cells of a table with its top-left corner at (x0, y0), red positive."""
    cw = size / G
    out = []
    for i, row in enumerate(t):
        for j, v in enumerate(row):
            share = int(round(100 * min(abs(v) / scale, 1.0)))
            color = f"{'hpos' if v > 0 else 'hneg'}!{share}!white"
            out.append(rf"\fill[{color}] ({x0 + j * cw:.3f},{y0 - (i + 1) * cw:.3f}) "
                       rf"rectangle +({cw:.3f},{cw:.3f});")
    for k in range(1, G):
        out.append(rf"\draw[{grid}, line width=0.25pt] ({x0 + k * cw:.3f},{y0 - size:.3f}) -- +(0,{size:.3f});")
        out.append(rf"\draw[{grid}, line width=0.25pt] ({x0:.3f},{y0 - k * cw:.3f}) -- +({size:.3f},0);")
    out.append(rf"\draw[line width={width}, {frame}] ({x0:.3f},{y0 - size:.3f}) "
               rf"rectangle +({size:.3f},{size:.3f});")
    return out


def main() -> None:
    rd = [[1.0, -1.0], [-1.0, 1.0]]
    rc = scaled(rd, 0.92)
    qd = [[-0.72, 0.72], [0.72, -0.72]]
    qc = scaled(qd, 0.03)
    a1 = norm(rd)
    theta1 = inner(rd, rc) / inner(rd, rd)
    theta2 = inner(qd, qc) / inner(qd, qd)
    surface = [[1.15 * x + 0.55 * y + rd[i][j] for j, y in enumerate(GRID)]
               for i, x in enumerate(GRID)]
    drawn = max(abs(a - b) for ra, rb in zip(project(surface), rd) for a, b in zip(ra, rb))
    assert drawn < 1e-9, "the residual drawn must be R of the surface drawn"
    scale = max(abs(v) for t in (rd, rc, qd, qc) for r in t for v in r)
    sscale = 0.85 * max(abs(v) for r in surface for v in r)

    H, TOP = 4.85, 4.47
    out = [
        r"\begin{tikzpicture}[x=1cm, y=1cm, font=\sffamily\scriptsize,",
        r"  ptitle/.style={font=\sffamily\footnotesize\bfseries, text=hnavy},",
        r"  flow/.style={-{Stealth[length=4.5pt, width=4pt]}, line width=0.8pt, hnavy},",
        r"  thin/.style={-{Stealth[length=3pt, width=3pt]}, line width=0.5pt, hnavy},",
        r"  note/.style={font=\sffamily\scriptsize, text=hnavy!75, align=center}]",
    ]

    # Soft regions, titles, and the arrows between them.
    regions = [(0.0, 3.25, "hblue!7", "(a) Fit twice"),
               (3.5, 4.1, "hgreen!7", "(b) Remove main effects"),
               (7.85, 6.35, "horange!7", "(c) Compare the two models"),
               (14.45, 3.65, "hnavy!6", "(d) Decide")]
    for x, w, color, title in regions:
        out.append(rf"\fill[{color}, rounded corners=5pt] ({x},0) rectangle +({w},{H});")
        out.append(rf"\node[ptitle] at ({x + w / 2:.3f},{TOP}) {{{title}}};")
    for x in (3.25, 7.6, 14.2):
        out.append(rf"\draw[flow] ({x - 0.05},2.15) -- ({x + 0.3},2.15);")

    # Stage 1: one data table split into three disjoint parts.
    x0, cw, ch, top = 0.22, 0.68, 0.345, 3.88
    for k, name in enumerate(["x_1", "x_2"]):
        out.append(rf"\node[font=\tiny, text=hnavy] at ({x0 + (k + 0.5) * cw:.3f},{top + 0.14:.3f}) {{${name}$}};")
    parts = [("hblue", "D"), ("horange", "C"), ("black", "E")]
    y = top
    centers = []
    for color, name in parts:
        h = 3 * ch
        tint = "black!14" if color == "black" else f"{color}!22"
        out.append(rf"\fill[{tint}] ({x0},{y - h:.3f}) rectangle +({2 * cw},{h:.3f});")
        for r in range(1, 3):
            out.append(rf"\draw[white, line width=0.5pt] ({x0},{y - r * ch:.3f}) -- +({2 * cw},0);")
        for c in range(1, 2):
            out.append(rf"\draw[white, line width=0.5pt] ({x0 + c * cw:.3f},{y - h:.3f}) -- +(0,{h:.3f});")
        text = "black!65" if color == "black" else color
        out.append(rf"\node[font=\footnotesize\bfseries, text={text}] at ({x0 + cw},{y - h / 2:.3f}) {{${name}$}};")
        centers.append(y - h / 2)
        y -= h
    bx = 2.4
    for (color, name), cy in zip(parts[:2], centers[:2]):
        out.append(rf"\node[draw={color}, fill=white, rounded corners=2.5pt, line width=0.6pt, "
                   rf"minimum width=0.95cm, minimum height=0.62cm, text={color}] (m{name}) at ({bx + 0.25},{cy:.3f}) {{\footnotesize $\hat f_{name}$}};")
        out.append(rf"\draw[thin] ({x0 + 2 * cw + 0.05},{cy:.3f}) -- (m{name}.west);")
    out.append(rf"\node[note, anchor=west] at ({x0 + 2 * cw + 0.06},{centers[2]:.3f}) {{evaluation\\contexts}};")
    out.append(rf"\node[note] at (1.625,0.34) {{two models, disjoint data}};")

    # Stage 2: the projection on one pair.
    out.append(rf"\node[note, font=\sffamily\scriptsize] at (5.55,4.03) {{pair $u=(x_1,x_2)$}};")
    size, top2 = 1.6, 3.8
    out += heatmap(surface, 3.68, top2, size, sscale)
    out += heatmap(rd, 5.82, top2, size, scale)
    out.append(rf"\draw[flow] (5.32,{top2 - size / 2}) -- (5.78,{top2 - size / 2}) node[midway, above=-1pt] {{\footnotesize $R$}};")
    out.append(rf"\node[text=hnavy] at (4.48,{top2 - size - 0.2}) {{surface $F_u$}};")
    out.append(rf"\node[text=hnavy] at (6.62,{top2 - size - 0.2}) {{residual table $r_u$}};")
    out.append(r"\node[note] at (5.55,0.86) {only the non-additive part is left};")
    out.append(r"\node[note] at (5.55,0.34) {its energy $\|r_u\|^2$ ranks the pairs};")

    # Stage 3: the cross inner product, the visual centre of the figure.
    size3, xa, xb, xt = 1.72, 8.05, 10.5, 12.3
    rows = [(rd, rc, 3.9, r"x_1\times x_2", theta1, "hgreen"),
            (qd, qc, 2.0, r"x_3\times x_4", theta2, "hred")]
    for ra, rb, y0, name, theta, color in rows:
        out += heatmap(ra, xa, y0, size3, scale, frame="hblue", width="0.9pt")
        # A nearly empty table needs a visible grid, or it reads as a blank box.
        grid = "white" if theta > 0.5 else "horange!60"
        out += heatmap(rb, xb, y0, size3, scale, frame="horange", width="0.9pt", grid=grid)
        cy = y0 - size3 / 2
        out.append(rf"\node[text=hnavy, font=\small] at ({(xa + size3 + xb) / 2:.3f},{cy:.3f}) {{$\langle\cdot,\cdot\rangle$}};")
        out.append(rf"\node[anchor=west, text=hnavy, align=left] at ({xt},{cy:.3f}) "
                   rf"{{\footnotesize ${name}$\\[1pt]reproduced\\[-0.5pt]{{\footnotesize\bfseries\color{{{color}}}{theta * 100:.0f}\%}}}};")
    out.append(rf"\node[font=\scriptsize, text=hblue] at ({xa + size3 / 2},4.07) {{$\hat f_D$}};")
    out.append(rf"\node[font=\scriptsize, text=horange] at ({xb + size3 / 2},4.07) {{$\hat f_C$}};")

    # Stage 4: ordered by energy, kept by the test.
    a = {"x1": a1, "x3": norm(qd)}
    items = [(r"x_1{\times}x_2", a["x1"], True), (r"x_3{\times}x_4", a["x3"], False),
             (r"x_5{\times}x_7", 0.52 * a1, True), (r"x_2{\times}x_6", 0.2 * a1, False)]
    out.append(r"\node[note] at (16.27,4.03) {ordered by energy};")
    for k, (name, val, kept) in enumerate(items):
        y = 3.55 - k * 0.52
        out.append(rf"\node[anchor=west, text=hnavy] at (14.55,{y:.2f}) {{${name}$}};")
        color = "hblue" if kept else "black!28"
        out.append(rf"\fill[{color}, rounded corners=1pt] (15.55,{y - 0.1:.2f}) rectangle +({1.8 * val / a1:.3f},0.2);")
        mark = r"\color{hgreen}\ding{51}" if kept else r"\color{hred}\ding{55}"
        out.append(rf"\node at (17.75,{y:.2f}) {{\footnotesize {mark}}};")
    out.append(r"\node[note] at (16.27,1.0) {strong is not enough:\\the second pair is dropped};")
    out.append(r"\node[note] at (16.27,0.34) {the test sets the list length};")
    out.append(r"\end{tikzpicture}")

    (HERE / "fig_overview.tex").write_text("\n".join(out) + "\n")
    print(f"theta kept {theta1:.3f}, theta dropped {theta2:.3f}, "
          f"a kept {a1:.3f}, a dropped {norm(qd):.3f}")


if __name__ == "__main__":
    main()
