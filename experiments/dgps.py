"""Unified DGP registry for interaction-testing benchmarks.

Covers:
  - The 3 in-house DGPs used in the hybrid-vs-exact 32-seed study
    (Additive, Friedman1, Smooth) — d = 10
  - F1..F10 from the Tsang et al. 2018 / Diamond (Nat Mach Intell 2025)
    benchmark suite, ported **verbatim** from the Diamond official repo:
        https://github.com/batmen-lab/diamond/blob/main/src/simulation/gen_Y.py
    using the Diamond default sampler
        https://github.com/batmen-lab/diamond/blob/main/src/simulation/gen_X.py
        (UniformSampler, p = 30, features ~ U(0, 1)^p)
    and the Diamond default sizes from gen_sim_data.py (n = 10000, p = 30).

Ground truth:
  Diamond gen_Y.py reports higher-order interaction groups, e.g. {0, 1, 2}.
  We decompose each group into its constituent pairs with
  ``itertools.combinations(sorted(group), 2)`` — exactly the same code
  Diamond uses in ``src/sim.py``.

Noise:
  Diamond uses Y = f(X) deterministically (no explicit additive noise;
  only a StandardScaler before training). We follow that for F1..F10.
  The in-house DGPs keep their original Gaussian noise so the 32-seed
  hybrid-vs-exact numbers stay directly reproducible.

Determinism:
  Each DGP callable owns a private ``np.random.RandomState(seed)`` and
  never touches global RNG state.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class DGP:
    """Immutable record describing one realization of a data-generating process."""
    name: str
    X: np.ndarray              # shape (n, p)
    y: np.ndarray              # shape (n,)
    true_pairs: list           # list[tuple[int, int]], sorted unique pairs
    active_features: set       # set of feature indices appearing in f
    interaction_groups: list   # raw higher-order groups from the source paper
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Ground-truth decomposition (matches Diamond src/sim.py)
# ---------------------------------------------------------------------------

def _pairs_from_groups(groups):
    """Decompose higher-order interaction sets into sorted unique (i, j) pairs.

    Matches Diamond src/sim.py:
        true_pairs = list(set([
            (i, j) for features in inter_gt
                   for i, j in itertools.combinations(sorted(list(features)), 2)
        ]))
    """
    pairs = set()
    for g in groups:
        g_sorted = sorted(g)
        for i, j in itertools.combinations(g_sorted, 2):
            pairs.add((int(i), int(j)))
    return sorted(pairs)


# ---------------------------------------------------------------------------
# Diamond F1..F10 (ported verbatim from batmen-lab/diamond gen_Y.py)
# ---------------------------------------------------------------------------
#
# Each ``_fN`` below takes ``x`` of shape ``(p, n)`` — i.e. features stacked
# along axis 0, same convention as Diamond's ``gen_Y.py`` after
# ``X.transpose()``. The wrapper ``_make_diamond_dgp`` handles the transpose.

def _f1(x):
    y = (np.pi ** (x[0] * x[1]) * np.sqrt(2 * x[2])
         - np.arcsin(x[3])
         + np.log(1 + x[2] + x[4])
         - (x[8] / (1 + x[9])) * np.sqrt(x[6] / (1 + x[7]))
         - x[1] * x[6])
    groups = [{0, 1, 2}, {2, 4}, {6, 7, 8, 9}, {1, 6}]
    active = {0, 1, 2, 3, 4, 6, 7, 8, 9}
    return y, groups, active


def _f2(x):
    y = (np.pi ** (x[0] * x[1]) * np.sqrt(2 * np.abs(x[2]))
         - np.arcsin(0.5 * x[3])
         + np.log(np.abs(x[2] + x[4]) + 1)
         - (x[8] / (1 + np.abs(x[9]))) * np.sqrt(np.abs(x[6]) / (1 + np.abs(x[7])))
         - x[1] * x[6])
    groups = [{0, 1, 2}, {2, 4}, {6, 7, 8, 9}, {1, 6}]
    active = {0, 1, 2, 3, 4, 6, 7, 8, 9}
    return y, groups, active


def _f3(x):
    y = (np.exp(np.abs(x[0] - x[1]))
         + np.abs(x[1] * x[2])
         - ((x[2]) ** 2) ** np.absolute(x[3])
         + np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2)
         + x[8]
         + 1.0 / (1 + x[9] ** 2))
    groups = [{0, 1}, {1, 2}, {2, 3}, {3, 4, 6, 7}]
    active = {0, 1, 2, 3, 4, 6, 7, 8, 9}
    return y, groups, active


def _f4(x):
    y = (np.exp(np.abs(x[0] - x[1]))
         + np.abs(x[1] * x[2])
         - ((x[2]) ** 2) ** np.absolute(x[3])
         + (x[0] * x[3]) ** 2
         + np.log(x[3] ** 2 + x[4] ** 2 + x[6] ** 2 + x[7] ** 2)
         + x[8]
         + 1.0 / (1 + x[9] ** 2))
    groups = [{0, 1}, {1, 2}, {2, 3}, {0, 3}, {3, 4, 6, 7}]
    active = {0, 1, 2, 3, 4, 6, 7, 8, 9}
    return y, groups, active


def _f5(x):
    y = (1.0 / (1 + x[0] ** 2 + x[1] ** 2 + x[2] ** 2)
         + np.sqrt(np.abs(x[3] + x[4]))
         + np.abs(x[5] + x[6])
         + x[7] * x[8] * x[9])
    groups = [{0, 1, 2}, {3, 4}, {5, 6}, {7, 8, 9}]
    active = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return y, groups, active


def _f6(x):
    y = (np.exp(np.abs(x[0] * x[1]) + 1)
         - np.exp(np.abs(x[2] + x[3]) + 1)
         + np.cos(x[4] + x[5] - x[7])
         + np.sqrt(x[7] ** 2 + x[8] ** 2 + x[9] ** 2))
    groups = [{0, 1}, {2, 3}, {4, 5, 7}, {7, 8, 9}]
    active = {0, 1, 2, 3, 4, 5, 7, 8, 9}
    return y, groups, active


def _f7(x):
    # Diamond gen_Y.py: np.max((x[2]*x[3]+x[5], np.zeros(x.shape[1])), 0)
    # This is an elementwise ``max(·, 0)``. We use np.maximum for clarity
    # and so it works for both 1-D and 2-D inputs.
    zeros = np.zeros_like(x[2])
    relu_term = np.maximum(x[2] * x[3] + x[5], zeros)
    y = ((np.arctan(x[0]) + np.arctan(x[1])) ** 2
         + relu_term
         - (1 + (x[3] * x[4] * x[5] * x[6] * x[7])) ** (-2)
         + (np.abs(x[6]) / (1 + np.abs(x[8]))) ** 5
         + np.sum(x[:10], axis=0))
    groups = [{0, 1}, {2, 3, 5}, {3, 4, 5, 6, 7}, {6, 8}]
    active = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return y, groups, active


def _f8(x):
    y = (x[0] * x[1]
         + 2 ** (x[2] + x[4] + x[5])
         + 2 ** (x[2] + x[3] + x[4] + x[6])
         + np.sin(x[6] * np.sin(x[7] + x[8]))
         + np.arccos(0.9 * x[9]))
    groups = [{0, 1}, {2, 4, 5}, {2, 3, 4, 6}, {6, 7, 8}]
    active = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return y, groups, active


def _f9(x):
    y = (np.tanh(x[0] * x[1] + x[2] * x[3]) * np.sqrt(np.abs(x[4]))
         + np.exp(x[4] + x[5])
         + np.log((x[5] * x[6] * x[7]) ** 2 + 1)
         + x[8] * x[9]
         + 1.0 / (1 + np.abs(x[9])))
    groups = [{0, 1, 2, 3, 4}, {4, 5}, {5, 6, 7}, {8, 9}]
    active = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
    return y, groups, active


def _f10(x):
    y = (np.sinh(x[0] + x[1])
         + np.arccos(np.tanh(x[2] + x[4] + x[6]))
         + np.cos(x[3] + x[4])
         + 1.0 / np.cos(x[6] * x[8]))
    groups = [{0, 1}, {2, 4, 6}, {3, 4}, {6, 8}]
    active = {0, 1, 2, 3, 4, 6, 8}
    return y, groups, active


_DIAMOND_FUNCS: dict[str, Callable] = {
    "F1": _f1, "F2": _f2, "F3": _f3, "F4": _f4, "F5": _f5,
    "F6": _f6, "F7": _f7, "F8": _f8, "F9": _f9, "F10": _f10,
}


def noise_free_response(name: str, X: np.ndarray) -> np.ndarray:
    """Evaluate the deterministic signal ``F*`` for a registered DGP.

    This is the public oracle-evaluation path for v2 run-matched metrics.  It
    never draws noise and never consults labels from a generated realization.
    """
    features = np.asarray(X, dtype=np.float64)
    if features.ndim != 2:
        raise ValueError(f"X must be two-dimensional, got {features.shape}")
    if not np.all(np.isfinite(features)):
        raise ValueError("X contains NaN or inf")
    if name == "Additive":
        if features.shape[1] < 5:
            raise ValueError("Additive requires at least five features")
        values = np.sin(features[:, :5]).sum(axis=1)
    elif name == "Friedman1":
        if features.shape[1] < 5:
            raise ValueError("Friedman1 requires at least five features")
        values = (
            10 * np.sin(np.pi * features[:, 0] * features[:, 1])
            + 20 * (features[:, 2] - 0.5) ** 2
            + 10 * features[:, 3]
            + 5 * features[:, 4]
        )
    elif name == "Smooth":
        if features.shape[1] < 8:
            raise ValueError("Smooth requires at least eight features")
        values = (
            np.sin(features[:, 0] * features[:, 1])
            + np.exp(np.clip(features[:, 2] + features[:, 3], -3, 3))
            + 0.5 * features[:, 4:8].sum(axis=1)
        )
    elif name in _DIAMOND_FUNCS:
        values, _, _ = _DIAMOND_FUNCS[name](features.T)
    else:
        raise KeyError(f"Unknown DGP name: {name!r}. Known: {DGP_NAMES}")
    output = np.asarray(values, dtype=np.float64).reshape(-1)
    if output.shape != (len(features),) or not np.all(np.isfinite(output)):
        raise ValueError(f"noise-free response for {name} is invalid")
    return output


def _make_diamond_dgp(name: str, seed: int, n: int = 10000, p: int = 30) -> DGP:
    """Diamond / Tsang F-series: U(0, 1)^p sampler, deterministic, no noise.

    Matches gen_sim_data.py defaults (p = 30, n = 10000) and gen_X.py
    UniformSampler([0, 1]^p).
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(low=0.0, high=1.0, size=(n, p))
    f = _DIAMOND_FUNCS[name]
    y_raw, groups, active = f(X.T)         # x[i] is column i, shape (n,)
    y = np.asarray(y_raw, dtype=np.float64)
    true_pairs = _pairs_from_groups(groups)
    meta = {
        "source": "diamond/gen_Y.py",
        "n_features_active": len(active),
        "n_features_total": p,
        "sampler": "UniformSampler [0,1]^p",
        "noise": "none (Y = f(X))",
    }
    return DGP(
        name=name, X=X, y=y,
        true_pairs=true_pairs,
        active_features=set(active),
        interaction_groups=groups,
        meta=meta,
    )


# ---------------------------------------------------------------------------
# In-house DGPs (d = 10) — byte-identical to multiseed_omnibus.py so the
# 32-seed hybrid-vs-exact numbers remain directly reproducible.
# ---------------------------------------------------------------------------

def _make_inhouse(seed: int, name: str, n: int = 10000, d: int = 10) -> DGP:
    """Reproduce the historical in-house RNG stream exactly.

    The 32-seed omnibus study generated Additive, Friedman1, and Smooth from a
    single ``RandomState(seed)`` in that order. To keep those results
    byte-identical, later DGPs must consume the same preceding random draws
    even when requested individually.
    """
    rng = np.random.RandomState(seed)

    X = rng.randn(n, d)
    y = sum(np.sin(X[:, j]) for j in range(5)) + rng.randn(n) * 0.1
    if name == "Additive":
        return DGP(
            name="Additive", X=X, y=np.asarray(y, dtype=np.float64),
            true_pairs=[],
            active_features={0, 1, 2, 3, 4},
            interaction_groups=[],
            meta={"source": "in-house", "n_features_total": d,
                  "sampler": "N(0, I_d)", "noise": "sigma=0.1"},
        )

    X = rng.uniform(0, 1, (n, d))
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1])
         + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3]
         + 5 * X[:, 4]
         + rng.randn(n) * 1.0)
    if name == "Friedman1":
        return DGP(
            name="Friedman1", X=X, y=np.asarray(y, dtype=np.float64),
            true_pairs=[(0, 1)],
            active_features={0, 1, 2, 3, 4},
            interaction_groups=[{0, 1}],
            meta={"source": "Friedman 1991", "n_features_total": d,
                  "sampler": "U(0,1)^d", "noise": "sigma=1.0"},
        )

    X = rng.randn(n, d)
    y = (np.sin(X[:, 0] * X[:, 1])
         + np.exp(np.clip(X[:, 2] + X[:, 3], -3, 3))
         + sum(0.5 * X[:, j] for j in range(4, 8))
         + rng.randn(n) * 0.1)
    if name == "Smooth":
        return DGP(
            name="Smooth", X=X, y=np.asarray(y, dtype=np.float64),
            true_pairs=[(0, 1), (2, 3)],
            active_features={0, 1, 2, 3, 4, 5, 6, 7},
            interaction_groups=[{0, 1}, {2, 3}],
            meta={"source": "in-house", "n_features_total": d,
                  "sampler": "N(0, I_d)", "noise": "sigma=0.1"},
        )

    raise KeyError(f"Unknown in-house DGP name: {name!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

#: Default DGP order used by the benchmark runner.
DGP_NAMES: list[str] = [
    "Additive", "Friedman1", "Smooth",
    "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10",
]


def generate(name: str, seed: int, n: int | None = None, p: int | None = None) -> DGP:
    """Return a DGP bundle deterministically for ``(name, seed)``.

    Size defaults:
      - In-house (Additive/Friedman1/Smooth): n = 10000, d = 10
      - Diamond F1..F10:                      n = 10000, p = 30
    """
    if name == "Additive":
        return _make_inhouse(seed, name, n=n or 10000, d=p or 10)
    if name == "Friedman1":
        return _make_inhouse(seed, name, n=n or 10000, d=p or 10)
    if name == "Smooth":
        return _make_inhouse(seed, name, n=n or 10000, d=p or 10)
    if name in _DIAMOND_FUNCS:
        return _make_diamond_dgp(name, seed, n=n or 10000, p=p or 30)
    raise KeyError(f"Unknown DGP name: {name!r}. Known: {DGP_NAMES}")


def generate_all(seed: int, names: list[str] | None = None) -> list[DGP]:
    """Generate every DGP in ``names`` (default = all 13) for a single seed."""
    if names is None:
        names = DGP_NAMES
    return [generate(n, seed) for n in names]


if __name__ == "__main__":
    # Tiny sanity script: for each DGP at seed=0, print name / shape / #true
    for name in DGP_NAMES:
        dgp = generate(name, seed=0)
        finite = bool(np.isfinite(dgp.y).all())
        print(f"{name:10s}  X={dgp.X.shape}  "
              f"y∈[{dgp.y.min():+.3f},{dgp.y.max():+.3f}]  "
              f"#true_pairs={len(dgp.true_pairs):2d}  "
              f"active={len(dgp.active_features):2d}  finite={finite}")
