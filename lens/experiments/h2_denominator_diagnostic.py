"""Per-pair diagnostic of why Friedman H2 ranks worse than LENS energy.

On the product grid the joint PD surface splits orthogonally into two main
effects and the interaction residual, so the H2 denominator equals
M + Delta, where Delta is the LENS raw energy and M is the main-effect
energy of the pair.  This script reruns the Stage-1 discovery of the formal
synthetic protocol (CPU XGBoost, so Teachers may differ slightly from the
CUDA formal run; the number of evaluation contexts is a parameter because
CPU prediction is slow) and stores Delta, H2 and M for every pair.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from experiments.dgps import generate
from lens.splits import make_three_way_split
from lens.synthetic_rankers import discover_synthetic_rankers
from lens.teachers import XGBoostTeacherFactory

DGPS = ("Friedman1", "Smooth", "F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10")


def run_cell(dgp: str, seed: int, output: Path, *, contexts: int, threads: int) -> None:
    destination = output / f"{dgp}_seed{seed}.json"
    if destination.exists():
        return
    started = time.perf_counter()
    generated = generate(dgp, seed=seed, n=None)
    split = make_three_way_split(len(generated.y), seed=seed)
    discovery = discover_synthetic_rankers(
        generated.X,
        generated.y,
        split=split,
        feature_kinds=("continuous",) * generated.X.shape[1],
        teacher_factory=XGBoostTeacherFactory("regression", params={"device": "cpu", "n_jobs": threads}),
        seed=seed,
        top_k=15,
        grid_points=12,
        grid_batch_size=144,
        ranking_max_samples=contexts,
        rankers=("LENS-raw", "Friedman-H2"),
    )
    energy = discovery.scores["LENS-raw"]
    h2 = discovery.scores["Friedman-H2"]
    truth = {tuple(pair) for pair in generated.true_pairs}
    active = set(generated.active_features)
    rows = []
    for pair in energy:
        delta = float(energy[pair])
        ratio = float(h2[pair])
        rows.append(
            {
                "pair": list(pair),
                "true": pair in truth,
                "n_active": int(pair[0] in active) + int(pair[1] in active),
                "delta": delta,
                "h2": ratio,
                "main": delta / ratio - delta if ratio > 0 else None,
            }
        )
    payload = {
        "dgp": dgp,
        "seed": seed,
        "p": int(generated.X.shape[1]),
        "contexts": contexts,
        "lens_ranking": [list(pair) for pair in discovery.rankings["LENS-raw"]],
        "h2_ranking": [list(pair) for pair in discovery.rankings["Friedman-H2"]],
        "pairs": rows,
        "elapsed_seconds": time.perf_counter() - started,
    }
    tmp = destination.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(destination)
    print(dgp, seed, f"{payload['elapsed_seconds']:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, default=8)
    parser.add_argument("--contexts", type=int, default=3000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--n-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    cells = [(dgp, seed) for dgp in DGPS for seed in range(args.seeds)]
    for index, (dgp, seed) in enumerate(cells):
        if index % args.n_shards == args.shard_index:
            run_cell(dgp, seed, output, contexts=args.contexts, threads=args.threads)


if __name__ == "__main__":
    main()
