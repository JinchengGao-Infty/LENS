"""Target-aligned pair ranking."""

from __future__ import annotations

from typing import Literal, Mapping

from .interaction_vectors import InteractionSample

RankingScore = Literal["raw_energy", "debiased_energy"]


def score_interaction(sample: InteractionSample, score: RankingScore) -> float:
    if score == "raw_energy":
        return sample.raw_energy
    if score == "debiased_energy":
        return sample.debiased_energy
    raise ValueError(f"unknown ranking score: {score!r}")


def rank_interactions(
    samples: Mapping[tuple[int, int], InteractionSample],
    *,
    score: RankingScore = "raw_energy",
) -> list[tuple[tuple[int, int], float]]:
    """Sort by score descending with pair order as deterministic tie-breaker."""
    rows = [(pair, score_interaction(sample, score)) for pair, sample in samples.items()]
    return sorted(rows, key=lambda item: (-item[1], item[0]))
