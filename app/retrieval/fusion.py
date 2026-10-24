"""Reciprocal Rank Fusion (Cormack et al., 2009).

RRF combines rankings using only ranks, so BM25 scores (unbounded) and cosine
scores (bounded) never need to be put on the same scale:

    rrf(d) = sum over rankings r of 1 / (k + rank_r(d))
"""

from __future__ import annotations

from collections.abc import Sequence


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """Fuse ranked lists of IDs. Duplicates within one list count only at their best rank."""
    scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    order = 0
    for ranking in rankings:
        seen: set[str] = set()
        for rank, item in enumerate(ranking, start=1):
            if item in seen:
                continue
            seen.add(item)
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
            if item not in first_seen:
                first_seen[item] = order
                order += 1
    # Ties are broken by first appearance to keep results deterministic.
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
