"""Dataset-independent structural relation schema."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations


@dataclass(frozen=True)
class RelationSchema:
    """Foreground-node relations using zero-based foreground indices."""

    num_foreground: int
    positive_pairs: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        expected = set(range(self.num_foreground))
        for left, right in self.positive_pairs:
            if left == right or not {left, right} <= expected:
                raise ValueError(f"invalid relation pair {(left, right)}")

    @property
    def candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(combinations(range(self.num_foreground), 2))
