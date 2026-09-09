"""Backward-compatible import path for the public SCQR-Net method core.

Dataset definitions, relation pairs, paths, training settings, and evaluation
protocols are intentionally not included.
"""

from scqr import (
    AdaptiveQueryChain,
    RelationSchema,
    adaptive_loss,
    masked_relation_bce,
)

__all__ = [
    "AdaptiveQueryChain",
    "RelationSchema",
    "adaptive_loss",
    "masked_relation_bce",
]
