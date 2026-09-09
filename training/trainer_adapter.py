"""Framework-neutral training integration skeleton."""

from __future__ import annotations

from torch import nn

from scqr import AdaptiveQueryChain, RelationSchema
from .settings import TrainingSettings


def build_model(
    backbone: nn.Module,
    num_classes: int,
    schema: RelationSchema,
    settings: TrainingSettings,
) -> AdaptiveQueryChain:
    """Build SCQR-Net only after private settings have been supplied."""
    settings.require_complete()
    return AdaptiveQueryChain(
        base=backbone,
        num_classes=num_classes,
        schema=schema,
        query_dim=settings.query_dim,
        correction_limit=settings.correction_limit,
    )


def train(*args, **kwargs):
    """Intentionally empty: connect this function to a private training stack."""
    raise NotImplementedError("Training loop and parameters are intentionally omitted")
