"""Generic loss functions for SCQR-Net."""

from __future__ import annotations

from typing import Iterable, Optional

from torch import Tensor, nn
import torch
import torch.nn.functional as F

from .model import AdaptiveQueryChain


def masked_relation_bce(
    logits: Tensor,
    targets: Tensor,
    pairs: Iterable[tuple[int, int]],
    present_mask: Optional[Tensor] = None,
) -> Tensor:
    target = targets.view(1, -1).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if present_mask is None:
        return loss.mean()

    pair_mask = torch.stack(
        [present_mask[:, left] & present_mask[:, right] for left, right in pairs],
        dim=1,
    ).to(loss.dtype)
    return (loss * pair_mask).sum() / pair_mask.sum().clamp_min(1.0)


def adaptive_loss(
    model: AdaptiveQueryChain,
    output: Tensor | list[Tensor],
    target: Tensor,
    segmentation_loss: nn.Module,
    relation_weight: float,
    present_mask: Optional[Tensor] = None,
) -> dict[str, Tensor]:
    primary = output[0] if isinstance(output, (tuple, list)) else output
    segmentation = segmentation_loss(primary, target)
    if model.last_relation_logits is None:
        raise RuntimeError("forward must run before adaptive_loss")
    relation_target = model.relation_targets(model.last_relation_logits.device)
    relation = masked_relation_bce(
        model.last_relation_logits,
        relation_target,
        model.pairs,
        present_mask,
    )
    return {
        "loss": segmentation + relation_weight * relation,
        "segmentation_loss": segmentation.detach(),
        "relation_loss": relation.detach(),
    }
