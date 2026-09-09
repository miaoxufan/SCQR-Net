"""Dataset-agnostic Adaptive Query+chain template for paper preparation.

This is a conceptual, runnable PyTorch module. The caller supplies an nnU-Net-
like base network, a foreground relation schema, and the dataset-specific
segmentation loss. It intentionally does not name a dataset or assume L4/L5
semantics. For anonymous labels, provide a verified schema externally.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class RelationSchema:
    """Foreground node relations, using zero-based foreground indices."""

    num_foreground: int
    positive_pairs: frozenset[tuple[int, int]]

    def __post_init__(self) -> None:
        expected = set(range(self.num_foreground))
        for i, j in self.positive_pairs:
            if i == j or not {i, j} <= expected:
                raise ValueError(f"invalid relation pair {(i, j)}")

    @property
    def candidate_pairs(self) -> tuple[tuple[int, int], ...]:
        return tuple(combinations(range(self.num_foreground), 2))


class AdaptiveQueryChain(nn.Module):
    """Backbone plus probability-pooled structural Query+chain residual."""

    def __init__(self, base: nn.Module, num_classes: int,
                 schema: RelationSchema, query_dim: int = 32,
                 correction_limit: float = 0.1) -> None:
        super().__init__()
        if num_classes != schema.num_foreground + 1:
            raise ValueError("num_classes must equal background + schema nodes")
        self.base = base
        self.num_classes = num_classes
        self.num_foreground = schema.num_foreground
        self.schema = schema
        self.pairs = schema.candidate_pairs
        self.correction_limit = correction_limit

        # Converts class logits into voxel features used to retrieve Queries.
        self.pixel_proj = nn.Conv3d(num_classes, query_dim, kernel_size=1)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * query_dim, query_dim), nn.GELU(),
            nn.Linear(query_dim, 1),
        )
        self.relation_update = nn.Sequential(
            nn.Linear(2 * query_dim, query_dim), nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )

        # Zero initialization preserves the base segmentation function.
        self.correction_scale = nn.Parameter(torch.zeros(self.num_foreground))
        self.last_relation_logits: Optional[Tensor] = None

    def _queries_from_logits(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        # Keep global pooling in FP32: large 3-D volumes can overflow FP16.
        logits32 = logits.float()
        probs = logits32.softmax(dim=1)[:, 1:]
        pixels = self.pixel_proj(logits32).flatten(2)       # B,D,N
        weights = probs.flatten(2)                          # B,C,N
        queries = torch.einsum("bcn,bdn->bcd", weights, pixels)
        queries = queries / weights.sum(-1, keepdim=True).clamp_min(1.0)
        return queries, pixels

    def refine(self, logits: Tensor) -> Tensor:
        queries, pixels = self._queries_from_logits(logits)
        refined = queries.clone()
        relation_logits = []

        for i, j in self.pairs:
            pair = torch.cat((queries[:, i], queries[:, j]), dim=1)
            edge_logit = self.edge_head(pair).squeeze(1)
            relation_logits.append(edge_logit)
            gate = edge_logit.sigmoid().unsqueeze(1)
            delta = self.relation_update(pair)
            refined[:, i] += gate * delta
            refined[:, j] += gate * delta

        self.last_relation_logits = torch.stack(relation_logits, dim=1)
        correction = torch.einsum(
            "bcd,bdn->bcn", F.normalize(refined, dim=2),
            F.normalize(pixels, dim=1),
        )
        correction = correction.view(logits.shape[0], self.num_foreground,
                                     *logits.shape[2:])
        scale = self.correction_limit * self.correction_scale.tanh()
        correction = correction * scale.view(1, self.num_foreground, 1, 1, 1)
        background = torch.zeros_like(logits[:, :1].float())
        return (logits.float() + torch.cat((background, correction), dim=1))

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        output = self.base(x)
        if isinstance(output, (tuple, list)):
            return [self.refine(output[0]), *output[1:]]
        return self.refine(output)

    def relation_targets(self, device: torch.device,
                         present_mask: Optional[Tensor] = None) -> Tensor:
        """Create BCE targets; optionally mask unavailable node pairs.

        present_mask is B x C and is required for truncated/anonymous datasets
        when a node is not observable in a sampled patch. The caller should use
        the same mask to exclude those pairs from BCE reduction.
        """
        target = torch.tensor(
            [float((i, j) in self.schema.positive_pairs) for i, j in self.pairs],
            device=device,
        )
        return target


def masked_relation_bce(logits: Tensor, targets: Tensor,
                        pairs: Iterable[tuple[int, int]],
                        present_mask: Optional[Tensor] = None) -> Tensor:
    """Relation BCE with optional B x C node-presence masking."""
    target = targets.view(1, -1).expand_as(logits)
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    if present_mask is not None:
        pair_mask = torch.stack(
            [present_mask[:, i] & present_mask[:, j] for i, j in pairs], dim=1
        ).to(loss.dtype)
        loss = loss * pair_mask
        return loss.sum() / pair_mask.sum().clamp_min(1.0)
    return loss.mean()


def adaptive_loss(model: AdaptiveQueryChain, output: Tensor | list[Tensor],
                  target: Tensor, segmentation_loss: nn.Module,
                  relation_weight: float = 0.1,
                  present_mask: Optional[Tensor] = None) -> dict[str, Tensor]:
    """Combine the normal segmentation loss and structural relation loss."""
    primary = output[0] if isinstance(output, (tuple, list)) else output
    seg = segmentation_loss(primary, target)
    if model.last_relation_logits is None:
        raise RuntimeError("forward must run before adaptive_loss")
    relation_target = model.relation_targets(model.last_relation_logits.device,
                                              present_mask)
    rel = masked_relation_bce(model.last_relation_logits, relation_target,
                              model.pairs, present_mask)
    return {"loss": seg + relation_weight * rel,
            "segmentation_loss": seg.detach(),
            "relation_loss": rel.detach()}


__all__ = [
    "RelationSchema", "AdaptiveQueryChain", "masked_relation_bce",
    "adaptive_loss",
]
