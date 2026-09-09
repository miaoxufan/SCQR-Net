"""Dataset-independent SCQR-Net model core."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .schema import RelationSchema


class AdaptiveQueryChain(nn.Module):
    """Wrap a segmentation backbone with structural-query refinement."""

    def __init__(
        self,
        base: nn.Module,
        num_classes: int,
        schema: RelationSchema,
        query_dim: int,
        correction_limit: float,
    ) -> None:
        super().__init__()
        if num_classes != schema.num_foreground + 1:
            raise ValueError("num_classes must equal background plus schema nodes")
        self.base = base
        self.num_classes = num_classes
        self.num_foreground = schema.num_foreground
        self.schema = schema
        self.pairs = schema.candidate_pairs
        self.correction_limit = correction_limit

        self.pixel_proj = nn.Conv3d(num_classes, query_dim, kernel_size=1)
        self.edge_head = nn.Sequential(
            nn.Linear(2 * query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, 1),
        )
        self.relation_update = nn.Sequential(
            nn.Linear(2 * query_dim, query_dim),
            nn.GELU(),
            nn.Linear(query_dim, query_dim),
        )
        self.correction_scale = nn.Parameter(torch.zeros(self.num_foreground))
        self.last_relation_logits: Optional[Tensor] = None

    def _queries_from_logits(self, logits: Tensor) -> tuple[Tensor, Tensor]:
        logits32 = logits.float()
        probabilities = logits32.softmax(dim=1)[:, 1:]
        pixels = self.pixel_proj(logits32).flatten(2)
        weights = probabilities.flatten(2)
        queries = torch.einsum("bcn,bdn->bcd", weights, pixels)
        queries = queries / weights.sum(-1, keepdim=True).clamp_min(1.0)
        return queries, pixels

    def refine(self, logits: Tensor) -> Tensor:
        queries, pixels = self._queries_from_logits(logits)
        refined = queries.clone()
        relation_logits = []

        for left, right in self.pairs:
            pair = torch.cat((queries[:, left], queries[:, right]), dim=1)
            edge_logit = self.edge_head(pair).squeeze(1)
            relation_logits.append(edge_logit)
            update = edge_logit.sigmoid().unsqueeze(1) * self.relation_update(pair)
            refined[:, left] += update
            refined[:, right] += update

        self.last_relation_logits = torch.stack(relation_logits, dim=1)
        correction = torch.einsum(
            "bcd,bdn->bcn",
            F.normalize(refined, dim=2),
            F.normalize(pixels, dim=1),
        )
        correction = correction.view(
            logits.shape[0], self.num_foreground, *logits.shape[2:]
        )
        scale = self.correction_limit * self.correction_scale.tanh()
        correction = correction * scale.view(1, self.num_foreground, 1, 1, 1)
        background = torch.zeros_like(logits[:, :1].float())
        return logits.float() + torch.cat((background, correction), dim=1)

    def forward(self, inputs: Tensor) -> Tensor | list[Tensor]:
        output = self.base(inputs)
        if isinstance(output, (tuple, list)):
            return [self.refine(output[0]), *output[1:]]
        return self.refine(output)

    def relation_targets(self, device: torch.device) -> Tensor:
        return torch.tensor(
            [float(pair in self.schema.positive_pairs) for pair in self.pairs],
            device=device,
        )
