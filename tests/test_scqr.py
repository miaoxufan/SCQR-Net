"""Shape and zero-initialization tests using synthetic tensors only."""

import torch
from torch import nn

from scqr import AdaptiveQueryChain, RelationSchema


def test_zero_initialized_refinement_preserves_backbone_output() -> None:
    backbone = nn.Conv3d(1, 4, kernel_size=1)
    schema = RelationSchema(3, frozenset({(0, 1), (1, 2)}))
    model = AdaptiveQueryChain(backbone, 4, schema, query_dim=8, correction_limit=1.0)
    inputs = torch.randn(2, 1, 4, 4, 4)

    with torch.no_grad():
        expected = backbone(inputs).float()
        actual = model(inputs)

    torch.testing.assert_close(actual, expected)
