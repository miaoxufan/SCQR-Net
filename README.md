# SCQR-Net

SCQR-Net (Spine Chain Query Refinement Network) is a dataset-agnostic PyTorch template for structure-aware spine segmentation refinement.

The module wraps an arbitrary segmentation backbone and performs four steps:

1. Convert multi-class segmentation logits into voxel features.
2. Pool one case-specific query for every foreground structure.
3. Predict pairwise structural relations and exchange query-level messages.
4. Project the refined queries back to voxel space as a bounded residual update.

The residual scale is zero-initialized, so the wrapped model initially preserves the backbone segmentation function. Training combines the original segmentation loss with a relation-classification loss.

## Files

- `adaptive_query_chain_template.py`: generic SCQR-Net module, relation schema, masked relation loss, and combined training loss.

## Dataset-independent design

The code does not hard-code a dataset, label names, patch size, image spacing, checkpoint, or nnU-Net plan. A caller supplies:

- a segmentation backbone;
- the number of output classes, including background;
- a verified foreground relation schema;
- a segmentation loss;
- an optional node-presence mask for truncated or partially observed anatomy.

For datasets with anonymous labels, structural relations must come from a verified external schema rather than inferred anatomical names.

## Minimal example

```python
from adaptive_query_chain_template import (
    AdaptiveQueryChain,
    RelationSchema,
    adaptive_loss,
)

schema = RelationSchema(
    num_foreground=5,
    positive_pairs=frozenset({(0, 3), (3, 1), (1, 4), (4, 2)}),
)

model = AdaptiveQueryChain(
    base=segmentation_backbone,
    num_classes=6,
    schema=schema,
    query_dim=32,
)

output = model(images)
losses = adaptive_loss(
    model=model,
    output=output,
    target=targets,
    segmentation_loss=segmentation_loss,
    relation_weight=0.1,
)
losses["loss"].backward()
```

The example relation indices are illustrative. Replace them with a validated schema for the dataset being trained.

## Scope

This repository contains the reusable method core. Dataset preparation, nnU-Net trainer integration, training schedules, evaluation protocols, and dataset-specific schemas should remain separate configuration layers.
