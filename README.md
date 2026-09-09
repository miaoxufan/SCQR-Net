# SCQR-Net

SCQR-Net (Spine Chain Query Refinement Network) is a dataset-agnostic PyTorch
template for structure-aware spine segmentation refinement.

The module wraps an arbitrary segmentation backbone and performs four steps:

1. Convert multi-class segmentation logits into voxel features.
2. Pool one case-specific query for every foreground structure.
3. Predict pairwise structural relations and exchange query-level messages.
4. Project the refined queries back to voxel space as a bounded residual update.

The residual scale is zero-initialized, so the wrapped model initially preserves
the backbone segmentation function. Training combines the original segmentation
loss with a relation-classification loss.

## Repository layout

- scqr/: reusable model, relation schema, and loss functions.
- training/: integration skeleton with every private setting left as None.
- evaluation/: callback-based evaluation interface without a fixed protocol.
- tests/: synthetic-tensor tests; no real samples or metadata.
- configs/: intentionally empty except for a zero-byte .gitkeep file.
- adaptive_query_chain_template.py: backward-compatible import entry point.

## Dataset-independent design

The code does not hard-code a dataset, label names, patch size, image spacing,
checkpoint, relation graph, or training plan. A caller privately supplies the
backbone, class count, verified relation schema, losses, and training settings.

## Minimal model example

    from scqr import AdaptiveQueryChain, RelationSchema, adaptive_loss

    schema = RelationSchema(
        num_foreground=NUM_FOREGROUND,
        positive_pairs=VERIFIED_PRIVATE_PAIRS,
    )

    model = AdaptiveQueryChain(
        base=segmentation_backbone,
        num_classes=NUM_CLASSES,
        schema=schema,
        query_dim=QUERY_DIM,
        correction_limit=CORRECTION_LIMIT,
    )

    output = model(images)
    losses = adaptive_loss(
        model=model,
        output=output,
        target=targets,
        segmentation_loss=segmentation_loss,
        relation_weight=RELATION_WEIGHT,
    )
    losses["loss"].backward()

All capitalized values above are private placeholders and must be supplied by
the caller.

## Public-release boundary

This repository intentionally omits dataset preparation, label definitions,
real relation schemas, data splits, paths, checkpoints, schedules, tuned
hyperparameters, experiment logs, results, and evaluation protocols.
