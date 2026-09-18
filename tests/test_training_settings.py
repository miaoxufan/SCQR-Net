"""Tests for complete SCQR-Net construction settings."""

from training.settings import TrainingSettings


def test_training_settings_are_complete_without_checkpoint_field() -> None:
    settings = TrainingSettings(
        data_root="data",
        output_root="outputs",
        dataset_name="example",
        label_schema={},
        split_definition={},
        epochs=1,
        batch_size=1,
        learning_rate=1e-2,
        weight_decay=3e-5,
        relation_weight=0.1,
        query_dim=32,
        correction_limit=0.1,
        seed=0,
    )

    settings.require_complete()
