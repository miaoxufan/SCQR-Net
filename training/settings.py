"""Empty training settings template.

Populate these values privately. No dataset path, label mapping, schedule, or
experiment-specific hyperparameter is included in the public repository.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TrainingSettings:
    data_root: Optional[str] = None
    output_root: Optional[str] = None
    dataset_name: Optional[str] = None
    label_schema: Optional[dict[str, Any]] = None
    split_definition: Optional[dict[str, Any]] = None
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    relation_weight: Optional[float] = None
    query_dim: Optional[int] = None
    correction_limit: Optional[float] = None
    checkpoint_path: Optional[str] = None
    seed: Optional[int] = None

    def require_complete(self) -> None:
        missing = [name for name, value in vars(self).items() if value is None]
        if missing:
            raise ValueError("Private training settings are required: " + ", ".join(missing))
