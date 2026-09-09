"""Evaluation callback without dataset paths or protocol-specific settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import Tensor


def evaluate_case(
    prediction: Tensor,
    target: Tensor,
    metric_fn: Callable[[Tensor, Tensor], dict[str, Any]],
) -> dict[str, Any]:
    """Delegate scoring to a caller-provided private evaluation protocol."""
    return metric_fn(prediction, target)
