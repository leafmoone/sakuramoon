"""Exact named-state deletion for external EMA adapters (no EMA reconstruction)."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from sakuramoon.checkpoint.schema import CheckpointError

FINAL_NORM_FQN = "dit.output_head.norm.weight"


def remove_final_norm_named_state(
    saved: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Preserve tensor objects exactly, removing only the old final-norm weight.

    Suitable for a *named* EMA shadow mapping once an external EMA adapter has
    established that mapping's identity. This repository has no model EMA owner
    or serialization; this function neither guesses one nor resets EMA history.
    The caller preserves all EMA counters/decay metadata unchanged.
    """
    if set(saved) - set(target) != {FINAL_NORM_FQN} or set(target) - set(saved):
        raise CheckpointError("named state delta is not exactly the final norm weight")
    removed = saved[FINAL_NORM_FQN]
    projection = target.get("dit.output_head.projection.weight")
    if (
        projection is None
        or projection.ndim != 2
        or removed.shape != (projection.shape[1],)
        or removed.dtype != torch.float32
    ):
        raise CheckpointError("removed final norm weight shape or dtype differs")
    for name, expected in target.items():
        tensor = saved[name]
        if tensor.shape != expected.shape or tensor.dtype != expected.dtype:
            raise CheckpointError(f"named state shape or dtype differs: {name}")
    return {name: saved[name] for name in target}
