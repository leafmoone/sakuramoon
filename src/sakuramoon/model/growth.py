"""Block slot topology helpers.

Slot ids are opaque non-negative integers chosen by the config
(``model.dit.depth`` / ``model.dit.active_slot_ids``) or preserved from a
source checkpoint on resume.  Which slots are "new" (and therefore ramped
with a growth alpha) is the set difference between the target topology and
the source topology — it is never inferred from a depth table.
"""

from __future__ import annotations

import math

import torch

from sakuramoon.model.slots import (
    ACTIVE_SLOT_IDS,
    BASE_SLOT_IDS,
    G1_NEW_SLOT_IDS,
    G2_NEW_SLOT_IDS,
    NEW_SLOT_IDS,
    active_slot_ids,
    slot_name,
)


def new_slot_ids(
    target_slots: tuple[int, ...] | list[int],
    source_slots: tuple[int, ...] | list[int],
) -> tuple[int, ...]:
    """Slots added by a growth step: present in the target, absent in the source."""

    added = set(target_slots) - set(source_slots)
    return tuple(sorted(added))




def slot_growth(
    active_slots: tuple[int, ...] | list[int],
    new_slots: tuple[int, ...] | list[int],
    slot_id: int,
    growth_alpha: float,
) -> float:
    if type(growth_alpha) is not float or not 0.0 <= growth_alpha <= 1.0:
        raise ValueError("growth_alpha must be a float in [0,1]")
    if slot_id not in active_slots:
        raise ValueError("slot is not active in the selected topology")
    return growth_alpha if slot_id in new_slots else 1.0


def packed_growth_alpha(
    active_slots: tuple[int, ...] | list[int],
    growth_alpha: float,
    reference: torch.Tensor,
) -> torch.Tensor:
    """Materialize dynamic growth as a device scalar for regional compilation."""

    if type(growth_alpha) is not float or not 0.0 <= growth_alpha <= 1.0:
        raise ValueError("growth_alpha must be a float in [0,1]")
    if not active_slots:
        raise ValueError("packed growth requires an active slot topology")
    if not isinstance(reference, torch.Tensor) or not reference.is_floating_point():
        raise TypeError("packed growth requires a floating-point reference tensor")
    return reference.new_tensor(growth_alpha)


def half_cosine_growth_alpha(elapsed_updates: int, ramp_updates: int) -> float:
    """Half-cosine ramp over ``ramp_updates`` (any positive length)."""

    if type(elapsed_updates) is not int or elapsed_updates < 0:
        raise ValueError("elapsed updates must be a nonnegative integer")
    if type(ramp_updates) is not int or ramp_updates <= 0:
        raise ValueError("ramp updates must be a positive integer")
    progress = min(elapsed_updates, ramp_updates) / ramp_updates
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def new_slot_fqn_prefixes(new_slots: tuple[int, ...] | list[int]) -> tuple[str, ...]:
    return tuple(
        prefix
        for slot_id in new_slots
        for prefix in (
            f"dit.blocks.{slot_name(slot_id)}.",
            f"dit.conditioner.block_biases.{slot_name(slot_id)}",
        )
    )


def is_new_slot_fqn(new_slots: tuple[int, ...] | list[int], name: str) -> bool:
    return any(
        name.startswith(allowed) if allowed.endswith(".") else name == allowed
        for allowed in new_slot_fqn_prefixes(new_slots)
    )


__all__ = [
    "ACTIVE_SLOT_IDS",
    "BASE_SLOT_IDS",
    "G1_NEW_SLOT_IDS",
    "G2_NEW_SLOT_IDS",
    "NEW_SLOT_IDS",
    "active_slot_ids",
    "half_cosine_growth_alpha",
    "is_new_slot_fqn",
    "new_slot_fqn_prefixes",
    "new_slot_ids",
    "packed_growth_alpha",
    "slot_growth",
    "slot_name",
]
