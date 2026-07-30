"""Stable block slot names for the approved 16 -> 20 -> 24 topology."""

from __future__ import annotations

import math

BASE_SLOT_IDS = (0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22)
G1_NEW_SLOT_IDS = (2, 8, 14, 20)
G2_NEW_SLOT_IDS = (5, 11, 17, 23)

ACTIVE_SLOT_IDS: dict[int, tuple[int, ...]] = {
    16: BASE_SLOT_IDS,
    20: tuple(sorted((*BASE_SLOT_IDS, *G1_NEW_SLOT_IDS))),
    24: tuple(range(24)),
}
NEW_SLOT_IDS: dict[int, tuple[int, ...]] = {
    16: (),
    20: G1_NEW_SLOT_IDS,
    24: G2_NEW_SLOT_IDS,
}


def active_slot_ids(depth: int) -> tuple[int, ...]:
    try:
        return ACTIVE_SLOT_IDS[depth]
    except KeyError as error:
        raise ValueError("depth must be one of 16, 20, or 24") from error


def new_slot_ids(depth: int) -> tuple[int, ...]:
    try:
        return NEW_SLOT_IDS[depth]
    except KeyError as error:
        raise ValueError("depth must be one of 16, 20, or 24") from error


def slot_name(slot_id: int) -> str:
    if slot_id < 0 or slot_id >= 24:
        raise ValueError("stable slot id must be in [0,23]")
    return f"slot_{slot_id:02d}"


def slot_growth(depth: int, slot_id: int, growth_alpha: float) -> float:
    if not 0.0 <= growth_alpha <= 1.0:
        raise ValueError("growth_alpha must be in [0,1]")
    if slot_id not in active_slot_ids(depth):
        raise ValueError("slot is not active at the selected depth")
    return growth_alpha if slot_id in new_slot_ids(depth) else 1.0


def growth_ramp_updates(planned_updates: int) -> int:
    if type(planned_updates) is not int or planned_updates <= 0:
        raise ValueError("planned updates must be a positive integer")
    return min(5000, max(1000, math.ceil(planned_updates * 0.02)))


def half_cosine_growth_alpha(elapsed_updates: int, ramp_updates: int) -> float:
    if type(elapsed_updates) is not int or elapsed_updates < 0:
        raise ValueError("elapsed updates must be a nonnegative integer")
    if type(ramp_updates) is not int or not 1000 <= ramp_updates <= 5000:
        raise ValueError("ramp updates must be in [1000,5000]")
    progress = min(elapsed_updates, ramp_updates) / ramp_updates
    return 0.5 - 0.5 * math.cos(math.pi * progress)


def new_slot_fqn_prefixes(depth: int) -> tuple[str, ...]:
    if depth == 16:
        raise ValueError("base depth has no growth slots")
    return tuple(
        prefix
        for slot_id in new_slot_ids(depth)
        for prefix in (
            f"dit.blocks.{slot_name(slot_id)}.",
            f"dit.conditioner.block_biases.{slot_name(slot_id)}",
        )
    )


def is_new_slot_fqn(depth: int, name: str) -> bool:
    return any(
        name.startswith(allowed) if allowed.endswith(".") else name == allowed
        for allowed in new_slot_fqn_prefixes(depth)
    )


__all__ = [
    "ACTIVE_SLOT_IDS",
    "BASE_SLOT_IDS",
    "G1_NEW_SLOT_IDS",
    "G2_NEW_SLOT_IDS",
    "NEW_SLOT_IDS",
    "active_slot_ids",
    "growth_ramp_updates",
    "half_cosine_growth_alpha",
    "is_new_slot_fqn",
    "new_slot_fqn_prefixes",
    "new_slot_ids",
    "slot_growth",
    "slot_name",
]
