"""Stable block slot names for the approved 16 -> 20 -> 24 topology."""

from __future__ import annotations

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


__all__ = [
    "ACTIVE_SLOT_IDS",
    "BASE_SLOT_IDS",
    "G1_NEW_SLOT_IDS",
    "G2_NEW_SLOT_IDS",
    "NEW_SLOT_IDS",
    "active_slot_ids",
    "new_slot_ids",
    "slot_growth",
    "slot_name",
]
