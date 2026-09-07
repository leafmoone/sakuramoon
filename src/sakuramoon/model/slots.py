"""Pure slot-topology vocabulary (no torch dependency).

Historical depths (16/20/24) have established non-contiguous slot topologies
so existing checkpoints resume against the same FQNs; any other positive
depth uses the plain contiguous id range.  Slot names keep the checkpoint's
original numbering on resume.
"""

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
    """Resolve the slot ids for a depth (historical map, contiguous fallback)."""

    if type(depth) is not int or depth <= 0:
        raise ValueError("depth must be a positive integer")
    try:
        return ACTIVE_SLOT_IDS[depth]
    except KeyError:
        return tuple(range(depth))


def slot_name(slot_id: int) -> str:
    if type(slot_id) is not int or slot_id < 0:
        raise ValueError("slot id must be a nonnegative integer")
    return f"slot_{slot_id:02d}"


__all__ = [
    "ACTIVE_SLOT_IDS",
    "BASE_SLOT_IDS",
    "G1_NEW_SLOT_IDS",
    "G2_NEW_SLOT_IDS",
    "NEW_SLOT_IDS",
    "active_slot_ids",
    "slot_name",
]
