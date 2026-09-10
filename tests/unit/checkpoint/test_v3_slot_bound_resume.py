"""Regression: v3 artifacts with the historical 24 slot pin resume on current code.

The slot-topology refactor (stable_slot_count becomes the derived exclusive
slot-id bound, max(active_slot_ids) + 1) must not break resume of pre-existing
v3 documents that record the configured 24 pin for the G1 lineage: the bound
is validation metadata only (no tensor is sized from it), so the parameter
contract comparison strips it exactly like ``new_slot_ids``.  The real slot
structure stays strict through ``active_slot_ids``.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import cast

from sakuramoon.checkpoint.artifact import architectures_share_parameter_contract

# The G1 depth-20 active slot set (20 ids inside 0..22).
G1_ACTIVE_SLOT_IDS = [0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22]


def _current_v3_document() -> dict[str, object]:
    return {
        "schema_version": 3,
        "class": "TrainableComposite",
        "dit": {
            "active_slot_ids": list(G1_ACTIVE_SLOT_IDS),
            # Derived exclusive bound for the G1 topology: max(22) + 1 = 23.
            "stable_slot_count": 23,
            "new_slot_ids": [],
            "attention_backend": "das_fa2_varlen",
            "hidden_size": 2560,
        },
        "text": {"adapter_size": 1024},
    }


def _v3_artifact_document() -> dict[str, object]:
    """A pre-existing v3 document: historical 24 pin, no new_slot_ids key."""
    document = _current_v3_document()
    raw_dit = document["dit"]
    assert isinstance(raw_dit, Mapping)
    dit = cast("dict[str, object]", raw_dit)
    dit["stable_slot_count"] = 24
    del dit["new_slot_ids"]
    return document


def test_v3_24_pin_shares_contract_with_derived_23_module() -> None:
    current = _current_v3_document()
    artifact = _v3_artifact_document()
    assert architectures_share_parameter_contract(current, artifact) is True


def test_v3_24_pin_shares_contract_with_allow_drop() -> None:
    current = _current_v3_document()
    artifact = _v3_artifact_document()
    assert (
        architectures_share_parameter_contract(
            current, artifact, allow_irepa_auxiliary_drop=True
        )
        is True
    )


def test_new_slot_ids_still_stripped() -> None:
    current = _current_v3_document()
    raw_dit = current["dit"]
    assert isinstance(raw_dit, Mapping)
    dit = cast("dict[str, object]", raw_dit)
    dit["new_slot_ids"] = [20]
    artifact = _v3_artifact_document()
    assert architectures_share_parameter_contract(current, artifact) is True


def test_active_slot_ids_remain_strict() -> None:
    current = _current_v3_document()
    artifact = _v3_artifact_document()
    broken = copy.deepcopy(artifact)
    raw_dit = broken["dit"]
    assert isinstance(raw_dit, Mapping)
    dit = cast("dict[str, object]", raw_dit)
    raw_slots = dit["active_slot_ids"]
    assert isinstance(raw_slots, list)
    assert raw_slots and all(type(slot) is int for slot in raw_slots)
    raw_slots.pop()  # drop the last active slot: real structure changed
    assert architectures_share_parameter_contract(current, broken) is False


def test_non_slot_fields_remain_strict() -> None:
    current = _current_v3_document()
    artifact = _v3_artifact_document()
    broken = copy.deepcopy(artifact)
    raw_dit = broken["dit"]
    assert isinstance(raw_dit, Mapping)
    dit = cast("dict[str, object]", raw_dit)
    dit["hidden_size"] = 9999
    assert architectures_share_parameter_contract(current, broken) is False
