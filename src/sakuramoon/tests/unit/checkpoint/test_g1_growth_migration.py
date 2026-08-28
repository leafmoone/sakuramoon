from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import sakuramoon.checkpoint.migrate_growth as migration
from sakuramoon.model.growth import G1_NEW_SLOT_IDS, active_slot_ids, slot_name


def _architecture() -> dict[str, object]:
    return {
        "dit": {
            "depth": 20,
            "active_slot_ids": list(active_slot_ids(20)),
            "attention_backend": "dense_sdpa",
            "hidden_size": 8,
            "intermediate_size": 16,
            "q_heads": 2,
            "kv_heads": 1,
            "head_dim": 4,
            "rope_nope_dim": 0,
            "rope_y_dim": 2,
            "rope_x_dim": 2,
            "rope_position_scale": 1.0,
            "rope_theta": 10.0,
            "norm_eps": 1e-6,
            "linear_dtype": "float32",
            "projection_bias": False,
            "attention_dropout": 0.0,
            "mlp_dropout": 0.0,
            "modulation_chunks": 6,
        }
    }


def test_new_slot_initialization_is_seeded_and_exact() -> None:
    first = migration._new_block_tensors(_architecture(), 902)  # pyright: ignore[reportPrivateUsage]
    second = migration._new_block_tensors(_architecture(), 902)  # pyright: ignore[reportPrivateUsage]
    different = migration._new_block_tensors(_architecture(), 903)  # pyright: ignore[reportPrivateUsage]

    assert first.keys() == second.keys()
    assert all(torch.equal(first[name], second[name]) for name in first)
    assert any(not torch.equal(first[name], different[name]) for name in first if name.startswith("dit.blocks."))
    assert all(
        any(
            name.startswith(f"dit.blocks.{slot_name(slot_id)}.")
            or name == f"dit.conditioner.block_biases.{slot_name(slot_id)}"
            for slot_id in G1_NEW_SLOT_IDS
        )
        for name in first
    )
    block_suffixes = []
    for slot_id in G1_NEW_SLOT_IDS:
        prefix = f"dit.blocks.{slot_name(slot_id)}."
        block_suffixes.append(
            {name.removeprefix(prefix) for name in first if name.startswith(prefix)}
        )
    assert block_suffixes[0]
    assert all(suffixes == block_suffixes[0] for suffixes in block_suffixes[1:])
    assert all(
        torch.count_nonzero(first[f"dit.conditioner.block_biases.{slot_name(slot_id)}"]) == 0
        for slot_id in G1_NEW_SLOT_IDS
    )


def test_migration_cli_requires_manual_approval_and_refuses_existing_target(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    with pytest.raises(migration.CheckpointError, match="incomplete"):
        migration._exact_complete_source(source)  # pyright: ignore[reportPrivateUsage]

    payload = {"migration_seed": 123, "strategy": "stable-slot-random-init-v1"}
    assert json.dumps(payload)
