from __future__ import annotations

import copy
import json

import torch

import sakuramoon.checkpoint.migrate_growth as migration


def test_optimizer_migration_preserves_old_state_and_leaves_new_state_lazy(
    tmp_path, monkeypatch
) -> None:
    train_state = tmp_path / "train_state"
    train_state.mkdir()
    old_moment = torch.arange(4, dtype=torch.float32)
    optimizer = {
        "state": {
            11: {
                "step": torch.tensor(7.0),
                "exp_avg": old_moment.clone(),
                "exp_avg_sq": old_moment.square(),
            },
            12: {},
        },
        "param_groups": [
            {
                "group_name": "matrix_decay",
                "param_names": ["old.weight"],
                "params": [11],
                "lr": 1.0,
            },
            {
                "group_name": "sensitive_no_decay",
                "param_names": ["old.norm"],
                "params": [12],
                "lr": 1.0,
            },
        ],
    }
    torch.save(optimizer, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "groups": [
                    {"group_name": "matrix_decay", "param_names": ["old.weight"]},
                    {"group_name": "sensitive_no_decay", "param_names": ["old.norm"]},
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        migration,
        "_target_group_names",
        lambda _architecture: {
            "matrix_decay": ["old.weight", "dit.blocks.slot_02.mlp.fc1.weight"],
            "sensitive_no_decay": [
                "old.norm",
                "dit.conditioner.block_biases.slot_02",
            ],
        },
    )

    migration._migrate_optimizer(tmp_path, architecture={})  # pyright: ignore[reportPrivateUsage]

    migrated = torch.load(train_state / "optimizer.pt", map_location="cpu", weights_only=True)
    assert torch.equal(migrated["state"][0]["exp_avg"], old_moment)
    assert set(migrated["state"]) == {0, 2}
    groups = {group["group_name"]: group for group in migrated["param_groups"]}
    assert groups["matrix_decay"]["param_names"] == [
        "old.weight",
        "dit.blocks.slot_02.mlp.fc1.weight",
    ]
    assert groups["sensitive_no_decay"]["param_names"] == [
        "old.norm",
        "dit.conditioner.block_biases.slot_02",
    ]
    assert groups["matrix_decay"]["params"] == [0, 1]
    assert groups["sensitive_no_decay"]["params"] == [2, 3]
    assert set(migrated["state"]) == {0, 2}
