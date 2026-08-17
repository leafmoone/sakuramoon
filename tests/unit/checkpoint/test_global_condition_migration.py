from __future__ import annotations

from typing import cast

import pytest
import torch

import sakuramoon.checkpoint.migrate_global_condition as migration
from sakuramoon.checkpoint.schema import CheckpointError


def _source_architecture() -> dict[str, object]:
    return {
        "schema_version": 2,
        "class": "TrainableComposite",
        "dit": {
            "attention_backend": "das_fa2_varlen",
            "hidden_size": 32,
            "condition_hidden_size": 16,
        },
        "text": {"input_size": 8},
        "condition_tokens": {"output_size": 32},
    }


def _source_optimizer() -> tuple[dict[str, object], dict[str, object]]:
    groups = [
        {
            "group_name": "matrix_decay",
            "param_names": ["dit.input_projection.weight"],
            "params": [7],
            "lr": 1e-4,
            "weight_decay": 0.01,
        },
        {
            "group_name": "sensitive_no_decay",
            "param_names": ["condition_tokens.queries"],
            "params": [3],
            "lr": 1e-4,
            "weight_decay": 0.0,
        },
    ]
    schema_groups = [
        {
            "group_name": cast(str, group["group_name"]),
            "param_names": cast(list[str], group["param_names"]),
        }
        for group in groups
    ]
    return (
        {
            "state": {
                7: {"step": torch.tensor(9)},
                3: {"step": torch.tensor(10)},
            },
            "param_groups": groups,
        },
        {"schema_version": 1, "groups": schema_groups},
    )


def test_architecture_migration_changes_only_the_architecture_schema() -> None:
    source = _source_architecture()

    migrated = migration.migrate_architecture(source)

    assert migrated == {**source, "schema_version": 3}
    assert source["schema_version"] == 2


def test_new_global_condition_tensors_have_locked_initialization() -> None:
    target = migration.migrate_architecture(_source_architecture())

    tensors = migration._new_model_tensors(target)  # pyright: ignore[reportPrivateUsage]

    assert set(tensors) == {
        "dit.conditioner.condition_global_norm.weight",
        "dit.conditioner.condition_global_projection.weight",
    }
    assert tensors["dit.conditioner.condition_global_norm.weight"].shape == (32,)
    assert tensors["dit.conditioner.condition_global_projection.weight"].shape == (
        16,
        32,
    )
    assert all(tensor.dtype == torch.float32 for tensor in tensors.values())
    assert torch.count_nonzero(
        tensors["dit.conditioner.condition_global_norm.weight"] - 1.0
    ) == 0
    assert torch.count_nonzero(
        tensors["dit.conditioner.condition_global_projection.weight"]
    ) == 0


def test_optimizer_migration_rebuilds_ids_and_preserves_old_moments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_state, source_schema = _source_optimizer()
    monkeypatch.setattr(
        migration,
        "_target_group_names",
        lambda _architecture: {
            "matrix_decay": ["dit.input_projection.weight"],
            "sensitive_no_decay": [
                "condition_tokens.queries",
                "dit.conditioner.condition_global_norm.weight",
                "dit.conditioner.condition_global_projection.weight",
            ],
        },
    )

    migrated, migrated_schema = migration.migrate_optimizer_state(
        source_state,
        source_schema,
        migration.migrate_architecture(_source_architecture()),
    )

    groups = cast(list[dict[str, object]], migrated["param_groups"])
    assert groups[0]["params"] == [0]
    assert groups[1]["params"] == [1, 2, 3]
    assert groups[1]["param_names"] == [
        "condition_tokens.queries",
        "dit.conditioner.condition_global_norm.weight",
        "dit.conditioner.condition_global_projection.weight",
    ]
    states = cast(dict[int, object], migrated["state"])
    assert set(states) == {0, 1}
    assert cast(dict[str, torch.Tensor], states[0])["step"].item() == 9
    assert cast(dict[str, torch.Tensor], states[1])["step"].item() == 10
    assert cast(list[dict[str, object]], migrated_schema["groups"])[1][
        "param_names"
    ] == groups[1]["param_names"]


def test_optimizer_migration_rejects_any_unexpected_parameter_delta(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_state, source_schema = _source_optimizer()
    monkeypatch.setattr(
        migration,
        "_target_group_names",
        lambda _architecture: {
            "matrix_decay": ["dit.input_projection.weight"],
            "sensitive_no_decay": [
                "condition_tokens.queries",
                "dit.conditioner.condition_global_norm.weight",
                "dit.conditioner.condition_global_projection.weight",
                "dit.unexpected.weight",
            ],
        },
    )

    with pytest.raises(CheckpointError, match="parameter delta"):
        migration.migrate_optimizer_state(
            source_state,
            source_schema,
            migration.migrate_architecture(_source_architecture()),
        )
