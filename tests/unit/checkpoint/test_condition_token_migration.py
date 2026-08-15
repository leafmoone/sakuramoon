from __future__ import annotations

import pytest
import torch

from sakuramoon.checkpoint.migrate_condition_tokens import (
    migrate_architecture,
    migrate_model_tensors,
    migrate_optimizer_state,
)
from sakuramoon.checkpoint.schema import CheckpointError


def _legacy_architecture() -> dict[str, object]:
    return {
        "class": "TrainableComposite",
        "dit": {"attention_backend": "das_fa2_varlen"},
        "text": {"input_size": 2048},
        "style": {
            "query_count": 4,
            "hidden_size": 3,
            "output_size": 5,
            "init_std": 0.02,
        },
    }


def test_architecture_migration_is_exact_and_declares_eight_tokens() -> None:
    migrated = migrate_architecture(_legacy_architecture())

    assert migrated["schema_version"] == 2
    assert "style" not in migrated
    condition = migrated["condition_tokens"]
    assert isinstance(condition, dict)
    assert condition["token_count"] == 8
    assert "query_count" not in condition
    dit = migrated["dit"]
    assert isinstance(dit, dict)
    assert dit["condition_token_count"] == 8


def test_architecture_migration_rejects_every_non_four_legacy_count() -> None:
    legacy = _legacy_architecture()
    style = legacy["style"]
    assert isinstance(style, dict)
    style["query_count"] = 8

    with pytest.raises(CheckpointError, match="exactly four"):
        migrate_architecture(legacy)


def test_model_tensor_migration_preserves_old_rows_and_initializes_new_rows() -> None:
    queries = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    null_tokens = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    legacy = {
        "dit.modality.style": torch.arange(5, dtype=torch.float32),
        "style.queries": queries,
        "style.null_tokens": null_tokens,
        "style.style_mlp.norm.weight": torch.ones(3),
    }

    first = migrate_model_tensors(legacy, init_std=0.02)
    second = migrate_model_tensors(legacy, init_std=0.02)

    assert set(first) == {
        "dit.modality.condition",
        "condition_tokens.queries",
        "condition_tokens.null_tokens",
        "condition_tokens.condition_mlp.norm.weight",
    }
    assert first["condition_tokens.queries"].shape == (8, 3)
    assert first["condition_tokens.null_tokens"].shape == (8, 5)
    torch.testing.assert_close(first["condition_tokens.queries"][:4], queries)
    torch.testing.assert_close(first["condition_tokens.null_tokens"][:4], null_tokens)
    torch.testing.assert_close(
        first["condition_tokens.queries"],
        second["condition_tokens.queries"],
        atol=0,
        rtol=0,
    )
    assert not torch.equal(first["condition_tokens.queries"][4:], queries)


def test_optimizer_migration_renames_sorts_and_drops_only_expanded_moments() -> None:
    state = {
        "state": {
            0: {"value": torch.tensor(10)},
            1: {"value": torch.tensor(11)},
            2: {"value": torch.tensor(12)},
            3: {"value": torch.tensor(13)},
        },
        "param_groups": [
            {
                "group_name": "matrix_decay",
                "param_names": ["dit.input_projection.weight", "style.input_projection.weight"],
                "params": [0, 1],
                "lr": 1e-4,
            },
            {
                "group_name": "sensitive_no_decay",
                "param_names": ["style.queries", "style.null_tokens"],
                "params": [2, 3],
                "lr": 1e-4,
            },
        ],
    }
    schema = {
        "schema_version": 1,
        "groups": [
            {
                "group_name": "matrix_decay",
                "param_names": ["dit.input_projection.weight", "style.input_projection.weight"],
            },
            {
                "group_name": "sensitive_no_decay",
                "param_names": ["style.queries", "style.null_tokens"],
            },
        ],
    }

    migrated, migrated_schema = migrate_optimizer_state(state, schema)

    groups = migrated["param_groups"]
    assert isinstance(groups, list)
    matrix = groups[0]
    sensitive = groups[1]
    assert isinstance(matrix, dict) and isinstance(sensitive, dict)
    assert matrix["param_names"] == [
        "condition_tokens.input_projection.weight",
        "dit.input_projection.weight",
    ]
    assert sensitive["param_names"] == [
        "condition_tokens.null_tokens",
        "condition_tokens.queries",
    ]
    migrated_state = migrated["state"]
    assert isinstance(migrated_state, dict)
    assert set(migrated_state) == {0, 1}
    schema_groups = migrated_schema["groups"]
    assert isinstance(schema_groups, list)
    assert schema_groups[0]["param_names"] == matrix["param_names"]
    assert schema_groups[1]["param_names"] == sensitive["param_names"]
