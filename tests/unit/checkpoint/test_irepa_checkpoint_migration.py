from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

import sakuramoon.checkpoint.migrate_irepa_checkpoint as migration
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
)


def _source_architecture() -> dict[str, object]:
    return {
        "schema_version": 3,
        "class": "TrainableComposite",
        "dit": {
            "hidden_size": 32,
        },
        "text": {"input_size": 8},
        "condition_tokens": {"output_size": 32},
    }


def test_architecture_migration_is_v3_to_v4_and_only_adds_the_auxiliary() -> None:
    source = _source_architecture()

    migrated = migration._migrate_architecture_v4(source)  # pyright: ignore[reportPrivateUsage]

    assert migrated["schema_version"] == 4
    assert migrated["class"] == "TrainableComposite"
    assert source["schema_version"] == 3
    auxiliaries = cast(dict[str, object], migrated["training_auxiliaries"])
    assert set(auxiliaries) == {"irepa"}
    irepa = cast(dict[str, object], auxiliaries["irepa"])
    assert irepa["in_channels"] == 32
    assert irepa["out_channels"] == 768
    assert irepa["weight_dtype"] == "bfloat16"
    assert irepa["bias_dtype"] == "float32"


def test_architecture_migration_rejects_non_v3_source() -> None:
    v2 = {**_source_architecture(), "schema_version": 2}
    with pytest.raises(CheckpointError, match="schema v3"):
        migration._migrate_architecture_v4(v2)  # pyright: ignore[reportPrivateUsage]
    already = _source_architecture()
    already["training_auxiliaries"] = {}
    with pytest.raises(CheckpointError, match="schema v3"):
        migration._migrate_architecture_v4(already)  # pyright: ignore[reportPrivateUsage]


def test_deterministic_projector_is_seeded_and_rng_isolated() -> None:
    baseline = torch.get_rng_state()

    first = migration._deterministic_projector(32, 1234)  # pyright: ignore[reportPrivateUsage]
    second = migration._deterministic_projector(32, 1234)  # pyright: ignore[reportPrivateUsage]
    other = migration._deterministic_projector(32, 9999)  # pyright: ignore[reportPrivateUsage]

    assert torch.equal(first.projector.weight, second.projector.weight)
    assert torch.equal(first.projector.bias, second.projector.bias)
    assert not torch.equal(first.projector.weight, other.projector.weight)
    assert torch.equal(torch.get_rng_state(), baseline)


def test_deterministic_projector_rejects_negative_seed() -> None:
    with pytest.raises(ValueError, match="nonnegative"):
        migration._deterministic_projector(32, -1)  # pyright: ignore[reportPrivateUsage]


def test_projector_tensors_have_locked_fqns_and_dtypes() -> None:
    alignment = migration._deterministic_projector(32, 7)  # pyright: ignore[reportPrivateUsage]

    tensors = migration._projector_tensors(alignment)  # pyright: ignore[reportPrivateUsage]

    assert set(tensors) == {
        "irepa_alignment.projector.weight",
        "irepa_alignment.projector.bias",
    }
    weight = tensors["irepa_alignment.projector.weight"]
    bias = tensors["irepa_alignment.projector.bias"]
    assert tuple(weight.shape) == (768, 32, 3, 3)
    assert weight.dtype is torch.bfloat16
    assert tuple(bias.shape) == (768,)
    assert bias.dtype is torch.float32


def test_irepa_state_document_persists_the_zero_lambda_anchor() -> None:
    manifest = CheckpointManifest(
        kind=CheckpointKind.RAW,
        identity=CheckpointIdentity("raw-100-abc", 100),
        files=(FileRecord("resolved_config.toml", 1),),
    )

    document = migration._irepa_state_document(  # pyright: ignore[reportPrivateUsage]
        source=manifest,
        successful_updates=100,
        migration_seed=42,
    )

    assert document == {
        "schema_version": 1,
        "start_successful_update": 101,
        "source_checkpoint_id": "raw-100-abc",
        "source_update": 100,
        "migration_seed": 42,
    }


def test_source_no_irepa_rejects_an_existing_projector(tmp_path: Path) -> None:
    source = tmp_path / "ckpt_10_raw"
    model = source / "model"
    (model).mkdir(parents=True)
    (source / "train_state").mkdir()
    index = {
        "metadata": {"total_size": 10},
        "weight_map": {"irepa_alignment.projector.weight": "x.safetensors"},
    }
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "config.json").write_text(
        json.dumps({"architecture": {"schema_version": 3, "class": "TrainableComposite"}})
    )

    with pytest.raises(CheckpointError, match="already contains iREPA"):
        migration._validate_source_no_irepa(source)  # pyright: ignore[reportPrivateUsage]


def test_source_no_irepa_rejects_an_existing_state_sidecar(tmp_path: Path) -> None:
    source = tmp_path / "ckpt_10_raw"
    model = source / "model"
    model.mkdir(parents=True)
    (source / "train_state").mkdir()
    index = {"metadata": {"total_size": 10}, "weight_map": {"dit.a.weight": "x.safetensors"}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "config.json").write_text(
        json.dumps({"architecture": {"schema_version": 3, "class": "TrainableComposite"}})
    )
    (source / "train_state" / "irepa_state.json").write_text("{}")

    with pytest.raises(CheckpointError, match="iREPA state sidecar"):
        migration._validate_source_no_irepa(source)  # pyright: ignore[reportPrivateUsage]


def test_source_no_irepa_rejects_an_already_v4_architecture(tmp_path: Path) -> None:
    source = tmp_path / "ckpt_10_raw"
    model = source / "model"
    model.mkdir(parents=True)
    (source / "train_state").mkdir()
    index = {"metadata": {"total_size": 10}, "weight_map": {"dit.a.weight": "x.safetensors"}}
    (model / "model.safetensors.index.json").write_text(json.dumps(index))
    (model / "config.json").write_text(
        json.dumps(
            {
                "architecture": {
                    "schema_version": 4,
                    "class": "TrainableComposite",
                    "training_auxiliaries": {"irepa": {}},
                }
            }
        )
    )

    with pytest.raises(CheckpointError, match="no-iREPA v3"):
        migration._validate_source_no_irepa(source)  # pyright: ignore[reportPrivateUsage]


def _fake_hybrid(
    decay_fqns: list[str], sensitive_fqns: list[str]
) -> tuple[dict[str, object], dict[str, object]]:
    next_id = 0
    groups = []
    schema_groups = []
    state: dict[int, object] = {}
    for group_name, names in (
        ("matrix_decay", decay_fqns),
        ("sensitive_no_decay", sensitive_fqns),
    ):
        ids = list(range(next_id, next_id + len(names)))
        next_id += len(ids)
        groups.append(
            {
                "group_name": group_name,
                "param_names": names,
                "params": ids,
                "lr": 1e-4,
                "weight_decay": 0.01 if group_name == "matrix_decay" else 0.0,
            }
        )
        schema_groups.append({"group_name": group_name, "param_names": names})
        for name, param_id in zip(names, ids, strict=True):
            state[param_id] = {"step": torch.tensor(hash(name) % 7)}
    return (
        {
            "cmuon": {"momenta": {}},
            "routing": {
                "cmuon": [],
                "adamw": [
                    {"name": n, "group": "matrix_decay", "weight_decay": 0.01}
                    for n in decay_fqns
                ]
                + [
                    {"name": n, "group": "sensitive_no_decay", "weight_decay": 0.0}
                    for n in sensitive_fqns
                ],
                "counts": {"total": next_id, "cmuon": 0, "adamw": next_id},
            },
            "optimizer": {"state": state, "param_groups": groups},
            "sr_rng": {"device_type": "cuda", "device_index": 0, "state": torch.zeros(8, dtype=torch.uint8)},
            "transition": None,
            "hybrid_cmuon_schema_version": 1,
        },
        {"groups": schema_groups, "schema_version": 1},
    )


class _FakeRouting:
    """A routing manifest where the CMuon set is empty and the AdamW set is
    the pre-existing FQNs plus the iREPA projector (which sorts last)."""

    def __init__(self, decay_fqns: list[str], sensitive_fqns: list[str]) -> None:
        self._decay = decay_fqns
        self._sensitive = sensitive_fqns

    def routing_manifest(self) -> dict[str, object]:
        decay = list(self._decay)
        sensitive = list(self._sensitive)
        return {
            "cmuon": [],
            "adamw": [
                {"name": n, "group": "matrix_decay", "weight_decay": 0.01}
                for n in decay
            ]
            + [
                {
                    "name": "irepa_alignment.projector.weight",
                    "group": "matrix_decay",
                    "weight_decay": 0.01,
                }
            ]
            + [
                {"name": n, "group": "sensitive_no_decay", "weight_decay": 0.0}
                for n in sensitive
            ]
            + [
                {
                    "name": "irepa_alignment.projector.bias",
                    "group": "sensitive_no_decay",
                    "weight_decay": 0.0,
                }
            ],
            "counts": {
                "total": len(decay) + len(sensitive) + 2,
                "cmuon": 0,
                "adamw": len(decay) + len(sensitive) + 2,
            },
        }


def test_irepa_optimizer_migration_appends_projector_and_preserves_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decay = ["dit.blocks.slot_01.attn.q_proj.weight", "dit.blocks.slot_02.attn.q_proj.weight"]
    sensitive = ["dit.blocks.slot_01.attn.q_proj.bias"]
    hybrid, schema = _fake_hybrid(decay, sensitive)

    checkpoint = tmp_path / "ckpt"
    train_state = checkpoint / "train_state"
    train_state.mkdir(parents=True)
    torch.save(hybrid, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(json.dumps(schema))

    # The projector canonical FQNs sort after every dit.* name, so each AdamW
    # group is unchanged except for the projector appended at its tail.
    monkeypatch.setattr(
        migration,
        "_target_adamw_group_names",
        lambda _architecture: {
            "matrix_decay": decay + ["irepa_alignment.projector.weight"],
            "sensitive_no_decay": sensitive + ["irepa_alignment.projector.bias"],
        },
    )

    class _Routing:
        def routing_manifest(self) -> dict[str, object]:
            return {
                "cmuon": [],
                "adamw": [
                    {"name": n, "group": "matrix_decay", "weight_decay": 0.01}
                    for n in decay
                ]
                + [
                    {"name": "irepa_alignment.projector.weight", "group": "matrix_decay", "weight_decay": 0.01}
                ]
                + [
                    {"name": n, "group": "sensitive_no_decay", "weight_decay": 0.0}
                    for n in sensitive
                ]
                + [
                    {"name": "irepa_alignment.projector.bias", "group": "sensitive_no_decay", "weight_decay": 0.0}
                ],
                "counts": {"total": len(decay) + len(sensitive) + 2, "cmuon": 0, "adamw": len(decay) + len(sensitive) + 2},
            }

    monkeypatch.setattr(
        migration,
        "build_trainable_composite",
        lambda _architecture, device: object(),
    )
    monkeypatch.setattr(
        migration,
        "route_cmuon_parameters",
        lambda _module, *, matrix_weight_decay, sensitive_weight_decay: _Routing(),
    )

    migration._migrate_irepa_optimizer(  # pyright: ignore[reportPrivateUsage]
        checkpoint,
        architecture={},
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )

    migrated = cast(dict[str, object], torch.load(train_state / "optimizer.pt", map_location="cpu"))
    inner = cast(dict[str, object], migrated["optimizer"])
    groups = cast(list[dict[str, object]], inner["param_groups"])
    assert [g["group_name"] for g in groups] == ["matrix_decay", "sensitive_no_decay"]
    assert groups[0]["params"] == [0, 1, 2]
    assert groups[0]["param_names"] == decay + ["irepa_alignment.projector.weight"]
    assert groups[1]["params"] == [3, 4]
    assert groups[1]["param_names"] == sensitive + ["irepa_alignment.projector.bias"]
    state = cast(dict[int, object], inner["state"])
    # Pre-existing parameters keep their exact numeric ids and state; the
    # projector parameters (ids 2 and 4) are lazy (no state entry).
    assert set(state) == {0, 1, 3}
    assert cast(dict[str, torch.Tensor], state[0])["step"].item() == hash(decay[0]) % 7
    routing = cast(dict[str, object], migrated["routing"])
    adamw = cast(list[object], routing["adamw"])
    assert len(adamw) == 5
    counts = cast(dict[str, object], routing["counts"])
    assert counts["adamw"] == 5

    schema_after = cast(dict[str, object], json.loads((train_state / "optimizer_schema.json").read_text()))
    groups_after = cast(list[dict[str, object]], schema_after["groups"])
    assert groups_after[0]["param_names"] == groups[0]["param_names"]
    assert groups_after[1]["param_names"] == groups[1]["param_names"]


def test_irepa_optimizer_migration_preserves_v2_schema_blocks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The production save path writes a schema-v2 document for hybrid
    # optimizers (groups + hybrid_cmuon [+ guarded_canonical]).  The
    # migration must consume `groups` and preserve the algorithm blocks
    # verbatim; rewriting a v2 document as v1 would make the migrated
    # checkpoint unloadable.
    decay = ["dit.blocks.slot_01.attn.q_proj.weight"]
    sensitive = ["dit.blocks.slot_01.attn.q_proj.bias"]
    hybrid, schema = _fake_hybrid(decay, sensitive)
    hybrid_cmuon_block = {
        "momentum": 0.95,
        "nesterov": True,
        "eps": 1e-8,
        "momentum_dtype": "bfloat16",
        "chunk_rescale_sqrt_n": False,
        "qkv_group_rescale": False,
        "ns_steps": {"matrix": 4},
    }
    guarded_block = {
        "schema_version": 1,
        "config": {"guard_ratio": 0.1, "reference_decay": 0.9},
        "owner_mapping_version": 1,
        "world_size": 1,
        "ns_mode": "canonical_owner_rank",
        "ns_steps": {"matrix": 4},
    }
    v2_schema = {
        "schema_version": 2,
        "groups": cast(list[object], schema["groups"]),
        "hybrid_cmuon": hybrid_cmuon_block,
        "guarded_canonical": guarded_block,
    }

    checkpoint = tmp_path / "ckpt"
    train_state = checkpoint / "train_state"
    train_state.mkdir(parents=True)
    torch.save(hybrid, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(json.dumps(v2_schema))

    monkeypatch.setattr(
        migration,
        "_target_adamw_group_names",
        lambda _architecture: {
            "matrix_decay": decay + ["irepa_alignment.projector.weight"],
            "sensitive_no_decay": sensitive + ["irepa_alignment.projector.bias"],
        },
    )
    monkeypatch.setattr(
        migration,
        "build_trainable_composite",
        lambda _architecture, device: object(),
    )
    monkeypatch.setattr(
        migration,
        "route_cmuon_parameters",
        lambda _module, *, matrix_weight_decay, sensitive_weight_decay: _FakeRouting(
            decay, sensitive
        ),
    )

    migration._migrate_irepa_optimizer(  # pyright: ignore[reportPrivateUsage]
        checkpoint,
        architecture={},
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )

    schema_after = cast(
        dict[str, object],
        json.loads((train_state / "optimizer_schema.json").read_text()),
    )
    assert schema_after["schema_version"] == 2
    assert cast(dict[str, object], schema_after["hybrid_cmuon"]) == hybrid_cmuon_block
    assert cast(dict[str, object], schema_after["guarded_canonical"]) == guarded_block
    groups_after = cast(list[dict[str, object]], schema_after["groups"])
    assert groups_after[0]["param_names"] == decay + ["irepa_alignment.projector.weight"]
    assert (
        groups_after[1]["param_names"]
        == sensitive + ["irepa_alignment.projector.bias"]
    )


def test_irepa_optimizer_migration_rejects_an_unknown_schema_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decay = ["dit.blocks.slot_01.attn.q_proj.weight"]
    sensitive = ["dit.blocks.slot_01.attn.q_proj.bias"]
    hybrid, _schema = _fake_hybrid(decay, sensitive)
    checkpoint = tmp_path / "ckpt"
    train_state = checkpoint / "train_state"
    train_state.mkdir(parents=True)
    torch.save(hybrid, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(
        json.dumps({"groups": [], "schema_version": 3})
    )
    with pytest.raises(CheckpointError, match="optimizer schema is invalid"):
        migration._migrate_irepa_optimizer(  # pyright: ignore[reportPrivateUsage]
            checkpoint,
            architecture={},
            matrix_weight_decay=0.01,
            sensitive_weight_decay=0.0,
        )


def test_irepa_optimizer_migration_rejects_missing_hybrid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "ckpt"
    train_state = checkpoint / "train_state"
    train_state.mkdir(parents=True)
    torch.save({"state": {}, "param_groups": []}, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(
        json.dumps({"groups": [], "schema_version": 1})
    )
    with pytest.raises(CheckpointError, match="hybrid CMuon"):
        migration._migrate_irepa_optimizer(  # pyright: ignore[reportPrivateUsage]
            checkpoint,
            architecture={},
            matrix_weight_decay=0.01,
            sensitive_weight_decay=0.0,
        )
