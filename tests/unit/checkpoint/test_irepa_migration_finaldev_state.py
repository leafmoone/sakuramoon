"""iREPA Phase 6A — recursive exact-state audit on the final-dev schema.

The production optimizer is the guarded canonical hybrid
(``hybrid_cmuon_canonical_ns4_fp32_rescue``).  Its persistent state carries
more than the base hybrid: a full ``cmuon`` configuration block, a ``guard``
block (references, counters, bootstrap mode, owner mapping), a
``transition`` document, the SR RNG, and the schema-v2 algorithm blocks in
``optimizer_schema.json``.  The iREPA migration must preserve EVERY one of
these verbatim; the only additions are the two projector FQNs (AdamW-routed,
stateless until their first step) and the irepa sidecar.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest
import torch

from sakuramoon.checkpoint import migrate_irepa_checkpoint as migration

PROJECTOR_WEIGHT = "irepa_alignment.projector.weight"
PROJECTOR_BIAS = "irepa_alignment.projector.bias"


def _deep_equal(a: Any, b: Any) -> bool:
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and a.shape == b.shape
            and torch.equal(a, b)
        )
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_deep_equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_deep_equal(x, y) for x, y in zip(a, b))
    if isinstance(a, float) and isinstance(b, float):
        return a == b
    return type(a) is type(b) and a == b


def _finaldev_guarded_hybrid() -> tuple[
    dict[str, object], dict[str, object], list[str], list[str]
]:
    """A synthetic FINAL-DEV production-schema hybrid optimizer state.

    Key tree mirrors the guarded canonical state_dict() exactly:
    optimizer / sr_rng / cmuon / routing / transition /
    hybrid_cmuon_schema_version / guard / guarded_canonical_schema_version.
    """
    decay = [
        "dit.blocks.slot_01.mlp.up_proj.weight",
        "dit.blocks.slot_02.mlp.up_proj.weight",
    ]
    sensitive = [
        "dit.blocks.slot_01.attn.q_proj.bias",
        "dit.blocks.slot_02.attn.k_proj.bias",
        "text.conditioner.lm_head.weight",
    ]
    cmuon_fqns = [
        "dit.blocks.slot_01.attn.q_proj.weight",
        "dit.blocks.slot_02.attn.q_proj.weight",
    ]

    adamw_names = decay + sensitive
    state: dict[int, dict[str, torch.Tensor]] = {}
    for next_id, name in enumerate(adamw_names):
        seed = torch.Generator().manual_seed(abs(hash(name)) % (2**31))
        state[next_id] = {
            "step": torch.tensor(17.0),
            "exp_avg": torch.randn(4, 3, generator=seed, dtype=torch.float32),
            "exp_avg_sq": torch.randn(4, 3, generator=seed, dtype=torch.float32).abs(),
        }

    groups = [
        {"group_name": "matrix_decay", "param_names": list(decay),
         "params": list(range(len(decay))), "lr": 2e-4, "weight_decay": 0.01},
        {"group_name": "sensitive_no_decay", "param_names": list(sensitive),
         "params": list(range(len(decay), len(adamw_names))), "lr": 2e-4,
         "weight_decay": 0.0},
    ]
    schema_groups = [
        {"group_name": "matrix_decay", "param_names": list(decay)},
        {"group_name": "sensitive_no_decay", "param_names": list(sensitive)},
    ]

    ref_generator = torch.Generator().manual_seed(42)
    hybrid: dict[str, object] = {
        "optimizer": {"state": state, "param_groups": groups},
        "sr_rng": {
            "device_type": "cuda",
            "device_index": 0,
            "state": torch.randint(0, 256, (8,), generator=ref_generator, dtype=torch.uint8),
        },
        "cmuon": {
            "momenta": {
                name: torch.randn(5, 5, dtype=torch.bfloat16,
                                  generator=torch.Generator().manual_seed(i))
                for i, name in enumerate(cmuon_fqns)
            },
            "ns_steps": {"matrix": 4, "qkv": 5},
            "ns_coefficients": [2.3683882, -1.6573476, 0.6025598],
            "momentum": 0.95,
            "nesterov": True,
            "eps": 1e-8,
            "momentum_dtype": "bfloat16",
            "chunk_rescale_sqrt_n": False,
            "qkv_group_rescale": False,
        },
        "routing": {
            "cmuon": [
                {"name": name, "role": "matrix", "ns_steps": 4, "chunk_dim": 0,
                 "chunk_size": 5}
                for name in cmuon_fqns
            ],
            "adamw": [
                {"name": n, "group": "matrix_decay", "weight_decay": 0.01}
                for n in decay
            ]
            + [
                {"name": n, "group": "sensitive_no_decay", "weight_decay": 0.0}
                for n in sensitive
            ],
            "counts": {"total": len(cmuon_fqns) + len(adamw_names),
                       "cmuon": len(cmuon_fqns), "adamw": len(adamw_names)},
        },
        "transition": {
            "from_adamw8bit": False,
            "preserved_adamw_params": [],
            "dropped_cmuon_params": [],
            "note": "native hybrid state",
        },
        "hybrid_cmuon_schema_version": 1,
        "guard": {
            "schema_version": 1,
            "config": {
                "guard_ratio": 0.1,
                "reference_decay": 0.9,
                "min_reference": 1e-3,
                "numerical_floor": 1e-6,
                "warmup_observations": 50,
                "invariant_check": True,
            },
            "references": {
                f"{cmuon_fqns[0]}#chunk0": 0.1234567,
                f"{cmuon_fqns[1]}#chunk1": 0.9876543,
            },
            "skip_total": 3,
            "skip_by_role": {"matrix": 2, "q": 1},
            "skip_by_fqn": {cmuon_fqns[0]: 2},
            "observations": 200,
            "bootstrap_mode": "observation",
            "owner_mapping_version": 1,
            "world_size": 2,
            "canonical_ns_mode": True,
            "ns_map": {"matrix": 4, "qkv": 5},
        },
        "guarded_canonical_schema_version": 1,
    }

    schema_document = {
        "schema_version": 2,
        "groups": schema_groups,
        "hybrid_cmuon": {
            "momentum": 0.95,
            "nesterov": True,
            "eps": 1e-8,
            "momentum_dtype": "bfloat16",
            "chunk_rescale_sqrt_n": False,
            "qkv_group_rescale": False,
            "ns_steps": {"matrix": 4, "qkv": 5},
        },
        "guarded_canonical": {
            "schema_version": 1,
            "config": {
                "guard_ratio": 0.1,
                "reference_decay": 0.9,
                "min_reference": 1e-3,
                "numerical_floor": 1e-6,
                "warmup_observations": 50,
                "invariant_check": True,
            },
            "owner_mapping_version": 1,
            "world_size": 2,
            "ns_mode": "canonical_owner_rank",
            "ns_steps": {"matrix": 4, "qkv": 5},
        },
    }
    return hybrid, schema_document, decay, sensitive


class _TargetRouting:
    def __init__(self, cmuon_fqns: list[str], decay: list[str], sensitive: list[str]) -> None:
        self._manifest = {
            "cmuon": [
                {"name": n, "role": "matrix", "ns_steps": 4, "chunk_dim": 0,
                 "chunk_size": 5}
                for n in cmuon_fqns
            ],
            "adamw": [
                {"name": n, "group": "matrix_decay", "weight_decay": 0.01}
                for n in decay
            ]
            + [{"name": PROJECTOR_WEIGHT, "group": "matrix_decay", "weight_decay": 0.01}]
            + [
                {"name": n, "group": "sensitive_no_decay", "weight_decay": 0.0}
                for n in sensitive
            ]
            + [{"name": PROJECTOR_BIAS, "group": "sensitive_no_decay", "weight_decay": 0.0}],
            "counts": {
                "total": len(cmuon_fqns) + len(decay) + len(sensitive) + 2,
                "cmuon": len(cmuon_fqns),
                "adamw": len(decay) + len(sensitive) + 2,
            },
        }

    def routing_manifest(self) -> dict[str, object]:
        return self._manifest


def test_finaldev_guarded_state_is_recursive_exact_through_migration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    hybrid, schema_document, decay, sensitive = _finaldev_guarded_hybrid()
    cmuon_fqns = [
        "dit.blocks.slot_01.attn.q_proj.weight",
        "dit.blocks.slot_02.attn.q_proj.weight",
    ]

    checkpoint = tmp_path / "ckpt"
    train_state = checkpoint / "train_state"
    train_state.mkdir(parents=True)
    torch.save(hybrid, train_state / "optimizer.pt")
    (train_state / "optimizer_schema.json").write_text(json.dumps(schema_document))

    monkeypatch.setattr(
        migration,
        "_target_adamw_group_names",
        lambda _architecture: {
            "matrix_decay": decay + [PROJECTOR_WEIGHT],
            "sensitive_no_decay": sensitive + [PROJECTOR_BIAS],
        },
    )
    monkeypatch.setattr(
        migration, "build_trainable_composite", lambda _a, device: object()
    )
    monkeypatch.setattr(
        migration,
        "route_cmuon_parameters",
        lambda _module, *, matrix_weight_decay, sensitive_weight_decay: _TargetRouting(
            cmuon_fqns, decay, sensitive
        ),
    )

    migration._migrate_irepa_optimizer(  # pyright: ignore[reportPrivateUsage]
        checkpoint,
        architecture={},
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
    )

    migrated = cast(dict[str, object], torch.load(train_state / "optimizer.pt", map_location="cpu"))

    # 1. Top-level key set: nothing added, nothing dropped.
    assert set(migrated) == set(hybrid)

    # 2. CMuon block: recursive bit-exact (momenta tensors + all config).
    assert _deep_equal(migrated["cmuon"], hybrid["cmuon"])

    # 3. SR RNG, transition, schema versions: verbatim.
    assert _deep_equal(migrated["sr_rng"], hybrid["sr_rng"])
    assert _deep_equal(migrated["transition"], hybrid["transition"])
    assert migrated["hybrid_cmuon_schema_version"] == 1
    assert migrated["guarded_canonical_schema_version"] == 1

    # 4. Guard block: recursive exact (references fp32, counters, mode,
    #    owner mapping, world size).
    assert _deep_equal(migrated["guard"], hybrid["guard"])

    # 5. AdamW: every pre-existing parameter's state (step + exp_avg +
    #    exp_avg_sq) is bit-exact, matched 1:1 by FQN across the id remap.
    source = cast(dict[str, object], hybrid["optimizer"])
    target = cast(dict[str, object], migrated["optimizer"])
    source_groups = cast(list[dict[str, object]], source["param_groups"])
    target_groups = cast(list[dict[str, object]], target["param_groups"])
    source_state = cast(dict[int, dict[str, torch.Tensor]], source["state"])
    target_state = cast(dict[int, dict[str, torch.Tensor]], target["state"])

    old_id_by_name: dict[str, int] = {}
    for group in source_groups:
        for name, pid in zip(group["param_names"], group["params"], strict=True):
            old_id_by_name[str(name)] = int(pid)
    new_id_by_name: dict[str, int] = {}
    for group in target_groups:
        for name, pid in zip(group["param_names"], group["params"], strict=True):
            new_id_by_name[str(name)] = int(pid)

    existing = set(old_id_by_name)
    added = set(new_id_by_name) - existing
    assert added == {PROJECTOR_WEIGHT, PROJECTOR_BIAS}
    assert set(new_id_by_name) == set(old_id_by_name) | added

    for name, old_id in old_id_by_name.items():
        new_id = new_id_by_name[name]
        old_entry = source_state[old_id]
        new_entry = target_state[new_id]
        assert _deep_equal(new_entry, old_entry), name

    # One state entry per pre-existing parameter, none for the projector:
    # the id set is exactly the name-mapped image of the source ids.
    assert len(target_state) == len(source_state)
    assert set(target_state) == {new_id_by_name[name] for name in old_id_by_name}

    # Group hyperparameters survive the remap.
    for source_group, target_group in zip(source_groups, target_groups, strict=True):
        for key in ("group_name", "lr", "weight_decay"):
            assert target_group[key] == source_group[key]
        assert target_group["param_names"] == source_group["param_names"] + [
            PROJECTOR_WEIGHT if source_group["group_name"] == "matrix_decay" else PROJECTOR_BIAS
        ]

    # 6. Routing: CMuon list verbatim; AdamW list = source + 2 projector
    #    entries; counts updated consistently.
    source_routing = cast(dict[str, object], hybrid["routing"])
    target_routing = cast(dict[str, object], migrated["routing"])
    assert _deep_equal(target_routing["cmuon"], source_routing["cmuon"])
    target_adamw = cast(list[dict[str, object]], target_routing["adamw"])
    assert {e["name"] for e in target_adamw} == set(old_id_by_name) | {
        PROJECTOR_WEIGHT,
        PROJECTOR_BIAS,
    }
    counts = cast(dict[str, object], target_routing["counts"])
    assert counts["cmuon"] == 2
    assert counts["adamw"] == len(old_id_by_name) + 2

    # 7. optimizer_schema.json: schema version + algorithm blocks verbatim;
    #    only the AdamW `groups` FQN lists change.
    schema_after = cast(
        dict[str, object], json.loads((train_state / "optimizer_schema.json").read_text())
    )
    assert schema_after["schema_version"] == 2
    assert _deep_equal(schema_after["hybrid_cmuon"], schema_document["hybrid_cmuon"])
    assert _deep_equal(schema_after["guarded_canonical"], schema_document["guarded_canonical"])
    groups_after = cast(list[dict[str, object]], schema_after["groups"])
    for source_group, group_after in zip(source_groups, groups_after, strict=True):
        suffix = (
            [PROJECTOR_WEIGHT]
            if source_group["group_name"] == "matrix_decay"
            else [PROJECTOR_BIAS]
        )
        assert group_after["param_names"] == source_group["param_names"] + suffix
