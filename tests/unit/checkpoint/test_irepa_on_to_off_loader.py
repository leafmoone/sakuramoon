"""Direct iREPA ON->OFF resume: the loader may drop exactly the declared
iREPA auxiliary (model FQNs + optimizer state) from a v4 artifact into a
no-iREPA module, and nothing else."""

from __future__ import annotations

import types
from typing import Any, cast

import pytest

import sakuramoon.checkpoint.load as loader
from sakuramoon.checkpoint.artifact import (
    architectures_share_parameter_contract,
)
from sakuramoon.checkpoint.migrate_irepa_checkpoint import (  # pyright: ignore[reportPrivateUsage]
    _migrate_architecture_v4,
)
from sakuramoon.checkpoint.schema import CheckpointError
from sakuramoon.model.irepa import irepa_auxiliary_fqns

PROJECTOR_FQNS = frozenset(
    {
        "irepa_alignment.projector.weight",
        "irepa_alignment.projector.bias",
    }
)


def _v3_architecture() -> dict[str, object]:
    return {
        "schema_version": 3,
        "class": "TrainableComposite",
        "dit": {
            "hidden_size": 32,
            "attention_backend": "dense_sdpa",
        },
        "text": {"input_size": 8},
        "condition_tokens": {"output_size": 32},
    }


def _v4_architecture() -> dict[str, object]:
    migrated = _migrate_architecture_v4(_v3_architecture())  # pyright: ignore[reportPrivateUsage]
    assert migrated["schema_version"] == 4
    return migrated


def _module(irepa: object) -> Any:
    return types.SimpleNamespace(irepa_alignment=irepa)


def test_irepa_auxiliary_fqns_returns_the_locked_set() -> None:
    v4 = _v4_architecture()
    auxiliary = cast(dict[str, object], v4["training_auxiliaries"])
    assert irepa_auxiliary_fqns(auxiliary["irepa"]) == PROJECTOR_FQNS


def test_irepa_auxiliary_fqns_rejects_non_locked_documents() -> None:
    v4 = _v4_architecture()
    auxiliary = cast(dict[str, object], v4["training_auxiliaries"])
    broken = dict(auxiliary["irepa"])
    broken["bias"] = False
    with pytest.raises(ValueError, match="locked v1"):
        irepa_auxiliary_fqns(broken)
    with pytest.raises(ValueError, match="object"):
        irepa_auxiliary_fqns("irepa")


@pytest.mark.parametrize(
    ("irepa", "architecture", "expected"),
    [
        (None, _v3_architecture(), frozenset()),
        (None, _v4_architecture(), PROJECTOR_FQNS),
        (types.SimpleNamespace(), _v4_architecture(), frozenset()),
        (types.SimpleNamespace(), _v3_architecture(), frozenset()),
    ],
)
def test_declared_dropped_auxiliary_fqns(
    irepa: object,
    architecture: dict[str, object],
    expected: frozenset[str],
) -> None:
    assert (
        loader._declared_dropped_auxiliary_fqns(  # pyright: ignore[reportPrivateUsage]
            _module(irepa), architecture
        )
        == expected
    )


def test_declared_dropped_rejects_non_irepa_auxiliaries() -> None:
    architecture = _v3_architecture()
    architecture["training_auxiliaries"] = {"other": {"class": "Other"}}
    with pytest.raises(CheckpointError, match="non-iREPA training auxiliaries"):
        loader._declared_dropped_auxiliary_fqns(  # pyright: ignore[reportPrivateUsage]
            _module(None), architecture
        )


def test_declared_dropped_rejects_malformed_irepa_metadata() -> None:
    architecture = _v3_architecture()
    architecture["training_auxiliaries"] = {
        "irepa": {"class": "IRepaAlignment", "schema_version": 9, "bias": True}
    }
    with pytest.raises(CheckpointError, match="locked v1"):
        loader._declared_dropped_auxiliary_fqns(  # pyright: ignore[reportPrivateUsage]
            _module(None), architecture
        )


def test_contract_is_strict_for_raw_v3_vs_v4_documents() -> None:
    # The document-level comparison stays strict: a raw v3 document and a
    # raw v4 document do NOT share a parameter contract.  The auxiliary
    # drop is an explicit loader opt-in (allow_irepa_auxiliary_drop), not a
    # property of the documents themselves.
    assert (
        architectures_share_parameter_contract(_v3_architecture(), _v4_architecture())
        is False
    )
    assert (
        architectures_share_parameter_contract(_v4_architecture(), _v3_architecture())
        is False
    )


def test_contract_accepts_v4_artifact_with_locked_irepa_against_v3_module() -> None:
    # With the explicit loader opt-in, the v4 document differs from the v3
    # trunk only by the locked auxiliary key and the schema-version marker;
    # both are stripped before the trunk comparison.
    assert (
        architectures_share_parameter_contract(
            _v3_architecture(),
            _v4_architecture(),
            allow_irepa_auxiliary_drop=True,
        )
        is True
    )


def test_contract_rejects_v4_artifact_with_other_auxiliaries_against_v3_module() -> (
    None
):
    v4 = _v4_architecture()
    v4["training_auxiliaries"] = {"other": {"class": "Other"}}  # type: ignore[assignment]
    assert (
        architectures_share_parameter_contract(
            _v3_architecture(), v4, allow_irepa_auxiliary_drop=True
        )
        is False
    )


def test_contract_keeps_reverse_direction_strict() -> None:
    # An iREPA module against a v3 artifact stays strict (OFF->ON is the
    # explicit migration, not a loader drop) - even with the opt-in set,
    # because the drop branch only applies to a no-iREPA left side.
    assert (
        architectures_share_parameter_contract(
            _v4_architecture(), _v3_architecture(), allow_irepa_auxiliary_drop=True
        )
        is False
    )


def test_verify_group_diff_accepts_exactly_the_declared_drop() -> None:
    current = [
        ("matrix_decay", ("a.weight", "b.weight")),
        ("sensitive_no_decay", ("a.bias", "c.bias")),
    ]
    saved = [
        (
            "matrix_decay",
            ("a.weight", "b.weight", "irepa_alignment.projector.weight"),
        ),
        (
            "sensitive_no_decay",
            ("a.bias", "c.bias", "irepa_alignment.projector.bias"),
        ),
    ]
    loader._verify_group_diff(  # pyright: ignore[reportPrivateUsage]
        saved, current, frozenset(), PROJECTOR_FQNS
    )


def test_verify_group_diff_rejects_unexpected_extra_fqns() -> None:
    current = [("matrix_decay", ("a.weight",)), ("sensitive_no_decay", ("a.bias",))]
    saved = [
        ("matrix_decay", ("a.weight", "rogue.weight")),
        ("sensitive_no_decay", ("a.bias",)),
    ]
    with pytest.raises(CheckpointError, match="does not have"):
        loader._verify_group_diff(  # pyright: ignore[reportPrivateUsage]
            saved, current, frozenset(), PROJECTOR_FQNS
        )


def test_verify_group_diff_rejects_declared_drop_absent_from_saved() -> None:
    current = [("matrix_decay", ("a.weight",)), ("sensitive_no_decay", ("a.bias",))]
    saved = [
        ("matrix_decay", ("a.weight", "irepa_alignment.projector.weight")),
        ("sensitive_no_decay", ("a.bias",)),
    ]
    with pytest.raises(CheckpointError, match="do not carry exactly"):
        loader._verify_group_diff(  # pyright: ignore[reportPrivateUsage]
            saved, current, frozenset(), PROJECTOR_FQNS
        )


def test_verify_group_diff_keeps_growth_slot_behavior() -> None:
    current = [("matrix_decay", ("a.weight", "slot_00.x")), ("sensitive_no_decay", ())]
    saved = [("matrix_decay", ("a.weight",)), ("sensitive_no_decay", ())]
    loader._verify_group_diff(  # pyright: ignore[reportPrivateUsage]
        saved, current, frozenset({"slot_00.x"})
    )


def _routing_manifests(
    saved_names: list[str],
    current_names: list[str],
) -> tuple[dict[str, object], dict[str, object]]:
    def entry(name: str) -> dict[str, object]:
        return {"name": name, "group": "sensitive_no_decay", "weight_decay": 0.0}

    # First item = the checkpoint (saved) side, second = the current
    # module side.  In an ON->OFF resume the saved side carries the
    # projector entries and the current side does not.
    return (
        {
            "cmuon": [],
            "adamw": [entry(name) for name in saved_names],
        },
        {
            "cmuon": [],
            "adamw": [entry(name) for name in current_names],
        },
    )


def test_verify_routing_manifest_accepts_exactly_the_declared_drop() -> None:
    saved, current = _routing_manifests(
        [
            "a",
            "c",
            "irepa_alignment.projector.weight",
            "irepa_alignment.projector.bias",
        ],
        ["a", "c"],
    )
    loader._verify_routing_manifest(  # pyright: ignore[reportPrivateUsage]
        saved, current, frozenset(), PROJECTOR_FQNS
    )


def test_verify_routing_manifest_rejects_unexpected_drop() -> None:
    # The saved manifest carries only one of the two declared projector
    # entries: the drop set is incomplete, not the declared one.
    saved, current = _routing_manifests(
        ["a", "c", "irepa_alignment.projector.weight"],
        ["a", "c"],
    )
    with pytest.raises(CheckpointError, match="differ from the declared"):
        loader._verify_routing_manifest(  # pyright: ignore[reportPrivateUsage]
            saved, current, frozenset(), PROJECTOR_FQNS
        )


def test_verify_routing_manifest_rejects_rogue_entry() -> None:
    saved, current = _routing_manifests(
        ["a", "c", "rogue.weight"],
        ["a", "c"],
    )
    with pytest.raises(CheckpointError, match="no current counterpart"):
        loader._verify_routing_manifest(  # pyright: ignore[reportPrivateUsage]
            saved, current, frozenset(), PROJECTOR_FQNS
        )


def _adamw_state(
    names: list[str],
    current_names: list[str],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    saved_ids = {name: index for index, name in enumerate(names)}
    current_ids = {name: index for index, name in enumerate(current_names)}
    state: dict[int, object] = {
        saved_ids[name]: {
            "step": 4.0,
            "exp_avg": f"avg-{name}",
            "exp_avg_sq": f"sq-{name}",
        }
        for name in names
    }
    current_groups = [
        {
            "params": [current_ids[name] for name in current_names],
            "param_names": list(current_names),
            "lr": 1e-5,
        }
    ]
    document = {
        "state": state,
        "param_groups": [
            {
                "params": list(saved_ids.values()),
                "param_names": list(names),
                "lr": 1e-5,
            }
        ],
    }
    return document, current_groups


def test_remap_drops_exactly_the_declared_auxiliary_state() -> None:
    saved_names = [
        "a",
        "irepa_alignment.projector.weight",
        "c",
        "irepa_alignment.projector.bias",
    ]
    current_names = ["a", "c"]
    document, current_groups = _adamw_state(saved_names, current_names)
    remapped = loader._remap_state_to_current_ids(  # pyright: ignore[reportPrivateUsage]
        document, current_groups, PROJECTOR_FQNS
    )
    remapped_state = cast(dict[int, object], remapped["state"])
    assert set(remapped_state) == {0, 1}
    assert cast(dict[str, object], remapped_state[0])["exp_avg"] == "avg-a"
    assert cast(dict[str, object], remapped_state[1])["exp_avg"] == "avg-c"
    assert remapped["param_groups"] == current_groups


def test_remap_still_rejects_unknown_fqns_when_dropping() -> None:
    saved_names = ["a", "rogue.weight", "c"]
    current_names = ["a", "c"]
    document, current_groups = _adamw_state(saved_names, current_names)
    with pytest.raises(CheckpointError, match="absent from the current optimizer"):
        loader._remap_state_to_current_ids(  # pyright: ignore[reportPrivateUsage]
            document, current_groups, PROJECTOR_FQNS
        )
