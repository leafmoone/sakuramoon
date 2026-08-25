"""P8 tests for the data-policy transition preflight helpers.

A governed data-policy cutover may change a run's operational identity and
the tables of the data policies enabled in either document, and nothing
else; these tests pin the cutover envelope, the leaf-path diffing, and the
append-only transition artifact contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakuramoon.train.preflight import (
    DATA_POLICY_TRANSITION_KIND,
    RESUME_TRANSITION_OPERATIONAL_LEAVES,
    audit_resolved_config_transition,
    diff_resolved_toml_paths,
    record_data_policy_transition,
)

BASE = (
    "[stage]\n"
    "base_lr = 0.0000475\n"
    "planned_updates = 110000\n"
    "\n"
    "[data.image]\n"
    "min_crop_retention = 0.8\n"
)
WITH_SPATIAL = BASE + "\n[data.spatial_crop]\nenabled = true\nprobability = 0.25\n"
WITH_TRANSPARENT = (
    BASE + "\n[data.transparent_background]\nenabled = true\n"
)


class TestTomlPathDiff:
    def test_identical_texts_have_no_changed_paths(self) -> None:
        assert diff_resolved_toml_paths(BASE, BASE) == ()

    def test_changed_scalar_reports_leaf_path(self) -> None:
        other = BASE.replace("base_lr = 0.0000475", "base_lr = 0.00008")
        assert diff_resolved_toml_paths(BASE, other) == ("stage.base_lr",)

    def test_added_table_reports_leaf_paths_only(self) -> None:
        changed = diff_resolved_toml_paths(BASE, WITH_SPATIAL)
        assert changed == ("data.spatial_crop.enabled", "data.spatial_crop.probability")
        # The table path itself is never a reported path: a non-empty table
        # contributes exactly its leaves, so allowlist checks cannot be
        # bypassed by renaming a whole table.
        assert "data.spatial_crop" not in changed

    def test_removed_table_reports_removed_leaf_paths(self) -> None:
        changed = diff_resolved_toml_paths(WITH_SPATIAL, BASE)
        assert changed == ("data.spatial_crop.enabled", "data.spatial_crop.probability")

    def test_empty_table_change_reports_table_path(self) -> None:
        previous = BASE + "\n[data.spatial_crop]\n"
        changed = diff_resolved_toml_paths(previous, WITH_SPATIAL)
        assert "data.spatial_crop" in changed
        assert "data.spatial_crop.enabled" in changed


class TestAuditTransition:
    def _identity_cutover(self) -> str:
        return BASE + (
            "\n"
            "[run]\n"
            'run_id = "g1_transparent_white"\n'
            "\n"
            "[paths]\n"
            'run_dir = "runs/g1_transparent_white"\n'
            'checkpoint_dir = "output_model/g1_transparent_white"\n'
            'artifact_dir = "artifacts/g1_transparent_white"\n'
            "\n"
            "[logging]\n"
            'local_jsonl_path = "artifacts/g1_transparent_white/metrics.jsonl"\n'
            "\n"
            "[wandb]\n"
            'retry_jsonl_path = "artifacts/g1_transparent_white/wandb-retry.jsonl"\n'
            "\n"
            "[evaluation]\n"
            'output_dir = "output_model/evaluation/g1_transparent_white"\n'
        )

    def test_identical_documents_pass(self) -> None:
        assert audit_resolved_config_transition(BASE, BASE) == ()

    def test_operational_identity_leaves_pass(self) -> None:
        changed = audit_resolved_config_transition(BASE, self._identity_cutover())
        assert len(changed) == len(RESUME_TRANSITION_OPERATIONAL_LEAVES)
        assert set(changed) == set(RESUME_TRANSITION_OPERATIONAL_LEAVES)

    def test_policy_enabled_in_current_passes(self) -> None:
        changed = audit_resolved_config_transition(BASE, WITH_SPATIAL)
        assert changed == (
            "data.spatial_crop.enabled",
            "data.spatial_crop.probability",
        )

    def test_policy_enabled_in_previous_passes(self) -> None:
        # A governed cutover may disable a policy the source checkpoint ran
        # with; the root is allowed while it is enabled on either side.
        changed = audit_resolved_config_transition(WITH_SPATIAL, BASE)
        assert changed == (
            "data.spatial_crop.enabled",
            "data.spatial_crop.probability",
        )

    def test_transparent_policy_leaves_pass(self) -> None:
        changed = audit_resolved_config_transition(BASE, WITH_TRANSPARENT)
        assert changed == ("data.transparent_background.enabled",)

    @pytest.mark.parametrize(
        "drift",
        [
            "stage.base_lr",
            "data.image.min_crop_retention",
        ],
    )
    def test_out_of_envelope_leaf_fails(self, drift: str) -> None:
        if drift == "stage.base_lr":
            other = BASE.replace("base_lr = 0.0000475", "base_lr = 0.00008")
        else:
            other = BASE.replace("min_crop_retention = 0.8", "min_crop_retention = 0.5")
        with pytest.raises(ValueError, match="outside the data policy allowlist"):
            audit_resolved_config_transition(BASE, other)

    def test_out_of_envelope_fails_even_when_a_policy_is_enabled(self) -> None:
        other = WITH_SPATIAL.replace("base_lr = 0.0000475", "base_lr = 0.00008")
        with pytest.raises(ValueError, match="stage.base_lr"):
            audit_resolved_config_transition(WITH_SPATIAL, other)

    def test_disabled_policy_drift_fails(self) -> None:
        previous = BASE + "\n[data.spatial_crop]\nenabled = false\nmin_equivalent_zoom = 1.2\n"
        current = BASE + "\n[data.spatial_crop]\nenabled = false\nmin_equivalent_zoom = 1.3\n"
        with pytest.raises(ValueError, match="min_equivalent_zoom"):
            audit_resolved_config_transition(previous, current)

    def test_foreign_policy_root_fails(self) -> None:
        other = BASE + "\n[data.future_policy]\nenabled = true\n"
        with pytest.raises(ValueError, match="future_policy"):
            audit_resolved_config_transition(BASE, other)

    def test_envelope_moves_together_with_policy_and_identity(self) -> None:
        # The realistic transparent cut-over: a new run identity, the
        # transparent policy enabled, and the spatial policy disabled.
        current = self._identity_cutover() + "\n[data.transparent_background]\nenabled = true\n"
        source = WITH_SPATIAL
        changed = audit_resolved_config_transition(source, current)
        assert set(changed) == set(RESUME_TRANSITION_OPERATIONAL_LEAVES) | {
            "data.spatial_crop.enabled",
            "data.spatial_crop.probability",
            "data.transparent_background.enabled",
        }


class TestRecordTransition:
    def _record(self, **overrides: object) -> dict[str, object]:
        base: dict[str, object] = {
            "kind": "checkpoint-resume",
            "policy_class": "data-only",
            "resume_checkpoint": "ckpt_75800",
            "spatial_crop": "p25",
            "recorded_at_unix_ns": 1_750_000_000_000_000_000,
        }
        base.update(overrides)
        return base

    def test_creates_and_appends(self, tmp_path: Path) -> None:
        artifact = tmp_path / "nested" / "data_policy_transition.json"
        record_data_policy_transition(artifact, self._record())
        record_data_policy_transition(
            artifact,
            self._record(resume_checkpoint="ckpt_76800"),
        )
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["kind"] == DATA_POLICY_TRANSITION_KIND
        assert len(payload["records"]) == 2
        assert payload["records"][0]["resume_checkpoint"] == "ckpt_75800"
        assert payload["records"][1]["resume_checkpoint"] == "ckpt_76800"
        # Atomic write leaves no temporary file behind.
        leftovers = [
            name
            for name in artifact.parent.iterdir()
            if name.name != artifact.name
        ]
        assert leftovers == []

    def test_duplicate_last_record_is_a_noop(self, tmp_path: Path) -> None:
        artifact = tmp_path / "data_policy_transition.json"
        record_data_policy_transition(artifact, self._record())
        before = artifact.read_bytes()
        result = record_data_policy_transition(
            artifact,
            self._record(recorded_at_unix_ns=999_999_999_999_999_999),
            skip_if_duplicate_of_last=(
                "kind",
                "resume_checkpoint",
                "spatial_crop",
            ),
        )
        assert result == artifact
        assert artifact.read_bytes() == before
        assert len(json.loads(before)["records"]) == 1

    def test_missing_required_fields_are_rejected(self, tmp_path: Path) -> None:
        artifact = tmp_path / "data_policy_transition.json"
        with pytest.raises(ValueError, match="kind"):
            record_data_policy_transition(artifact, self._record(kind=""))
        with pytest.raises(ValueError, match="policy_class"):
            record_data_policy_transition(
                artifact,
                {k: v for k, v in self._record().items() if k != "policy_class"},
            )
        with pytest.raises(ValueError, match="timestamp"):
            record_data_policy_transition(
                artifact,
                self._record(recorded_at_unix_ns=0),
            )

    def test_tampered_artifact_shape_is_rejected(self, tmp_path: Path) -> None:
        artifact = tmp_path / "data_policy_transition.json"
        artifact.write_text(
            json.dumps({"kind": "something.else", "records": []}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown shape"):
            record_data_policy_transition(artifact, self._record())
        artifact.write_text(
            json.dumps({"kind": DATA_POLICY_TRANSITION_KIND, "records": {}}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="unknown shape"):
            record_data_policy_transition(artifact, self._record())

    def test_corrupt_artifact_restarts_fresh(self, tmp_path: Path) -> None:
        artifact = tmp_path / "data_policy_transition.json"
        artifact.write_text("{not json", encoding="utf-8")
        record_data_policy_transition(artifact, self._record())
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        assert payload["kind"] == DATA_POLICY_TRANSITION_KIND
        assert len(payload["records"]) == 1
