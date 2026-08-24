"""P8 tests for the data-policy transition preflight helpers.

The shifted-bucket cutover may change exactly the ``data.spatial_crop``
resolved-config table and nothing else; these tests pin the allowlist, the
leaf-path diffing, and the append-only transition artifact contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sakuramoon.train.preflight import (
    DATA_POLICY_TRANSITION_KIND,
    SPATIAL_TRANSITION_ROOT,
    diff_resolved_toml_paths,
    record_data_policy_transition,
    require_spatial_transition_allowlist,
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


class TestAllowlist:
    def test_spatial_paths_pass(self) -> None:
        require_spatial_transition_allowlist(())
        require_spatial_transition_allowlist((SPATIAL_TRANSITION_ROOT,))
        require_spatial_transition_allowlist(
            (
                "data.spatial_crop.enabled",
                "data.spatial_crop.probability",
                "data.spatial_crop.max_equivalent_zoom",
            )
        )

    @pytest.mark.parametrize(
        "path",
        [
            "stage.base_lr",
            "data.image.min_crop_retention",
            "data.spatial_cropX.enabled",
        ],
    )
    def test_any_other_path_fails(self, path: str) -> None:
        with pytest.raises(ValueError, match="outside the data policy allowlist"):
            require_spatial_transition_allowlist((path,))


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
            artifact, self._record(resume_checkpoint="ckpt_76800")
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
                artifact, self._record(recorded_at_unix_ns=0)
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
