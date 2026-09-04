"""Camera cutover contract tests for the data-policy transition artifact.

C1 scope is contract-and-tests only: no production transition may run.
The allowlist mirrors the frozen ruling: a camera cutover may change only
data.spatial_crop.{enabled,probability}, data.camera_viewport.*, and the
isolated run/path/log/eval identity -- never optimizer, LR, batch,
dropout, objective, model, stage depth, resolution, compile, or iREPA.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from sakuramoon.config import load_config
from sakuramoon.train.preflight import DATA_POLICY_TRANSITION_KIND
from sakuramoon.train.production import _record_data_policy_resume_transition

CONFIG_ROOT = Path("config")

ALLOWED_PREFIXES = (
    "data.spatial_crop.",
    "data.camera_viewport.",
    "run.",
    "paths.",
    "logging.",
    "wandb.",
    "evaluation.",
)
FORBIDDEN_PREFIXES = (
    "optimizer.",
    "stage.",
    "model.",
    "objective.",
    "sampling.",
    "timestep.",
    "gpu.",
    "data.caption_dropout",
    "data.image.",
    "data.buckets",
    "data.spatial_crop.zoom",
)


def _load(name: str):
    return load_config(Path(name), config_root=CONFIG_ROOT, validate_secrets=False)


def _artifact(tmp_path: Path, loaded) -> Path:
    return tmp_path / loaded.config.paths.artifact_dir / "data_policy_transition.json"


def test_camera_cutover_records_the_camera_table(tmp_path: Path) -> None:
    loaded = _load("train_g1_camera_v2_p25.toml")
    resume = tmp_path / "resume"
    resume.mkdir()
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    artifact = _artifact(tmp_path, loaded)
    assert artifact.is_file()
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert payload["kind"] == DATA_POLICY_TRANSITION_KIND
    record = payload["records"][-1]
    assert record["policy_class"] == "data-only"
    assert record["resume_checkpoint"] == str(resume)
    assert "spatial_crop" not in record  # spatial is disabled in the canary
    camera = record["camera_viewport"]
    assert camera["enabled"] is True
    assert camera["mode"] == "hdm_shifted_square_v2"
    assert camera["probability"] == 0.25
    assert camera["min_equivalent_zoom"] == 1.10
    assert camera["max_equivalent_zoom"] == 1.50


def test_spatial_only_legacy_record_is_unchanged(tmp_path: Path) -> None:
    loaded = _load("train_g1_cmuon_production.toml")
    resume = tmp_path / "resume"
    resume.mkdir()
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    artifact = _artifact(tmp_path, loaded)
    record = json.loads(artifact.read_text(encoding="utf-8"))["records"][-1]
    assert "spatial_crop" in record
    assert "camera_viewport" not in record


def test_no_policy_writes_no_record(tmp_path: Path) -> None:
    loaded = _load("train_g1_camera_v2_p25.toml")
    data = loaded.config.data.model_copy(
        update={
            "camera_viewport": None,
            "spatial_crop": loaded.config.data.spatial_crop.model_copy(
                update={"enabled": False, "probability": 0.0}
            ),
            "transparent_background": loaded.config.data.transparent_background.model_copy(
                update={"enabled": False}
            ),
        }
    )
    loaded = dataclasses.replace(loaded, config=loaded.config.model_copy(update={"data": data}))
    resume = tmp_path / "resume"
    resume.mkdir()
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    assert not _artifact(tmp_path, loaded).is_file()


def test_disabled_camera_writes_no_record(tmp_path: Path) -> None:
    loaded = _load("train_g1_camera_v2_p25.toml")
    camera = loaded.config.data.camera_viewport
    assert camera is not None
    data = loaded.config.data.model_copy(
        update={
            "camera_viewport": camera.model_copy(
                update={"enabled": False, "probability": 0.0}
            ),
            "transparent_background": loaded.config.data.transparent_background.model_copy(
                update={"enabled": False}
            ),
        }
    )
    loaded = dataclasses.replace(loaded, config=loaded.config.model_copy(update={"data": data}))
    resume = tmp_path / "resume"
    resume.mkdir()
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    assert not _artifact(tmp_path, loaded).is_file()


def test_restart_at_same_checkpoint_is_a_no_op(tmp_path: Path) -> None:
    loaded = _load("train_g1_camera_v2_p25.toml")
    resume = tmp_path / "resume"
    resume.mkdir()
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    payload = json.loads(_artifact(tmp_path, loaded).read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1


def test_camera_cutover_diff_stays_inside_the_allowlist(tmp_path: Path) -> None:
    # The source sidecar carries the production resolved config (spatial
    # p50, no camera); the target is the p25 canary lineage.
    source = _load("train_g1_cmuon_production.toml")
    loaded = _load("train_g1_camera_v2_p25.toml")
    resume = tmp_path / "resume"
    resume.mkdir()
    (resume / "resolved_config.toml").write_text(
        source.resolved_toml, encoding="utf-8"
    )
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    record = json.loads(
        _artifact(tmp_path, loaded).read_text(encoding="utf-8")
    )["records"][-1]
    changed = record.get("resolved_config_changed_toml_paths", [])
    assert changed, "the cutover must record its changed leaves"
    # The two mandatory camera/spatial moves are present.
    assert "data.spatial_crop.enabled" in changed
    assert any(
        path.startswith("data.camera_viewport.") for path in changed
    )
    for path in changed:
        assert any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES), path
        assert not any(path.startswith(prefix) for prefix in FORBIDDEN_PREFIXES), path
