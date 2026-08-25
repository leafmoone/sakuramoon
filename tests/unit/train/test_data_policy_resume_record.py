"""Tests for the checkpoint-resume data-policy transition artifact.

``_record_data_policy_resume_transition`` appends one record per distinct
(checkpoint, policy configuration) pair, audits a changed resolved-config
sidecar against the cutover envelope, and dedupes restarts at the same
checkpoint with the same configuration.
"""

from __future__ import annotations

import json
import tomllib
from copy import deepcopy
from pathlib import Path

import pytest

from sakuramoon.config import ConfigurationError, LoadedConfig, load_config
from sakuramoon.train.production import _record_data_policy_resume_transition

CONFIG_ROOT = Path(__file__).resolve().parents[3] / "config"


def _load(config_name: str) -> LoadedConfig:
    return load_config(
        CONFIG_ROOT / config_name,
        config_root=CONFIG_ROOT,
        validate_secrets=False,
    )


def _render_toml(document: dict[str, object]) -> str:
    """Render a two/three-level resolved-config document back to TOML text."""

    lines: list[str] = []

    def scalar(value: object) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return repr(value)
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, list):
            return "[" + ", ".join(scalar(item) for item in value) + "]"
        raise TypeError(f"unsupported TOML value: {value!r}")

    def emit_table(prefix: str, table: dict[str, object]) -> None:
        lines.append(f"[{prefix}]")
        for key, value in table.items():
            if isinstance(value, dict):
                continue
            lines.append(f"{key} = {scalar(value)}")
        lines.append("")
        for key, value in table.items():
            if isinstance(value, dict):
                emit_table(f"{prefix}.{key}", value)

    root_scalars = {k: v for k, v in document.items() if not isinstance(v, dict)}
    if root_scalars:
        for key, value in root_scalars.items():
            lines.append(f"{key} = {scalar(value)}")
        lines.append("")
    for key, value in document.items():
        if isinstance(value, dict):
            emit_table(key, value)
    return "\n".join(lines)


def _source_sidecar(loaded: LoadedConfig, **spatial: object) -> str:
    """A same-protocol spatial-canary resolved config: the current document
    with the spatial policy switched on (a realistic p50 to transparent
    cutover keeps the LR/batch protocol, so the B1 contract stays green).
    """

    document = deepcopy(tomllib.loads(loaded.resolved_toml))
    spatial_table = document["data"]["spatial_crop"]
    for key, value in spatial.items():
        spatial_table[key] = value
    return _render_toml(document)


def test_transparent_resume_records_diff_and_dedupes(tmp_path: Path) -> None:
    loaded = _load("train_g1_transparent_white.toml")
    assert loaded.config.data.transparent_background.enabled
    assert not loaded.config.data.spatial_crop.enabled

    resume = tmp_path / "ckpt_78800_raw-78800"
    resume.mkdir()
    (resume / "resolved_config.toml").write_text(
        _source_sidecar(loaded, enabled=True, probability=0.5),
        encoding="utf-8",
    )

    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    artifact = (
        tmp_path
        / loaded.config.paths.artifact_dir
        / "data_policy_transition.json"
    )
    payload = json.loads(artifact.read_text(encoding="utf-8"))
    assert len(payload["records"]) == 1
    record = payload["records"][0]
    assert record["kind"] == "checkpoint-resume"
    assert record["resume_checkpoint"] == str(resume)
    assert "transparent_background" in record
    assert "spatial_crop" not in record
    # Only the flipped spatial leaves differ: the sidecar source is the
    # same protocol document with spatial enabled, so the operational
    # identity and the transparent policy are unchanged on both sides.
    assert record["resolved_config_changed_toml_paths"] == [
        "data.spatial_crop.enabled",
        "data.spatial_crop.probability",
    ]

    # Restarting at the same checkpoint with the same configuration is a
    # no-op; a new checkpoint appends.
    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    assert len(json.loads(artifact.read_text(encoding="utf-8"))["records"]) == 1
    next_resume = tmp_path / "ckpt_79800_raw-79800"
    next_resume.mkdir()
    (next_resume / "resolved_config.toml").write_text(
        (resume / "resolved_config.toml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    _record_data_policy_resume_transition(loaded, tmp_path, next_resume)
    assert len(json.loads(artifact.read_text(encoding="utf-8"))["records"]) == 2


def test_identical_sidecar_records_without_diff_paths(tmp_path: Path) -> None:
    loaded = _load("train_g1_transparent_white.toml")
    resume = tmp_path / "ckpt_78800_raw-78800"
    resume.mkdir()
    (resume / "resolved_config.toml").write_text(
        loaded.resolved_toml, encoding="utf-8"
    )

    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    artifact = (
        tmp_path
        / loaded.config.paths.artifact_dir
        / "data_policy_transition.json"
    )
    record = json.loads(artifact.read_text(encoding="utf-8"))["records"][0]
    assert "transparent_background" in record
    assert "resolved_config_changed_toml_paths" not in record


def test_out_of_envelope_resume_fails(tmp_path: Path) -> None:
    loaded = _load("train_g1_transparent_white.toml")
    source = tomllib.loads(loaded.resolved_toml)
    source["stage"]["base_lr"] = 0.00005
    resume = tmp_path / "ckpt_78800_raw-78800"
    resume.mkdir()
    (resume / "resolved_config.toml").write_text(
        _render_toml(source), encoding="utf-8"
    )

    with pytest.raises(ConfigurationError, match="resume config transition"):
        _record_data_policy_resume_transition(loaded, tmp_path, resume)


def test_disabled_policies_record_nothing(tmp_path: Path) -> None:
    loaded = _load("train_g1.toml")
    assert not loaded.config.data.spatial_crop.enabled
    assert not loaded.config.data.transparent_background.enabled
    resume = tmp_path / "ckpt_80400_raw-80400"
    resume.mkdir()

    _record_data_policy_resume_transition(loaded, tmp_path, resume)
    artifact = (
        tmp_path
        / loaded.config.paths.artifact_dir
        / "data_policy_transition.json"
    )
    assert not artifact.is_file()
