from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from sakuramoon.train import preflight as preflight_module
from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.preflight import (
    PREFLIGHT_CHECKS,
    AcceptedPreflight,
    PreflightError,
    build_single_gpu_preflight_checks,
    require_accepted_preflight,
    run_single_gpu_preflight,
)


def _passing_checks() -> dict[str, Callable[[], None]]:
    return {name: (lambda: None) for name in PREFLIGHT_CHECKS}


def test_single_gpu_preflight_runs_every_fixed_check_in_order(tmp_path: Path) -> None:
    calls: list[str] = []
    checks: dict[str, Callable[[], None]] = {
        name: (lambda current=name: calls.append(current)) for name in PREFLIGHT_CHECKS
    }

    accepted = run_single_gpu_preflight(checks, tmp_path / "preflight.json")
    report = accepted.report

    assert report.passed is True
    require_accepted_preflight(accepted)
    assert calls == list(PREFLIGHT_CHECKS)
    assert tuple(result.name for result in report.checks) == PREFLIGHT_CHECKS
    assert all(result.passed for result in report.checks)


def test_preflight_failure_stops_without_bypass_and_redacts_message(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    checks: dict[str, Callable[[], None]] = {
        name: (lambda current=name: calls.append(current)) for name in PREFLIGHT_CHECKS
    }

    def fail() -> None:
        calls.append("dataset_revision")
        raise RuntimeError("secret-shaped diagnostic text")

    checks["dataset_revision"] = fail
    destination = tmp_path / "preflight.json"

    with pytest.raises(PreflightError, match="dataset_revision"):
        run_single_gpu_preflight(checks, destination)

    assert calls == ["resolved_config", "local_assets", "dataset_revision"]
    text = destination.read_text()
    assert "secret-shaped" not in text
    payload = json.loads(text)
    assert payload["passed"] is False
    assert payload["checks"][-1] == {
        "error_type": "RuntimeError",
        "name": "dataset_revision",
        "passed": False,
    }


def test_preflight_rejects_missing_or_unknown_checks(tmp_path: Path) -> None:
    checks = _passing_checks()
    checks.pop("sample")
    checks["force"] = lambda: None

    with pytest.raises(ValueError, match="every fixed check"):
        run_single_gpu_preflight(checks, tmp_path / "preflight.json")


def test_preflight_handle_cannot_be_constructed_or_forged() -> None:
    with pytest.raises(TypeError, match="created only"):
        AcceptedPreflight(  # pyright: ignore[reportCallIssue]
            object()  # pyright: ignore[reportArgumentType]
        )
    forged = object.__new__(AcceptedPreflight)
    with pytest.raises(PreflightError, match="process-local"):
        require_accepted_preflight(forged)


def test_gpu_identity_requires_healthy_exact_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def healthy_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            returncode=0,
            stdout="NVIDIA GeForce RTX 5090, 580.105.08, 32607\n",
        )

    monkeypatch.setattr(
        preflight_module.subprocess,
        "run",
        healthy_run,
    )
    assert preflight_module._nvidia_smi_identity() == (  # pyright: ignore[reportPrivateUsage]
        "NVIDIA GeForce RTX 5090",
        "580.105.08",
        32607,
    )

    def failed_run(*_args: object, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=1, stdout="")

    monkeypatch.setattr(preflight_module.subprocess, "run", failed_run)
    with pytest.raises(RuntimeError, match="healthy GPU"):
        preflight_module._nvidia_smi_identity()  # pyright: ignore[reportPrivateUsage]


def test_nvme_identity_rejects_network_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def network_mount(_path: Path) -> tuple[Path, str, str]:
        return Path("/"), "nfs", "server:/workspace"

    monkeypatch.setattr(
        preflight_module,
        "_mount_identity",
        network_mount,
    )
    with pytest.raises(RuntimeError, match="local NVMe"):
        preflight_module._require_nvme(tmp_path)  # pyright: ignore[reportPrivateUsage]


def test_preflight_builder_requires_measured_checkpoint_size(tmp_path: Path) -> None:
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("[run]\n")
    with pytest.raises(ValueError, match="measured raw checkpoint bytes"):
        build_single_gpu_preflight_checks(
            cast(Any, object()),
            repository_root=tmp_path,
            resolved_config_path=resolved,
            data_client=cast(Any, object()),
            qwen=object(),
            vae=object(),
            trainable_module=cast(Any, object()),
            parameter_schema=lambda: None,
            image_shapes=lambda: None,
            text_shapes=lambda: None,
            zero_update_loss=lambda: None,
            optimizer_step=lambda: None,
            sample=lambda: None,
            checkpoint_round_trip=lambda: None,
            checkpoint_payload_bytes=0,
        )


def test_preflight_does_not_replace_report_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "preflight.json"
    target = tmp_path / "target.json"
    destination.symlink_to(target)

    with pytest.raises(FileExistsError, match="already exists"):
        run_single_gpu_preflight(_passing_checks(), destination)

    assert destination.is_symlink()
    assert not target.exists()


def test_preflight_rejects_relative_symlink_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="parent may not be a symlink"):
        run_single_gpu_preflight(_passing_checks(), Path("linked/preflight.json"))

    assert not (real / "preflight.json").exists()


def test_failure_bundle_is_atomic_immutable_and_omits_exception_message(
    tmp_path: Path,
) -> None:
    snapshot = FailureSnapshot("failure-1", "update", "OutOfMemoryError", 3, 2, 8)

    bundle = write_failure_bundle(tmp_path, snapshot)

    assert (bundle / "COMPLETE").is_file()
    assert json.loads((bundle / "failure.json").read_text())["error_type"] == (
        "OutOfMemoryError"
    )
    with pytest.raises(FileExistsError):
        write_failure_bundle(tmp_path, snapshot)


def test_failure_bundle_does_not_replace_a_dangling_symlink(tmp_path: Path) -> None:
    snapshot = FailureSnapshot("failure-1", "update", "RuntimeError", 1, 0, 0)
    target = tmp_path / snapshot.failure_id
    target.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="already exists"):
        write_failure_bundle(tmp_path, snapshot)

    assert target.is_symlink()
