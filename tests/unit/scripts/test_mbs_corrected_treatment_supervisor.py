"""Tracked MBS treatment supervisor: stop-env hardening unit tests.

Pins the fix-review supervisor contract (stdlib only, no real kills):

- build_stop_env carries all 14 operational variables PLUS the explicit
  VENV_ROOT (the missing-VENV_ROOT bug that orphaned the workload on the
  previous untracked supervisor);
- PYTHON_BIN/ACCELERATE_BIN/MANAGEMENT_PYTHON are derived from VENV_ROOT
  with the exact training_stack.sh convention when absent, and preserved
  when the launch environment carries them;
- the --self-test-stop-env validation: PASS on a valid environment, FAIL
  on a missing VENV_ROOT, a non-0600 WORKLOAD_ENV_FILE, a symlink env
  file, and out-of-range port/interval;
- the supervisor invokes training_stack.sh with ``stop`` only -- no
  auto-restart path exists in the source.
"""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[3] / "scripts" / "mbs_corrected_treatment_supervisor.py"
)


@pytest.fixture(scope="module")
def supervisor_module():
    spec = importlib.util.spec_from_file_location(
        "mbs_supervisor_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _namespace(
    tmp_path: Path,
    *,
    venv_root: str | None = None,
    workload_env_file: str | None = None,
    interval_seconds: int = 86400,
    main_process_port: int = 29500,
    required_host_substring: str = "come7",
    config_name: str = "train_g1_camera_v2_p25_mirror_v2_canary.toml",
) -> argparse.Namespace:
    root = tmp_path / "work"
    root.mkdir(parents=True, exist_ok=True)
    return argparse.Namespace(
        project_root=str(root),
        runtime_root=str(tmp_path),
        config_root=str(root / "config"),
        config_name=config_name,
        log_root=str(root / "logs"),
        run_root=str(root / "logs" / "run"),
        workload_env_file=str(workload_env_file or root / "env.nul"),
        resume_checkpoint=str(root / "ckpt"),
        publish_state_root=str(root / "pub"),
        publish_last_published=str(root / "pub" / "last-published.txt"),
        repo_path="experiments/camera-v2-mbs-corrected-118100-118200",
        interval_seconds=interval_seconds,
        required_host_substring=required_host_substring,
        main_process_port=main_process_port,
        venv_root=str(venv_root if venv_root is not None else root / "venv"),
        self_test_stop_env=False,
    )


def _make_venv(venv_root: Path) -> None:
    (venv_root / "bin").mkdir(parents=True)
    for name in ("python", "accelerate"):
        target = venv_root / "bin" / name
        target.write_text("#!/bin/sh\n")
        target.chmod(0o755)


def _make_env_file(path: Path, mode: int = 0o600) -> None:
    path.write_text("KEY=value\n")
    path.chmod(mode)


def test_stop_env_carries_all_operational_vars_and_venv_root(
    supervisor_module, tmp_path: Path
) -> None:
    ns = _namespace(tmp_path)
    env = supervisor_module.build_stop_env(ns)

    for key in supervisor_module.STOP_ENV_KEYS:
        assert key in env, key
    assert env["VENV_ROOT"] == ns.venv_root
    assert env["INTERVAL_SECONDS"] == "86400"
    assert env["MAIN_PROCESS_PORT"] == "29500"
    assert env["PROJECT_ROOT"] == ns.project_root
    assert env["REQUIRED_HOST_SUBSTRING"] == "come7"


def test_stop_env_derives_bins_from_venv_root(
    supervisor_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in ("PYTHON_BIN", "ACCELERATE_BIN", "MANAGEMENT_PYTHON"):
        monkeypatch.delenv(key, raising=False)
    ns = _namespace(tmp_path)
    env = supervisor_module.build_stop_env(ns)

    assert env["PYTHON_BIN"] == f"{ns.venv_root}/bin/python"
    assert env["ACCELERATE_BIN"] == f"{ns.venv_root}/bin/accelerate"
    assert env["MANAGEMENT_PYTHON"] == env["PYTHON_BIN"]


def test_stop_env_preserves_launch_bins(
    supervisor_module, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PYTHON_BIN", "/launch/python")
    monkeypatch.setenv("ACCELERATE_BIN", "/launch/accelerate")
    monkeypatch.setenv("MANAGEMENT_PYTHON", "/launch/management")
    ns = _namespace(tmp_path)
    env = supervisor_module.build_stop_env(ns)

    assert env["PYTHON_BIN"] == "/launch/python"
    assert env["ACCELERATE_BIN"] == "/launch/accelerate"
    assert env["MANAGEMENT_PYTHON"] == "/launch/management"
    assert env["VENV_ROOT"] == ns.venv_root


def test_self_test_passes_on_valid_environment(
    supervisor_module, tmp_path: Path
) -> None:
    ns = _namespace(tmp_path)
    _make_venv(Path(ns.venv_root))
    _make_env_file(Path(ns.workload_env_file))

    ok, problems, env = supervisor_module.self_test_stop_env(ns)

    assert ok, problems
    assert problems == []
    assert env["VENV_ROOT"] == ns.venv_root


def test_self_test_fails_on_missing_venv_root(
    supervisor_module, tmp_path: Path
) -> None:
    ns = _namespace(tmp_path, venv_root=str(tmp_path / "does-not-exist"))
    _make_env_file(Path(ns.workload_env_file))

    ok, problems, _ = supervisor_module.self_test_stop_env(ns)

    assert not ok
    assert any("VENV_ROOT" in problem for problem in problems)
    assert any("PYTHON_BIN" in problem for problem in problems)


def test_self_test_fails_on_non_600_env_file(
    supervisor_module, tmp_path: Path
) -> None:
    ns = _namespace(tmp_path)
    _make_venv(Path(ns.venv_root))
    _make_env_file(Path(ns.workload_env_file), mode=0o644)

    ok, problems, _ = supervisor_module.self_test_stop_env(ns)

    assert not ok
    assert any("0600" in problem for problem in problems)


def test_self_test_fails_on_symlink_env_file(
    supervisor_module, tmp_path: Path
) -> None:
    ns = _namespace(tmp_path)
    _make_venv(Path(ns.venv_root))
    real = tmp_path / "real-env.nul"
    _make_env_file(real)
    link = tmp_path / "link-env.nul"
    link.symlink_to(real)
    ns.workload_env_file = str(link)

    ok, problems, _ = supervisor_module.self_test_stop_env(ns)

    assert not ok
    assert any("symlink" in problem for problem in problems)


def test_self_test_fails_on_bad_port_and_interval(
    supervisor_module, tmp_path: Path
) -> None:
    ns = _namespace(
        tmp_path,
        interval_seconds=0,
        main_process_port=70000,
    )
    _make_venv(Path(ns.venv_root))
    _make_env_file(Path(ns.workload_env_file))

    ok, problems, _ = supervisor_module.self_test_stop_env(ns)

    assert not ok
    assert any("INTERVAL_SECONDS" in problem for problem in problems)
    assert any("MAIN_PROCESS_PORT" in problem for problem in problems)


def test_supervisor_invokes_stack_stop_only_never_restarts() -> None:
    source = SCRIPT_PATH.read_text(encoding="utf-8")

    assert '"scripts/training_stack.sh"), "stop"' in source
    assert '"scripts/training_stack.sh"), "start"' not in source
    assert '"restart"' not in source
