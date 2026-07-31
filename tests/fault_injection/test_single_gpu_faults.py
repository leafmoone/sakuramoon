from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from sakuramoon.fault_injection import (
    FaultScenario,
    run_expected_exit,
    run_until_ready_and_sigkill,
)
from sakuramoon.fault_injection.single_gpu_worker import OOM_EXIT_CODE

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)

_SOURCE_ROOT = Path(__file__).parents[2] / "src"
_CONTROL = {
    "accumulation_steps": 1,
    "attention_backend": "dense_sdpa",
    "checkpoint_every_updates": 1000,
    "learning_rate": 2e-5,
    "local_batch": 1,
    "optimizer_name": "TorchAO AdamW8bit",
    "resolved_config_sha256": "a" * 64,
    "world_size": 1,
}


def test_recovery_worker_uses_weights_only_deserialization() -> None:
    from sakuramoon.fault_injection import single_gpu_worker

    assert single_gpu_worker.__file__ is not None
    source = Path(single_gpu_worker.__file__).read_text(encoding="utf-8")
    assert "weights_only=True" in source
    assert "weights_only=False" not in source


def _worker_command(arguments: tuple[str, ...]) -> tuple[str, ...]:
    script = (
        f"import sys; sys.path.insert(0, {_SOURCE_ROOT.as_posix()!r}); "
        "from sakuramoon.fault_injection.single_gpu_worker import main; "
        f"raise SystemExit(main({arguments!r}))"
    )
    return (sys.executable, "-c", script)


@pytest.mark.parametrize(
    ("phase", "scenario"),
    [
        ("microbatch", FaultScenario.MICROBATCH_SIGKILL),
        ("optimizer", FaultScenario.OPTIMIZER_SIGKILL),
        ("checkpoint", FaultScenario.CHECKPOINT_SIGKILL),
    ],
)
def test_real_torchao_kill_then_fresh_process_recovers_explicit_complete_parent(
    tmp_path: Path, phase: str, scenario: FaultScenario
) -> None:
    workspace = tmp_path / phase
    run_until_ready_and_sigkill(
        _worker_command(("kill", "--phase", phase, "--workspace", str(workspace))),
        scenario=scenario,
        timeout_seconds=30.0,
    )

    parent = workspace / "parent_0"
    assert (parent / "COMPLETE").read_bytes() == b"complete\n"
    assert json.loads((parent / "control.json").read_bytes()) == _CONTROL
    assert tuple(workspace.glob("*/COMPLETE")) == (parent / "COMPLETE",)
    recovery = workspace / "recovery.json"
    completed = subprocess.run(
        _worker_command(
            ("recover", "--parent", str(parent), "--output", str(recovery))
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0
    payload = json.loads(recovery.read_bytes())
    assert payload == {
        "control_after": _CONTROL,
        "parent_checkpoint_id": "parent_0",
        "successful_updates": 1,
    }


def test_real_cuda_allocator_oom_is_process_local_and_context_is_reclaimed(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "oom-workspace"
    run_until_ready_and_sigkill(
        _worker_command(
            ("kill", "--phase", "checkpoint", "--workspace", str(workspace))
        ),
        scenario=FaultScenario.CHECKPOINT_SIGKILL,
        timeout_seconds=30.0,
    )
    parent = workspace / "parent_0"
    output = tmp_path / "oom.json"
    evidence = run_expected_exit(
        _worker_command(
            ("oom", "--parent", str(parent), "--output", str(output))
        ),
        scenario=FaultScenario.CUDA_OOM,
        expected_returncode=OOM_EXIT_CODE,
        timeout_seconds=30.0,
    )

    assert evidence.returncode == OOM_EXIT_CODE
    payload = json.loads(output.read_bytes())
    assert payload == {
        "control_after": _CONTROL,
        "control_before": _CONTROL,
        "error_type": "OutOfMemoryError",
        "parent_checkpoint_id": "parent_0",
        "parent_complete": True,
    }
    recovery = tmp_path / "oom-recovery.json"
    completed = subprocess.run(
        _worker_command(
            ("recover", "--parent", str(parent), "--output", str(recovery))
        ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
        timeout=30.0,
    )
    assert completed.returncode == 0
    assert json.loads(recovery.read_bytes())["parent_checkpoint_id"] == "parent_0"
    probe = torch.ones(16, device="cuda")
    assert bool(torch.isfinite(probe).all().item())
