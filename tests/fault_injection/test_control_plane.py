from __future__ import annotations

import errno
import hashlib
import json
import os
import signal
import sys
from dataclasses import replace
from pathlib import Path

import pytest

import sakuramoon.fault_injection.recovery as recovery_module
import sakuramoon.fault_injection.report as report_module
from sakuramoon.checkpoint.schema import (
    CheckpointError,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
)
from sakuramoon.fault_injection import (
    CPU_SCENARIOS,
    FOUR_GPU_SCENARIOS,
    ONE_GPU_SCENARIOS,
    ExecutedFaultEvidence,
    FaultMatrixReport,
    FaultOutcome,
    FaultProcessError,
    FaultScenario,
    FaultStatus,
    HardwareLevel,
    ReplayEvidence,
    TrainingControlSnapshot,
    load_executed_fault_evidence,
    publish_fault_matrix_from_evidence,
    run_expected_exit,
    run_until_ready_and_sigkill,
    select_complete_raw_parent,
    write_executed_fault_evidence,
    write_fault_matrix,
)

_HASH = "a" * 64
_SOURCE_ROOT = Path(__file__).parents[2] / "src"


def _control() -> TrainingControlSnapshot:
    return TrainingControlSnapshot(
        resolved_config_sha256=_HASH,
        local_batch=2,
        accumulation_steps=4,
        attention_backend="dense_sdpa",
        world_size=1,
        optimizer_name="TorchAO AdamW8bit",
        learning_rate=2e-5,
        checkpoint_every_updates=1000,
    )


def _replay() -> ReplayEvidence:
    return ReplayEvidence(
        parent_checkpoint_id="parent",
        parent_successful_update=10,
        completed_shards=("release/a.tar",),
        active_shard="release/b.tar",
        replayed_shards=1,
        replayed_samples=17,
    )


def _matrix() -> FaultMatrixReport:
    control = _control()
    outcomes = [
        FaultOutcome(
            scenario=scenario,
            status=FaultStatus.PASSED,
            hardware_level=HardwareLevel.CPU,
            failure_type="InjectedFailure",
            control_before=control,
            control_after=control,
            replay=None,
            evidence_file=f"{scenario.value}.json",
            evidence_sha256=_HASH,
            blockers=(),
        )
        for scenario in CPU_SCENARIOS
    ]
    outcomes.extend(
        FaultOutcome(
            scenario=scenario,
            status=FaultStatus.PASSED,
            hardware_level=HardwareLevel.ONE_GPU,
            failure_type="InjectedFailure",
            control_before=control,
            control_after=control,
            replay=_replay(),
            evidence_file=f"{scenario.value}.json",
            evidence_sha256=_HASH,
            blockers=(),
        )
        for scenario in ONE_GPU_SCENARIOS
    )
    outcomes.extend(
        FaultOutcome(
            scenario=scenario,
            status=FaultStatus.BLOCKED,
            hardware_level=HardwareLevel.FOUR_GPU,
            failure_type=None,
            control_before=None,
            control_after=None,
            replay=None,
            evidence_file=None,
            evidence_sha256=None,
            blockers=("FOUR-GPU-AVAILABLE",),
        )
        for scenario in FOUR_GPU_SCENARIOS
    )
    return FaultMatrixReport("T054", tuple(outcomes))


@pytest.mark.parametrize(
    "scenario",
    [
        FaultScenario.MICROBATCH_SIGKILL,
        FaultScenario.OPTIMIZER_SIGKILL,
        FaultScenario.CHECKPOINT_SIGKILL,
    ],
)
def test_driver_sends_real_sigkill_after_inherited_pipe_barrier(
    scenario: FaultScenario,
) -> None:
    script = (
        f"import sys; sys.path.insert(0, {_SOURCE_ROOT.as_posix()!r}); import time; "
        "from sakuramoon.fault_injection import signal_ready_from_environment; "
        "signal_ready_from_environment(); time.sleep(30)"
    )

    evidence = run_until_ready_and_sigkill(
        (sys.executable, "-c", script), scenario=scenario, timeout_seconds=5.0
    )

    assert evidence.ready_observed is True
    assert evidence.returncode == -signal.SIGKILL
    assert evidence.duration_seconds < 5.0


def test_driver_times_out_and_reaps_worker_without_a_barrier() -> None:
    with pytest.raises(FaultProcessError, match="readiness"):
        run_until_ready_and_sigkill(
            (sys.executable, "-c", "import time; time.sleep(30)"),
            scenario=FaultScenario.MICROBATCH_SIGKILL,
            timeout_seconds=0.1,
        )


def test_expected_exit_driver_is_bounded_and_uses_credential_free_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MODELSCOPE_API_TOKEN", "test-only-sentinel")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test-only-sentinel")
    script = (
        "import os; forbidden=('MODELSCOPE_API_TOKEN','AWS_SECRET_ACCESS_KEY'); "
        "raise SystemExit(74 if any(name in os.environ for name in forbidden) else 73)"
    )
    evidence = run_expected_exit(
        (sys.executable, "-c", script),
        scenario=FaultScenario.CUDA_OOM,
        expected_returncode=73,
        timeout_seconds=5.0,
    )

    assert evidence.returncode == 73
    assert evidence.ready_observed is False
    assert os.environ["MODELSCOPE_API_TOKEN"] == "test-only-sentinel"
    with pytest.raises(ValueError, match="expected-exit"):
        run_expected_exit(
            (sys.executable, "-c", "raise SystemExit(73)"),
            scenario=FaultScenario.NCCL_RANK_FAILURE,
            expected_returncode=73,
            timeout_seconds=5.0,
        )


def test_matrix_requires_identical_controls_and_explicit_four_gpu_blocks() -> None:
    matrix = _matrix()
    payload = matrix.to_dict()

    assert payload["status"] == "cpu_single_gpu_complete_four_gpu_blocked"
    assert len(matrix.outcomes) == 17
    changed = replace(_control(), accumulation_steps=8)
    with pytest.raises(ValueError, match="invariant"):
        FaultOutcome(
            scenario=FaultScenario.NONFINITE_LOSS,
            status=FaultStatus.PASSED,
            hardware_level=HardwareLevel.CPU,
            failure_type="FloatingPointError",
            control_before=_control(),
            control_after=changed,
            replay=None,
            evidence_file="nonfinite_loss.json",
            evidence_sha256=_HASH,
            blockers=(),
        )


def test_runner_binds_every_executed_scenario_before_matrix_publication(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "scenarios"
    evidence_root.mkdir()
    test_report = tmp_path / "test-report.json"
    test_report.write_bytes(b'{"status":"passed"}\n')
    report_sha256 = hashlib.sha256(test_report.read_bytes()).hexdigest()
    for scenario in CPU_SCENARIOS + ONE_GPU_SCENARIOS:
        hardware = (
            HardwareLevel.CPU
            if scenario in CPU_SCENARIOS
            else HardwareLevel.ONE_GPU
        )
        evidence = ExecutedFaultEvidence(
            scenario=scenario,
            hardware_level=hardware,
            failure_type="InjectedFailure",
            control_before=_control(),
            control_after=_control(),
            replay=None if scenario in CPU_SCENARIOS else _replay(),
            test_selector=f"tests/fault_injection::{scenario.value}",
            test_report_sha256=report_sha256,
        )
        path = evidence_root / f"{scenario.value}.json"
        write_executed_fault_evidence(path, evidence)
        assert load_executed_fault_evidence(path) == evidence

    output = tmp_path / "fault-matrix.json"
    report = publish_fault_matrix_from_evidence(
        evidence_root, test_report, output
    )

    assert output.is_file()
    assert tuple(item.scenario for item in report.outcomes) == (
        CPU_SCENARIOS + ONE_GPU_SCENARIOS + FOUR_GPU_SCENARIOS
    )
    assert all(
        item.evidence_sha256 is not None
        for item in report.outcomes[: len(CPU_SCENARIOS) + len(ONE_GPU_SCENARIOS)]
    )
    first_path = evidence_root / f"{CPU_SCENARIOS[0].value}.json"
    assert report.outcomes[0].evidence_sha256 == hashlib.sha256(
        first_path.read_bytes()
    ).hexdigest()
    assert all(
        item.status is FaultStatus.BLOCKED
        for item in report.outcomes[-len(FOUR_GPU_SCENARIOS) :]
    )


def test_runner_rejects_missing_or_mislabeled_scenario_evidence(
    tmp_path: Path,
) -> None:
    evidence_root = tmp_path / "scenarios"
    evidence_root.mkdir()
    first = CPU_SCENARIOS[0]
    evidence = ExecutedFaultEvidence(
        scenario=first,
        hardware_level=HardwareLevel.CPU,
        failure_type="InjectedFailure",
        control_before=_control(),
        control_after=_control(),
        replay=None,
        test_selector="tests/fault_injection::download_interruption",
        test_report_sha256=_HASH,
    )
    wrong_path = evidence_root / f"{CPU_SCENARIOS[1].value}.json"
    wrong_path.write_text(json.dumps(evidence.to_dict()))
    with pytest.raises(ValueError, match="filename"):
        load_executed_fault_evidence(wrong_path)
    with pytest.raises(ValueError, match="opened"):
        publish_fault_matrix_from_evidence(
            evidence_root,
            tmp_path / "missing-test-report.json",
            tmp_path / "fault-matrix.json",
        )


def test_fault_matrix_publication_is_durable_no_clobber_and_enospc_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "fault-matrix.json"
    write_fault_matrix(destination, _matrix())

    assert json.loads(destination.read_bytes())["task_id"] == "T054"
    with pytest.raises(FileExistsError):
        write_fault_matrix(destination, _matrix())

    enospc_destination = tmp_path / "enospc.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError(errno.ENOSPC, "injected disk full")

    monkeypatch.setattr(report_module, "_fsync_file", fail_fsync)
    with pytest.raises(OSError) as captured:
        write_fault_matrix(enospc_destination, _matrix())
    assert captured.value.errno == errno.ENOSPC
    assert not enospc_destination.exists()
    assert not tuple(tmp_path.glob(".enospc.json.*.tmp"))


def test_resume_selector_requires_one_exact_complete_raw_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = CheckpointIdentity("parent", 10, _HASH, "b" * 64, "c" * 64)
    manifest = CheckpointManifest(
        CheckpointKind.RAW,
        identity,
        (FileRecord("payload", 1, "d" * 64),),
    )
    parent = tmp_path / "ckpt_10_parent"
    def discover(_root: Path) -> tuple[Path, ...]:
        return (parent,)

    def read_manifest(_path: Path) -> CheckpointManifest:
        return manifest

    monkeypatch.setattr(recovery_module, "discover_complete_checkpoints", discover)
    monkeypatch.setattr(recovery_module, "read_checkpoint_manifest", read_manifest)

    selected = select_complete_raw_parent(
        tmp_path, checkpoint_id="parent", successful_update=10
    )

    assert selected.path == parent
    assert selected.identity == identity
    with pytest.raises(CheckpointError, match="exact COMPLETE"):
        select_complete_raw_parent(
            tmp_path, checkpoint_id="parent", successful_update=9
        )
