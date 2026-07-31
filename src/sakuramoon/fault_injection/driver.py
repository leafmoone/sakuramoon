"""Bounded subprocess drivers for destructive fault-injection workers."""

from __future__ import annotations

import os
import select
import signal
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass

from sakuramoon.fault_injection.schema import FaultScenario

_READY_FD_ENVIRONMENT = "SAKURAMOON_FAULT_READY_FD"
_READY_MESSAGE = b"ready\n"
_PASSTHROUGH_ENVIRONMENT = (
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "LANG",
    "LC_ALL",
    "LD_LIBRARY_PATH",
    "LD_PRELOAD",
    "PATH",
    "PYTHONPATH",
    "TZ",
)


class FaultProcessError(RuntimeError):
    """A fault worker did not reach or preserve the requested process boundary."""


@dataclass(frozen=True, slots=True)
class ProcessFaultEvidence:
    scenario: FaultScenario
    pid: int
    returncode: int
    ready_observed: bool
    duration_seconds: float

    def __post_init__(self) -> None:
        if (
            type(self.pid) is not int
            or self.pid <= 0
            or type(self.returncode) is not int
            or type(self.duration_seconds) is not float
            or self.duration_seconds < 0.0
        ):
            raise ValueError("process fault evidence is invalid")


def _validate_command(command: Sequence[str], timeout_seconds: float) -> tuple[str, ...]:
    normalized = tuple(command)
    if (
        not normalized
        or any(not value or "\0" in value for value in normalized)
        or type(timeout_seconds) is not float
        or not 0.0 < timeout_seconds <= 60.0
    ):
        raise ValueError("fault command or timeout is invalid")
    return normalized


def _subprocess_environment(
    additions: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the fixed runtime allowlist shared by every fault worker."""

    environment = {
        name: value
        for name in _PASSTHROUGH_ENVIRONMENT
        if (value := os.environ.get(name)) is not None
    }
    if additions is not None:
        environment.update(additions)
    return environment


def signal_ready_from_environment() -> None:
    """Signal the parent barrier without exposing command output or runtime config."""

    raw_descriptor = os.environ.get(_READY_FD_ENVIRONMENT)
    if raw_descriptor is None:
        raise FaultProcessError("fault readiness descriptor is unavailable")
    try:
        descriptor = int(raw_descriptor)
    except ValueError:
        raise FaultProcessError("fault readiness descriptor is invalid") from None
    offset = 0
    while offset < len(_READY_MESSAGE):
        written = os.write(descriptor, _READY_MESSAGE[offset:])
        if written <= 0:
            raise FaultProcessError("fault readiness signal could not be written")
        offset += written
    os.close(descriptor)


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGKILL)
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            raise FaultProcessError("fault process could not be stopped") from None


def run_until_ready_and_sigkill(
    command: Sequence[str],
    *,
    scenario: FaultScenario,
    timeout_seconds: float,
) -> ProcessFaultEvidence:
    """Wait for an inherited-pipe barrier, then send a real SIGKILL."""

    if scenario not in {
        FaultScenario.DOWNLOAD_INTERRUPTION,
        FaultScenario.MICROBATCH_SIGKILL,
        FaultScenario.OPTIMIZER_SIGKILL,
        FaultScenario.CHECKPOINT_SIGKILL,
    }:
        raise ValueError("scenario does not use the SIGKILL driver")
    normalized = _validate_command(command, timeout_seconds)
    read_descriptor, write_descriptor = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    environment = _subprocess_environment(
        {_READY_FD_ENVIRONMENT: str(write_descriptor)}
    )
    try:
        process = subprocess.Popen(
            normalized,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=(write_descriptor,),
            env=environment,
            start_new_session=True,
        )
        os.close(write_descriptor)
        write_descriptor = -1
        readable, _, _ = select.select(
            [read_descriptor], [], [], timeout_seconds
        )
        if not readable:
            raise FaultProcessError("fault process did not reach its readiness barrier")
        if os.read(read_descriptor, len(_READY_MESSAGE)) != _READY_MESSAGE:
            raise FaultProcessError("fault process exited before its readiness barrier")
        os.killpg(process.pid, signal.SIGKILL)
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            raise FaultProcessError("fault process did not stop after SIGKILL") from None
        if returncode != -signal.SIGKILL:
            raise FaultProcessError("fault process did not report a SIGKILL exit")
        return ProcessFaultEvidence(
            scenario=scenario,
            pid=process.pid,
            returncode=returncode,
            ready_observed=True,
            duration_seconds=float(time.monotonic() - started),
        )
    finally:
        if process is not None:
            _stop_process(process)
        os.close(read_descriptor)
        if write_descriptor >= 0:
            os.close(write_descriptor)


def run_expected_exit(
    command: Sequence[str],
    *,
    scenario: FaultScenario,
    expected_returncode: int,
    timeout_seconds: float,
) -> ProcessFaultEvidence:
    """Run a bounded worker whose expected fault is represented by an exit code."""

    if scenario is not FaultScenario.CUDA_OOM:
        raise ValueError("scenario does not use the expected-exit driver")
    if type(expected_returncode) is not int or expected_returncode == 0:
        raise ValueError("expected fault return code must be a nonzero integer")
    normalized = _validate_command(command, timeout_seconds)
    started = time.monotonic()
    process = subprocess.Popen(
        normalized,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=True,
        env=_subprocess_environment(),
        start_new_session=True,
    )
    try:
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            raise FaultProcessError("fault process exceeded its timeout") from None
        if returncode != expected_returncode:
            raise FaultProcessError("fault process returned an unexpected status")
        return ProcessFaultEvidence(
            scenario=scenario,
            pid=process.pid,
            returncode=returncode,
            ready_observed=False,
            duration_seconds=float(time.monotonic() - started),
        )
    finally:
        _stop_process(process)


__all__ = [
    "FaultProcessError",
    "ProcessFaultEvidence",
    "run_expected_exit",
    "run_until_ready_and_sigkill",
    "signal_ready_from_environment",
]
