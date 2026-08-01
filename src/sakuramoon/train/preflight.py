"""Ordered, non-bypassable single-GPU preflight orchestration."""

from __future__ import annotations

import json
import os
import subprocess
import weakref
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from torch import nn

from sakuramoon.assets import require_local_qwen, require_local_vae
from sakuramoon.config.load import LoadedConfig
from sakuramoon.data.collate import DataLeaseClient
from sakuramoon.storage import require_training_storage
from sakuramoon.train.runtime import require_single_gpu_config

PREFLIGHT_CHECKS = (
    "resolved_config",
    "local_assets",
    "dataset_revision",
    "single_gpu_runtime",
    "storage_capacity",
    "frozen_encoders",
    "parameter_schema",
    "image_shapes",
    "text_shapes",
    "zero_update_loss",
    "optimizer_step",
    "sample",
    "checkpoint_round_trip",
)


class PreflightError(RuntimeError):
    """A mandatory preflight check failed."""


@dataclass(frozen=True, slots=True)
class PreflightCheckResult:
    name: str
    passed: bool
    error_type: str | None


@dataclass(frozen=True, slots=True)
class PreflightReport:
    schema_version: int
    hardware: str
    passed: bool
    checks: tuple[PreflightCheckResult, ...]


class AcceptedPreflight:
    """Process-local proof that every mandatory preflight check passed."""

    __slots__ = ("__weakref__", "report")

    report: PreflightReport

    def __init__(self, report: PreflightReport) -> None:
        del report
        raise TypeError("accepted preflight handles are created only by preflight")


_ACCEPTED_PREFLIGHTS: weakref.WeakSet[AcceptedPreflight] = weakref.WeakSet()


def _accepted_preflight(report: PreflightReport) -> AcceptedPreflight:
    accepted = object.__new__(AcceptedPreflight)
    accepted.report = report
    _ACCEPTED_PREFLIGHTS.add(accepted)
    return accepted


def require_accepted_preflight(value: AcceptedPreflight) -> None:
    if (
        type(value) is not AcceptedPreflight
        or value not in _ACCEPTED_PREFLIGHTS
        or value.report.passed is not True
    ):
        raise PreflightError("training requires an accepted process-local preflight")


def _write_report(report: PreflightReport, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("preflight report already exists")
    absolute = destination.absolute()
    current = Path(absolute.anchor)
    for part in absolute.parent.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ValueError("preflight report parent may not be a symlink")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("preflight temporary report already exists")
    payload = (
        json.dumps(asdict(report), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    try:
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination, follow_symlinks=False)
        directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def run_single_gpu_preflight(
    checks: Mapping[str, Callable[[], None]], destination: Path
) -> AcceptedPreflight:
    """Run every mandatory check in fixed order and stop at the first failure."""

    if set(checks) != set(PREFLIGHT_CHECKS) or len(checks) != len(PREFLIGHT_CHECKS):
        raise ValueError("preflight requires every fixed check exactly once")
    results: list[PreflightCheckResult] = []
    for name in PREFLIGHT_CHECKS:
        try:
            checks[name]()
        except Exception as exc:
            results.append(PreflightCheckResult(name, False, type(exc).__name__))
            report = PreflightReport(1, "1GPU", False, tuple(results))
            _write_report(report, destination)
            raise PreflightError(f"mandatory preflight check failed: {name}") from exc
        results.append(PreflightCheckResult(name, True, None))
    report = PreflightReport(1, "1GPU", True, tuple(results))
    _write_report(report, destination)
    return _accepted_preflight(report)


_GPU_NAME = "NVIDIA GeForce RTX 5090"
_DRIVER_VERSION = "580.105.08"
_CUDA_VERSION = "12.8"
_COMPUTE_CAPABILITY = (12, 0)
_MIN_GPU_MEMORY_MIB = 32_000
_MIN_LOGICAL_CPUS = 14
_MIN_RAM_BYTES = 120 * 1024**3
class _CudaDeviceProperties(Protocol):
    name: str
    major: int
    minor: int


def _nvidia_smi_identity() -> tuple[str, str, int]:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=10.0,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError("nvidia-smi identity check failed") from exc
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    if result.returncode != 0 or len(rows) != 1:
        raise RuntimeError("nvidia-smi did not return one healthy GPU")
    fields = [field.strip() for field in rows[0].split(",")]
    if len(fields) != 3:
        raise RuntimeError("nvidia-smi identity output is malformed")
    try:
        memory_mib = int(fields[2])
    except ValueError:
        raise RuntimeError("nvidia-smi memory output is malformed") from None
    return fields[0], fields[1], memory_mib


def _memory_identity() -> tuple[int, int]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
            name, separator, raw = line.partition(":")
            if separator and name in {"MemTotal", "SwapTotal"}:
                amount, unit = raw.split()
                if unit != "kB":
                    raise RuntimeError("host memory unit is unsupported")
                values[name] = int(amount) * 1024
    except (OSError, ValueError) as exc:
        raise RuntimeError("host memory identity is unreadable") from exc
    if set(values) != {"MemTotal", "SwapTotal"}:
        raise RuntimeError("host memory identity is incomplete")
    return values["MemTotal"], values["SwapTotal"]


def build_single_gpu_preflight_checks(
    loaded: LoadedConfig,
    *,
    repository_root: Path,
    resolved_config_path: Path,
    data_client: DataLeaseClient,
    qwen: object,
    vae: object,
    trainable_module: torch.nn.Module,
    parameter_schema: Callable[[], None],
    image_shapes: Callable[[], None],
    text_shapes: Callable[[], None],
    zero_update_loss: Callable[[], None],
    optimizer_step: Callable[[], None],
    sample: Callable[[], None],
    checkpoint_round_trip: Callable[[], None],
    checkpoint_payload_bytes: int,
) -> dict[str, Callable[[], None]]:
    """Construct the fixed production preflight categories with no bypasses."""

    if not resolved_config_path.is_file() or resolved_config_path.is_symlink():
        raise ValueError("resolved config must be an existing regular file")
    if type(checkpoint_payload_bytes) is not int or checkpoint_payload_bytes <= 0:
        raise ValueError("measured raw checkpoint bytes must be a positive integer")

    def resolved_config() -> None:
        payload = resolved_config_path.read_bytes()
        if payload != loaded.resolved_toml.encode("utf-8"):
            raise ValueError("resolved config bytes differ from loaded identity")

    def local_assets() -> None:
        require_local_qwen(repository_root)
        require_local_vae(repository_root)

    def dataset_revision() -> None:
        if data_client.identity.manifest_sha256 != loaded.config.data.manifest.sha256:
            raise ValueError("data service manifest identity differs from config")
        if (
            data_client.identity.worker_count
            != loaded.config.data.cache.persistent_workers_per_rank
        ):
            raise ValueError("data service worker topology differs from config")
        data_client.health()

    def single_gpu_runtime() -> None:
        require_single_gpu_config(loaded.config)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        if torch.cuda.device_count() != 1:
            raise RuntimeError("single-GPU preflight requires exactly one visible GPU")
        if torch.cuda.current_device() != loaded.config.evaluation.gpu_index:
            raise RuntimeError("selected evaluation GPU differs from current device")
        properties = cast(
            _CudaDeviceProperties,
            torch.cuda.get_device_properties(  # pyright: ignore[reportUnknownMemberType]
                torch.cuda.current_device()
            ),
        )
        if (
            properties.name != _GPU_NAME
            or (properties.major, properties.minor) != _COMPUTE_CAPABILITY
            or torch.version.cuda != _CUDA_VERSION
        ):
            raise RuntimeError(
                "CUDA runtime GPU identity differs from the environment lock"
            )
        name, driver, memory_mib = _nvidia_smi_identity()
        if (
            name != _GPU_NAME
            or driver != _DRIVER_VERSION
            or memory_mib < _MIN_GPU_MEMORY_MIB
        ):
            raise RuntimeError("GPU driver or memory differs from the environment lock")
        logical_cpus = os.cpu_count()
        ram_bytes, swap_bytes = _memory_identity()
        if (
            logical_cpus is None
            or logical_cpus < _MIN_LOGICAL_CPUS
            or ram_bytes < _MIN_RAM_BYTES
            or swap_bytes != 0
        ):
            raise RuntimeError("host CPU, RAM, or swap differs from the training floor")

    def storage_capacity() -> None:
        require_training_storage(
            loaded.config,
            repository_root,
            checkpoint_payload_bytes=checkpoint_payload_bytes,
        )

    def frozen_encoders() -> None:
        for encoder in (qwen, vae):
            if not isinstance(encoder, nn.Module):
                raise TypeError("frozen encoder must be a torch module")
            if getattr(encoder, "training", True):
                raise RuntimeError("frozen encoder is in training mode")
            if any(parameter.requires_grad for parameter in encoder.parameters()):
                raise RuntimeError("frozen encoder exposes trainable parameters")

    def trainable_schema() -> None:
        if not any(
            parameter.requires_grad for parameter in trainable_module.parameters()
        ):
            raise RuntimeError("trainable module has no trainable parameters")
        parameter_schema()

    return {
        "resolved_config": resolved_config,
        "local_assets": local_assets,
        "dataset_revision": dataset_revision,
        "single_gpu_runtime": single_gpu_runtime,
        "storage_capacity": storage_capacity,
        "frozen_encoders": frozen_encoders,
        "parameter_schema": trainable_schema,
        "image_shapes": image_shapes,
        "text_shapes": text_shapes,
        "zero_update_loss": zero_update_loss,
        "optimizer_step": optimizer_step,
        "sample": sample,
        "checkpoint_round_trip": checkpoint_round_trip,
    }


__all__ = [
    "PREFLIGHT_CHECKS",
    "AcceptedPreflight",
    "PreflightCheckResult",
    "PreflightError",
    "PreflightReport",
    "build_single_gpu_preflight_checks",
    "require_accepted_preflight",
    "run_single_gpu_preflight",
]
