"""Runtime environment fingerprint for performance baselines (P0).

Records the exact software/hardware environment that produced a
benchmark result.  Fields that cannot be discovered honestly are recorded
as ``None`` (JSON null) — never guessed.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import platform
import socket
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SCHEMA_VERSION = 1
_UNAVAILABLE = "unavailable"


def _module_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    return None


def _module_path(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    path = getattr(module, "__file__", None)
    if isinstance(path, str) and path:
        return path
    return None


def _dtk_identity() -> str:
    """Best-effort DTK runtime identity from environment or filesystem."""

    env_value = os.environ.get("DTK_VERSION", "").strip()
    if env_value:
        return env_value
    for candidate in (Path("/opt/dtk/VERSION"), Path("/opt/dtk/version.txt")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value.splitlines()[0].strip()
    opt = Path("/opt")
    if opt.is_dir():
        for entry in sorted(opt.iterdir()):
            if entry.is_dir() and entry.name.startswith("dtk-") and entry.name != "dtk":
                return entry.name
    return _UNAVAILABLE


@dataclass(frozen=True, slots=True)
class RuntimeFingerprint:
    """Exact software/hardware identity of one benchmark machine."""

    schema_version: int
    git_sha: str
    hostname: str
    python_version: str
    torch_version: str
    torch_hip_version: str | None
    device_name: str
    device_count: int
    device_total_memory_bytes: tuple[int, ...] | None
    dtk_identity: str
    transformers_version: str | None
    flash_attn_version: str | None
    flash_attn_path: str | None
    fla_version: str | None
    triton_version: str | None
    causal_conv1d_available: bool
    distributed_backend: str | None
    world_size: int | None
    cuda_visible_devices: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_json(self, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            self.to_dict(), ensure_ascii=True, indent=2, sort_keys=True
        )
        destination.write_text(encoded + "\n", encoding="utf-8")
        return destination


def capture_runtime_fingerprint(
    *,
    git_sha: str | None = None,
    distributed_backend: str | None = None,
    world_size: int | None = None,
    device_index: int = 0,
) -> RuntimeFingerprint:
    """Capture the machine environment without guessing unavailable fields.

    ``git_sha`` may be supplied by the caller (the benchmark harness passes
    the exact worktree HEAD); otherwise it is recorded as unavailable.
    """

    if git_sha is not None and (type(git_sha) is not str or not git_sha):
        raise ValueError("git_sha must be a non-empty string or None")
    if device_index < 0:
        raise ValueError("device_index must be non-negative")
    hip_version = getattr(torch.version, "hip", None)
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    device_name = _UNAVAILABLE
    if cuda_available and device_index < device_count:
        device_name = torch.cuda.get_device_name(device_index)
    memory: tuple[int, ...] | None = None
    if cuda_available:
        memory = tuple(
            int(torch.cuda.get_device_properties(i).total_memory)  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            for i in range(device_count)
        )
    return RuntimeFingerprint(
        schema_version=SCHEMA_VERSION,
        git_sha=git_sha if git_sha is not None else _UNAVAILABLE,
        hostname=socket.gethostname(),
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_hip_version=hip_version if isinstance(hip_version, str) else None,
        device_name=device_name,
        device_count=device_count,
        device_total_memory_bytes=memory,
        dtk_identity=_dtk_identity(),
        transformers_version=_module_version("transformers"),
        flash_attn_version=_module_version("flash_attn"),
        flash_attn_path=_module_path("flash_attn"),
        fla_version=_module_version("fla"),
        triton_version=_module_version("triton"),
        causal_conv1d_available=importlib.util.find_spec("causal_conv1d") is not None,
        distributed_backend=distributed_backend,
        world_size=world_size,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
    )


__all__ = [
    "SCHEMA_VERSION",
    "RuntimeFingerprint",
    "capture_runtime_fingerprint",
]
