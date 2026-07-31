"""Tiny real-CUDA worker used only by bounded T054 engineering faults."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from sakuramoon.fault_injection.driver import signal_ready_from_environment
from sakuramoon.fault_injection.schema import TrainingControlSnapshot
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit

OOM_EXIT_CODE = 73
_CONTROL = TrainingControlSnapshot(
    resolved_config_sha256="a" * 64,
    local_batch=1,
    accumulation_steps=1,
    attention_backend="dense_sdpa",
    world_size=1,
    optimizer_name="TorchAO AdamW8bit",
    learning_rate=2e-5,
    checkpoint_every_updates=1000,
)


class _MixedModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.matrix = nn.Linear(
            64, 64, bias=False, device="cuda", dtype=torch.bfloat16
        )
        self.sensitive = nn.Parameter(torch.ones(64, device="cuda"))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.matrix(inputs.to(torch.bfloat16)).float() * self.sensitive


def _optimizer(module: nn.Module) -> IsolatedAdamW8bit:
    return build_adamw8bit(
        module,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=5400,
    )


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    body = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())


def _write_torch(path: Path, payload: dict[str, object]) -> None:
    with path.open("xb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_parent(
    workspace: Path, module: _MixedModule, optimizer: IsolatedAdamW8bit
) -> Path:
    parent = workspace / "parent_0"
    parent.mkdir(parents=True)
    _write_torch(
        parent / "state.pt",
        {"model": module.state_dict(), "optimizer": optimizer.state_dict()},
    )
    _write_json(parent / "control.json", asdict(_CONTROL))
    with (parent / "COMPLETE").open("xb") as marker:
        marker.write(b"complete\n")
        marker.flush()
        os.fsync(marker.fileno())
    _fsync_directory(parent)
    _fsync_directory(workspace)
    return parent


def _batch_loss(module: _MixedModule) -> torch.Tensor:
    inputs = torch.full((2, 64), 0.125, device="cuda")
    targets = torch.full((2, 64), -0.25, device="cuda")
    return (module(inputs) - targets).square().mean()


def _wait_for_kill() -> None:
    signal_ready_from_environment()
    time.sleep(60)
    raise RuntimeError("fault worker was not killed")


def run_kill_worker(phase: str, workspace: Path) -> None:
    if phase not in {"microbatch", "optimizer", "checkpoint"}:
        raise ValueError("single-GPU kill phase is invalid")
    if workspace.exists() or workspace.is_symlink():
        raise FileExistsError("single-GPU fault workspace already exists")
    workspace.mkdir(parents=True)
    torch.manual_seed(5401)  # pyright: ignore[reportUnknownMemberType]
    module = _MixedModule()
    optimizer = _optimizer(module)
    _publish_parent(workspace, module, optimizer)

    loss = _batch_loss(module)
    if not bool(torch.isfinite(loss).item()):
        raise FloatingPointError("engineering worker loss is nonfinite")
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    if phase == "microbatch":
        _wait_for_kill()
    if phase == "optimizer":
        original_run_step = optimizer.sr_rng.run_step

        def stop_inside_optimizer(step: object) -> object:
            _wait_for_kill()
            return original_run_step(step)

        optimizer.sr_rng.run_step = stop_inside_optimizer
        optimizer.step()
        raise RuntimeError("optimizer fault barrier returned")

    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    candidate = workspace / ".candidate_1.tmp"
    candidate.mkdir()
    _write_torch(
        candidate / "state.pt",
        {"model": module.state_dict(), "optimizer": optimizer.state_dict()},
    )
    _fsync_directory(candidate)
    _wait_for_kill()


def _load_parent(
    parent: Path, module: _MixedModule, optimizer: IsolatedAdamW8bit
) -> None:
    if parent.is_symlink() or not parent.is_dir():
        raise RuntimeError("recovery parent is not a real directory")
    if {path.name for path in parent.iterdir()} != {
        "COMPLETE",
        "control.json",
        "state.pt",
    }:
        raise RuntimeError("recovery parent file set is invalid")
    if (parent / "COMPLETE").read_bytes() != b"complete\n":
        raise RuntimeError("recovery parent is incomplete")
    control = cast(dict[str, Any], json.loads((parent / "control.json").read_bytes()))
    if control != asdict(_CONTROL):
        raise RuntimeError("recovery parent changed protected controls")
    state = cast(
        dict[str, object],
        torch.load(parent / "state.pt", map_location="cpu", weights_only=True),
    )
    module.load_state_dict(cast(dict[str, torch.Tensor], state.get("model")))
    optimizer.load_state_dict(cast(dict[str, object], state.get("optimizer")))


def run_recovery_worker(parent: Path, output: Path) -> None:
    if output.exists() or output.is_symlink():
        raise FileExistsError("single-GPU recovery evidence already exists")
    module = _MixedModule()
    optimizer = _optimizer(module)
    _load_parent(parent, module, optimizer)
    loss = _batch_loss(module)
    loss.backward()  # pyright: ignore[reportUnknownMemberType]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    audit = optimizer.audit_state()
    if not audit or any(not item.initialized or item.step != 1 for item in audit):
        raise RuntimeError("recovered optimizer did not complete exactly one update")
    _write_json(
        output,
        {
            "control_after": asdict(_CONTROL),
            "parent_checkpoint_id": parent.name,
            "successful_updates": 1,
        },
    )
    _fsync_directory(output.parent)


def run_oom_worker(parent: Path, output: Path) -> int:
    if output.exists() or output.is_symlink():
        raise FileExistsError("OOM evidence already exists")
    module = _MixedModule()
    optimizer = _optimizer(module)
    _load_parent(parent, module, optimizer)
    free_bytes, _ = torch.cuda.mem_get_info()
    requested_bytes = free_bytes + 1024**3
    try:
        torch.empty(requested_bytes // 2, device="cuda", dtype=torch.bfloat16)
    except torch.OutOfMemoryError:
        torch.cuda.empty_cache()
        _write_json(
            output,
            {
                "control_after": asdict(_CONTROL),
                "control_before": asdict(_CONTROL),
                "error_type": "OutOfMemoryError",
                "parent_checkpoint_id": parent.name,
                "parent_complete": True,
            },
        )
        _fsync_directory(output.parent)
        return OOM_EXIT_CODE
    return OOM_EXIT_CODE + 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)
    kill = subparsers.add_parser("kill")
    kill.add_argument("--phase", choices=("microbatch", "optimizer", "checkpoint"), required=True)
    kill.add_argument("--workspace", type=Path, required=True)
    recover = subparsers.add_parser("recover")
    recover.add_argument("--parent", type=Path, required=True)
    recover.add_argument("--output", type=Path, required=True)
    oom = subparsers.add_parser("oom")
    oom.add_argument("--parent", type=Path, required=True)
    oom.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    callbacks: dict[str, Callable[[], int]] = {
        "kill": lambda: (run_kill_worker(args.phase, args.workspace), 0)[1],
        "recover": lambda: (run_recovery_worker(args.parent, args.output), 0)[1],
        "oom": lambda: run_oom_worker(args.parent, args.output),
    }
    return callbacks[args.mode]()


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "OOM_EXIT_CODE",
    "main",
    "run_kill_worker",
    "run_oom_worker",
    "run_recovery_worker",
]
