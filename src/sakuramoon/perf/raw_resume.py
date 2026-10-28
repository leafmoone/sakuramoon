"""Isolated synthetic RAW continuation; no production service or publication."""
from __future__ import annotations

import json
import os
import time
import tomllib
from collections.abc import Callable, Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import torch

from sakuramoon.checkpoint.load import read_raw_checkpoint_state
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    CheckpointReason,
    RawCheckpointState,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.resolve import resolved_config_bytes
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.train import production
from sakuramoon.train.loop import SingleGpuTrainingLoop, SuccessfulLoopObservation
from sakuramoon.train.preflight import RestoredSingleGpuCheckpoint
from sakuramoon.train.runtime import require_checkpoint_resume_binding
from sakuramoon.train.stage import canonical_growth_alpha


def validate_paths(source: Path, output_root: Path) -> tuple[Path, Path]:
    """Never follow output symlinks or reuse a directory/source ancestor."""
    for candidate in (source.absolute(), output_root.absolute()):
        for part in (candidate, *candidate.parents):
            if part.is_symlink():
                raise ValueError("resume and output paths must not contain symlinks")
    source = source.resolve(strict=True)
    output_root = output_root.resolve(strict=False)
    if not source.is_dir():
        raise ValueError("resume must be a RAW checkpoint directory")
    if output_root.exists():
        raise FileExistsError("output root must be a new isolated directory")
    if source.is_relative_to(output_root) or output_root.is_relative_to(source):
        raise ValueError("source and output root must not overlap")
    if "output_model" not in output_root.parts:
        raise ValueError("synthetic RAW outputs must be under an output_model directory")
    return source, output_root


def validate_resume_config(loaded: LoadedConfig, source: Path, updates: int) -> RawCheckpointState:
    """Narrow supported continuation, with no hidden batch/LR/topology rewrite."""
    if type(updates) is not int or updates <= 0:
        raise ValueError("updates must be positive")
    config = loaded.config
    _, state = read_raw_checkpoint_state(source)
    architecture = json.loads((source / "model" / "config.json").read_text())["architecture"]
    if architecture.get("training_auxiliaries"):
        raise ValueError("synthetic RAW continuation does not support source training auxiliaries")
    old = tomllib.loads((source / "resolved_config.toml").read_text())
    current = tomllib.loads(loaded.resolved_toml)
    if (config.irepa is not None and config.irepa.enabled) or bool(old.get("irepa", {}).get("enabled", False)):
        raise ValueError("synthetic RAW continuation does not support iREPA; anchor/teacher integration is required")
    if state.growth.alpha != 1.0 or canonical_growth_alpha(state.growth, state.trainer.successful_updates) != 1.0 or canonical_growth_alpha(state.growth, state.trainer.successful_updates + 1) != 1.0:
        raise ValueError("synthetic RAW continuation requires a mature checkpoint without active growth")
    if tuple(config.model.dit.active_slot_ids or ()) != state.growth.active_slot_ids:
        raise ValueError("synthetic RAW continuation does not permit growth/topology cutover")
    for section in ("optimizer", "scheduler", "gradient", "timestep", "objective", "distributed", "growth"):
        if current.get(section) != old.get(section):
            raise ValueError(f"synthetic RAW continuation requires unchanged {section} config")
    for field in ("global_batch", "local_batch", "accumulation", "resolution"):
        if current["train"].get(field) != old["train"].get(field):
            raise ValueError(f"synthetic RAW continuation requires unchanged train.{field}")
    if config.model.dit.final_output_norm != "none":
        raise ValueError("norm-removal validation requires model.dit.final_output_norm=none")
    for field in ("attention_backend", "qwen_attention_backend"):
        if current["kernels"].get(field) != old["kernels"].get(field):
            raise ValueError(f"synthetic RAW continuation requires unchanged kernels.{field}")
    if current["train"]["activation_checkpoint_mode"] != old["train"]["activation_checkpoint_mode"]:
        raise ValueError("synthetic RAW continuation requires unchanged activation checkpoint mode")
    if config.run.seed != old["run"]["seed"]:
        raise ValueError("synthetic RAW continuation requires unchanged run.seed")
    if config.checkpoint.full_every_updates != state.checkpoint_cadence.every_successful_updates:
        raise ValueError("synthetic RAW continuation requires unchanged checkpoint cadence")
    if config.distributed.world_size != state.growth.world_size:
        raise ValueError("synthetic RAW continuation requires unchanged checkpoint world size")
    if config.train.max_updates < state.trainer.successful_updates + updates:
        raise ValueError("config train.max_updates does not allow requested continuation")
    forensic = config.optimizer.cmuon_forensic
    if forensic is not None and forensic.enabled:
        raise ValueError("external forensic outputs are unsupported in isolated continuation")
    if os.environ.get("SAKURAMOON_CONDITION_SPIKE_LOG"):
        raise ValueError("unset SAKURAMOON_CONDITION_SPIKE_LOG for isolated continuation")
    return state


def isolate_optimizer_diagnostics(optimizer: Any, output_root: Path) -> None:
    """Redirect only documented failure-output destinations, never optimizer math."""
    os.environ["SAKURAMOON_FORENSIC_DIR"] = str(output_root / "diagnostics" / "guard")
    for attribute, directory in (
        ("hard_fail_artifact_root", "hard-fail"),
        ("legacy_forensic_dir", "optimizer"),
        ("emergency_capsule_root", "emergency"),
    ):
        if hasattr(optimizer, attribute):
            setattr(optimizer, attribute, output_root / "diagnostics" / directory)


def advance_synthetic(
    *, loaded: LoadedConfig, restored: RestoredSingleGpuCheckpoint,
    runtime: Any, batches: Iterable[Any], updates: int, output_root: Path,
    checkpoint_id: str, rank: int = 0, world_size: int = 1,
    backward: Callable[[torch.Tensor], None] | None = None,
    no_sync: Callable[[], Any] | None = None,
    barrier: Callable[[], None] = lambda: None,
) -> Path:
    """Run real optimizer steps from restored state, then atomically save once.

    The caller owns hardware/model/data assembly. Tests inject a tiny CPU loss;
    the CLI supplies the existing Qwen/VAE/objective runtime and synthetic RGB.
    """
    config: RuntimeConfig = loaded.config
    raw = restored.state
    require_checkpoint_resume_binding(config, raw, runtime_growth_alpha=runtime.growth_alpha)
    if raw.trainer.successful_updates + updates > config.train.max_updates:
        raise ValueError("requested update exceeds configured terminal")
    scheduler = production._SuccessfulUpdateLrScheduler(  # pyright: ignore[reportPrivateUsage]
        config, restored.optimizer, production._config_learning_rate,  # pyright: ignore[reportPrivateUsage]
        restored_successful_update=raw.trainer.successful_updates, fresh=False,
    )
    next_update = raw.trainer.successful_updates + 1
    def start(_timer: Any) -> None:
        runtime.set_growth_alpha(canonical_growth_alpha(raw.growth, next_update))
    def loss(batch: Any, microbatch: int) -> torch.Tensor:
        return runtime.measure(batch, update_number=next_update, is_final_microbatch=microbatch == config.train.accumulation - 1).per_sample_loss
    def observe(event: SuccessfulLoopObservation) -> None:
        nonlocal next_update
        next_update = event.update.state.successful_updates + 1
        if rank == 0:
            print(f"synthetic update={event.update.state.successful_updates} loss={float(event.update.mean_loss):.8g} grad_norm={float(event.update.clip.pre_clip_norm):.8g}", flush=True)
    loop = SingleGpuTrainingLoop(
        module=restored.module, optimizer=restored.optimizer, loss_fn=loss,
        accumulation_steps=config.train.accumulation,
        target_successful_updates=raw.trainer.successful_updates + updates,
        checkpoint_every_successful_updates=raw.checkpoint_cadence.every_successful_updates,
        scheduler_step=scheduler, checkpoint=lambda _update: None,
        diagnostic_root=output_root / "diagnostics", failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
        state=raw.trainer, update_started=start, successful_update_observer=observe,
        effective_sample_multiplier=world_size,
        growth_alpha_for_update=lambda update: canonical_growth_alpha(raw.growth, update),
        backward=backward, no_sync=no_sync, global_clip_norm=config.gradient.clip_norm,
        condition_optimizer=config.optimizer.condition,
    )
    final = loop.run(batches).state
    if not bool(torch.stack([torch.isfinite(parameter).all() for parameter in restored.module.parameters()]).all().item()):
        raise FloatingPointError("synthetic continuation produced nonfinite model parameters")
    alpha = canonical_growth_alpha(raw.growth, final.successful_updates)
    cadence = raw.checkpoint_cadence.committed(
        successful_update=final.successful_updates,
        wall_clock_unix_seconds=max(time.time(), raw.checkpoint_cadence.last_wall_clock_unix_seconds),
        reason=CheckpointReason.STAGE_FINALIZE,
    )
    state = replace(raw, trainer=final, growth=replace(raw.growth, alpha=alpha), checkpoint_cadence=cadence)
    identity = CheckpointIdentity(checkpoint_id, final.successful_updates)
    expected_path = output_root / f"ckpt_{identity.update}_{identity.checkpoint_id}"
    barrier()
    if rank == 0:
        # Save-only isolation metadata: numerical config and optimizer are unchanged.
        saved_config = config.model_copy(update={
            "run": config.run.model_copy(update={"run_id": f"synthetic-{checkpoint_id}", "label": "synthetic RAW continuation validation"}),
            "paths": config.paths.model_copy(update={
                "run_dir": str(output_root / "runs"),
                "cache_dir": str(output_root / "cache"),
                "checkpoint_dir": str(output_root),
                "artifact_dir": str(output_root / "artifacts"),
            }),
        })
        result = save_raw_checkpoint(output_root, identity, restored.module, restored.optimizer, state, resolved_config=resolved_config_bytes(saved_config))
        print("saved config marked synthetic; run identity and output paths isolated, numerical config unchanged", flush=True)
        if result.path != expected_path:
            raise RuntimeError("saved RAW path differs from planned destination")
        print(f"synthetic RAW saved: {result.path}", flush=True)
        print(f"manifest: {result.path / 'manifest.json'}", flush=True)
    barrier()
    return expected_path
