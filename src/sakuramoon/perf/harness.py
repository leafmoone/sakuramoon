"""Standalone logical-update benchmark harness (P0 observatory).

Assembles the CURRENT production components — config-driven DiT composite,
Qwen/VAE encoders, objective, optimizer, real DDP — and drives the
production :class:`~sakuramoon.train.loop.SingleGpuTrainingLoop` over
deterministic in-memory synthetic batches.

Isolation rules (P0 boundary):

* NO production DataService, NO production checkpoint, NO W&B, NO
  publisher, NO evaluation, NO persistent model save.
* The loop is built with ``cadence=None`` and a no-op checkpoint callback,
  so no model is ever written.
* Model load, compile setup, first allocation and lazy optimizer state
  live entirely in the warmup stage; the measured stage is steady state.
* ``growth_alpha`` is held at the ramp-complete steady value 1.0 (the
  canonical baseline of a mature G1 run); this is a benchmark assumption,
  documented in the report, not a production behavior change.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from sakuramoon.config import load_config
from sakuramoon.config.assembly import build_trainable_composite_from_config
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
)
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.model.growth import new_slot_ids as growth_new_slot_ids
from sakuramoon.perf.sample import PerformanceSample
from sakuramoon.perf.synthetic import SyntheticBatchSource
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train import production
from sakuramoon.train.loop import SingleGpuTrainingLoop, SuccessfulLoopObservation
from sakuramoon.train.runtime import (
    RuntimeMeasurement,
    SingleGpuBatchRuntime,
    compile_packed_dit_blocks,
    require_distributed_forward_module,
    require_train_topology,
)
from sakuramoon.train.step import SingleGpuUpdateState, StepOptimizer

BENCHMARK_GROWTH_ALPHA = 1.0
_NO_CHECKPOINT_INTERVAL = 10**9
_RANK_SEED_STRIDE = 1_000_003


def git_head_sha(repository_root: Path) -> str:
    """Exact worktree HEAD, or "unavailable" when git is not readable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    value = result.stdout.strip()
    return value if len(value) == 40 else "unavailable"


def single_rank_benchmark_config(config: RuntimeConfig) -> RuntimeConfig:
    """Derive the in-memory benchmark-only single-rank variant of a config.

    Backend/world_size become the single-process native topology and
    ``global_batch`` is re-derived as ``local_batch * accumulation * 1``.
    Everything else (model, precision, optimizer, objective, encoders) is
    preserved byte-for-byte.  This is an in-memory copy only: no config
    file is written.
    """

    distributed = config.distributed
    if distributed.world_size == 1:
        return config
    train = config.train
    return config.model_copy(
        update={
            "distributed": distributed.model_copy(
                update={"backend": "native", "world_size": 1}
            ),
            "train": train.model_copy(
                update={"global_batch": train.local_batch * train.accumulation}
            ),
        }
    )


@dataclass(frozen=True, slots=True)
class BenchmarkAssembly:
    """Everything the loop needs, assembled from production components."""

    config: RuntimeConfig
    module: nn.Module
    optimizer: StepOptimizer
    runtime: SingleGpuBatchRuntime
    scheduler: Any
    rank: int
    world_size: int
    device: torch.device
    backward: Callable[[torch.Tensor], None] | None
    no_sync: Callable[[], Any] | None
    batches: tuple[Any, ...]
    samples_per_update: int


def _topology(config: RuntimeConfig) -> tuple[int, int]:
    """(rank, world_size) from the launcher environment (native or accelerate)."""

    require_train_topology(config)
    world_size = config.distributed.world_size
    if world_size == 1:
        return 0, 1
    local_rank_env = os.environ.get("LOCAL_RANK")
    rank_env = os.environ.get("RANK")
    world_env = os.environ.get("WORLD_SIZE")
    if local_rank_env is None or rank_env is None or world_env is None:
        raise RuntimeError(
            "multi-rank benchmark requires an accelerate/torchrun launcher "
            "(LOCAL_RANK/RANK/WORLD_SIZE are unset)"
        )
    rank = int(rank_env)
    world = int(world_env)
    if world != world_size:
        raise RuntimeError(
            f"launcher world size {world} differs from resolved config {world_size}"
        )
    return rank, world


def assemble_benchmark(
    *,
    config_path: Path,
    config_root: Path,
    repository_root: Path,
    environment: Mapping[str, str] | None = None,
    seed: int = 44,
    single_rank: bool = False,
    warmup_updates: int = 5,
    measure_updates: int = 10,
    config_override: Callable[[RuntimeConfig], RuntimeConfig] | None = None,
) -> BenchmarkAssembly:
    """Build the current production stack on scratch state (no I/O beyond models).

    ``config_override`` (tests only) transforms the loaded config in memory
    AFTER the single-rank benchmark transform, e.g. to a small-model recipe.
    """

    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative int")
    if type(warmup_updates) is not int or warmup_updates <= 0:
        raise ValueError("warmup_updates must be a positive int")
    if type(measure_updates) is not int or measure_updates <= 0:
        raise ValueError("measure_updates must be a positive int")

    loaded = load_config(
        config_path,
        config_root=config_root,
        environment=dict(environment or {}),
    )
    config = (
        single_rank_benchmark_config(loaded.config) if single_rank else loaded.config
    )
    if config_override is not None:
        config = config_override(config)
    rank, world_size = _topology(config)

    accelerator: Any = None
    if world_size > 1:
        from datetime import timedelta

        # accelerate ships no py.typed in this venv: the unresolved import is
        # the established production noise class (production.py carries it),
        # silenced here so the harness adds no new pyright findings.
        from accelerate import (  # pyright: ignore[reportMissingImports]
            Accelerator,  # pyright: ignore[reportUnknownVariableType]
        )
        from accelerate.utils import (  # pyright: ignore[reportMissingImports]
            DistributedDataParallelKwargs,  # pyright: ignore[reportUnknownVariableType]
            InitProcessGroupKwargs,  # pyright: ignore[reportUnknownVariableType]
        )

        accelerator = Accelerator(  # pyright: ignore[reportUnknownVariableType]
            mixed_precision="no",
            kwargs_handlers=[
                # Condition-token encoding is data-dependent (mirrors the
                # production DDP kwargs): a rank whose microbatch has only
                # null-condition samples leaves those parameters unused.
                DistributedDataParallelKwargs(find_unused_parameters=True),
                InitProcessGroupKwargs(timeout=timedelta(minutes=15)),
            ],
        )
        if accelerator.num_processes != world_size:  # pyright: ignore[reportUnknownMemberType]
            raise RuntimeError("accelerate topology differs from resolved config")
        device = cast(torch.device, accelerator.device)  # pyright: ignore[reportUnknownMemberType]
        if device.type != "cuda":
            raise RuntimeError("benchmark requires CUDA/DCU devices")
    else:
        device = torch.device("cuda", 0)
    local_index = int(device.index or 0)

    target_slots = config.model.dit.active_slot_ids
    if target_slots is None:
        raise ValueError("model.dit did not resolve to active slots")
    new_growth_slots = growth_new_slot_ids(target_slots, ())
    module = build_trainable_composite_from_config(
        config, device=device, new_slot_ids=new_growth_slots
    )
    # The benchmark reuses the production optimizer/scheduler assembly
    # paths directly (module-private by design).
    optimizer = production._build_optimizer(  # pyright: ignore[reportPrivateUsage]
        config, module, rank=rank, world_size=world_size
    )
    qwen = load_local_qwen(
        repository_root,
        device,
        attention_backend=config.kernels.qwen_attention_backend,
    )
    vae = load_local_mage_vae(repository_root, device)

    rank_seed = seed + rank * _RANK_SEED_STRIDE
    torch.manual_seed(rank_seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.default_generators[local_index].manual_seed(rank_seed)

    backward: Callable[[torch.Tensor], None] | None = None
    no_sync: Callable[[], Any] | None = None
    forward_module: nn.Module | None = None
    if world_size > 1:
        prepared = cast(
            tuple[nn.Module, Any],
            accelerator.prepare(  # pyright: ignore[reportUnknownMemberType]
                module,
                optimizer.optimizer,  # pyright: ignore[reportUnknownMemberType]
            ),
        )
        forward_module = prepared[0]
        require_distributed_forward_module(module, forward_module)
        optimizer.optimizer = prepared[1]
        backward = cast(Callable[[torch.Tensor], None], accelerator.backward)  # pyright: ignore[reportUnknownMemberType]
        distributed_sync = require_distributed_forward_module(module, forward_module)
        no_sync = distributed_sync.no_sync

    if config.kernels.torch_compile_enabled:
        compile_packed_dit_blocks(
            module,
            backend=config.kernels.torch_compile_backend,
            mode=config.kernels.torch_compile_mode,
            dynamic=config.kernels.torch_compile_dynamic,
        )

    runtime = SingleGpuBatchRuntime(
        qwen=cast(Any, qwen.encoder),
        vae=vae,
        composite=module,
        forward_module=forward_module,
        device=device,
        generator=torch.cuda.default_generators[local_index],
        p_mean=config.timestep.p_mean,
        p_std=config.timestep.p_std,
        noise_scale=config.timestep.noise_scale,
        t_eps=config.timestep.t_eps,
        noise_observation_boundary=config.logging.noise_observation_boundary,
        growth_alpha=BENCHMARK_GROWTH_ALPHA,
        torch_compile_enabled=config.kernels.torch_compile_enabled,
        torch_compile_backend=config.kernels.torch_compile_backend,
        torch_compile_mode=config.kernels.torch_compile_mode,
        torch_compile_dynamic=config.kernels.torch_compile_dynamic,
    )

    scheduler = production._SuccessfulUpdateLrScheduler(  # pyright: ignore[reportPrivateUsage]
        config,
        optimizer,
        production._config_learning_rate,  # pyright: ignore[reportPrivateUsage]
        restored_successful_update=0,
        fresh=True,
    )

    raw_pad_token_id = qwen.tokenizer.pad_token_id  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
    if not isinstance(raw_pad_token_id, int):
        raise TypeError("Qwen tokenizer padding identity is unavailable")
    padding_token_id = raw_pad_token_id
    source = SyntheticBatchSource(
        seed=rank_seed,
        target_height=config.train.resolution,
        target_width=config.train.resolution,
        local_batch=config.train.local_batch,
        tokenizer=qwen.tokenizer,
        framing=FramingContract(
            EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, padding_token_id
        ),
    )
    total_updates = warmup_updates + measure_updates
    # Margin of one full update so loop retries never exhaust the supply.
    batch_count = (total_updates + 1) * config.train.accumulation
    batches = source.pregenerate(batch_count, pin=True)
    samples_per_update = config.train.local_batch * config.train.accumulation
    return BenchmarkAssembly(
        config=config,
        module=module,
        optimizer=optimizer,
        runtime=runtime,
        scheduler=scheduler,
        rank=rank,
        world_size=world_size,
        device=device,
        backward=backward,
        no_sync=no_sync,
        batches=batches,
        samples_per_update=samples_per_update,
    )


def _batch_iterator(batches: tuple[Any, ...]) -> Iterator[Any]:
    if not batches:
        raise ValueError("benchmark batch supply is empty")
    position = 0
    while True:
        yield batches[position % len(batches)]
        position += 1


def run_benchmark_stage(
    assembly: BenchmarkAssembly,
    *,
    state: SingleGpuUpdateState,
    target_successful_updates: int,
    diagnostic_root: Path,
    collect: list[PerformanceSample] | None = None,
    wrap_workload: Callable[[Callable[[], None]], Any] | None = None,
) -> SingleGpuUpdateState:
    """Run one benchmark stage (warmup or measured) of the production loop.

    ``collect`` receives one :class:`PerformanceSample` per successful
    update (pass a list for measured stages).  ``wrap_workload`` receives
    the zero-argument loop workload and is expected to invoke it, wrapped
    (the profiler capture uses this hook).  Returns the final update state.
    """

    if (
        type(target_successful_updates) is not int
        or target_successful_updates <= state.successful_updates
    ):
        raise ValueError("stage target must exceed the restored update")

    config = assembly.config
    device = assembly.device
    runtime = assembly.runtime
    pending: list[RuntimeMeasurement] = []
    active_timer: PhaseTimer | None = None

    def update_started(timer: Any) -> None:
        nonlocal active_timer
        if not isinstance(timer, PhaseTimer):
            raise TypeError("benchmark stage expects an active PhaseTimer")
        active_timer = timer
        runtime.set_growth_alpha(BENCHMARK_GROWTH_ALPHA)

    def measure_batch(batch: Any) -> torch.Tensor:
        timer = active_timer
        if timer is None:
            raise RuntimeError("benchmark update phase timer was not initialized")
        measurement = runtime.measure(batch, phase_timer=timer)
        pending.append(measurement)
        return measurement.per_sample_loss

    def observe(observation: SuccessfulLoopObservation) -> None:
        timer = active_timer
        if timer is None or observation.phase_timer is not timer:
            raise RuntimeError("benchmark observation phase timer identity changed")
        # finish_update synced the device (nonfinite-flag read), so every
        # event pair of this update is complete; collect_ready() therefore
        # adds no synchronization of its own.
        phases = timer.collect_ready()
        if collect is None:
            pending.clear()
            return
        measurements = tuple(pending)
        pending.clear()
        if not measurements:
            raise RuntimeError("benchmark observation lost its microbatches")
        collect.append(
            PerformanceSample(
                update=observation.update.state.successful_updates,
                wall_seconds=observation.update_wall_seconds,
                samples=len(measurements) * measurements[0].per_sample_loss.numel(),
                image_tokens=sum(item.image_tokens for item in measurements),
                text_tokens=sum(item.text_tokens for item in measurements),
                dit_forward_matmul_flops=sum(item.dit_flops for item in measurements),
                phases=dict(phases),
                memory_allocated_bytes=int(torch.cuda.memory_allocated(device)),
                memory_reserved_bytes=int(torch.cuda.memory_reserved(device)),
            )
        )

    loop = SingleGpuTrainingLoop(
        module=assembly.module,
        optimizer=assembly.optimizer,
        loss_fn=measure_batch,
        accumulation_steps=config.train.accumulation,
        target_successful_updates=target_successful_updates,
        checkpoint_every_successful_updates=_NO_CHECKPOINT_INTERVAL,
        scheduler_step=assembly.scheduler,
        checkpoint=lambda _update: None,
        diagnostic_root=diagnostic_root,
        failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
        state=state,
        phase_timer=PhaseTimer(device=device),
        update_started=update_started,
        successful_update_observer=observe,
        effective_sample_multiplier=assembly.world_size,
        growth_alpha_for_update=lambda _update: BENCHMARK_GROWTH_ALPHA,
        backward=assembly.backward,
        no_sync=assembly.no_sync,
    )

    final_state: list[SingleGpuUpdateState] = []

    def workload() -> None:
        result = loop.run(_batch_iterator(assembly.batches))
        final_state.append(result.state)

    if wrap_workload is not None:
        wrap_workload(workload)
    else:
        workload()
    if not final_state:
        raise RuntimeError("benchmark stage produced no loop state")
    return final_state[0]


def write_rank_samples(samples: Sequence[PerformanceSample], destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "rank_samples": [item.to_dict() for item in samples],
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return destination


def load_rank_samples(source: Path) -> list[PerformanceSample]:
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or not isinstance(
        payload.get("rank_samples"), list
    ):
        raise ValueError(f"rank sample file is malformed: {source}")
    return [
        PerformanceSample.from_dict(item)
        for item in cast(list[Any], payload["rank_samples"])
    ]


__all__ = [
    "BENCHMARK_GROWTH_ALPHA",
    "BenchmarkAssembly",
    "assemble_benchmark",
    "git_head_sha",
    "load_rank_samples",
    "run_benchmark_stage",
    "single_rank_benchmark_config",
    "write_rank_samples",
]
