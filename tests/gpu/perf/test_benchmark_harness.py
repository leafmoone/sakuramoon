# pyright: reportPrivateUsage=false
"""GPU smoke test for the P0 benchmark harness (small model, real encoders).

Exercises the REAL production path end to end on a scratch state:

  config load (train_s0.toml) -> in-memory small-model override
  -> build_trainable_composite_from_config -> _build_optimizer
  -> real Qwen (2B, FA2 via FA_SO_PATH) + real Mage VAE
  -> production SingleGpuTrainingLoop, two stages (1 warmup / 2 measured)

Deliberately NOT covered here: the multi-rank DDP path (covered by the
2-GPU RUN B) and torch.compile (smoke stays eager; production runs use
the configured max-autotune compile).
"""

from __future__ import annotations

import glob
import os
import sysconfig
from pathlib import Path

# Single visible device: the production topology check requires
# torch.cuda.device_count() == config.distributed.world_size (1 here).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import pytest
import torch

from sakuramoon.config import load_config
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import EvaluationDisabledConfig, RuntimeConfig
from sakuramoon.train.step import SingleGpuUpdateState

# Production Qwen loads FA2 through FA_SO_PATH only (DTK/DCU runtime).
if "FA_SO_PATH" not in os.environ:
    _pattern = os.path.join(sysconfig.get_paths()["purelib"], "flash_attn_2_cuda*.so")
    _matches = sorted(glob.glob(_pattern))
    if _matches:
        os.environ["FA_SO_PATH"] = _matches[0]

REPOSITORY_ROOT = Path(__file__).parents[3]
MODEL_ROOT = Path(
    os.environ.get("SAKURAMOON_TEST_MODEL_ROOT", "/sakuramoon-runtime/model")
)

SMALL_LOCAL_BATCH = 2
SMALL_ACCUMULATION = 2
WARMUP_UPDATES = 1
MEASURE_UPDATES = 2

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        not MODEL_ROOT.is_dir(),
        reason="local Qwen/VAE assets are required (SAKURAMOON_TEST_MODEL_ROOT)",
    ),
]


def _small_config(config: RuntimeConfig) -> RuntimeConfig:
    """Small-DiT, eager, single-rank, small-batch benchmark variant."""

    dit = config.model.dit.model_copy(
        update={
            "hidden_size": 32,
            "intermediate_size": 64,
            "q_heads": 2,
            "kv_heads": 1,
            "head_dim": 16,
        }
    )
    model = config.model.model_copy(
        update={
            "dit": dit,
            "rope": config.model.rope.model_copy(
                update={"head_dim": 16, "nope_dim": 4, "y_dim": 6, "x_dim": 6}
            ),
            "head": config.model.head.model_copy(update={"final_modulation_size": 64}),
            "text": config.model.text.model_copy(update={"output_size": 32}),
            "condition_tokens": config.model.condition_tokens.model_copy(
                update={"output_size": 32}
            ),
        }
    )
    train = config.train.model_copy(
        update={
            "local_batch": SMALL_LOCAL_BATCH,
            "accumulation": SMALL_ACCUMULATION,
            "global_batch": SMALL_LOCAL_BATCH * SMALL_ACCUMULATION,
        }
    )
    return config.model_copy(
        update={
            "model": model,
            "distributed": config.distributed.model_copy(
                update={"backend": "native", "world_size": 1}
            ),
            "train": train,
            "kernels": config.kernels.model_copy(
                update={
                    "attention_backend": "dense_sdpa_reference",
                    "torch_compile_enabled": False,
                    "torch_compile_dynamic": False,
                    "vae_torch_compile": False,
                }
            ),
            "optimizer": config.optimizer.model_copy(update={"name": "hybrid_cmuon"}),
            "evaluation": EvaluationDisabledConfig(enabled=False),
        }
    )


@pytest.fixture(scope="module")
def loaded_config() -> LoadedConfig:
    return load_config(
        REPOSITORY_ROOT / "config" / "train_s0.toml",
        config_root=REPOSITORY_ROOT / "config",
        environment={
            "MODELSCOPE_API_TOKEN": "placeholder-perf-smoke",
            "WANDB_API_KEY": "placeholder-perf-smoke",
        },
    )


@pytest.fixture(scope="module")
def repository_root(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("perf-repo")
    (root / "model").symlink_to(MODEL_ROOT)
    return root


@pytest.fixture(scope="module")
def assembly(
    loaded_config: LoadedConfig,
    repository_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    from sakuramoon.perf.harness import assemble_benchmark

    return assemble_benchmark(
        config_path=REPOSITORY_ROOT / "config" / "train_s0.toml",
        config_root=REPOSITORY_ROOT / "config",
        repository_root=repository_root,
        environment={
            "MODELSCOPE_API_TOKEN": "placeholder-perf-smoke",
            "WANDB_API_KEY": "placeholder-perf-smoke",
        },
        seed=44,
        single_rank=True,
        warmup_updates=WARMUP_UPDATES,
        measure_updates=MEASURE_UPDATES,
        config_override=_small_config,
    )


def test_single_rank_benchmark_config_rederives_global_batch(
    loaded_config: LoadedConfig,
) -> None:
    from sakuramoon.perf.harness import single_rank_benchmark_config

    config = single_rank_benchmark_config(loaded_config.config)
    assert config.distributed.backend == "native"
    assert config.distributed.world_size == 1
    assert config.train.global_batch == (
        loaded_config.config.train.local_batch * loaded_config.config.train.accumulation
    )
    # Model/optimizer/encoders must be preserved byte-for-byte.
    assert config.model == loaded_config.config.model
    assert config.optimizer == loaded_config.config.optimizer


def test_full_harness_two_stage_run(assembly) -> None:
    import tempfile

    from sakuramoon.perf.harness import (
        BENCHMARK_GROWTH_ALPHA,
        run_benchmark_stage,
    )
    from sakuramoon.perf.summary import summarize

    with tempfile.TemporaryDirectory(prefix="perf-diag-") as workdir:
        diag = Path(workdir)
        warm_state = run_benchmark_stage(
            assembly,
            state=SingleGpuUpdateState.initial(),
            target_successful_updates=WARMUP_UPDATES,
            diagnostic_root=diag,
        )
        assert warm_state.successful_updates == WARMUP_UPDATES

        measured: list = []
        final_state = run_benchmark_stage(
            assembly,
            state=warm_state,
            target_successful_updates=WARMUP_UPDATES + MEASURE_UPDATES,
            diagnostic_root=diag,
            collect=measured,
            reset_peak_memory_per_update=True,
        )

    assert final_state.successful_updates == WARMUP_UPDATES + MEASURE_UPDATES
    assert len(measured) == MEASURE_UPDATES
    assert [sample.update for sample in measured] == [2, 3]

    for sample in measured:
        assert sample.wall_seconds > 0.0
        assert sample.samples == SMALL_LOCAL_BATCH * SMALL_ACCUMULATION
        # 256x256 bucket -> 16x16 image tokens per sample.
        assert sample.image_tokens == sample.samples * 16 * 16
        assert sample.text_tokens > 0
        assert sample.dit_forward_matmul_flops > 0
        for phase in ("qwen", "vae", "dit_forward", "backward", "optimizer"):
            assert phase in sample.phases, f"missing measured phase {phase}"
            assert sample.phases[phase] > 0.0
        assert sample.memory_allocated_bytes > 0
        assert sample.memory_reserved_bytes >= sample.memory_allocated_bytes
        # True per-update peak counters: peak >= final always holds.
        assert sample.peak_memory_allocated_bytes >= sample.memory_allocated_bytes
        assert sample.peak_memory_reserved_bytes >= sample.memory_reserved_bytes

    summary = summarize(
        {assembly.rank: measured},
        warmup_iterations=WARMUP_UPDATES,
        world_size=assembly.world_size,
    )
    assert summary.measured_iterations == MEASURE_UPDATES
    assert summary.global_samples_per_second > 0.0
    assert summary.phase_seconds["dit_forward"]["share_of_step"] is not None
    memory = summary.memory["per_rank"][assembly.rank]
    assert memory["peak_allocated_bytes"] >= memory["final_allocated_bytes"]
    assert memory["peak_reserved_bytes"] >= memory["final_reserved_bytes"]
    assert assembly.runtime.growth_alpha == BENCHMARK_GROWTH_ALPHA

    # Separate profiler scratch stage (post-baseline): it must execute an
    # update (wrap called, state advances) WITHOUT appending to the
    # measured baseline.  The production LR scheduler requires consecutive
    # update ids, so the scratch stage continues the SAME sequence
    # (update 4) instead of restarting.
    wrap_calls = {"count": 0}

    def wrap_scratch(workload) -> None:
        wrap_calls["count"] += 1
        workload()

    with tempfile.TemporaryDirectory(prefix="perf-diag-scratch-") as scratch_dir:
        scratch_state = run_benchmark_stage(
            assembly,
            state=final_state,
            target_successful_updates=final_state.successful_updates + 1,
            diagnostic_root=Path(scratch_dir),
            wrap_workload=wrap_scratch,
        )
    assert scratch_state.successful_updates == final_state.successful_updates + 1
    assert wrap_calls["count"] == 1
    assert len(measured) == MEASURE_UPDATES  # measured baseline untouched


def test_peak_memory_counters_differ_from_current_memory(assembly) -> None:
    """Lock the peak/current distinction with a real transient allocation.

    Reset the peak counters, transiently allocate well above the final
    resident allocation, free it, and prove that
    ``max_memory_allocated`` stays above ``memory_allocated`` while the
    current counter drops back down.
    """

    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats(device)
    # ~2 GiB of bf16: far above this small-model harness's resident
    # allocation, small relative to a 64 GiB device.
    transient = torch.empty(1_000_000_000, dtype=torch.bfloat16, device="cuda")
    del transient
    torch.cuda.empty_cache()
    peak = int(torch.cuda.max_memory_allocated(device))
    current = int(torch.cuda.memory_allocated(device))
    assert peak > current, (
        "peak_memory_allocated must exceed the current allocation after a "
        f"transient (peak={peak}, current={current})"
    )


def test_no_checkpoints_are_written(assembly, repository_root: Path) -> None:
    """The benchmark loop must never write into the repository checkout."""

    entries = sorted(entry.name for entry in repository_root.iterdir())
    assert entries == ["model"]  # only the asset symlink; nothing was written


@pytest.fixture(scope="module")
def alternate_shape_assembly(
    repository_root: Path,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Small config (2x2) + IN-MEMORY override to 4x1 (same 4 samples/update)."""

    from sakuramoon.perf.harness import assemble_benchmark

    return assemble_benchmark(
        config_path=REPOSITORY_ROOT / "config" / "train_s0.toml",
        config_root=REPOSITORY_ROOT / "config",
        repository_root=repository_root,
        environment={
            "MODELSCOPE_API_TOKEN": "placeholder-perf-smoke",
            "WANDB_API_KEY": "placeholder-perf-smoke",
        },
        seed=44,
        single_rank=True,
        warmup_updates=WARMUP_UPDATES,
        measure_updates=MEASURE_UPDATES,
        config_override=_small_config,
        local_batch=4,
        accumulation=1,
        expected_global_batch=SMALL_LOCAL_BATCH * SMALL_ACCUMULATION,
    )


def test_alternate_batch_shape_same_logical_population(
    assembly, alternate_shape_assembly
) -> None:
    """P1-R1A identity: 4x1 consumes the SAME sample identities per
    logical update as 2x2 (same seed, same population per rank/update)."""

    assert alternate_shape_assembly.config.train.local_batch == 4
    assert alternate_shape_assembly.config.train.accumulation == 1
    assert alternate_shape_assembly.config.train.global_batch == (
        SMALL_LOCAL_BATCH * SMALL_ACCUMULATION
    )
    assert alternate_shape_assembly.samples_per_update == (
        SMALL_LOCAL_BATCH * SMALL_ACCUMULATION
    )
    assert alternate_shape_assembly.samples_per_update == assembly.samples_per_update

    base_batches = assembly.batches
    alt_batches = alternate_shape_assembly.batches
    total_updates = WARMUP_UPDATES + MEASURE_UPDATES
    for update in range(total_updates):
        base_ids: set[int] = set()
        for microbatch in range(SMALL_ACCUMULATION):
            batch = base_batches[
                (update * SMALL_ACCUMULATION + microbatch) % len(base_batches)
            ]
            base_ids.update(int(value) for value in batch.sample_ids.tolist())
        # accumulation=1: exactly one microbatch per logical update.
        alt_batch = alt_batches[update % len(alt_batches)]
        alt_ids = {int(value) for value in alt_batch.sample_ids.tolist()}
        assert len(base_ids) == SMALL_LOCAL_BATCH * SMALL_ACCUMULATION
        assert alt_ids == base_ids


def test_alternate_batch_shape_runs_through_harness(alternate_shape_assembly) -> None:
    """One alternate shape end to end through the production loop."""

    import tempfile

    from sakuramoon.perf.harness import run_benchmark_stage

    with tempfile.TemporaryDirectory(prefix="perf-alt-shape-") as workdir:
        diag = Path(workdir)
        warm_state = run_benchmark_stage(
            alternate_shape_assembly,
            state=SingleGpuUpdateState.initial(),
            target_successful_updates=WARMUP_UPDATES,
            diagnostic_root=diag,
        )
        measured: list = []
        final_state = run_benchmark_stage(
            alternate_shape_assembly,
            state=warm_state,
            target_successful_updates=WARMUP_UPDATES + MEASURE_UPDATES,
            diagnostic_root=diag,
            collect=measured,
            reset_peak_memory_per_update=True,
        )
    assert final_state.successful_updates == WARMUP_UPDATES + MEASURE_UPDATES
    assert len(measured) == MEASURE_UPDATES
    for sample in measured:
        assert sample.samples == SMALL_LOCAL_BATCH * SMALL_ACCUMULATION
        assert sample.image_tokens == sample.samples * 16 * 16
