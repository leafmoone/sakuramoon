from __future__ import annotations

from collections.abc import Iterator
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.schema import StageBudgetCheckpointState
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.train.preflight import (
    PREFLIGHT_CHECKS,
    AcceptedPreflight,
    PreflightError,
    run_single_gpu_preflight,
)
from sakuramoon.train.runtime import (
    SingleGpuBatchRuntime,
    require_single_gpu_config,
    run_single_gpu_training,
)
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite


class _FakeQwen(torch.nn.Module):
    def forward(self, _input_ids: torch.Tensor, _mask: torch.Tensor) -> object:
        return object()


class _FakeVae(torch.nn.Module):
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return image


class _SgdAdapter:
    def __init__(self, parameters: Iterator[torch.nn.Parameter]) -> None:
        self.optimizer = torch.optim.SGD(tuple(parameters), lr=0.01)

    def step(self) -> None:
        self.optimizer.step()  # pyright: ignore[reportUnknownMemberType]

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.optimizer.zero_grad(set_to_none=set_to_none)


def _accepted(tmp_path: Path) -> AcceptedPreflight:
    return run_single_gpu_preflight(
        {name: (lambda: None) for name in PREFLIGHT_CHECKS},
        tmp_path / "accepted-preflight.json",
    )


def test_single_gpu_config_requires_native_s0() -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
    )
    require_single_gpu_config(cast(RuntimeConfig, config))

    changed = SimpleNamespace(
        run=SimpleNamespace(stage="S1"),
        stage=SimpleNamespace(world_size=4),
        distributed=SimpleNamespace(backend="ddp", world_size=4),
        failure=SimpleNamespace(allow_force_bypass=False),
    )
    with pytest.raises(ValueError, match="topology"):
        require_single_gpu_config(cast(RuntimeConfig, changed))


def test_training_boundary_accepts_only_d025_preassembled_batches() -> None:
    parameters = signature(run_single_gpu_training).parameters

    assert "batches" in parameters
    assert {
        "batch_size",
        "worker_count",
        "ready_batches",
        "pin_memory",
        "drop_last",
        "pipeline",
        "data_client",
    }.isdisjoint(parameters)


def test_training_rejects_forged_preflight_before_runtime_use(tmp_path: Path) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1, accumulation=1, planned_updates=1),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
        checkpoint=SimpleNamespace(full_every_updates=1000),
    )
    forged = object.__new__(AcceptedPreflight)
    with pytest.raises(PreflightError, match="process-local"):
        run_single_gpu_training(
            cast(RuntimeConfig, config),
            preflight=forged,
            runtime=cast(SingleGpuBatchRuntime, object()),
            module=torch.nn.ParameterList(),
            optimizer=cast(Any, object()),
            batches=iter(()),
            scheduler_step=lambda _update: None,
            checkpoint=lambda _update: None,
            diagnostic_root=tmp_path,
            failure_id=lambda phase, _state: phase,
            state=SingleGpuUpdateState.initial(),
            stage_budget=StageBudgetCheckpointState(0, 1),
            cadence=CheckpointCadence(0, 0.0),
        )


def test_training_rejects_checkpoint_cadence_state_drift(tmp_path: Path) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1, accumulation=1, planned_updates=5),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
        checkpoint=SimpleNamespace(full_every_updates=1000),
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    module = torch.nn.ParameterList([parameter])

    class _Runtime:
        composite = module

    with pytest.raises(ValueError, match="does not match trainer state"):
        run_single_gpu_training(
            cast(RuntimeConfig, config),
            preflight=_accepted(tmp_path),
            runtime=cast(SingleGpuBatchRuntime, _Runtime()),
            module=module,
            optimizer=cast(Any, object()),
            batches=iter(()),
            scheduler_step=lambda _update: None,
            checkpoint=lambda _update: None,
            diagnostic_root=tmp_path,
            failure_id=lambda phase, _state: phase,
            state=SingleGpuUpdateState(4, 3, 3),
            stage_budget=StageBudgetCheckpointState(0, 5),
            cadence=CheckpointCadence(2, 0.0),
        )


def test_mid_stage_resume_stops_at_original_absolute_budget(tmp_path: Path) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1, accumulation=1, planned_updates=5),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
        checkpoint=SimpleNamespace(full_every_updates=1000),
    )
    module = torch.nn.Linear(1, 1, bias=False)

    class _Runtime:
        composite = module

        @staticmethod
        def measure(batch: torch.Tensor) -> object:
            return SimpleNamespace(per_sample_loss=module(batch).square().reshape(1))

    result = run_single_gpu_training(
        cast(RuntimeConfig, config),
        preflight=_accepted(tmp_path),
        runtime=cast(SingleGpuBatchRuntime, _Runtime()),
        module=module,
        optimizer=_SgdAdapter(iter(module.parameters())),
        batches=cast(Iterator[TrainingBatch], iter(torch.ones(3, 1, 1))),
        scheduler_step=lambda _update: None,
        checkpoint=lambda _update: None,
        diagnostic_root=tmp_path,
        failure_id=lambda phase, _state: phase,
        state=SingleGpuUpdateState(3, 3, 3),
        stage_budget=StageBudgetCheckpointState(0, 5),
        cadence=CheckpointCadence(3, 0.0),
        clock=lambda: 1.0,
    )

    assert result.state == SingleGpuUpdateState(5, 5, 5)


@pytest.mark.parametrize(
    ("stage_budget", "message"),
    [
        (StageBudgetCheckpointState(4, 9), "trainer state"),
        (StageBudgetCheckpointState(0, 6), "resolved config"),
    ],
)
def test_training_rejects_restored_stage_budget_drift(
    tmp_path: Path,
    stage_budget: StageBudgetCheckpointState,
    message: str,
) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1, accumulation=1, planned_updates=5),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
        checkpoint=SimpleNamespace(full_every_updates=1000),
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    module = torch.nn.ParameterList([parameter])

    class _Runtime:
        composite = module

    with pytest.raises(ValueError, match=message):
        run_single_gpu_training(
            cast(RuntimeConfig, config),
            preflight=_accepted(tmp_path),
            runtime=cast(SingleGpuBatchRuntime, _Runtime()),
            module=module,
            optimizer=cast(Any, object()),
            batches=iter(()),
            scheduler_step=lambda _update: None,
            checkpoint=lambda _update: None,
            diagnostic_root=tmp_path,
            failure_id=lambda phase, _state: phase,
            state=SingleGpuUpdateState(3, 3, 3),
            stage_budget=stage_budget,
            cadence=CheckpointCadence(3, 0.0),
        )


def test_runtime_publishes_exact_proposed_cadence_before_commit(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1, accumulation=1, planned_updates=1),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
        checkpoint=SimpleNamespace(full_every_updates=1000),
    )
    module = torch.nn.Linear(1, 1, bias=False)

    class _Runtime:
        composite = module

        @staticmethod
        def measure(batch: torch.Tensor) -> object:
            return SimpleNamespace(per_sample_loss=module(batch).square().reshape(1))

    proposed: list[tuple[int, CheckpointReason, CheckpointCadence]] = []
    cadence = CheckpointCadence(0, 0.0)
    result = run_single_gpu_training(
        cast(RuntimeConfig, config),
        preflight=_accepted(tmp_path),
        runtime=cast(SingleGpuBatchRuntime, _Runtime()),
        module=module,
        optimizer=_SgdAdapter(iter(module.parameters())),
        batches=cast(Iterator[TrainingBatch], iter((torch.ones(1, 1),))),
        scheduler_step=lambda _update: None,
        checkpoint=lambda _update: None,
        diagnostic_root=tmp_path,
        failure_id=lambda phase, _state: phase,
        state=SingleGpuUpdateState.initial(),
        stage_budget=StageBudgetCheckpointState(0, 1),
        cadence=cadence,
        checkpoint_cadence_event=lambda update, reason, next_cadence: proposed.append(
            (update, reason, next_cadence)
        ),
        clock=lambda: 6.0 * 3600.0,
    )

    expected = CheckpointCadence(1, 6.0 * 3600.0)
    assert proposed == [(1, CheckpointReason.WALL_CADENCE, expected)]
    assert result.cadence == expected
    assert cadence == CheckpointCadence(0, 0.0)


def test_runtime_failed_cadence_publication_keeps_restored_anchor(
    tmp_path: Path,
) -> None:
    config = SimpleNamespace(
        run=SimpleNamespace(stage="S0"),
        stage=SimpleNamespace(world_size=1, accumulation=1, planned_updates=1),
        distributed=SimpleNamespace(backend="native", world_size=1),
        failure=SimpleNamespace(allow_force_bypass=False),
        checkpoint=SimpleNamespace(full_every_updates=1000),
    )
    module = torch.nn.Linear(1, 1, bias=False)

    class _Runtime:
        composite = module

        @staticmethod
        def measure(batch: torch.Tensor) -> object:
            return SimpleNamespace(per_sample_loss=module(batch).square().reshape(1))

    cadence = CheckpointCadence(0, 0.0)

    def fail_save(
        _update: int,
        _reason: CheckpointReason,
        proposed: CheckpointCadence,
    ) -> None:
        assert proposed == CheckpointCadence(1, 6.0 * 3600.0)
        raise OSError("synthetic durable save failure")

    with pytest.raises(OSError, match="durable save failure"):
        run_single_gpu_training(
            cast(RuntimeConfig, config),
            preflight=_accepted(tmp_path),
            runtime=cast(SingleGpuBatchRuntime, _Runtime()),
            module=module,
            optimizer=_SgdAdapter(iter(module.parameters())),
            batches=cast(Iterator[TrainingBatch], iter((torch.ones(1, 1),))),
            scheduler_step=lambda _update: None,
            checkpoint=lambda _update: None,
            diagnostic_root=tmp_path,
            failure_id=lambda phase, _state: phase,
            state=SingleGpuUpdateState.initial(),
            stage_budget=StageBudgetCheckpointState(0, 1),
            cadence=cadence,
            checkpoint_cadence_event=fail_save,
            clock=lambda: 6.0 * 3600.0,
        )

    assert cadence == CheckpointCadence(0, 0.0)


def test_runtime_requires_checkpointed_default_cuda_generator() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    from sakuramoon.conditioning.style_resampler import StyleResampler
    from sakuramoon.conditioning.text_mixer import TextConditioner
    from sakuramoon.model.dit import PackedDiT

    composite = TrainableComposite(
        dit=PackedDiT(
            depth=16,
            input_channels=128,
            hidden_size=2560,
            intermediate_size=6912,
            q_heads=20,
            kv_heads=5,
            head_dim=128,
            rope_nope_dim=32,
            rope_y_dim=48,
            rope_x_dim=48,
            rope_position_scale=16.0,
            rope_theta=1000.0,
            norm_eps=1e-6,
            timestep_dim=256,
            size_dim=64,
            aspect_dim=64,
            condition_hidden_size=1024,
            stable_slot_count=24,
            modulation_chunks=6,
            final_modulation_size=5120,
            out_channels=128,
            modality_init_std=0.02,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
            projection_bias=False,
            attention_dropout=0.0,
            mlp_dropout=0.0,
            output_weight_zero_init=True,
            output_bias_zero_init=True,
        ).cuda(),
        text=TextConditioner(
            input_size=2048,
            adapter_size=1024,
            output_size=2560,
            groups=8,
            attention_heads=16,
            norm_eps=1e-6,
            mix_gate_init=0.0,
            layer_scale_init=1.0,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ).cuda(),
        style=StyleResampler(
            input_size=2048,
            hidden_size=1024,
            intermediate_size=2048,
            output_size=2560,
            query_count=4,
            attention_heads=16,
            norm_eps=1e-6,
            init_std=0.02,
            projection_bias=False,
            linear_dtype=torch.bfloat16,
            sensitive_dtype=torch.float32,
        ).cuda(),
    )
    fake_qwen = _FakeQwen().cuda()
    fake_vae = _FakeVae().cuda()
    with pytest.raises(ValueError, match="default CUDA generator"):
        SingleGpuBatchRuntime(
            qwen=fake_qwen,
            vae=fake_vae,
            composite=composite,
            device=torch.device("cuda"),
            generator=torch.Generator(device="cuda").manual_seed(1),
            p_mean=-0.8,
            p_std=0.8,
            noise_scale=1.0,
            t_eps=0.05,
            noise_observation_boundary=0.95,
            growth_alpha=0.0,
        )
