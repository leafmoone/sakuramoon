"""CPU integration coverage for live condition-diagnostics producer ownership."""

from __future__ import annotations

# The CUDA-only constructor/assets and batch preparation are bypassed here;
# measure, composite, objective, loop, step, activation facts and recorder are real.
# pyright: reportPrivateUsage=false
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from sakuramoon.conditioning.condition_tokens import (
    ConditionTokenEncoder,
    ConditionTokenOutput,
)
from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.model.dit import DenseDiT
from sakuramoon.telemetry.metrics import DurableJsonlSink
from sakuramoon.telemetry.timers import AnyPhaseTimer, NoopPhaseTimer
from sakuramoon.train import condition_diagnostics_stream as diagnostics
from sakuramoon.train import runtime as runtime_module
from sakuramoon.train import step as step_module
from sakuramoon.train.condition_diagnostics_stream import ConditionDiagnosticsRecorder
from sakuramoon.train.loop import SingleGpuTrainingLoop, SuccessfulLoopObservation
from sakuramoon.train.production import _construct_condition_diagnostics
from sakuramoon.train.runtime import (
    ActualDitFlopCounter,
    PreparedTrainingBatch,
    RuntimeMeasurement,
    SingleGpuBatchRuntime,
    SuccessfulTrainingObservation,
)
from sakuramoon.train.step import (
    SingleGpuUpdateState,
    TrainableComposite,
    TrainableCompositeInputs,
)


@pytest.fixture(autouse=True)
def _small_cpu_threads() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _composite() -> TrainableComposite:
    module = TrainableComposite(
        dit=DenseDiT(
            depth=1, active_slot_ids=(0,), input_channels=8, hidden_size=16,
            intermediate_size=32, q_heads=2, kv_heads=1, head_dim=8,
            rope_nope_dim=0, rope_y_dim=4, rope_x_dim=4,
            rope_position_scale=1.0, rope_theta=10.0, norm_eps=1e-6,
            timestep_dim=8, size_dim=8, aspect_dim=8, condition_hidden_size=32,
            stable_slot_count=1, modulation_chunks=6, final_modulation_size=32,
            out_channels=8, condition_token_count=2, modality_init_std=0.02,
            linear_dtype=torch.float32, sensitive_dtype=torch.float32,
            projection_bias=False, attention_dropout=0.0, mlp_dropout=0.0,
            output_weight_zero_init=True, output_bias_zero_init=True,
        ),
        text=TextConditioner(
            input_size=8, adapter_size=16, output_size=16, groups=2,
            attention_heads=2, norm_eps=1e-6, mix_gate_init=0.1,
            layer_scale_init=1.0, projection_bias=False,
            linear_dtype=torch.float32, sensitive_dtype=torch.float32,
        ),
        condition_tokens=ConditionTokenEncoder(
            input_size=8, hidden_size=16, intermediate_size=32, output_size=16,
            token_count=2, attention_heads=2, norm_eps=1e-6, init_std=0.02,
            projection_bias=False, linear_dtype=torch.float32,
            sensitive_dtype=torch.float32,
        ),
    )
    with torch.no_grad():
        module.dit.output_head.projection.weight.normal_(std=0.03)
        for name, parameter in module.dit.conditioner.named_parameters():
            if name.startswith(("condition_mlp.", "shared_block_projection.", "block_biases.")):
                parameter.normal_(std=0.03)
    return module


def _batch() -> TrainingBatch:
    # Metadata is not read by the replaced preparation/assets boundary.
    return cast(TrainingBatch, SimpleNamespace(
        images=torch.zeros(2, 3, 16, 16), target_height=16, target_width=16,
        dense_length=4, main_token_lengths=(2, 2), sample_ids=torch.tensor([11, 12]),
        captions=(), dropout_hits=None, condition_routes=None, spatial_crop=None,
        camera_viewport=None, transparent=None,
    ))


def _prepare(
    _self: SingleGpuBatchRuntime, _batch: TrainingBatch,
    *, phase_timer: AnyPhaseTimer | None = None,
) -> PreparedTrainingBatch:
    del phase_timer
    inputs = TrainableCompositeInputs(
        qwen_states=torch.randn(2, 4, 7, 8),
        main_token_indices=torch.tensor([[0, 1], [0, 1]]),
        main_mask=torch.ones(2, 2, dtype=torch.bool), main_token_lengths=(2, 2),
        condition_token_indices=torch.tensor([[2, 3], [2, 3]]),
        condition_mask=torch.ones(2, 2, dtype=torch.bool),
        use_null_condition=torch.tensor([False, True]),
        active_condition_sample_indices=torch.tensor([0]),
        latents=tuple(torch.randn(2, 8, 1, 1).unbind()),
        image_coordinates=(image_coordinates(1, 1, device=torch.device("cpu")),) * 2,
        timestep=torch.tensor([0.25, 0.75]), size_scale=torch.ones(2),
        aspect=torch.zeros(2), growth_alpha=1.0,
    )
    return PreparedTrainingBatch(
        inputs=inputs, clean_latents=tuple(torch.randn(2, 8, 1, 1).unbind()),
        states=inputs.latents,
    )


def _runtime(
    module: TrainableComposite, predicate: Callable[[int], bool] | None,
) -> SingleGpuBatchRuntime:
    runtime = SingleGpuBatchRuntime.__new__(SingleGpuBatchRuntime)
    runtime.composite = module
    runtime.forward_module = module
    runtime.device = torch.device("cpu")
    runtime.dit_flop_counter = ActualDitFlopCounter(module.dit)
    runtime.irepa_teacher = None
    runtime.irepa_schedule = None
    runtime.irepa_weight = 0.0
    runtime.t_eps = 0.05
    runtime.noise_observation_boundary = 0.95
    runtime.condition_diagnostics_every = None
    runtime.condition_diagnostics_should_record = predicate
    return runtime


@dataclass
class _Run:
    counts: list[tuple[int, ...]]
    losses: list[torch.Tensor]
    parameters: torch.Tensor
    rng: torch.Tensor
    checkpoint_updates: tuple[int, ...]
    scheduler_updates: list[int]
    records: list[dict[str, Any]]


def _train(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
    *, failure: tuple[str, Exception] | None = None,
    grad_share: bool | None = None,
) -> _Run:
    torch.manual_seed(811)  # pyright: ignore[reportUnknownMemberType]
    module = _composite()
    state_keys = tuple(module.state_dict())
    parameter_ids = tuple(id(parameter) for parameter in module.parameters())
    recorder: ConditionDiagnosticsRecorder | None = None

    def should_record(update: int) -> bool:
        return recorder is not None and recorder.should_record(update)

    runtime = _runtime(module, None if mode == "off" else should_record)
    # Six counters: fact computes, enable, actual captures, disable, union scans,
    # mandatory norm scans. All wrappers delegate to the real implementation.
    counters = [0] * 6
    counts: list[tuple[int, ...]] = []
    active_microbatch = -1
    original_facts = runtime_module.compute_condition_activation_facts
    original_conditioning = TrainableComposite.forward_conditioning
    original_enable = TrainableComposite.enable_condition_diagnostics
    original_disable = TrainableComposite.disable_condition_diagnostics
    original_norm = step_module._parameter_grad_norm
    facts_before_step: dict[int, dict[str, float | int | None]] = {}
    pre_step_weights: dict[int, torch.Tensor] = {}
    pending: list[RuntimeMeasurement] = []
    losses: list[torch.Tensor] = []
    scheduler_updates: list[int] = []
    saved_updates: list[int] = []

    def facts(
        composite: TrainableComposite, inputs: TrainableCompositeInputs,
        condition: ConditionTokenOutput,
    ) -> dict[str, float | int | None]:
        counters[0] += 1
        assert active_microbatch == 1
        assert composite.last_conditioning is condition
        assert condition.tokens.grad_fn is not None
        if failure is not None and failure[0] == "facts":
            raise failure[1]
        result = original_facts(composite, inputs, condition)
        update = loop.state.successful_updates + 1
        facts_before_step[update] = result.copy()
        pre_step_weights[update] = composite.dit.conditioner.condition_global_norm.weight.detach().clone()
        assert result["condition_global_norm_weight_rms"] == pytest.approx(
            float(pre_step_weights[update].square().mean().sqrt())
        )
        return result

    def conditioning(
        composite: TrainableComposite, inputs: TrainableCompositeInputs,
    ) -> Any:
        result = original_conditioning(composite, inputs)
        if composite.last_conditioning is not None:
            counters[2] += 1
            assert active_microbatch == 1
            assert composite.last_conditioning is result[1]
            if failure is not None and failure[0] == "forward":
                raise failure[1]
        return result

    def enable(composite: TrainableComposite) -> None:
        counters[1] += 1
        original_enable(composite)

    def disable(composite: TrainableComposite) -> None:
        counters[3] += 1
        original_disable(composite)

    def norm(
        target: nn.Module, *, predicate: Callable[[str, nn.Parameter], bool],
        device: torch.device,
    ) -> torch.Tensor:
        probe = next(target.parameters())
        # Only the optional UNION selects both a token-encoder name and a
        # conditioner norm name; none of the four mandatory scans do.
        union = predicate("condition_tokens.probe", probe) and predicate(
            "dit.conditioner.condition_global_norm.weight", probe,
        )
        counters[4 if union else 5] += 1
        return original_norm(target, predicate=predicate, device=device)

    def loss_fn(batch: TrainingBatch, microbatch: int) -> torch.Tensor:
        nonlocal active_microbatch
        active_microbatch = microbatch
        measurement = runtime.measure(
            batch, update_number=loop.state.successful_updates + 1,
            is_final_microbatch=microbatch == 1,
        )
        assert module.last_conditioning is None
        assert not module._condition_diagnostics
        assert torch.equal(measurement.per_sample_loss, measurement.main_per_sample_loss)
        assert measurement.irepa_lambda == 0.0
        assert torch.count_nonzero(measurement.irepa_per_sample_loss) == 0
        if microbatch == 0:
            assert measurement.condition_diagnostics is None
        pending.append(measurement.detached())
        return measurement.per_sample_loss

    def observe(observation: SuccessfulLoopObservation) -> None:
        update = observation.update.state.successful_updates
        assert module.last_conditioning is None
        assert observation.update.irepa_projector_grad_norm == 0.0
        if recorder is not None:
            recorder.maybe_record(SuccessfulTrainingObservation(
                loop=observation, microbatches=tuple(pending),
                phase_timer=NoopPhaseTimer(device=torch.device("cpu")),
                learning_rate=0.1, gpu_memory_allocated_bytes=0,
                gpu_memory_reserved_bytes=0,
            ))
        if pending[-1].condition_diagnostics is not None:
            assert pending[-1].condition_diagnostics == facts_before_step[update]
        losses.append(observation.update.mean_loss.detach().clone())
        pending.clear()
        counts.append(tuple(counters))
        counters[:] = [0] * 6

    path = tmp_path / f"{mode}.jsonl"
    with monkeypatch.context() as patch:
        patch.setattr(SingleGpuBatchRuntime, "prepare", _prepare)
        patch.setattr(runtime_module, "compute_condition_activation_facts", facts)
        patch.setattr(TrainableComposite, "forward_conditioning", conditioning)
        patch.setattr(TrainableComposite, "enable_condition_diagnostics", enable)
        patch.setattr(TrainableComposite, "disable_condition_diagnostics", disable)
        patch.setattr(step_module, "_parameter_grad_norm", norm)
        if failure is not None and failure[0] == "loss":
            def fail_loss(*_args: Any, **_kwargs: Any) -> Any:
                if module.last_conditioning is not None:
                    raise failure[1]
                return original_loss(*_args, **_kwargs)
            original_loss = runtime._loss
            patch.setattr(runtime, "_loss", fail_loss)
        if mode == "constructor_io":
            def fail_construct(*_args: Any, **_kwargs: Any) -> None:
                raise OSError(28, "injected constructor disk full")
            patch.setattr(diagnostics, "DurableJsonlSink", fail_construct)
        if mode not in {"off", "no_consumer"}:
            recorder = _construct_condition_diagnostics(
                build=ConditionDiagnosticsRecorder, path=path, rank=0, world_size=1,
                log_every_n=2, flush_every_records=1, slot_ids=(0,),
            )
        if mode == "closed":
            assert recorder is not None
            recorder.close()
        if mode == "write_io" or (failure is not None and failure[0] == "record"):
            def fail_write(_sink: DurableJsonlSink, _payload: Mapping[str, object]) -> None:
                if failure is not None:
                    raise failure[1]
                raise OSError(28, "injected update-2 disk full")
            patch.setattr(DurableJsonlSink, "write", fail_write)
        loop = SingleGpuTrainingLoop(
            module=module, optimizer=torch.optim.SGD(module.parameters(), lr=0.1),
            loss_fn=loss_fn, accumulation_steps=2, target_successful_updates=4,
            checkpoint_every_successful_updates=2, scheduler_step=scheduler_updates.append,
            checkpoint=saved_updates.append, diagnostic_root=tmp_path / "failures",
            failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
            state=SingleGpuUpdateState.initial(), successful_update_observer=observe,
            condition_grad_share=(
                grad_share if grad_share is not None else
                False if mode == "off" else should_record
            ),
        )
        try:
            result = loop.run((_batch(),) * 8)
        finally:
            assert module.last_conditioning is None
            assert not module._condition_diagnostics
            assert tuple(module.state_dict()) == state_keys
            assert tuple(id(parameter) for parameter in module.parameters()) == parameter_ids
            if recorder is not None:
                recorder.close()
                assert not recorder.should_record(6)
    assert saved_updates == [2, 4]
    records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    return _Run(
        counts, losses, torch.cat([p.detach().flatten() for p in module.parameters()]),
        torch.get_rng_state(), result.checkpoint_updates, scheduler_updates, records,
    )


@pytest.mark.parametrize("mode", ["enabled", "write_io", "constructor_io", "closed", "no_consumer"])
def test_live_gate_bounds_real_producer_work_and_preserves_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    baseline = _train(tmp_path, monkeypatch, "off")
    observed = _train(tmp_path, monkeypatch, mode)
    idle = (0, 0, 0, 0, 0, 4)
    due = (1, 1, 1, 1, 2, 4)
    assert baseline.counts == [idle] * 4
    assert observed.counts == (
        [idle, due, idle, due] if mode == "enabled" else
        [idle, due, idle, idle] if mode == "write_io" else [idle] * 4
    )
    assert all(torch.equal(a, b) for a, b in zip(baseline.losses, observed.losses, strict=True))
    assert torch.equal(baseline.parameters, observed.parameters)
    assert torch.equal(baseline.rng, observed.rng)
    assert observed.checkpoint_updates == baseline.checkpoint_updates == (2, 4)
    assert observed.scheduler_updates == baseline.scheduler_updates == [1, 2, 3, 4]
    assert [record["successful_update"] for record in observed.records] == (
        [2, 4] if mode == "enabled" else []
    )


@pytest.mark.parametrize("phase", ["forward", "loss", "facts", "record"])
def test_structural_failures_propagate_identity_and_release_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    original = ValueError(f"injected {phase} contract failure")
    with pytest.raises(ValueError) as caught:
        _train(tmp_path, monkeypatch, "enabled", failure=(phase, original))
    assert caught.value is original


def test_direct_boolean_grad_share_interface_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed = _train(tmp_path, monkeypatch, "off", grad_share=True)
    assert observed.counts == [(0, 0, 0, 0, 2, 4)] * 4


@pytest.mark.parametrize("gate", ["off", "disabled", "cadence", "nonfinal"])
@pytest.mark.parametrize("failure_phase", [None, "prepare", "forward"])
def test_runtime_owns_pre_enabled_capture_on_ineligible_calls(
    monkeypatch: pytest.MonkeyPatch, gate: str, failure_phase: str | None,
) -> None:
    module = _composite()
    runtime = _runtime(module, (lambda _update: False) if gate == "disabled" else None)
    if gate in {"cadence", "nonfinal"}:
        runtime.condition_diagnostics_every = 2
    module.enable_condition_diagnostics()
    # Standalone consumers may leave a real graph reference before runtime.
    module.forward_conditioning(_prepare(runtime, _batch()).inputs)
    assert module.last_conditioning is not None
    original_conditioning = TrainableComposite.forward_conditioning
    captures: list[bool] = []
    facts: list[bool] = []
    original = ValueError("ineligible runtime failure")

    def conditioning(composite: TrainableComposite, inputs: TrainableCompositeInputs) -> Any:
        result = original_conditioning(composite, inputs)
        captures.append(composite.last_conditioning is not None)
        if failure_phase == "forward":
            raise original
        return result

    def prepare(*args: Any, **kwargs: Any) -> PreparedTrainingBatch:
        if failure_phase == "prepare":
            raise original
        return _prepare(*args, **kwargs)

    def unexpected_facts(*args: Any, **kwargs: Any) -> Any:
        facts.append(True)
        raise AssertionError("ineligible activation diagnostics must not run")

    monkeypatch.setattr(SingleGpuBatchRuntime, "prepare", prepare)
    monkeypatch.setattr(TrainableComposite, "forward_conditioning", conditioning)
    monkeypatch.setattr(runtime_module, "compute_condition_activation_facts", unexpected_facts)
    if failure_phase is None:
        measurement = runtime.measure(
            _batch(), update_number=2 if gate == "nonfinal" else 1,
            is_final_microbatch=gate != "nonfinal",
        )
        assert measurement.condition_diagnostics is None
    else:
        with pytest.raises(ValueError) as caught:
            runtime.measure(
                _batch(), update_number=2 if gate == "nonfinal" else 1,
                is_final_microbatch=gate != "nonfinal",
            )
        assert caught.value is original
    assert captures == ([] if failure_phase == "prepare" else [False])
    assert facts == []
    assert module.last_conditioning is None
    assert not module.condition_diagnostics_enabled


def test_cadence_only_runtime_interface_and_missing_update_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _composite()
    runtime = _runtime(module, None)
    runtime.condition_diagnostics_every = 2
    monkeypatch.setattr(SingleGpuBatchRuntime, "prepare", _prepare)
    assert runtime.measure(_batch(), update_number=1).condition_diagnostics is None
    assert runtime.measure(_batch(), update_number=2, is_final_microbatch=False).condition_diagnostics is None
    assert runtime.measure(_batch(), update_number=2).condition_diagnostics is not None
    assert module.last_conditioning is None
    with pytest.raises(RuntimeError, match="update number"):
        runtime.measure(_batch())
    assert module.last_conditioning is None
