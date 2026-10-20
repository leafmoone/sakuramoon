"""Real CPU dual-stream training; NOT native TorchAO/SR or checkpoint restore.

Only CUDA AdamW construction/SR and frozen-asset preparation are adapted. The
precision audit, routing, group split, clipping, NS, momentum and guards are real.
"""
from __future__ import annotations

# pyright: reportPrivateUsage=false
import copy
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from torch import nn

from sakuramoon.config.load import load_config
from sakuramoon.config.schema import OptimizerConfig, RuntimeConfig
from sakuramoon.data.caption import CaptionDropoutCounts, ConditionRouteCounts
from sakuramoon.data.spatial_crop import (
    SPATIAL_FALLBACK_REASONS,
    ZOOM_HISTOGRAM_LABELS,
    SpatialCropCounts,
)
from sakuramoon.data.transparent_white import TransparentWhiteCounts
from sakuramoon.optim import guarded_canonical as guarded
from sakuramoon.optim.cmuon import route_cmuon_parameters
from sakuramoon.optim.condition import condition_learning_rate
from sakuramoon.optim.groups import ParameterSpec
from sakuramoon.telemetry.metrics import DROPOUT_KEYS, DurableJsonlSink
from sakuramoon.telemetry.observer import UpdateMetricContext, build_training_metric
from sakuramoon.telemetry.timers import NoopPhaseTimer
from sakuramoon.train import production
from sakuramoon.train import runtime as runtime_module
from sakuramoon.train import step as step_module
from sakuramoon.train.condition_diagnostics_stream import ConditionDiagnosticsRecorder
from sakuramoon.train.guard_window_stream import (
    GuardWindowIdentity,
    GuardWindowRecorder,
)
from sakuramoon.train.loop import SingleGpuTrainingLoop, SuccessfulLoopObservation
from sakuramoon.train.runtime import (
    RuntimeMeasurement,
    SingleGpuBatchRuntime,
    SuccessfulTrainingObservation,
)
from sakuramoon.train.step import SingleGpuUpdateState, TrainableComposite


def _helpers(relative: str) -> Any:
    path = Path(__file__).parents[1] / relative
    spec = importlib.util.spec_from_file_location(f"_dual_stream_{path.stem}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass decorators resolve their module
    spec.loader.exec_module(module)
    return module


_tiny = _helpers("train/test_condition_diagnostics_lifecycle.py")
_cpu = _helpers("optim/test_guard_window_gloo.py")


@pytest.fixture(autouse=True)
def _small_cpu_threads() -> Any:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _locked_precision(module: TrainableComposite) -> None:
    # Keep the fixture's sensitive parameters FP32; exercise the REAL audit,
    # rather than patching it to bless an all-FP32 nonproduction topology.
    modules = dict(module.named_modules())  # pyright: ignore[reportUnknownArgumentType]
    for name, child in modules.items():
        ancestors = [modules[".".join(name.split(".")[:i])] for i in range(len(name.split(".")) + 1)]
        if any(type(parent).__name__ in {"GlobalConditioner", "FinalOutputHead"} for parent in ancestors):
            continue
        if isinstance(child, (nn.Linear, nn.MultiheadAttention)):
            for parameter in child.parameters(recurse=False):
                if parameter.ndim == 2:
                    parameter.data = parameter.data.to(torch.bfloat16)


def _config(module: TrainableComposite) -> RuntimeConfig:
    root = Path(__file__).parents[3]
    config = load_config(root / "config/train_g1_condition_optim_ab.example.toml", config_root=root, validate_secrets=False).config
    optimizer = config.optimizer.model_dump()
    optimizer["name"] = "hybrid_cmuon_guarded_canonical"
    optimizer["condition"].update(clip_norm=1e-4, rest_clip_norm=1e-3)
    routing = route_cmuon_parameters(module, matrix_weight_decay=0.0, sensitive_weight_decay=0.0)
    refs = {spec.name: 1e-12 for spec in routing.cmuon_specs}
    refs[routing.cmuon_specs[0].name] = 1.0  # a genuinely guarded low-signal input
    optimizer["cmuon_guard"].update(references=refs, min_reference=1e-12, numerical_floor=1e-12)
    return config.model_copy(update={"optimizer": OptimizerConfig.model_validate(optimizer)})


def _cpu_builder(specs: tuple[ParameterSpec, ...], **kwargs: Any) -> Any:
    adapter = _cpu._CPUAdamW(specs, _cpu._CPUSR())
    groups = [
        {"params": [s.parameter for s in subset], "param_names": [s.name for s in subset],
         "group_name": name, "weight_decay": decay}
        for name, subset, decay in (
            ("matrix_decay", adapter.audit.decay, kwargs["matrix_weight_decay"]),
            ("sensitive_no_decay", adapter.audit.sensitive, kwargs["sensitive_weight_decay"]),
        )
    ]
    adapter.optimizer = torch.optim.AdamW(groups, lr=kwargs["lr"], betas=kwargs["betas"], eps=kwargs["eps"], foreach=False)
    return adapter


def _equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        mapping = cast(dict[str, Any], left)
        assert mapping.keys() == right.keys()
        for key in mapping:
            _equal(mapping[key], right[key])
    elif isinstance(left, (list, tuple)):
        sequence = cast(list[Any] | tuple[Any, ...], left)
        assert len(sequence) == len(right)
        for a, b in zip(sequence, right, strict=True):
            _equal(a, b)
    else:
        assert left == right


def _run(root: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> dict[str, Any]:
    torch.manual_seed(811)  # pyright: ignore[reportUnknownMemberType]
    module: TrainableComposite = _tiny._composite()
    _locked_precision(module)
    config = _config(module)
    initial = copy.deepcopy(module.state_dict())
    counts = [0] * 6  # facts, enable, capture, disable, optional union, mandatory norms
    per_update: list[tuple[int, ...]] = []
    pending: list[RuntimeMeasurement] = []
    metrics: list[dict[str, object]] = []
    trajectories: list[Any] = []
    snapshots: list[dict[str, object]] = []
    scheduled: list[int] = []
    saved: list[int] = []
    writes: list[tuple[str, int]] = []
    sinks: dict[int, DurableJsonlSink] = {}
    closes: dict[int, int] = {}
    active = [0, 0]
    diagnostics: ConditionDiagnosticsRecorder | None = None
    guard: GuardWindowRecorder | None = None
    original_facts = runtime_module.compute_condition_activation_facts
    original_conditioning = TrainableComposite.forward_conditioning
    original_norm = step_module._parameter_grad_norm
    original_write, original_close = DurableJsonlSink.write, DurableJsonlSink.close
    original_ns = guarded.cmuon_zeroth_power

    def due(update: int) -> bool:
        return diagnostics is not None and diagnostics.should_record(update)

    def tracked(counter: int, fn: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            counts[counter] += 1
            return fn(*args, **kwargs)
        return call

    def facts(*args: Any, **kwargs: Any) -> Any:
        assert active[1] == 1 and due(active[0])
        return tracked(0, original_facts)(*args, **kwargs)

    def conditioning(self: TrainableComposite, *args: Any, **kwargs: Any) -> Any:
        result = original_conditioning(self, *args, **kwargs)
        if self.last_conditioning is not None:
            counts[2] += 1
            assert active[1] == 1 and due(active[0])
        return result

    def norm(target: nn.Module, *, predicate: Callable[[str, nn.Parameter], bool], device: torch.device) -> torch.Tensor:
        probe = next(target.parameters())
        union = predicate("condition_tokens.probe", probe) and predicate("dit.conditioner.condition_global_norm.weight", probe)
        counts[4 if union else 5] += 1
        return original_norm(target, predicate=predicate, device=device)

    def write(sink: DurableJsonlSink, payload: Mapping[str, object]) -> None:
        stream = "guard" if "guard_window" in sink.path.parts else "diagnostics"
        writes.append((stream, active[0]))
        sinks[id(sink)] = sink
        if active[0] == 2 and mode == f"{stream}_io":
            raise OSError(28, f"injected {stream} disk full at update 2")
        original_write(sink, payload)

    def close(sink: DurableJsonlSink) -> None:
        assert not sink._closed, "the underlying sink must not be closed twice"
        sinks[id(sink)] = sink
        closes[id(sink)] = closes.get(id(sink), 0) + 1
        original_close(sink)

    def loss_fn(batch: Any, microbatch: int) -> torch.Tensor:
        active[:] = [loop.state.successful_updates + 1, microbatch]
        # CPU autocast permits the tiny fixture's mixed sensitive/projection boundary.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            measurement = runtime.measure(batch, update_number=active[0], is_final_microbatch=microbatch == 1)
        assert module.last_conditioning is None and not module.condition_diagnostics_enabled
        assert (measurement.condition_diagnostics is not None) == (microbatch == 1 and due(active[0]))
        pending.append(measurement.detached())
        return measurement.per_sample_loss

    def schedule(update: int) -> None:
        scheduled.append(update)
        production._set_optimizer_learning_rate(optimizer, config.scaled_learning_rate() / (update + 1), config)

    def started(_timer: Any) -> None:
        nonlocal rates, condition_rate
        rates = runtime_module._optimizer_learning_rates(optimizer)
        condition_rate = condition_learning_rate(optimizer)

    def observe(observation: SuccessfulLoopObservation) -> None:
        update = observation.update
        assert update.condition_clip is not None and update.rest_clip is not None
        for clip in (update.condition_clip, update.rest_clip):
            assert 0.0 < float(clip.coefficient) < 1.0
            assert float(clip.post_clip_norm) < float(clip.pre_clip_norm)
        detached = tuple(pending)
        assert all(m.per_sample_loss.grad_fn is None for m in detached)
        # Wall time is not part of recorder neutrality; all model measurements
        # and update values remain real and detached.
        observation = replace(observation, data_wait_seconds=0.0, checkpoint_seconds=0.0)
        training = SuccessfulTrainingObservation(
            loop=observation, microbatches=detached, phase_timer=NoopPhaseTimer(device=torch.device("cpu")),
            learning_rate=rates[0], learning_rate_cmuon=rates[1], learning_rate_adamw=rates[2],
            learning_rate_condition=condition_rate, gpu_memory_allocated_bytes=0, gpu_memory_reserved_bytes=0,
        )
        metric = build_training_metric(training, context=UpdateMetricContext(
            dit_flops=sum(m.dit_flops for m in detached), samples_per_second=4.0,
            ready_queue_depth=0, supplemental_phase_seconds={},
        ), recorded_at_unix_ns=active[0])
        base = config.scaled_learning_rate() / active[0]
        assert (metric.learning_rate, metric.learning_rate_cmuon, metric.learning_rate_adamw, metric.learning_rate_condition) == (
            base, base * config.optimizer.cmuon_lr_multiplier, base * config.optimizer.adamw_lr_multiplier,
            base * config.optimizer.adamw_lr_multiplier * 0.23,
        )
        assert metric.condition_clip_fraction == metric.rest_clip_fraction == 1.0
        metrics.append(metric.as_json_mapping())
        if diagnostics is not None:
            diagnostics.maybe_record(training)
        if guard is not None:
            guard.maybe_record(training)
            assert guard.disabled == (mode == "guard_io" and active[0] >= 2)
        trajectories.append((copy.deepcopy(module.state_dict()), copy.deepcopy(optimizer.state_dict()),
                             [m.per_sample_loss for m in detached], torch.get_rng_state()))
        per_update.append(tuple(counts))
        counts[:] = [0] * 6
        pending.clear()

    with monkeypatch.context() as patch:
        patch.setattr(guarded, "_build_adamw8bit_for_specs", _cpu_builder)
        optimizer = production._build_optimizer(config, module)
        assert isinstance(optimizer, guarded.HybridCMuonGuardedCanonical)
        assert {g["condition_lr_group"] for g in optimizer.optimizer.param_groups} == {True, False}
        for group in optimizer.optimizer.param_groups:
            assert all(name.startswith("condition_tokens.") == group["condition_lr_group"] for name in group["param_names"])
        runtime: SingleGpuBatchRuntime = _tiny._runtime(module, None if mode == "off" else due)
        def prepare(self: SingleGpuBatchRuntime, batch: Any, **kwargs: Any) -> Any:
            prepared = _tiny._prepare(self, batch, **kwargs)
            latents = tuple(t.bfloat16() for t in prepared.inputs.latents)
            return replace(prepared, states=latents, inputs=replace(prepared.inputs,
                qwen_states=prepared.inputs.qwen_states.bfloat16(), latents=latents))
        patch.setattr(SingleGpuBatchRuntime, "prepare", prepare)
        patch.setattr(runtime_module, "compute_condition_activation_facts", facts)
        patch.setattr(TrainableComposite, "forward_conditioning", conditioning)
        patch.setattr(TrainableComposite, "enable_condition_diagnostics", tracked(1, TrainableComposite.enable_condition_diagnostics))
        patch.setattr(TrainableComposite, "disable_condition_diagnostics", tracked(3, TrainableComposite.disable_condition_diagnostics))
        patch.setattr(step_module, "_parameter_grad_norm", norm)
        patch.setattr(DurableJsonlSink, "write", write)
        patch.setattr(DurableJsonlSink, "close", close)
        ns_calls: list[int] = []
        def ns(*args: Any, **kwargs: Any) -> torch.Tensor:
            ns_calls.append(active[0])
            return original_ns(*args, **kwargs)
        patch.setattr(guarded, "cmuon_zeroth_power", ns)
        if mode != "off":
            diagnostics = ConditionDiagnosticsRecorder(root / "diagnostics.jsonl", rank=0, world_size=1, log_every_n=2, flush_every_records=1, slot_ids=(0,))
            guard = GuardWindowRecorder(artifact_dir=root, optimizer=optimizer, window_updates=2, identity=GuardWindowIdentity(
                "dual-stream", mode, 0, 1, "fresh-start", 0, "fresh_bootstrap"),
                last_published_checkpoint=lambda: str(saved[-1]) if saved else None)
            real_snapshot = guard._snapshot
            def snapshot() -> dict[str, object]:
                result = real_snapshot()
                snapshots.append(copy.deepcopy(result))
                return result
            patch.setattr(guard, "_snapshot", snapshot)
        rates, condition_rate = (0.0, 0.0, 0.0), None
        batch = _tiny._batch()
        batch.dropout_hits = CaptionDropoutCounts(**dict.fromkeys(DROPOUT_KEYS, 0))
        batch.condition_routes = ConditionRouteCounts(1, 0, 1)
        batch.spatial_crop = SpatialCropCounts(0, 0, {k: 2 if k == "not_selected" else 0 for k in SPATIAL_FALLBACK_REASONS}, dict.fromkeys(ZOOM_HISTOGRAM_LABELS, 0), 0.0, 0.0, 0.0, 0.0, 0)
        batch.transparent = TransparentWhiteCounts(0, 0, 0, 0, 0, 0)
        loop = SingleGpuTrainingLoop(module=module, optimizer=optimizer, loss_fn=loss_fn,
            accumulation_steps=2, target_successful_updates=4, checkpoint_every_successful_updates=2,
            scheduler_step=schedule, checkpoint=saved.append, diagnostic_root=root / "failures",
            failure_id=lambda phase, _state: phase, state=SingleGpuUpdateState.initial(),
            update_started=started, successful_update_observer=observe,
            condition_optimizer=config.optimizer.condition, condition_grad_share=False if mode == "off" else due)
        try:
            result = loop.run((batch,) * 8)
        finally:
            for _ in range(2):  # public cleanup must be idempotent, including after I/O disable
                if diagnostics is not None:
                    diagnostics.close()
                if guard is not None:
                    guard.close("clean")
        assert result.checkpoint_updates == tuple(saved) == (2, 4)
        assert scheduled == [1, 2, 3, 4]
        assert cast(Any, optimizer.sr_rng).calls == optimizer.observations == 4
        assert set(ns_calls) == {1, 2, 3, 4} and optimizer.skip_total > 0
        assert len(ns_calls) + optimizer.skip_total == 4 * sum(s.chunk_count for s in optimizer.routing.cmuon_specs)
        assert any(torch.count_nonzero(t).item() for t in optimizer._momenta.values())
        for specs in (optimizer.routing.cmuon_specs, optimizer.routing.adamw_specs):
            assert any(not torch.equal(initial[s.name], s.parameter) for s in specs)
        for state in optimizer.optimizer.state.values():
            assert float(state["step"]) == 4 and torch.count_nonzero(state["exp_avg"]) > 0
        assert closes == dict.fromkeys(sinks, 1)
        for sink in sinks.values():
            with pytest.raises(OSError):
                os.fstat(sink._fd)
    records = [json.loads(p.read_text()) for p in sorted(root.glob("guard_window/session-*/rank-*/guard-window-*.jsonl"))]
    diag_path = root / "diagnostics.jsonl"
    diag_records = [json.loads(line) for line in diag_path.read_text().splitlines()] if diag_path.exists() else []
    return {"trajectories": trajectories, "metrics": metrics, "rng": torch.get_rng_state(), "counts": per_update,
            "guard": records, "diagnostics": diag_records, "writes": writes, "snapshots": snapshots}


@pytest.mark.parametrize("mode", ["enabled", "diagnostics_io", "guard_io"])
def test_dual_streams_are_live_gated_and_training_neutral(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str) -> None:
    baseline = _run(tmp_path / "off", monkeypatch, "off")
    observed = _run(tmp_path / mode, monkeypatch, mode)
    for key in ("trajectories", "metrics", "rng"):
        _equal(baseline[key], observed[key])
    idle, due = (0, 0, 0, 0, 0, 4), (1, 1, 1, 1, 2, 4)
    assert baseline["counts"] == [idle] * 4
    assert observed["counts"] == [idle, due, idle, idle if mode == "diagnostics_io" else due]
    assert [r["successful_update"] for r in observed["diagnostics"]] == ([] if mode == "diagnostics_io" else [2, 4])
    assert observed["writes"] == [("diagnostics", 2), ("guard", 2)] + (
        [("diagnostics", 4)] if mode == "guard_io" else [("guard", 4)] if mode == "diagnostics_io" else [("diagnostics", 4), ("guard", 4)])
    assert len(observed["snapshots"]) == (2 if mode == "guard_io" else 4)
    assert len(observed["guard"]) == (0 if mode == "guard_io" else 2)
    for index, record in enumerate(observed["guard"]):
        assert (record["start_update"], record["end_update"]) == (index * 2 + 1, index * 2 + 2)
        assert record["baseline"]["source"] == "fresh"
        assert record["absolute"] == observed["snapshots"][index * 2 + 1]
        assert record["baseline"]["counts"]["observations"] == index * 2
        assert record["delta"]["observations"] == 2 and record["delta"]["skip_total"] > 0
        assert sum(record["delta"]["skip_by_role"].values()) == sum(record["delta"]["skip_by_fqn"].values()) == record["delta"]["skip_total"]
        assert set(record["delta_status"].values()) == {"ok"}
        assert record["launch"]["last_published_checkpoint"] == str(index * 2 + 2)
        assert record["complete"] and record["run_outcome"] is None
        assert record["owner_partition"] == {"0": sorted(f"{r['fqn']}#{c}" for r in record["routing"] for c in range(r["chunk_count"]))}
        assert "bf16_attempts" not in record["absolute"]  # base variant has no rescue-owner counters
