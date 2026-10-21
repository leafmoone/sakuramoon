# pyright: reportPrivateUsage=false
"""Contract tests for the production nested successful-update observer.

The real nested ``observe_successful_update`` function and the real
``calibration_observer`` binding statement are extracted from
``src/sakuramoon/train/production.py`` via AST and executed against
lightweight DI stubs.  No observer algorithm or event text is duplicated
here; the stubs only record call order, arguments, and raised identities.

Recording sentinels (config / growth / update property access) pin the
in-place read order: update first, then the optional benchmark append,
then the sampling-due config reads, then growth alpha reads at each
sample/eval event.  The same extraction and builder functions are
reusable by the external oracle via ``production_path``.
"""

from __future__ import annotations

import ast
import hashlib
import json
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Self, cast

import pytest

from sakuramoon.checkpoint.schema import GrowthCheckpointState
from sakuramoon.distributed import (
    DistributedProgress,
    DistributedProgressError,
)
from sakuramoon.train.benchmark import BenchmarkTrace
from sakuramoon.train.production_observer import (
    observe_production_successful_update,
)
from sakuramoon.train.runtime import SuccessfulTrainingObservation
from sakuramoon.train.sampling import TrainingSampleResult
from sakuramoon.train.stage import canonical_growth_alpha

DEFAULT_PRODUCTION_PATH = (
    Path(__file__).resolve().parents[3] / "src/sakuramoon/train/production.py"
)
OBSERVER_FUNCTION = "observe_successful_update"
CALIBRATION_BINDING = "calibration_observer"
STAGE_NAMES = ("training-samples", "telemetry", "concept-suite", "complete")
EVAL_PAYLOAD_KEYS = (
    "evaluation/fid",
    "evaluation/inception_score_mean",
    "evaluation/inception_score_std",
    "evaluation/kid_mean",
    "evaluation/kid_std",
    "evaluation/cmmd",
    "evaluation/sample_count",
    "evaluation/real_sample_count",
    "evaluation/is_splits",
)


class SamplerBoom(RuntimeError):
    """Injected sampler failure with a stable message for identity checks."""


class Store:
    """In-memory control-plane store with the DistributedProgress protocol."""

    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    def set(self, key: str, value: str) -> None:
        self.data[key] = value.encode("utf-8")

    def check(self, keys: Sequence[str]) -> bool:
        return any(key in self.data for key in keys)

    def get(self, key: str) -> bytes:
        return self.data[key]


class Clock:
    """Use real elapsed time so peer scheduling cannot exhaust a fake timeout."""

    def __init__(self, step_seconds: float) -> None:
        self.step_seconds = step_seconds

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(min(seconds, self.step_seconds))


class TimerContext:
    """Context manager recording stage enter/exit pairs into the event log."""

    def __init__(self, events: list[tuple[object, ...]], stage: str) -> None:
        self.events = events
        self.stage = stage

    def __enter__(self) -> Self:
        self.events.append((self.stage, "enter"))
        return self

    def __exit__(
        self,
        exc_type: object,
        exc: object,
        tb: object,
    ) -> bool:
        del exc_type, exc, tb
        self.events.append((self.stage, "exit"))
        return False


class Timer:
    """Recording stand-in for the observation phase timer."""

    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def record(self, stage: str) -> TimerContext:
        return TimerContext(self.events, stage)


class UpdateState:
    """Records each read of the successful-update counter in order."""

    def __init__(
        self, events: list[tuple[object, ...]], successful_updates: int
    ) -> None:
        self.events = events
        self._successful_updates = successful_updates

    @property
    def successful_updates(self) -> int:
        self.events.append(("update_read", self._successful_updates))
        return self._successful_updates


class Caption:
    """Minimal caption fact for the benchmark trace payload."""

    def __init__(self, text: str) -> None:
        self.text = text


class Microbatch:
    """Microbatch facts consumed by the real BenchmarkTrace.append."""

    def __init__(
        self,
        sample_ids: tuple[str, ...],
        shape_keys: tuple[str, ...],
        captions: tuple[Caption, ...],
        image_tokens: int,
        text_tokens: int,
        dit_flops: int,
    ) -> None:
        self.sample_ids = sample_ids
        self.shape_keys = shape_keys
        self.captions = captions
        self.image_tokens = image_tokens
        self.text_tokens = text_tokens
        self.dit_flops = dit_flops


class Observation:
    """Input observation exposing only the attributes the observer reads."""

    def __init__(
        self,
        update_state: UpdateState,
        microbatches: tuple[Microbatch, ...],
        timer: Timer,
        update_wall_seconds: float,
    ) -> None:
        self.loop = SimpleNamespace(
            update=SimpleNamespace(state=update_state),
            update_wall_seconds=update_wall_seconds,
        )
        self.microbatches = microbatches
        self.phase_timer = timer


class SamplingTrainingConfig:
    """Records each sampling-due config read in place."""

    def __init__(
        self,
        events: list[tuple[object, ...]],
        enabled: bool,
        every_updates: int,
    ) -> None:
        self.events = events
        self._enabled = enabled
        self._every_updates = every_updates

    @property
    def enabled(self) -> bool:
        self.events.append(("config_access", "enabled"))
        return self._enabled

    @property
    def every_updates(self) -> int:
        self.events.append(("config_access", "every_updates"))
        return self._every_updates


class Growth:
    """Mutable ramp state recording every in-place field access."""

    def __init__(
        self,
        events: list[tuple[object, ...]],
        ramp_start_successful_update: int | None,
        ramp_updates: int | None,
    ) -> None:
        self.events = events
        self._ramp_start = ramp_start_successful_update
        self._ramp_updates = ramp_updates

    @property
    def ramp_start_successful_update(self) -> int | None:
        self.events.append(("growth_read", "ramp_start"))
        return self._ramp_start

    @property
    def ramp_updates(self) -> int | None:
        self.events.append(("growth_read", "ramp_updates"))
        return self._ramp_updates


class BenchmarkDelegate:
    """Records the append event, then delegates to the real trace."""

    def __init__(
        self, events: list[tuple[object, ...]], inner: BenchmarkTrace
    ) -> None:
        self.events = events
        self.inner = inner

    def append(self, observation: object) -> None:
        self.events.append(("benchmark_append",))
        BenchmarkTrace.append(
            self.inner, cast(SuccessfulTrainingObservation, observation)
        )  # type: ignore[reportUnnecessaryCast]


class Sampler:
    """Stub training sampler capturing alpha writes and sample calls."""

    def __init__(
        self,
        events: list[tuple[object, ...]],
        result: TrainingSampleResult,
        exception: BaseException | None,
        on_alpha: Callable[[float], None] | None,
    ) -> None:
        self.events = events
        self.result = result
        self.exception = exception
        self.on_alpha = on_alpha
        self.alphas: list[float] = []
        self.calls: list[tuple[int, object]] = []

    def set_growth_alpha(self, alpha: float) -> None:
        self.alphas.append(alpha)
        self.events.append(("set_alpha", alpha))
        if self.on_alpha is not None:
            self.on_alpha(alpha)

    def sample(
        self, update: int, measurements: object
    ) -> TrainingSampleResult:
        if self.exception is not None:
            raise self.exception
        self.calls.append((update, measurements))
        self.events.append(("sample_call", update))
        return self.result


class Telemetry:
    """Stub telemetry capturing images, metric payloads, and the observer."""

    def __init__(
        self,
        events: list[tuple[object, ...]],
        images_exception: BaseException | None,
        metrics_exception: BaseException | None,
        observer_exception: BaseException | None,
    ) -> None:
        self.events = events
        self.images_exception = images_exception
        self.metrics_exception = metrics_exception
        self.observer_exception = observer_exception
        self.images: list[tuple[object, object, int]] = []
        self.metrics: list[tuple[dict[str, object], int]] = []
        self.observers: list[object] = []

    def submit_wandb_images(
        self,
        paths: object,
        captions: object,
        *,
        successful_update: int,
    ) -> None:
        if self.images_exception is not None:
            raise self.images_exception
        self.images.append((paths, captions, successful_update))
        self.events.append(("images", successful_update))

    def submit_wandb_metrics(
        self,
        payload: dict[str, object],
        *,
        successful_update: int,
    ) -> None:
        if self.metrics_exception is not None:
            raise self.metrics_exception
        self.metrics.append((dict(payload), successful_update))
        self.events.append(("metrics", successful_update))

    def observer(self, observation: object) -> None:
        if self.observer_exception is not None:
            raise self.observer_exception
        self.observers.append(observation)
        self.events.append(("observer", observation))


class Evaluation:
    """Metric values returned by the evaluator on the main rank."""

    def __init__(
        self,
        fid: float,
        inception_score_mean: float,
        inception_score_std: float,
        kid_mean: float,
        kid_std: float,
        cmmd: float,
        sample_count: int,
        real_sample_count: int,
    ) -> None:
        self.fid = fid
        self.inception_score_mean = inception_score_mean
        self.inception_score_std = inception_score_std
        self.kid_mean = kid_mean
        self.kid_std = kid_std
        self.cmmd = cmmd
        self.sample_count = sample_count
        self.real_sample_count = real_sample_count


class Evaluator:
    """Stub training evaluator capturing every in-place call."""

    def __init__(
        self,
        events: list[tuple[object, ...]],
        due_updates: frozenset[int],
        result: Evaluation | None,
        suite: dict[str, float] | None,
        evaluate_exception: BaseException | None,
        suite_exception: BaseException | None,
    ) -> None:
        self.events = events
        self.due_updates = due_updates
        self.result = result
        self.suite = suite
        self.evaluate_exception = evaluate_exception
        self.suite_exception = suite_exception
        self.alphas: list[float] = []
        self.calls: list[int] = []
        self.suite_calls: list[int] = []

    def due(self, update: int) -> bool:
        return update in self.due_updates

    def set_growth_alpha(self, alpha: float) -> None:
        self.alphas.append(alpha)
        self.events.append(("set_alpha_eval", alpha))

    def evaluate(self, update: int) -> Evaluation | None:
        if self.evaluate_exception is not None:
            raise self.evaluate_exception
        self.calls.append(update)
        self.events.append(("evaluate", update))
        return self.result

    def run_concept_suite(self, update: int) -> dict[str, float] | None:
        if self.suite_exception is not None:
            raise self.suite_exception
        self.suite_calls.append(update)
        self.events.append(("suite", update))
        return self.suite


def oracle_growth(
    ramp_start: int | None, ramp_updates: int | None
) -> GrowthCheckpointState:
    """Real persisted growth state used as the alpha value oracle."""
    return GrowthCheckpointState(
        active_slot_ids=(0,),
        alpha=0.5,
        stage="growth",
        world_size=2,
        resolution=512,
        ramp_start_successful_update=ramp_start,
        ramp_updates=ramp_updates,
    )


def _stage(update: int, name: str) -> str:
    return f"observer/update-{update}/{name}"


class RecordingProgress(DistributedProgress):
    """Real DistributedProgress that also records ordered method calls."""

    def __init__(
        self,
        store: Store | None,
        *,
        progress_namespace: str,
        rank: int,
        world_size: int,
        clock: Clock,
    ) -> None:
        super().__init__(
            store,
            namespace=progress_namespace,
            rank=rank,
            world_size=world_size,
            stall_timeout=timedelta(seconds=30),
            poll_interval_seconds=0.001,
            heartbeat_interval_seconds=0.002,
            monotonic=clock.monotonic,
            sleep=clock.sleep,
        )
        self.calls: list[str] = []

    def run_on_rank(
        self, stage: str, owner_rank: int, operation: object
    ) -> object:
        self.calls.append(stage)
        return super().run_on_rank(
            stage, owner_rank, cast(Callable[[], object], operation)
        )

    def synchronize(self, stage: str) -> None:
        self.calls.append(stage)
        return super().synchronize(stage)


def make_progress(
    store: Store | None,
    progress_namespace: str,
    rank: int,
    world_size: int,
    clock: Clock,
) -> RecordingProgress:
    return RecordingProgress(
        store,
        progress_namespace=progress_namespace,
        rank=rank,
        world_size=world_size,
        clock=clock,
    )


def extract_nested_function(
    production_path: Path,
    namespace: dict[str, object],
    name: str,
) -> None:
    """Exec the real nested function definition found by name into namespace."""
    source = production_path.read_text("utf-8")
    tree = ast.parse(source, filename=str(production_path))
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.FunctionDef) and item.name == name
        ),
        None,
    )
    if node is None:
        raise AssertionError(f"nested function {name!r} not found")
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError(f"nested function {name!r} has no source segment")
    module = ast.Module(body=[node], type_ignores=[])
    code = compile(module, str(production_path), "exec")
    exec(code, namespace)  # noqa: S102 - the extracted body IS the test subject


def extract_calibration_binding(
    production_path: Path,
    namespace: dict[str, object],
) -> None:
    """Exec the real calibration_observer assignment into namespace."""
    source = production_path.read_text("utf-8")
    tree = ast.parse(source, filename=str(production_path))
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == CALIBRATION_BINDING
        ),
        None,
    )
    if node is None:
        raise AssertionError(f"binding {CALIBRATION_BINDING!r} not found")
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError(f"binding {CALIBRATION_BINDING!r} has no segment")
    module = ast.parse(segment)
    code = compile(module, str(production_path), "exec")
    exec(code, namespace)  # noqa: S102 - the extracted binding IS the test subject


def build_observer_namespace(
    *,
    production_path: Path = DEFAULT_PRODUCTION_PATH,
    store: Store | None = None,
    progress_namespace: str,
    rank: int = 0,
    world_size: int = 2,
    clock: Clock,
    update: int,
    events: list[tuple[object, ...]],
    sampling_enabled: bool = True,
    every_updates: int = 10,
    growth: Growth,
    sampler: Sampler | None,
    telemetry: Telemetry,
    evaluator: Evaluator | None = None,
    is_splits: int | None = None,
    is_main_process: bool = True,
    benchmark: object = None,
    in_calibration: bool = False,
) -> dict[str, object]:
    """Assemble the closure namespace for the real nested observer."""
    namespace: dict[str, object] = {
        "observe_production_successful_update": observe_production_successful_update,
        "benchmark_trace": benchmark,
        "config": SimpleNamespace(
            sampling=SimpleNamespace(
                training=SamplingTrainingConfig(
                    events, sampling_enabled, every_updates
                )
            )
        ),
        "training_sampler": sampler,
        "restored": SimpleNamespace(state=SimpleNamespace(growth=growth)),
        "progress": make_progress(
            store, progress_namespace, rank, world_size, clock
        ),
        "telemetry": telemetry,
        "evaluator": evaluator,
        "evaluation_is_splits": is_splits,
        "is_main_process": is_main_process,
        "canonical_growth_alpha": canonical_growth_alpha,
        "in_calibration": in_calibration,
    }
    extract_nested_function(production_path, namespace, OBSERVER_FUNCTION)
    extract_calibration_binding(production_path, namespace)
    return namespace


def make_observation(
    events: list[tuple[object, ...]],
    update: int,
    microbatches: tuple[Microbatch, ...],
) -> Observation:
    return Observation(
        UpdateState(events, update),
        microbatches,
        Timer(events),
        1.5,
    )


def make_benchmark(
    tmp_path: Path, events: list[tuple[object, ...]], rank: int
) -> BenchmarkDelegate:
    """Pre-create the trace file the way from_environment does, then wrap."""
    path = tmp_path / f"rank-{rank}.jsonl"
    path.touch(mode=0o600)
    return BenchmarkDelegate(events, BenchmarkTrace(path, rank))


def standard_result(update: int) -> TrainingSampleResult:
    return TrainingSampleResult(
        update,
        (Path("a.png"), Path("b.png")),
        ("alpha", "beta"),
        {"cond": 0.5},
    )


@dataclass(frozen=True)
class RankOutcome:
    error: BaseException | None
    value: object


def run_interleaved(
    observe0: Callable[[Observation], None],
    observe1: Callable[[Observation], None],
    observation0: Observation,
    observation1: Observation,
    timeout_seconds: float,
) -> tuple[RankOutcome, RankOutcome]:
    """Run both ranks; rank 1 on a worker thread, rank 0 on the caller."""
    slot1: dict[str, object] = {}

    def worker() -> None:
        try:
            slot1["value"] = observe1(observation1)
        except BaseException as error:  # noqa: BLE001 - reported verbatim
            slot1["error"] = error

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    slot0: dict[str, object] = {}
    try:
        slot0["value"] = observe0(observation0)
    except BaseException as error:  # noqa: BLE001 - reported verbatim
        slot0["error"] = error
    thread.join(timeout_seconds)
    if thread.is_alive():
        pytest.fail("interleaved observer run did not finish (deadlock?)")

    def to_outcome(slot: dict[str, object]) -> RankOutcome:
        return RankOutcome(
            cast(BaseException | None, slot.get("error")),
            slot.get("value"),
        )

    return to_outcome(slot0), to_outcome(slot1)


def extract_in_calibration_assign(
    production_path: Path,
    namespace: dict[str, object],
) -> bool:
    """Exec the real in_calibration OR assignment into namespace."""
    source = production_path.read_text("utf-8")
    tree = ast.parse(source, filename=str(production_path))
    node = next(
        (
            item
            for item in ast.walk(tree)
            if isinstance(item, ast.Assign)
            and len(item.targets) == 1
            and isinstance(item.targets[0], ast.Name)
            and item.targets[0].id == "in_calibration"
        ),
        None,
    )
    if node is None:
        raise AssertionError("in_calibration assignment not found")
    segment = ast.get_source_segment(source, node)
    if segment is None:
        raise AssertionError("in_calibration assignment has no segment")
    code = compile(ast.parse(segment), str(production_path), "exec")
    exec(code, namespace)  # noqa: S102 - the real OR source under test
    return cast(bool, namespace["in_calibration"])


def bind_with_real_or(
    production_path: Path,
    namespace: dict[str, object],
    calibration: object,
    structural: object,
) -> Callable[[Observation], None]:
    """Compute in_calibration from the real OR, then re-execute the binding."""
    value = extract_in_calibration_assign(
        production_path, {"calibration": calibration, "structural": structural}
    )
    namespace["in_calibration"] = value
    extract_calibration_binding(production_path, namespace)
    return cast(Callable[[Observation], None], namespace[CALIBRATION_BINDING])


class ProgressBoom(RuntimeError):
    """Injected progress-method failure with a stable message."""


class FaultyProgress(RecordingProgress):
    """Real progress that faults one method at one named stage."""

    def __init__(
        self,
        store: Store | None,
        *,
        progress_namespace: str,
        rank: int,
        world_size: int,
        clock: Clock,
        fault_stage: str,
        fault_method: str,
        fault: ProgressBoom,
    ) -> None:
        super().__init__(
            store,
            progress_namespace=progress_namespace,
            rank=rank,
            world_size=world_size,
            clock=clock,
        )
        self.fault_stage = fault_stage
        self.fault_method = fault_method
        self.fault = fault

    def run_on_rank(
        self, stage: str, owner_rank: int, operation: object
    ) -> object:
        self.calls.append(stage)
        if self.fault_method == "run_on_rank" and stage == self.fault_stage:
            raise self.fault
        return DistributedProgress.run_on_rank(
            self, stage, owner_rank, cast(Callable[[], object], operation)
        )

    def synchronize(self, stage: str) -> None:
        self.calls.append(stage)
        if self.fault_method == "synchronize" and stage == self.fault_stage:
            raise self.fault
        return DistributedProgress.synchronize(self, stage)


class FaultyTelemetry(Telemetry):
    """Telemetry that faults one W&B metric payload kind by discriminant."""

    def __init__(
        self,
        events: list[tuple[object, ...]],
        images_exception: BaseException | None,
        metrics_exception: BaseException | None,
        observer_exception: BaseException | None,
        metrics_faults: dict[str, BaseException],
    ) -> None:
        super().__init__(
            events, images_exception, metrics_exception, observer_exception
        )
        self.metrics_faults = metrics_faults

    def submit_wandb_metrics(
        self,
        payload: dict[str, object],
        *,
        successful_update: int,
    ) -> None:
        if "eval" in self.metrics_faults and "evaluation/fid" in payload:
            raise self.metrics_faults["eval"]
        if "concept" in self.metrics_faults and any(
            key.startswith("evaluation/concept/") for key in payload
        ):
            raise self.metrics_faults["concept"]
        super().submit_wandb_metrics(payload, successful_update=successful_update)


def expected_due_events(
    *,
    with_benchmark: bool,
    sampling_due: bool,
    read_every_updates: bool,
    evaluation_due: bool,
    is_main: bool,
) -> list[str]:
    """Projected event-kind order from the real observer body."""
    kinds: list[str] = ["update_read"]
    if with_benchmark:
        kinds.extend(["benchmark_append", "update_read"])
    kinds.append("config_access")
    if read_every_updates:
        kinds.append("config_access")
    if sampling_due:
        kinds.extend(
            [
                "sample",
                "growth_read",
                "growth_read",
                "growth_read",
                "growth_read",
                "growth_read",
                "set_alpha",
                "sample_call",
                "sample",
                "images",
                "metrics",
            ]
        )
    kinds.append("observer")
    if evaluation_due:
        kinds.extend(
            [
                "growth_read",
                "growth_read",
                "growth_read",
                "growth_read",
                "growth_read",
                "set_alpha_eval",
                "evaluate",
                "suite",
            ]
        )
        if is_main:
            kinds.extend(["metrics", "metrics"])
    return kinds


def expected_nonowner_events(
    *,
    read_every_updates: bool,
    evaluation_due: bool,
) -> list[str]:
    """Pre-callback in-place reads performed before any rank-0 stage.

    The due expression itself reads enabled and (when update > 0) the
    modulo cadence once, regardless of the due outcome.
    """
    kinds: list[str] = ["update_read", "config_access"]
    if read_every_updates:
        kinds.append("config_access")
    if evaluation_due:
        kinds.extend(
            [
                "growth_read",
                "growth_read",
                "growth_read",
                "growth_read",
                "growth_read",
                "set_alpha_eval",
                "evaluate",
            ]
        )
    return kinds


def check_stages(progress: DistributedProgress, stages: tuple[str, ...]) -> None:
    assert tuple(sorted(progress._sequences)) == tuple(sorted(stages))


def check_store_has_detail(store: Store, detail: str) -> None:
    assert any(detail.encode("utf-8") in raw for raw in store.data.values())


def get_namespace_component(
    namespace: dict[str, object], name: str, cls: type
) -> object:
    value = namespace[name]
    assert isinstance(value, cls)
    return value


def expected_stage_calls(
    update: int, *, sampling_due: bool, evaluation_due: bool
) -> list[str]:
    """Exact ordered progress-method call list both ranks must perform."""
    calls: list[str] = []
    if sampling_due:
        calls.append(_stage(update, "training-samples"))
    calls.append(_stage(update, "telemetry"))
    if evaluation_due:
        calls.append(_stage(update, "concept-suite"))
    calls.append(_stage(update, "complete"))
    return calls


# ---------------------------------------------------------------------------
# REV2: normal due matrix (4 due groups x 2 ranks, benchmark on/off)
# ---------------------------------------------------------------------------


NORMAL_DUE_MATRIX: tuple[tuple[str, bool, bool, bool], ...] = (
    ("both_due_benchmark_on", True, True, True),
    ("both_due_benchmark_off", True, True, False),
    ("sampling_only", True, False, False),
    ("eval_only", False, True, False),
    ("neither_due_benchmark_on", False, False, True),
    ("neither_due_benchmark_off", False, False, False),
)


def _matrix_namespace(
    *,
    store: Store,
    progress_namespace: str,
    rank: int,
    is_main: bool,
    clock: Clock,
    update: int,
    events: list[tuple[object, ...]],
    sampling_due: bool,
    evaluation_due: bool,
    benchmark: object = None,
    telemetry: Telemetry | None = None,
) -> dict[str, object]:
    return build_observer_namespace(
        store=store,
        progress_namespace=progress_namespace,
        rank=rank,
        clock=clock,
        update=update,
        events=events,
        # enabled in every case; due is driven purely by the modulo
        every_updates=10 if sampling_due else 7,
        growth=Growth(events, 10, 100),
        sampler=Sampler(events, standard_result(update), None, None),
        telemetry=Telemetry(events, None, None, None) if telemetry is None else telemetry,
        evaluator=Evaluator(
            events,
            frozenset({update}) if evaluation_due else frozenset(),
            Evaluation(0.5, 8.0, 0.2, 0.02, 0.001, 1.0, 4, 4)
            if is_main
            else None,
            {"a": 1.0},
            None,
            None,
        ),
        is_splits=1,
        is_main_process=is_main,
        benchmark=benchmark,
    )


@pytest.mark.parametrize(
    "name,sampling_due,evaluation_due,benchmark_on",
    NORMAL_DUE_MATRIX,
    ids=[item[0] for item in NORMAL_DUE_MATRIX],
)
def test_normal_due_matrix(
    tmp_path: Path,
    name: str,
    sampling_due: bool,
    evaluation_due: bool,
    benchmark_on: bool,
) -> None:
    del name
    update = 10
    clock = Clock(0.001)
    store = Store()
    events0: list[tuple[object, ...]] = []
    events1: list[tuple[object, ...]] = []
    benchmark0 = make_benchmark(tmp_path, events0, 0) if benchmark_on else None
    namespace0 = _matrix_namespace(
        store=store,
        progress_namespace="d1-matrix",
        rank=0,
        is_main=True,
        clock=clock,
        update=update,
        events=events0,
        sampling_due=sampling_due,
        evaluation_due=evaluation_due,
        benchmark=benchmark0,
    )
    namespace1 = _matrix_namespace(
        store=store,
        progress_namespace="d1-matrix",
        rank=1,
        is_main=False,
        clock=clock,
        update=update,
        events=events1,
        sampling_due=sampling_due,
        evaluation_due=evaluation_due,
    )
    observation0 = make_observation(events0, update, ())
    observation1 = make_observation(events1, update, ())
    outcome0, outcome1 = run_interleaved(
        cast(Callable[[Observation], None], namespace0[OBSERVER_FUNCTION]),
        cast(Callable[[Observation], None], namespace1[OBSERVER_FUNCTION]),
        observation0,
        observation1,
        20.0,
    )
    assert outcome0.error is None
    assert outcome1.error is None

    expected_calls = expected_stage_calls(
        update, sampling_due=sampling_due, evaluation_due=evaluation_due
    )
    progress0 = cast(RecordingProgress, namespace0["progress"])
    progress1 = cast(RecordingProgress, namespace1["progress"])
    # Both ranks perform the exact ordered stage call list, no extras.
    assert progress0.calls == expected_calls
    assert progress1.calls == expected_calls
    check_stages(progress0, tuple(expected_calls))
    check_stages(progress1, (_stage(update, "complete"),))

    sampler0 = cast(Sampler, namespace0["training_sampler"])
    telemetry0 = cast(Telemetry, namespace0["telemetry"])
    evaluator0 = cast(Evaluator, namespace0["evaluator"])
    evaluator1 = cast(Evaluator, namespace1["evaluator"])
    if sampling_due:
        assert sampler0.calls == [(update, observation0.microbatches)]
        assert len(telemetry0.images) == 1
    else:
        assert sampler0.calls == []
        assert telemetry0.images == []
    expected_metric_count = int(sampling_due) + 2 * int(evaluation_due)
    assert len(telemetry0.metrics) == expected_metric_count
    assert evaluator0.suite_calls == ([update] if evaluation_due else [])
    # evaluate() runs on every rank when due.
    assert evaluator1.calls == ([update] if evaluation_due else [])

    assert [event[0] for event in events0] == expected_due_events(
        with_benchmark=benchmark_on,
        sampling_due=sampling_due,
        # enabled and update > 0 in every matrix case, so every_updates is
        # always read regardless of whether the modulo is due
        read_every_updates=True,
        evaluation_due=evaluation_due,
        is_main=True,
    )
    assert [event[0] for event in events1] == expected_nonowner_events(
        read_every_updates=True,
        evaluation_due=evaluation_due,
    )

    # Enabled-but-modulo-not-due still reads every_updates, never samples.
    if not sampling_due:
        assert ("config_access", "every_updates") in events0
        assert ("sample", "enter") not in events0
    trace_file = tmp_path / "rank-0.jsonl"
    if benchmark_on:
        payload = json.loads(trace_file.read_text("utf-8").splitlines()[0])
        assert payload["successful_update"] == update
    else:
        assert not trace_file.exists()


# ---------------------------------------------------------------------------
# REV2: progress-method failures (sentinel object, exact boundary)
# ---------------------------------------------------------------------------


def _progress_fault_namespace(
    *,
    update: int,
    clock: Clock,
    store: Store,
    events: list[tuple[object, ...]],
    sampling_due: bool,
    evaluation_due: bool,
    fault_stage: str,
    fault_method: str,
    fault: ProgressBoom,
) -> tuple[dict[str, object], FaultyProgress]:
    namespace = _matrix_namespace(
        store=store,
        progress_namespace="d1-pfail",
        rank=0,
        is_main=True,
        clock=clock,
        update=update,
        events=events,
        sampling_due=sampling_due,
        evaluation_due=evaluation_due,
    )
    faulty = FaultyProgress(
        store,
        progress_namespace="d1-pfail",
        rank=0,
        world_size=2,
        clock=clock,
        fault_stage=fault_stage,
        fault_method=fault_method,
        fault=fault,
    )
    namespace["progress"] = faulty
    return namespace, faulty


def test_progress_fault_at_training_samples_stage() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    fault = ProgressBoom("samples-stage-down")
    namespace, progress = _progress_fault_namespace(
        update=update,
        clock=clock,
        store=store,
        events=events,
        sampling_due=True,
        evaluation_due=False,
        fault_stage=_stage(update, "training-samples"),
        fault_method="run_on_rank",
        fault=fault,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(ProgressBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(
            observation
        )
    assert excinfo.value is fault
    # The fault fired before the owner callback and before any publication.
    assert [event[0] for event in events] == [
        "update_read",
        "config_access",
        "config_access",
    ]
    assert progress.calls == [_stage(update, "training-samples")]
    assert progress._sequences == {}
    sampler = cast(Sampler, namespace["training_sampler"])
    telemetry = cast(Telemetry, namespace["telemetry"])
    assert sampler.calls == []
    assert telemetry.observers == []
    assert store.data == {}


def test_progress_fault_at_telemetry_stage() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    fault = ProgressBoom("telemetry-stage-down")
    namespace, progress = _progress_fault_namespace(
        update=update,
        clock=clock,
        store=store,
        events=events,
        sampling_due=False,
        evaluation_due=False,
        fault_stage=_stage(update, "telemetry"),
        fault_method="run_on_rank",
        fault=fault,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(ProgressBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(
            observation
        )
    assert excinfo.value is fault
    assert [event[0] for event in events] == [
        "update_read",
        "config_access",
        "config_access",
    ]
    assert progress.calls == [_stage(update, "telemetry")]
    assert progress._sequences == {}
    telemetry = cast(Telemetry, namespace["telemetry"])
    assert telemetry.observers == []


def test_progress_fault_at_concept_suite_stage() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    fault = ProgressBoom("concept-stage-down")
    namespace, progress = _progress_fault_namespace(
        update=update,
        clock=clock,
        store=store,
        events=events,
        sampling_due=True,
        evaluation_due=True,
        fault_stage=_stage(update, "concept-suite"),
        fault_method="run_on_rank",
        fault=fault,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(ProgressBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(
            observation
        )
    assert excinfo.value is fault
    # Earlier stages completed; the suite callback never ran.
    assert progress.calls == expected_stage_calls(
        update, sampling_due=True, evaluation_due=True
    )[:3]
    check_stages(
        progress,
        (
            _stage(update, "training-samples"),
            _stage(update, "telemetry"),
        ),
    )
    evaluator = cast(Evaluator, namespace["evaluator"])
    telemetry = cast(Telemetry, namespace["telemetry"])
    assert evaluator.calls == [update]
    assert evaluator.suite_calls == []
    assert ("suite", update) not in events
    assert len(telemetry.metrics) == 1  # sampler metrics only


def test_progress_fault_at_complete_synchronize() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    fault = ProgressBoom("complete-barrier-down")
    namespace, progress = _progress_fault_namespace(
        update=update,
        clock=clock,
        store=store,
        events=events,
        sampling_due=True,
        evaluation_due=True,
        fault_stage=_stage(update, "complete"),
        fault_method="synchronize",
        fault=fault,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(ProgressBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(
            observation
        )
    assert excinfo.value is fault
    # All rank-0 work finished; the complete stage was never published.
    assert progress.calls == expected_stage_calls(
        update, sampling_due=True, evaluation_due=True
    )
    check_stages(
        progress,
        (
            _stage(update, "training-samples"),
            _stage(update, "telemetry"),
            _stage(update, "concept-suite"),
        ),
    )
    assert _stage(update, "complete") not in progress._sequences
    evaluator = cast(Evaluator, namespace["evaluator"])
    telemetry = cast(Telemetry, namespace["telemetry"])
    assert evaluator.suite_calls == [update]
    assert len(telemetry.metrics) == 3


# ---------------------------------------------------------------------------
# REV2: W&B evaluation / concept metric payload faults
# ---------------------------------------------------------------------------


def _wandb_payload_fault_namespace(
    *,
    update: int,
    clock: Clock,
    store: Store,
    events: list[tuple[object, ...]],
    sampling_due: bool,
    evaluation_due: bool,
    metrics_faults: dict[str, BaseException],
) -> tuple[dict[str, object], RecordingProgress]:
    namespace = _matrix_namespace(
        store=store,
        progress_namespace="d1-wandb",
        rank=0,
        is_main=True,
        clock=clock,
        update=update,
        events=events,
        sampling_due=sampling_due,
        evaluation_due=evaluation_due,
        telemetry=FaultyTelemetry(
            events, None, None, None, metrics_faults
        ),
    )
    return namespace, cast(RecordingProgress, namespace["progress"])


def test_wandb_eval_metrics_fault_after_suite_stage() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    fault = SamplerBoom("eval-metrics-down")
    namespace, progress = _wandb_payload_fault_namespace(
        update=update,
        clock=clock,
        store=store,
        events=events,
        sampling_due=False,
        evaluation_due=True,
        metrics_faults={"eval": fault},
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(
            observation
        )
    assert excinfo.value is fault
    # The concept-suite stage already ran; the eval payload aborted the
    # run before any further payload and before complete.
    check_stages(
        progress,
        (_stage(update, "telemetry"), _stage(update, "concept-suite")),
    )
    assert _stage(update, "complete") not in progress._sequences
    telemetry = cast(Telemetry, namespace["telemetry"])
    assert telemetry.metrics == []
    evaluator = cast(Evaluator, namespace["evaluator"])
    assert evaluator.suite_calls == [update]


def test_wandb_concept_metrics_fault_after_eval_payload() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    fault = SamplerBoom("concept-metrics-down")
    namespace, progress = _wandb_payload_fault_namespace(
        update=update,
        clock=clock,
        store=store,
        events=events,
        sampling_due=False,
        evaluation_due=True,
        metrics_faults={"concept": fault},
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(
            observation
        )
    assert excinfo.value is fault
    check_stages(
        progress,
        (_stage(update, "telemetry"), _stage(update, "concept-suite")),
    )
    assert _stage(update, "complete") not in progress._sequences
    telemetry = cast(Telemetry, namespace["telemetry"])
    # The evaluation payload was submitted; the concept payload was not.
    assert len(telemetry.metrics) == 1
    assert "evaluation/fid" in telemetry.metrics[0][0]


# ---------------------------------------------------------------------------
# REV2: real in_calibration OR source (calibration/structural combinations)
# ---------------------------------------------------------------------------


def test_in_calibration_real_or_four_combinations(tmp_path: Path) -> None:
    clock = Clock(0.001)
    store = Store()
    calibration = SimpleNamespace(kind="calibration")
    structural = SimpleNamespace(kind="structural")
    combinations: tuple[tuple[object, object, bool], ...] = (
        (None, None, False),
        (calibration, None, True),
        (None, structural, True),
        (calibration, structural, True),
    )
    for index, (cal_value, struct_value, expected) in enumerate(combinations):
        events: list[tuple[object, ...]] = []
        computed = extract_in_calibration_assign(
            DEFAULT_PRODUCTION_PATH,
            {"calibration": cal_value, "structural": struct_value},
        )
        assert computed is expected

        namespace = build_observer_namespace(
            store=store,
            progress_namespace=f"d1-or-{index}",
            clock=clock,
            update=10,
            events=events,
            growth=Growth(events, 10, 100),
            sampler=Sampler(events, standard_result(10), None, None),
            telemetry=Telemetry(events, None, None, None),
            benchmark=make_benchmark(tmp_path, events, 0)
            if index == 3
            else None,
        )
        binding = bind_with_real_or(
            DEFAULT_PRODUCTION_PATH, namespace, cal_value, struct_value
        )
        observation = make_observation(events, 10, ())
        if expected:
            assert binding(observation) is None
            # Full bypass under the real OR: no stages, no callbacks.
            assert store.data == {}
            assert events == []
            sampler = cast(Sampler, namespace["training_sampler"])
            telemetry = cast(Telemetry, namespace["telemetry"])
            assert sampler.calls == []
            assert sampler.alphas == []
            assert telemetry.observers == []
            progress = cast(RecordingProgress, namespace["progress"])
            assert progress.calls == []
            assert progress._sequences == {}
        else:
            # The binding IS the extracted real observer function.
            assert binding is namespace[OBSERVER_FUNCTION]
        store.data = {}


# ---------------------------------------------------------------------------
# Normal due groups
# ---------------------------------------------------------------------------


def test_due_update_rank0_full_flow_exact_stages_and_payloads(
    tmp_path: Path,
) -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events0: list[tuple[object, ...]] = []
    events1: list[tuple[object, ...]] = []
    sampler0 = Sampler(events0, standard_result(update), None, None)
    evaluator0 = Evaluator(
        events0,
        frozenset({update}),
        Evaluation(0.12, 9.0, 0.3, 0.01, 0.002, 1.5, 8, 8),
        {"a": 1.0, "b": 2.0},
        None,
        None,
    )
    evaluator1 = Evaluator(events1, frozenset({update}), None, None, None, None)
    microbatches = (
        Microbatch(
            ("s1", "s2"),
            ("k1", "k2"),
            (Caption("a"), Caption("b")),
            4,
            3,
            10,
        ),
    )
    namespace0 = build_observer_namespace(
        store=store,
        progress_namespace="d1-full",
        clock=clock,
        update=update,
        events=events0,
        growth=Growth(events0, 10, 100),
        sampler=sampler0,
        telemetry=Telemetry(events0, None, None, None),
        evaluator=evaluator0,
        is_splits=2,
        benchmark=make_benchmark(tmp_path, events0, 0),
    )
    namespace1 = build_observer_namespace(
        store=store,
        progress_namespace="d1-full",
        rank=1,
        clock=clock,
        update=update,
        events=events1,
        growth=Growth(events1, 10, 100),
        sampler=Sampler(events1, standard_result(update), None, None),
        telemetry=Telemetry(events1, None, None, None),
        evaluator=evaluator1,
        is_splits=2,
        is_main_process=False,
    )
    observation0 = make_observation(events0, update, microbatches)
    observation1 = make_observation(events1, update, ())
    outcome0, _outcome1 = run_interleaved(
        cast(Callable[[Observation], None], namespace0[OBSERVER_FUNCTION]),
        cast(Callable[[Observation], None], namespace1[OBSERVER_FUNCTION]),
        observation0,
        observation1,
        20.0,
    )
    assert outcome0.error is None

    # Exact stage names, no extras, on the owner rank.
    progress0 = get_namespace_component(
        namespace0, "progress", DistributedProgress
    )
    check_stages(
        cast(DistributedProgress, progress0),
        tuple(_stage(update, name) for name in STAGE_NAMES),
    )

    # In-place read order: update, benchmark, due config, then work.
    assert [event[0] for event in events0] == expected_due_events(
        with_benchmark=True,
        sampling_due=True,
        read_every_updates=True,
        evaluation_due=True,
        is_main=True,
    )

    telemetry0 = cast(Telemetry, namespace0["telemetry"])
    # Image payload and sampler metric payload carry the exact values/keys.
    assert telemetry0.images == [
        ((Path("a.png"), Path("b.png")), ("alpha", "beta"), update)
    ]
    assert telemetry0.metrics[0] == (
        {
            "training_samples/count": 2,
            "condition_diagnostics/cond": 0.5,
        },
        update,
    )
    assert telemetry0.observers == [observation0]

    # The sample call receives the observation microbatches verbatim.
    assert sampler0.calls == [(update, observation0.microbatches)]

    # Growth alpha is read in place at each event, not pre-captured.
    alpha100 = canonical_growth_alpha(oracle_growth(10, 100), update)
    assert sampler0.alphas == [alpha100]
    assert evaluator0.alphas == [alpha100]

    # Evaluation and concept-suite payloads carry the exact keys/values.
    eval_payload, eval_update = telemetry0.metrics[1]
    assert eval_update == update
    assert tuple(sorted(eval_payload)) == tuple(sorted(EVAL_PAYLOAD_KEYS))
    assert eval_payload == {
        "evaluation/fid": 0.12,
        "evaluation/inception_score_mean": 9.0,
        "evaluation/inception_score_std": 0.3,
        "evaluation/kid_mean": 0.01,
        "evaluation/kid_std": 0.002,
        "evaluation/cmmd": 1.5,
        "evaluation/sample_count": 8,
        "evaluation/real_sample_count": 8,
        "evaluation/is_splits": 2,
    }
    concept_payload, concept_update = telemetry0.metrics[2]
    assert concept_update == update
    assert concept_payload == {
        "evaluation/concept/a": 1.0,
        "evaluation/concept/b": 2.0,
    }
    assert len(telemetry0.metrics) == 3

    # Benchmark trace: real BenchmarkTrace wrote the exact JSONL payload.
    raw_lines = (tmp_path / "rank-0.jsonl").read_text("utf-8").splitlines()
    assert len(raw_lines) == 1
    payload = json.loads(raw_lines[0])
    assert payload == {
        "schema_version": 1,
        "rank": 0,
        "successful_update": update,
        "update_wall_seconds": 1.5,
        "microbatches": [
            {
                "sample_ids": ["s1", "s2"],
                "shape_keys": ["k1", "k2"],
                "caption_sha256": [
                    hashlib.sha256(b"a").hexdigest(),
                    hashlib.sha256(b"b").hexdigest(),
                ],
                "image_tokens": 4,
                "text_tokens": 3,
                "dit_flops": 10,
            }
        ],
    }


def test_due_update_nonzero_rank_skips_callbacks_stages_participate() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events0: list[tuple[object, ...]] = []
    events1: list[tuple[object, ...]] = []
    sampler1 = Sampler(events1, standard_result(update), None, None)
    telemetry1 = Telemetry(events1, None, None, None)
    evaluator0 = Evaluator(
        events0,
        frozenset({update}),
        Evaluation(0.5, 8.0, 0.2, 0.02, 0.001, 1.0, 4, 4),
        {"only": 3.0},
        None,
        None,
    )
    evaluator1 = Evaluator(events1, frozenset({update}), None, None, None, None)
    namespace0 = build_observer_namespace(
        store=store,
        progress_namespace="d1-nonzero",
        clock=clock,
        update=update,
        events=events0,
        growth=Growth(events0, 10, 100),
        sampler=Sampler(events0, standard_result(update), None, None),
        telemetry=Telemetry(events0, None, None, None),
        evaluator=evaluator0,
        is_splits=1,
    )
    namespace1 = build_observer_namespace(
        store=store,
        progress_namespace="d1-nonzero",
        rank=1,
        clock=clock,
        update=update,
        events=events1,
        growth=Growth(events1, 10, 100),
        sampler=sampler1,
        telemetry=telemetry1,
        evaluator=evaluator1,
        is_splits=1,
        is_main_process=False,
    )
    observation0 = make_observation(events0, update, ())
    observation1 = make_observation(events1, update, ())
    outcome0, outcome1 = run_interleaved(
        cast(Callable[[Observation], None], namespace0[OBSERVER_FUNCTION]),
        cast(Callable[[Observation], None], namespace1[OBSERVER_FUNCTION]),
        observation0,
        observation1,
        20.0,
    )
    assert outcome0.error is None
    assert outcome1.error is None
    assert outcome0.value is None
    assert outcome1.value is None

    # Rank 1 performs only the pre-stage in-place reads, never callbacks.
    assert [event[0] for event in events1] == expected_nonowner_events(
        read_every_updates=True,
        evaluation_due=True,
    )
    assert sampler1.alphas == []
    assert sampler1.calls == []
    assert telemetry1.images == []
    assert telemetry1.metrics == []
    assert telemetry1.observers == []

    # evaluate() runs on every rank; the concept suite runs only on rank 0.
    assert evaluator0.calls == [update]
    assert evaluator1.calls == [update]
    assert evaluator0.suite_calls == [update]
    assert evaluator1.suite_calls == []

    # Stage participation: rank 0 publishes all four stages; rank 1 only
    # publishes its own synchronize point and observes the owner stages.
    progress0 = cast(DistributedProgress, namespace0["progress"])
    progress1 = cast(DistributedProgress, namespace1["progress"])
    check_stages(
        progress0, tuple(_stage(update, name) for name in STAGE_NAMES)
    )
    check_stages(progress1, (_stage(update, "complete"),))

    # The evaluation payload is submitted only on the main rank.
    telemetry0 = cast(Telemetry, namespace0["telemetry"])
    assert len(telemetry0.metrics) == 3
    assert telemetry1.metrics == []


def test_due_update_eval_disabled_both_ranks() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events0: list[tuple[object, ...]] = []
    events1: list[tuple[object, ...]] = []
    namespace0 = build_observer_namespace(
        store=store,
        progress_namespace="d1-noeval",
        clock=clock,
        update=update,
        events=events0,
        growth=Growth(events0, 10, 100),
        sampler=Sampler(events0, standard_result(update), None, None),
        telemetry=Telemetry(events0, None, None, None),
    )
    namespace1 = build_observer_namespace(
        store=store,
        progress_namespace="d1-noeval",
        rank=1,
        clock=clock,
        update=update,
        events=events1,
        growth=Growth(events1, 10, 100),
        sampler=Sampler(events1, standard_result(update), None, None),
        telemetry=Telemetry(events1, None, None, None),
        is_main_process=False,
    )
    observation0 = make_observation(events0, update, ())
    observation1 = make_observation(events1, update, ())
    outcome0, outcome1 = run_interleaved(
        cast(Callable[[Observation], None], namespace0[OBSERVER_FUNCTION]),
        cast(Callable[[Observation], None], namespace1[OBSERVER_FUNCTION]),
        observation0,
        observation1,
        20.0,
    )
    assert outcome0.error is None
    assert outcome1.error is None

    progress0 = cast(DistributedProgress, namespace0["progress"])
    progress1 = cast(DistributedProgress, namespace1["progress"])
    check_stages(
        progress0,
        (
            _stage(update, "training-samples"),
            _stage(update, "telemetry"),
            _stage(update, "complete"),
        ),
    )
    check_stages(progress1, (_stage(update, "complete"),))

    telemetry0 = cast(Telemetry, namespace0["telemetry"])
    assert len(telemetry0.metrics) == 1  # sampler metrics only
    assert [event[0] for event in events0] == expected_due_events(
        with_benchmark=False,
        sampling_due=True,
        read_every_updates=True,
        evaluation_due=False,
        is_main=True,
    )


def test_update_zero_sampling_not_due() -> None:
    update = 0
    clock = Clock(0.001)
    store = Store()
    events0: list[tuple[object, ...]] = []
    events1: list[tuple[object, ...]] = []
    sampler0 = Sampler(events0, standard_result(1), None, None)
    namespace0 = build_observer_namespace(
        store=store,
        progress_namespace="d1-zero",
        clock=clock,
        update=update,
        events=events0,
        growth=Growth(events0, None, None),
        sampler=sampler0,
        telemetry=Telemetry(events0, None, None, None),
    )
    namespace1 = build_observer_namespace(
        store=store,
        progress_namespace="d1-zero",
        rank=1,
        clock=clock,
        update=update,
        events=events1,
        growth=Growth(events1, None, None),
        sampler=Sampler(events1, standard_result(1), None, None),
        telemetry=Telemetry(events1, None, None, None),
        is_main_process=False,
    )
    observation0 = make_observation(events0, update, ())
    observation1 = make_observation(events1, update, ())
    outcome0, outcome1 = run_interleaved(
        cast(Callable[[Observation], None], namespace0[OBSERVER_FUNCTION]),
        cast(Callable[[Observation], None], namespace1[OBSERVER_FUNCTION]),
        observation0,
        observation1,
        20.0,
    )
    assert outcome0.error is None
    assert outcome1.error is None

    # No sampling stage and no concept stage; only telemetry and complete.
    progress0 = cast(DistributedProgress, namespace0["progress"])
    check_stages(
        progress0,
        (_stage(update, "telemetry"), _stage(update, "complete")),
    )
    assert sampler0.calls == []
    assert sampler0.alphas == []

    telemetry0 = cast(Telemetry, namespace0["telemetry"])
    assert telemetry0.observers == [observation0]
    assert telemetry0.metrics == []
    assert telemetry0.images == []

    # The due expression short-circuits before reading every_updates.
    assert [event[0] for event in events0] == [
        "update_read",
        "config_access",
        "observer",
    ]
    assert ("config_access", "every_updates") not in events0


# ---------------------------------------------------------------------------
# Calibration binding
# ---------------------------------------------------------------------------


def test_calibration_disabled_binds_real_observer_identity() -> None:
    clock = Clock(0.001)
    events: list[tuple[object, ...]] = []
    namespace = build_observer_namespace(
        progress_namespace="d1-cal-off",
        world_size=1,
        clock=clock,
        update=10,
        events=events,
        sampling_enabled=False,
        growth=Growth(events, None, None),
        sampler=None,
        telemetry=Telemetry(events, None, None, None),
    )
    observer = namespace[OBSERVER_FUNCTION]
    binding = namespace[CALIBRATION_BINDING]
    assert binding is observer


def test_calibration_enabled_bypasses_observer_side_effects() -> None:
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    sampler = Sampler(events, standard_result(10), None, None)
    telemetry = Telemetry(events, None, None, None)
    namespace = build_observer_namespace(
        store=store,
        progress_namespace="d1-cal-on",
        clock=clock,
        update=10,
        events=events,
        growth=Growth(events, 10, 100),
        sampler=sampler,
        telemetry=telemetry,
        in_calibration=True,
    )
    binding = cast(Callable[[Observation], None], namespace[CALIBRATION_BINDING])
    observation = make_observation(events, 10, ())
    assert binding(observation) is None
    # Full bypass: no stages published, no callbacks, no config/growth reads.
    assert store.data == {}
    assert events == []
    assert sampler.alphas == []
    assert sampler.calls == []
    assert telemetry.observers == []
    progress = cast(DistributedProgress, namespace["progress"])
    assert progress._sequences == {}


def test_calibration_enabled_with_benchmark_still_bypasses(tmp_path: Path) -> None:
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    sampler = Sampler(events, standard_result(10), None, None)
    telemetry = Telemetry(events, None, None, None)
    namespace = build_observer_namespace(
        store=store,
        progress_namespace="d1-cal-on-bench",
        clock=clock,
        update=10,
        events=events,
        growth=Growth(events, 10, 100),
        sampler=sampler,
        telemetry=telemetry,
        benchmark=make_benchmark(tmp_path, events, 0),
        in_calibration=True,
    )
    binding = cast(Callable[[Observation], None], namespace[CALIBRATION_BINDING])
    observation = make_observation(events, 10, ())
    assert binding(observation) is None
    assert store.data == {}
    assert events == []
    assert sampler.alphas == []
    assert sampler.calls == []
    assert telemetry.observers == []
    assert (tmp_path / "rank-0.jsonl").read_text("utf-8") == ""


def test_calibration_disabled_with_benchmark_binds_real_observer(
    tmp_path: Path,
) -> None:
    clock = Clock(0.001)
    events: list[tuple[object, ...]] = []
    namespace = build_observer_namespace(
        progress_namespace="d1-cal-off-bench",
        world_size=1,
        clock=clock,
        update=10,
        events=events,
        sampling_enabled=False,
        growth=Growth(events, None, None),
        sampler=None,
        telemetry=Telemetry(events, None, None, None),
        benchmark=make_benchmark(tmp_path, events, 0),
    )
    assert namespace[CALIBRATION_BINDING] is namespace[OBSERVER_FUNCTION]


# ---------------------------------------------------------------------------
# Dynamic growth alpha reads
# ---------------------------------------------------------------------------


def test_growth_alpha_read_dynamically_at_each_event() -> None:
    update = 20
    clock = Clock(0.001)
    events: list[tuple[object, ...]] = []
    growth = Growth(events, 10, 100)

    def mutate_after_first_alpha(alpha: float) -> None:
        del alpha
        growth._ramp_updates = 200

    sampler = Sampler(events, standard_result(update), None, mutate_after_first_alpha)
    main_evaluation = Evaluation(0.5, 8.0, 0.2, 0.02, 0.001, 1.0, 4, 4)
    namespace = build_observer_namespace(
        progress_namespace="d1-alpha",
        world_size=1,
        clock=clock,
        update=update,
        events=events,
        growth=growth,
        sampler=sampler,
        telemetry=Telemetry(events, None, None, None),
        evaluator=Evaluator(
            events, frozenset({update}), main_evaluation, {"a": 1.0}, None, None
        ),
        is_splits=1,
    )
    observation = make_observation(events, update, ())
    cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)

    # Each event reads the current ramp state, never a pre-captured value.
    alpha_at_100 = canonical_growth_alpha(oracle_growth(10, 100), update)
    alpha_at_200 = canonical_growth_alpha(oracle_growth(10, 200), update)
    assert alpha_at_100 != alpha_at_200
    assert sampler.alphas == [alpha_at_100]
    assert cast(Evaluator, namespace["evaluator"]).alphas == [alpha_at_200]


# ---------------------------------------------------------------------------
# Exception object identity, exact stages, and store failure detail
# ---------------------------------------------------------------------------


def _single_rank_exception_namespace(
    *,
    update: int,
    sampler: Sampler | None,
    telemetry: Telemetry,
    evaluator: Evaluator | None = None,
    is_splits: int | None = None,
    sampling_due: bool = True,
    growth: Growth | None,
    events: list[tuple[object, ...]],
    clock: Clock,
    store: Store,
) -> dict[str, object]:
    return build_observer_namespace(
        store=store,
        progress_namespace="d1-exc",
        clock=clock,
        update=update,
        events=events,
        every_updates=10 if sampling_due else 1000,
        growth=cast(Growth, growth),
        sampler=sampler,
        telemetry=telemetry,
        evaluator=evaluator,
        is_splits=is_splits,
    )


def test_sampler_exception_same_object_and_failed_stage_only() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    injected = SamplerBoom("boom")
    sampler = Sampler(events, standard_result(update), injected, None)
    telemetry = Telemetry(events, None, None, None)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=telemetry,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert excinfo.value is injected
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (_stage(update, "training-samples"),),
    )
    assert telemetry.images == []
    assert telemetry.metrics == []
    assert telemetry.observers == []
    check_store_has_detail(store, "SamplerBoom: boom")


def test_missing_sampler_raises_exact_runtime_error() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    telemetry = Telemetry(events, None, None, None)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=None,
        telemetry=telemetry,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(RuntimeError) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert str(excinfo.value) == "rank zero training sampler is unavailable"
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (_stage(update, "training-samples"),),
    )
    assert telemetry.images == []


def test_wandb_images_exception_same_object() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    injected = SamplerBoom("images-down")
    sampler = Sampler(events, standard_result(update), None, None)
    telemetry = Telemetry(events, injected, None, None)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=telemetry,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert excinfo.value is injected
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (_stage(update, "training-samples"),),
    )
    assert telemetry.images == []
    assert telemetry.metrics == []


def test_wandb_metrics_exception_same_object_after_images() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    injected = SamplerBoom("metrics-down")
    sampler = Sampler(events, standard_result(update), None, None)
    telemetry = Telemetry(events, None, injected, None)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=telemetry,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert excinfo.value is injected
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (_stage(update, "training-samples"),),
    )
    # Images were already submitted before the metrics failure.
    assert len(telemetry.images) == 1
    assert telemetry.metrics == []


def test_telemetry_observer_exception_same_object() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    injected = SamplerBoom("observer-down")
    sampler = Sampler(events, standard_result(update), None, None)
    telemetry = Telemetry(events, None, None, injected)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=telemetry,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert excinfo.value is injected
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (
            _stage(update, "training-samples"),
            _stage(update, "telemetry"),
        ),
    )
    assert len(telemetry.images) == 1
    assert telemetry.observers == []


def test_evaluator_evaluate_exception_same_object() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    injected = SamplerBoom("eval-down")
    sampler = Sampler(events, standard_result(update), None, None)
    evaluator = Evaluator(
        events, frozenset({update}), None, None, injected, None
    )
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=Telemetry(events, None, None, None),
        evaluator=evaluator,
        is_splits=1,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert excinfo.value is injected
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (
            _stage(update, "training-samples"),
            _stage(update, "telemetry"),
        ),
    )
    assert evaluator.suite_calls == []


def test_concept_suite_exception_same_object_and_no_eval_payload() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    injected = SamplerBoom("suite-down")
    sampler = Sampler(events, standard_result(update), None, None)
    evaluator = Evaluator(
        events,
        frozenset({update}),
        Evaluation(0.5, 8.0, 0.2, 0.02, 0.001, 1.0, 4, 4),
        {"a": 1.0},
        None,
        injected,
    )
    telemetry = Telemetry(events, None, None, None)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=telemetry,
        evaluator=evaluator,
        is_splits=1,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(SamplerBoom) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert excinfo.value is injected
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (
            _stage(update, "training-samples"),
            _stage(update, "telemetry"),
            _stage(update, "concept-suite"),
        ),
    )
    # Sampler metrics were logged; no evaluation or concept payload.
    assert len(telemetry.metrics) == 1
    check_store_has_detail(store, "SamplerBoom: suite-down")


def test_invalid_rank_zero_evaluation_result() -> None:
    update = 11
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    sampler = Sampler(events, standard_result(update), None, None)
    evaluator = Evaluator(
        events,
        frozenset({update}),
        None,
        {"a": 1.0},
        None,
        None,
    )
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=Telemetry(events, None, None, None),
        evaluator=evaluator,
        is_splits=1,
        sampling_due=False,
        growth=Growth(events, 10, 100),
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(RuntimeError) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert str(excinfo.value) == "rank zero evaluation is unavailable"
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (
            _stage(update, "telemetry"),
            _stage(update, "concept-suite"),
        ),
    )
    assert sampler.calls == []
    telemetry = cast(Telemetry, namespace["telemetry"])
    assert telemetry.metrics == []
    assert evaluator.suite_calls == [update]


def test_growth_ramp_origin_error_propagates_before_sampling() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events: list[tuple[object, ...]] = []
    # Ramp origin 50 precedes update 10: the real canonical_growth_alpha
    # rejects the in-place read at the first alpha event.
    growth = Growth(events, 50, 100)
    sampler = Sampler(events, standard_result(update), None, None)
    namespace = _single_rank_exception_namespace(
        update=update,
        sampler=sampler,
        telemetry=Telemetry(events, None, None, None),
        growth=growth,
        events=events,
        clock=clock,
        store=store,
    )
    observation = make_observation(events, update, ())
    with pytest.raises(ValueError) as excinfo:
        cast(Callable[[Observation], None], namespace[OBSERVER_FUNCTION])(observation)
    assert str(excinfo.value) == "successful update precedes growth ramp origin"
    check_stages(
        cast(DistributedProgress, namespace["progress"]),
        (_stage(update, "training-samples"),),
    )
    assert sampler.calls == []
    assert sampler.alphas == []


def test_nonzero_rank_surfaces_owner_failure_as_progress_error() -> None:
    update = 10
    clock = Clock(0.001)
    store = Store()
    events0: list[tuple[object, ...]] = []
    events1: list[tuple[object, ...]] = []
    injected = SamplerBoom("owner-boom")
    sampler0 = Sampler(events0, standard_result(update), injected, None)
    sampler1 = Sampler(events1, standard_result(update), None, None)
    telemetry1 = Telemetry(events1, None, None, None)
    namespace0 = build_observer_namespace(
        store=store,
        progress_namespace="d1-fail",
        clock=clock,
        update=update,
        events=events0,
        growth=Growth(events0, 10, 100),
        sampler=sampler0,
        telemetry=Telemetry(events0, None, None, None),
    )
    namespace1 = build_observer_namespace(
        store=store,
        progress_namespace="d1-fail",
        rank=1,
        clock=clock,
        update=update,
        events=events1,
        growth=Growth(events1, 10, 100),
        sampler=sampler1,
        telemetry=telemetry1,
        is_main_process=False,
    )
    observation0 = make_observation(events0, update, ())
    observation1 = make_observation(events1, update, ())
    outcome0, outcome1 = run_interleaved(
        cast(Callable[[Observation], None], namespace0[OBSERVER_FUNCTION]),
        cast(Callable[[Observation], None], namespace1[OBSERVER_FUNCTION]),
        observation0,
        observation1,
        20.0,
    )
    # Owner re-raises the exact same exception object.
    assert outcome0.error is injected
    # The non-owner participates in the stage and surfaces the owner
    # failure as a typed progress error with the stage name.
    assert type(outcome1.error) is DistributedProgressError
    assert outcome1.error is not None
    message = str(outcome1.error)
    assert f"'{_stage(update, 'training-samples')}'" in message
    assert "failed on rank 0" in message
    assert "SamplerBoom: owner-boom" in message
    # Rank 1 performs only the pre-stage in-place reads before it sees the
    # owner failure; no callbacks ever ran.
    assert [event[0] for event in events1] == expected_nonowner_events(
        read_every_updates=True,
        evaluation_due=False,
    )
    assert sampler1.calls == []
    assert sampler1.alphas == []
    assert telemetry1.observers == []
