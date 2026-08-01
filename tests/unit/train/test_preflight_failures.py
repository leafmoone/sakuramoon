from __future__ import annotations

# pyright: reportPrivateUsage=false
import hashlib
import json
import multiprocessing.reduction
import secrets
from collections.abc import Callable, Iterator
from dataclasses import replace
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

import sakuramoon.data.production as production_module
from sakuramoon.checkpoint.policy import CheckpointReason
from sakuramoon.checkpoint.schema import (
    CheckpointCadence,
    CheckpointIdentity,
    CheckpointKind,
    CheckpointManifest,
    FileRecord,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
    manifest_to_dict,
)
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.production import (
    AcceptedProductionBatchStream,
    ConfiguredDataLoader,
    ProductionBatchStreamIdentity,
)
from sakuramoon.data.service_protocol import DataServiceSessionIdentity
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.train import preflight as preflight_module
from sakuramoon.train.failures import FailureSnapshot, write_failure_bundle
from sakuramoon.train.preflight import (
    PREFLIGHT_CHECKS,
    PreflightError,
    ProductionSingleGpuCheckpointPublisher,
    RestoredSingleGpuCheckpoint,
    SingleGpuPreflightPlan,
    build_single_gpu_preflight_workload,
    require_accepted_preflight,
    run_single_gpu_preflight,
)
from sakuramoon.train.preflight import (
    _build_single_gpu_preflight_checks as build_single_gpu_preflight_checks,
)
from sakuramoon.train.runtime import DenseDiTAdapter
from sakuramoon.train.step import SingleGpuUpdateState

_HASH_A = "a" * 64
_HASH_B = "b" * 64
_HASH_C = "c" * 64
_HASH_D = "d" * 64


class _CheckpointPublisher:
    def __init__(self, path: Path) -> None:
        self.path = path

    def publish_preflight(
        self, identity: CheckpointIdentity, state: RawCheckpointState
    ) -> Path:
        del identity, state
        return self.path

    def publish_update(
        self,
        state: SingleGpuUpdateState,
        reason: CheckpointReason,
        cadence: CheckpointCadence,
    ) -> Path:
        del state, reason, cadence
        return self.path

    def apply_verified_retention(
        self,
        checkpoint: Path,
        manifest: CheckpointManifest,
        state: RawCheckpointState,
    ) -> None:
        del checkpoint, manifest, state

    def discard_preflight(self, checkpoint: Path) -> None:
        del checkpoint


class _CloseableIterator(Iterator[Any]):
    def __init__(self, *, close_error: bool = False) -> None:
        self.close_calls = 0
        self.close_error = close_error

    def __next__(self) -> Any:
        raise StopIteration

    def close(self) -> None:
        self.close_calls += 1
        if self.close_error:
            raise OSError("synthetic stream close failure")


def _restored(
    module: torch.nn.Module,
    optimizer: object,
    *,
    config_sha256: str = _HASH_A,
) -> RestoredSingleGpuCheckpoint:
    state = RawCheckpointState(
        SingleGpuUpdateState.initial(),
        GrowthCheckpointState(BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None),
        StageBudgetCheckpointState(0, 1),
        CheckpointCadence(0, 0.0),
    )
    restored = object.__new__(RestoredSingleGpuCheckpoint)
    restored._manifest = CheckpointManifest(
        CheckpointKind.RAW,
        CheckpointIdentity("unit", 0, config_sha256, _HASH_B, _HASH_C),
        (FileRecord("payload", 37, _HASH_D),),
    )
    restored._module = module
    restored._optimizer = optimizer
    restored._owner_pid = preflight_module.os.getpid()
    restored._path = Path("/test/checkpoint")
    restored._payload_bytes = 37
    restored._state = state
    restored._token = secrets.token_hex(32)
    preflight_module._RESTORED_CHECKPOINTS[restored._token] = restored
    return restored


def _stream(
    iterator: Iterator[Any] | None = None,
) -> AcceptedProductionBatchStream:
    identity = ProductionBatchStreamIdentity(
        _HASH_A,
        ConfiguredDataLoader(1, 2, 2, False, True),
        _HASH_B,
        DataServiceSessionIdentity(_HASH_B, 2).sha256,
        _HASH_D,
    )
    return production_module._issue_batch_stream(
        iter(()) if iterator is None else iterator,
        identity,
    )


def _plan(
    checks: dict[str, Callable[[], None]],
    *,
    iterator: Iterator[Any] | None = None,
) -> tuple[SingleGpuPreflightPlan, RestoredSingleGpuCheckpoint]:
    module = torch.nn.Linear(1, 1)
    optimizer = object()
    restored = _restored(module, optimizer)
    stream = _stream(iterator)
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)
    bindings = preflight_module._PreflightBindings(
        config=object(),
        resolved_config_sha256=_HASH_A,
        batches=stream,
        runtime=runtime,
        qwen=qwen,
        vae=vae,
        module=module,
        optimizer=optimizer,
        restored=restored,
        checkpoint_publisher=object(),
    )
    plan = object.__new__(SingleGpuPreflightPlan)
    plan._bindings = bindings
    plan._checks = tuple((name, checks[name]) for name in PREFLIGHT_CHECKS)
    plan._manifest_sha256 = _HASH_B
    plan._owner_pid = preflight_module.os.getpid()
    plan._service_session_sha256 = DataServiceSessionIdentity(_HASH_B, 2).sha256
    plan._token = secrets.token_hex(32)
    preflight_module._PREFLIGHT_PLANS[plan._token] = plan
    return plan, restored


def _passing_checks() -> dict[str, Callable[[], None]]:
    return {name: (lambda: None) for name in PREFLIGHT_CHECKS}


def _checkpoint_round_trip_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    publisher: _CheckpointPublisher,
) -> tuple[SingleGpuPreflightPlan, RestoredSingleGpuCheckpoint]:
    from sakuramoon.checkpoint import load as checkpoint_load_module

    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            data=SimpleNamespace(
                manifest=SimpleNamespace(sha256=_HASH_B),
                cache=SimpleNamespace(persistent_workers_per_rank=2),
            )
        ),
    )
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)
    resolved = tmp_path / "resolved.toml"
    if not resolved.exists():
        resolved.write_text("resolved\n")
    client = SimpleNamespace(
        identity=DataServiceSessionIdentity(_HASH_B, 2), health=lambda: False
    )
    module = torch.nn.Linear(1, 1)
    optimizer = object.__new__(IsolatedAdamW8bit)
    restored = _restored(module, optimizer)
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)

    def load_restored(
        _path: Path,
        _module: torch.nn.Module,
        _optimizer: IsolatedAdamW8bit,
        _identity: CheckpointIdentity,
    ) -> RawCheckpointState:
        return restored.state

    monkeypatch.setattr(
        checkpoint_load_module,
        "load_raw_checkpoint",
        load_restored,
    )
    plan = build_single_gpu_preflight_checks(
        loaded,
        repository_root=tmp_path,
        resolved_config_path=resolved,
        data_client=cast(Any, client),
        batches=_stream(),
        runtime=cast(Any, runtime),
        qwen=qwen,
        vae=vae,
        trainable_module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        workload=preflight_module._issue_verified_preflight_workload(
            {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
            config=config,
            runtime=runtime,
            trainable_module=module,
            optimizer=optimizer,
        ),
        checkpoint_publisher=publisher,
    )
    return plan, restored


def test_single_gpu_preflight_runs_every_fixed_check_once_in_order(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    iterator = _CloseableIterator()
    plan, _restored_handle = _plan(
        {
            name: (lambda current=name: calls.append(current))
            for name in PREFLIGHT_CHECKS
        },
        iterator=iterator,
    )

    accepted = run_single_gpu_preflight(plan, tmp_path / "preflight.json")

    require_accepted_preflight(accepted)
    assert calls == list(PREFLIGHT_CHECKS)
    assert accepted.report.schema_version == 2
    assert accepted.report.checkpoint_id == "unit"
    assert iterator.close_calls == 0
    with pytest.raises(PreflightError, match="builder-issued"):
        run_single_gpu_preflight(plan, tmp_path / "second.json")
    plan._bindings.batches.close()
    assert iterator.close_calls == 1


def test_arbitrary_callback_mapping_cannot_issue_acceptance(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="builder-issued"):
        run_single_gpu_preflight(
            cast(SingleGpuPreflightPlan, cast(object, _passing_checks())),
            tmp_path / "preflight.json",
        )


def test_preflight_failure_stops_and_redacts_message(tmp_path: Path) -> None:
    calls: list[str] = []
    checks: dict[str, Callable[[], None]] = {
        name: (lambda current=name: calls.append(current)) for name in PREFLIGHT_CHECKS
    }

    def fail() -> None:
        calls.append("dataset_revision")
        raise RuntimeError("secret-shaped diagnostic text")

    checks["dataset_revision"] = fail
    iterator = _CloseableIterator()
    plan, _restored_handle = _plan(checks, iterator=iterator)
    destination = tmp_path / "preflight.json"
    with pytest.raises(PreflightError, match="dataset_revision"):
        run_single_gpu_preflight(plan, destination)

    assert calls == ["resolved_config", "local_assets", "dataset_revision"]
    text = destination.read_text()
    assert "secret-shaped" not in text
    assert json.loads(text)["checks"][-1]["error_type"] == "RuntimeError"
    assert iterator.close_calls == 1


def test_preflight_preserves_check_and_stream_close_failures(tmp_path: Path) -> None:
    checks = _passing_checks()

    def fail() -> None:
        raise RuntimeError("synthetic check failure")

    checks["dataset_revision"] = fail
    iterator = _CloseableIterator(close_error=True)
    plan, _restored_handle = _plan(checks, iterator=iterator)

    with pytest.raises(BaseExceptionGroup) as captured:
        run_single_gpu_preflight(plan, tmp_path / "preflight.json")

    assert [type(error) for error in captured.value.exceptions] == [
        PreflightError,
        OSError,
    ]
    assert isinstance(captured.value.exceptions[0].__cause__, RuntimeError)
    assert iterator.close_calls == 1


def test_preflight_preserves_report_and_stream_close_failures(tmp_path: Path) -> None:
    destination = tmp_path / "preflight.json"
    destination.write_text("occupied")
    iterator = _CloseableIterator(close_error=True)
    plan, _restored_handle = _plan(_passing_checks(), iterator=iterator)

    with pytest.raises(BaseExceptionGroup) as captured:
        run_single_gpu_preflight(plan, destination)

    assert [type(error) for error in captured.value.exceptions] == [
        FileExistsError,
        OSError,
    ]
    assert iterator.close_calls == 1


def test_preflight_handle_is_pid_bound_and_cannot_be_pickled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan, _restored_handle = _plan(_passing_checks())

    with pytest.raises(PreflightError, match="cannot be serialized"):
        multiprocessing.reduction.ForkingPickler.dumps(plan)
    with pytest.raises(PreflightError, match="cannot be serialized"):
        multiprocessing.reduction.ForkingPickler.dumps(_restored_handle)
    accepted = run_single_gpu_preflight(plan, tmp_path / "preflight.json")

    with pytest.raises(PreflightError, match="cannot be serialized"):
        multiprocessing.reduction.ForkingPickler.dumps(accepted)
    monkeypatch.setattr(preflight_module.os, "getpid", lambda: 999_999)
    with pytest.raises(PreflightError, match="process-local"):
        require_accepted_preflight(accepted)
    with pytest.raises(PreflightError, match="process-local"):
        _ = _restored_handle.state


def test_public_builder_has_no_arbitrary_workload_callbacks() -> None:
    parameters = signature(
        preflight_module.build_single_gpu_preflight_checks
    ).parameters
    assert (
        not {
            "parameter_schema",
            "image_shapes",
            "text_shapes",
            "zero_update_loss",
            "optimizer_step",
            "sample",
            "checkpoint_round_trip",
        }
        & parameters.keys()
    )
    workload_parameters = signature(build_single_gpu_preflight_workload).parameters
    assert set(workload_parameters) == {
        "config",
        "runtime",
        "trainable_module",
        "optimizer",
    }


@pytest.mark.parametrize(
    ("configured", "observed"),
    [
        ("fa4_varlen", "fa4_varlen"),
        ("dense_sdpa_reference", "dense_sdpa"),
    ],
)
def test_attention_backend_binding_accepts_exact_dit_artifact(
    configured: str,
    observed: str,
) -> None:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(kernels=SimpleNamespace(attention_backend=configured)),
    )
    module = cast(
        torch.nn.Module,
        SimpleNamespace(
            dit=SimpleNamespace(
                artifact_config=lambda: {"attention_backend": observed}
            )
        ),
    )

    preflight_module._require_attention_backend_binding(config, module)


def test_attention_backend_binding_accepts_dense_reference_adapter() -> None:
    class _DenseReference(torch.nn.Module):
        @staticmethod
        def model_metadata() -> dict[str, int | str]:
            return {"prediction_type": "x"}

        @staticmethod
        def artifact_config() -> dict[str, object]:
            return {"attention_backend": "dense_sdpa"}

    adapter = DenseDiTAdapter(cast(Any, _DenseReference()))
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            kernels=SimpleNamespace(attention_backend="dense_sdpa_reference")
        ),
    )
    module = cast(torch.nn.Module, SimpleNamespace(dit=adapter))

    preflight_module._require_attention_backend_binding(config, module)
    assert adapter.model_metadata() == {"prediction_type": "x"}
    assert adapter.artifact_config() == {"attention_backend": "dense_sdpa"}


@pytest.mark.parametrize(
    ("artifact", "error", "message"),
    [
        (None, TypeError, "artifact config is unavailable"),
        (lambda: (), TypeError, "artifact config is malformed"),
        (dict, TypeError, "artifact attention backend is missing or malformed"),
        (
            lambda: {"attention_backend": "dense_sdpa"},
            ValueError,
            "differs from DiT artifact",
        ),
    ],
)
def test_attention_backend_binding_rejects_missing_malformed_or_mismatched_artifact(
    artifact: Callable[[], object] | None,
    error: type[Exception],
    message: str,
) -> None:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(kernels=SimpleNamespace(attention_backend="fa4_varlen")),
    )
    dit = (
        SimpleNamespace()
        if artifact is None
        else SimpleNamespace(artifact_config=artifact)
    )
    module = cast(torch.nn.Module, SimpleNamespace(dit=dit))

    with pytest.raises(error, match=message):
        preflight_module._require_attention_backend_binding(config, module)


def test_public_builder_rejects_custom_checkpoint_publisher(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="ProductionSingleGpuCheckpointPublisher"):
        preflight_module.build_single_gpu_preflight_checks(
            cast(LoadedConfig, object()),
            repository_root=tmp_path,
            resolved_config_path=tmp_path / "resolved.toml",
            data_client=cast(Any, object()),
            batches=cast(Any, object()),
            runtime=cast(Any, object()),
            qwen=object(),
            vae=object(),
            trainable_module=torch.nn.Linear(1, 1),
            optimizer=object(),
            restored_checkpoint=cast(Any, object()),
            workload=cast(Any, object()),
            checkpoint_publisher=cast(Any, _CheckpointPublisher(tmp_path)),
        )


def test_preflight_builder_binds_stream_client_checkpoint_and_measured_capacity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            data=SimpleNamespace(
                manifest=SimpleNamespace(sha256=_HASH_B),
                cache=SimpleNamespace(persistent_workers_per_rank=2),
            )
        ),
    )
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("resolved\n")
    client = SimpleNamespace(
        identity=DataServiceSessionIdentity(_HASH_B, 2),
        health=lambda: True,
    )
    stream = _stream()
    qwen = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    vae = torch.nn.Linear(1, 1).eval().requires_grad_(False)
    module = torch.nn.Linear(1, 1)
    optimizer = object()
    restored = _restored(module, optimizer)
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)
    measured: list[int] = []

    def storage(
        _config: object,
        _root: Path,
        *,
        checkpoint_payload_bytes: int,
    ) -> None:
        measured.append(checkpoint_payload_bytes)

    monkeypatch.setattr(preflight_module, "require_training_storage", storage)
    plan = build_single_gpu_preflight_checks(
        loaded,
        repository_root=tmp_path,
        resolved_config_path=resolved,
        data_client=cast(Any, client),
        batches=stream,
        runtime=cast(Any, runtime),
        qwen=qwen,
        vae=vae,
        trainable_module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        workload=preflight_module._issue_verified_preflight_workload(
            {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
            config=config,
            runtime=runtime,
            trainable_module=module,
            optimizer=optimizer,
        ),
        checkpoint_publisher=_CheckpointPublisher(tmp_path / "published"),
    )

    dict(plan._checks)["storage_capacity"]()
    with pytest.raises(RuntimeError, match="no training lease"):
        dict(plan._checks)["dataset_revision"]()
    assert measured == [37]
    assert plan._bindings.batches is stream
    assert plan._bindings.restored is restored
    assert isinstance(plan._checks, tuple)


def test_preflight_runtime_check_rejects_checkpoint_drift_before_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            run=SimpleNamespace(intent="train", stage="S0"),
            stage=SimpleNamespace(
                name="S0",
                enabled=True,
                world_size=1,
                depth=16,
                resolution=256,
                activation_checkpoint_mode="none",
                planned_updates=1,
            ),
            distributed=SimpleNamespace(backend="native", world_size=1),
            failure=SimpleNamespace(allow_force_bypass=False),
            growth=SimpleNamespace(enabled=False),
            kernels=SimpleNamespace(attention_backend="fa4_varlen"),
            data=SimpleNamespace(
                manifest=SimpleNamespace(sha256=_HASH_B),
                cache=SimpleNamespace(persistent_workers_per_rank=2),
            ),
        ),
    )
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("resolved\n")
    client = SimpleNamespace(
        identity=DataServiceSessionIdentity(_HASH_B, 2),
        health=lambda: False,
    )

    class _Dit:
        @staticmethod
        def artifact_config() -> dict[str, object]:
            return {"attention_backend": "fa4_varlen"}

    class _Module(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dit = _Dit()

    module = _Module()
    optimizer = object()
    restored = _restored(module, optimizer)
    restored._state = RawCheckpointState(
        trainer=SingleGpuUpdateState(5, 5, 5),
        growth=GrowthCheckpointState(
            BASE_SLOT_IDS, 1.0, "S0", 1, 256, None, None
        ),
        stage_budget=StageBudgetCheckpointState(5, 6),
        checkpoint_cadence=CheckpointCadence(5, 0.0),
    )
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(
        qwen=qwen,
        vae=vae,
        composite=module,
        growth_alpha=1.0,
    )
    plan = build_single_gpu_preflight_checks(
        loaded,
        repository_root=tmp_path,
        resolved_config_path=resolved,
        data_client=cast(Any, client),
        batches=_stream(),
        runtime=cast(Any, runtime),
        qwen=qwen,
        vae=vae,
        trainable_module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        workload=preflight_module._issue_verified_preflight_workload(
            {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
            config=config,
            runtime=runtime,
            trainable_module=module,
            optimizer=optimizer,
        ),
        checkpoint_publisher=_CheckpointPublisher(tmp_path / "published"),
    )
    def accept_local_asset(_root: Path) -> None:
        pass

    monkeypatch.setattr(preflight_module, "require_local_qwen", accept_local_asset)
    monkeypatch.setattr(preflight_module, "require_local_vae", accept_local_asset)
    destination = tmp_path / "preflight.json"

    with pytest.raises(PreflightError, match="single_gpu_runtime"):
        run_single_gpu_preflight(plan, destination)

    report = json.loads(destination.read_text())
    assert report["passed"] is False
    assert [check["name"] for check in report["checks"]] == [
        "resolved_config",
        "local_assets",
        "dataset_revision",
        "single_gpu_runtime",
    ]
    assert report["checks"][-1]["error_type"] == "ValueError"


def test_checkpoint_round_trip_rejects_publisher_without_new_raw(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sakuramoon.checkpoint import load as checkpoint_load_module

    config = cast(
        RuntimeConfig,
        SimpleNamespace(
            data=SimpleNamespace(
                manifest=SimpleNamespace(sha256=_HASH_B),
                cache=SimpleNamespace(persistent_workers_per_rank=2),
            )
        ),
    )
    loaded = LoadedConfig(config, (), "resolved\n", _HASH_A)
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("resolved\n")
    client = SimpleNamespace(
        identity=DataServiceSessionIdentity(_HASH_B, 2), health=lambda: False
    )
    module = torch.nn.Linear(1, 1)
    optimizer = object.__new__(IsolatedAdamW8bit)
    restored = _restored(module, optimizer)
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)

    def load_restored(
        _path: Path,
        _module: torch.nn.Module,
        _optimizer: IsolatedAdamW8bit,
        _identity: CheckpointIdentity,
    ) -> RawCheckpointState:
        return restored.state

    monkeypatch.setattr(
        checkpoint_load_module,
        "load_raw_checkpoint",
        load_restored,
    )

    class _NoArtifactPublisher(_CheckpointPublisher):
        def publish_preflight(
            self, identity: CheckpointIdentity, state: RawCheckpointState
        ) -> Path:
            del identity, state
            return cast(Path, None)

    plan = build_single_gpu_preflight_checks(
        loaded,
        repository_root=tmp_path,
        resolved_config_path=resolved,
        data_client=cast(Any, client),
        batches=_stream(),
        runtime=cast(Any, runtime),
        qwen=qwen,
        vae=vae,
        trainable_module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        workload=preflight_module._issue_verified_preflight_workload(
            {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
            config=config,
            runtime=runtime,
            trainable_module=module,
            optimizer=optimizer,
        ),
        checkpoint_publisher=_NoArtifactPublisher(tmp_path / "unused"),
    )

    with pytest.raises(TypeError, match="new RAW path"):
        dict(plan._checks)["checkpoint_round_trip"]()


def test_production_checkpoint_publisher_retains_only_after_fresh_qualification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sakuramoon.checkpoint import policy as checkpoint_policy
    from sakuramoon.checkpoint import save as checkpoint_save

    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    resolved_config = b"resolved=true\n"
    config_sha256 = hashlib.sha256(resolved_config).hexdigest()
    module = torch.nn.Linear(1, 1)
    optimizer = object.__new__(IsolatedAdamW8bit)
    restored = _restored(module, optimizer, config_sha256=config_sha256)
    saved: list[tuple[CheckpointIdentity, RawCheckpointState]] = []
    retention: list[object] = []

    def save(
        root: Path,
        identity: CheckpointIdentity,
        _module: torch.nn.Module,
        _optimizer: IsolatedAdamW8bit,
        state: RawCheckpointState,
        *,
        resolved_config: bytes,
    ) -> object:
        assert root == checkpoint_root
        assert resolved_config == b"resolved=true\n"
        path = root / f"ckpt_{identity.update}_{identity.checkpoint_id}"
        path.mkdir()
        manifest = CheckpointManifest(
            CheckpointKind.RAW,
            identity,
            (FileRecord("payload", 1, _HASH_D),),
        )
        (path / "manifest.json").write_text(json.dumps(manifest_to_dict(manifest)))
        saved.append((identity, state))
        return SimpleNamespace(path=path)

    plan = object()
    monkeypatch.setattr(checkpoint_save, "save_raw_checkpoint", save)

    def plan_retention(
        _root: Path, *, accepted_checkpoint_ids: frozenset[str]
    ) -> object:
        retention.append(accepted_checkpoint_ids)
        return plan

    def apply_retention(
        _root: Path,
        observed: object,
        *,
        accepted_checkpoint_ids: frozenset[str],
    ) -> None:
        retention.extend((observed, accepted_checkpoint_ids))

    monkeypatch.setattr(
        checkpoint_policy,
        "plan_raw_retention",
        plan_retention,
    )
    monkeypatch.setattr(
        checkpoint_policy,
        "apply_raw_retention",
        apply_retention,
    )
    publisher = ProductionSingleGpuCheckpointPublisher(
        checkpoint_root=checkpoint_root,
        resolved_config=resolved_config,
        module=module,
        optimizer=optimizer,
        restored_checkpoint=restored,
        accepted_checkpoint_ids=frozenset({"accepted"}),
    )
    base = restored.manifest.identity
    preflight_identity = CheckpointIdentity(
        "preflight-test",
        base.update,
        base.config_sha256,
        base.dependency_sha256,
        base.parameter_schema_sha256,
    )

    temporary = publisher.publish_preflight(preflight_identity, restored.state)
    assert temporary.is_dir()
    publisher.discard_preflight(temporary)
    assert not temporary.exists()

    state = SingleGpuUpdateState(1, 1, 1)
    cadence = CheckpointCadence(1, 1.0)
    durable = publisher.publish_update(
        state,
        CheckpointReason.STAGE_FINALIZE,
        cadence,
    )
    assert durable.is_dir()
    assert saved[-1][1] == RawCheckpointState(
        state,
        restored.state.growth,
        restored.state.stage_budget,
        cadence,
    )
    assert retention == []
    durable_manifest = CheckpointManifest(
        CheckpointKind.RAW,
        saved[-1][0],
        (FileRecord("payload", 1, _HASH_D),),
    )
    with pytest.raises(PreflightError, match="pending update"):
        publisher.apply_verified_retention(
            checkpoint_root / "not-published",
            durable_manifest,
            saved[-1][1],
        )
    with pytest.raises(PreflightError, match="changed before retention"):
        publisher.apply_verified_retention(
            durable,
            replace(durable_manifest, identity=replace(saved[-1][0], update=2)),
            saved[-1][1],
        )
    with pytest.raises(PreflightError, match="changed before retention"):
        publisher.apply_verified_retention(
            durable,
            durable_manifest,
            restored.state,
        )
    assert retention == []

    publisher.apply_verified_retention(durable, durable_manifest, saved[-1][1])
    assert retention == [frozenset({"accepted"}), plan, frozenset({"accepted"})]


def test_checkpoint_round_trip_rejects_noop_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sakuramoon.checkpoint import load as checkpoint_load_module

    published = tmp_path / "published"

    class _NoopCleanupPublisher(_CheckpointPublisher):
        identity: CheckpointIdentity | None = None
        state: RawCheckpointState | None = None

        def publish_preflight(
            self, identity: CheckpointIdentity, state: RawCheckpointState
        ) -> Path:
            self.identity = identity
            self.state = state
            self.path.mkdir()
            return self.path

    publisher = _NoopCleanupPublisher(published)
    plan, restored = _checkpoint_round_trip_plan(tmp_path, monkeypatch, publisher)

    def read_back(_path: Path) -> tuple[CheckpointManifest, RawCheckpointState]:
        assert publisher.identity is not None and publisher.state is not None
        return (
            CheckpointManifest(
                CheckpointKind.RAW,
                publisher.identity,
                (FileRecord("payload", 1, _HASH_D),),
            ),
            publisher.state,
        )

    monkeypatch.setattr(checkpoint_load_module, "read_raw_checkpoint_state", read_back)
    with pytest.raises(RuntimeError, match="cleanup left"):
        dict(plan._checks)["checkpoint_round_trip"]()
    assert published.is_dir()
    assert restored.state.trainer == SingleGpuUpdateState.initial()


def test_checkpoint_round_trip_preserves_validation_and_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sakuramoon.checkpoint import load as checkpoint_load_module
    from sakuramoon.checkpoint.schema import CheckpointError

    class _FailingCleanupPublisher(_CheckpointPublisher):
        def discard_preflight(self, checkpoint: Path) -> None:
            del checkpoint
            raise OSError("cleanup failed")

    publisher = _FailingCleanupPublisher(tmp_path / "published")
    plan, _restored_handle = _checkpoint_round_trip_plan(
        tmp_path, monkeypatch, publisher
    )

    def fail_read(_path: Path) -> tuple[CheckpointManifest, RawCheckpointState]:
        raise CheckpointError("validation failed")

    monkeypatch.setattr(
        checkpoint_load_module,
        "read_raw_checkpoint_state",
        fail_read,
    )

    with pytest.raises(BaseExceptionGroup) as captured:
        dict(plan._checks)["checkpoint_round_trip"]()
    assert [type(error) for error in captured.value.exceptions] == [
        CheckpointError,
        OSError,
    ]


def test_builder_rejects_cross_client_or_checkpoint_config_identity(
    tmp_path: Path,
) -> None:
    loaded = cast(
        LoadedConfig,
        SimpleNamespace(
            config=SimpleNamespace(
                data=SimpleNamespace(
                    manifest=SimpleNamespace(sha256=_HASH_B),
                    cache=SimpleNamespace(persistent_workers_per_rank=2),
                )
            ),
            resolved_sha256=_HASH_A,
            resolved_toml="resolved\n",
        ),
    )
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("resolved\n")
    module = torch.nn.Linear(1, 1)
    optimizer = object()
    restored = _restored(module, optimizer, config_sha256=_HASH_D)
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)
    client = SimpleNamespace(
        identity=DataServiceSessionIdentity(_HASH_B, 2), health=lambda: True
    )
    with pytest.raises(ValueError, match="checkpoint resolved identity"):
        build_single_gpu_preflight_checks(
            loaded,
            repository_root=tmp_path,
            resolved_config_path=resolved,
            data_client=cast(Any, client),
            batches=_stream(),
            runtime=cast(Any, runtime),
            qwen=qwen,
            vae=vae,
            trainable_module=module,
            optimizer=optimizer,
            restored_checkpoint=restored,
            workload=preflight_module._issue_verified_preflight_workload(
                {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
                config=loaded.config,
                runtime=runtime,
                trainable_module=module,
                optimizer=optimizer,
            ),
            checkpoint_publisher=_CheckpointPublisher(tmp_path / "published"),
        )


@pytest.mark.parametrize(
    "binding", ["config", "runtime", "qwen", "vae", "module", "optimizer"]
)
def test_verified_workload_rejects_cross_resource_reuse(
    tmp_path: Path,
    binding: str,
) -> None:
    def make_config() -> RuntimeConfig:
        return cast(
            RuntimeConfig,
            SimpleNamespace(
                data=SimpleNamespace(
                    manifest=SimpleNamespace(sha256=_HASH_B),
                    cache=SimpleNamespace(persistent_workers_per_rank=2),
                )
            ),
        )

    config = make_config()
    module = torch.nn.Linear(1, 1)
    optimizer = object()
    qwen = object()
    vae = object()
    runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)
    workload = preflight_module._issue_verified_preflight_workload(
        {name: (lambda: None) for name in preflight_module._WORKLOAD_CHECKS},
        config=config,
        runtime=runtime,
        trainable_module=module,
        optimizer=optimizer,
    )

    plan_config = config
    plan_module = module
    plan_optimizer = optimizer
    plan_qwen = qwen
    plan_vae = vae
    plan_runtime = runtime
    if binding == "config":
        plan_config = make_config()
    elif binding == "runtime":
        plan_runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=module)
    elif binding == "qwen":
        plan_qwen = object()
        plan_runtime = SimpleNamespace(qwen=plan_qwen, vae=vae, composite=module)
    elif binding == "vae":
        plan_vae = object()
        plan_runtime = SimpleNamespace(qwen=qwen, vae=plan_vae, composite=module)
    elif binding == "module":
        plan_module = torch.nn.Linear(1, 1)
        plan_runtime = SimpleNamespace(qwen=qwen, vae=vae, composite=plan_module)
    elif binding == "optimizer":
        plan_optimizer = object()

    restored = _restored(plan_module, plan_optimizer)
    loaded = LoadedConfig(plan_config, (), "resolved\n", _HASH_A)
    resolved = tmp_path / "resolved.toml"
    resolved.write_text("resolved\n")
    client = SimpleNamespace(
        identity=DataServiceSessionIdentity(_HASH_B, 2), health=lambda: False
    )

    with pytest.raises(PreflightError, match="exact resources"):
        build_single_gpu_preflight_checks(
            loaded,
            repository_root=tmp_path,
            resolved_config_path=resolved,
            data_client=cast(Any, client),
            batches=_stream(),
            runtime=cast(Any, plan_runtime),
            qwen=plan_qwen,
            vae=plan_vae,
            trainable_module=plan_module,
            optimizer=plan_optimizer,
            restored_checkpoint=restored,
            workload=workload,
            checkpoint_publisher=_CheckpointPublisher(tmp_path / "published"),
        )


def test_preflight_does_not_replace_report_symlink(tmp_path: Path) -> None:
    destination = tmp_path / "preflight.json"
    destination.symlink_to(tmp_path / "target.json")
    plan, _restored_handle = _plan(_passing_checks())
    with pytest.raises(FileExistsError, match="already exists"):
        run_single_gpu_preflight(plan, destination)
    assert destination.is_symlink()


def test_preflight_rejects_relative_symlink_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "linked").symlink_to(real, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    plan, _restored_checkpoint = _plan(_passing_checks())

    with pytest.raises(ValueError, match="parent may not be a symlink"):
        run_single_gpu_preflight(plan, Path("linked/preflight.json"))

    assert not (real / "preflight.json").exists()


def test_failure_bundle_is_atomic_immutable_and_redacted(tmp_path: Path) -> None:
    snapshot = FailureSnapshot("failure-1", "update", "OutOfMemoryError", 3, 2, 8)
    bundle = write_failure_bundle(tmp_path, snapshot)
    assert (bundle / "COMPLETE").is_file()
    assert json.loads((bundle / "failure.json").read_text())["error_type"] == (
        "OutOfMemoryError"
    )
    with pytest.raises(FileExistsError):
        write_failure_bundle(tmp_path, snapshot)


def test_failure_bundle_does_not_replace_a_dangling_symlink(tmp_path: Path) -> None:
    snapshot = FailureSnapshot("failure-1", "update", "RuntimeError", 1, 0, 0)
    target = tmp_path / snapshot.failure_id
    target.symlink_to(tmp_path / "missing")

    with pytest.raises(FileExistsError, match="already exists"):
        write_failure_bundle(tmp_path, snapshot)

    assert target.is_symlink()
