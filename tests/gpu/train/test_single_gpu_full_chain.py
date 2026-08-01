from __future__ import annotations

import gc
import hashlib
import io
import json
import multiprocessing as mp
import os
import tarfile
import threading
import time
import tomllib
from pathlib import Path
from typing import Any, Protocol, cast

import pytest
import torch
from PIL import Image
from torch.utils.data import get_worker_info

from sakuramoon.checkpoint.load import (
    discover_complete_checkpoints,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.policy import CheckpointCadence, CheckpointReason
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.config.load import LoadedConfig
from sakuramoon.config.resolve import resolved_config_bytes, resolved_config_sha256
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
from sakuramoon.data.production import ProductionPipelineFactory
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX, FramingContract
from sakuramoon.data.service import (
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)
from sakuramoon.data.validation import VALIDATION_SAMPLE_COUNT
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.model.dit import PackedDiT
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.optim.adamw8bit import build_adamw8bit
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.preflight import (
    ProductionSingleGpuCheckpointPublisher,
    build_single_gpu_preflight_checks,
    build_single_gpu_preflight_workload,
    restore_single_gpu_checkpoint,
    run_single_gpu_preflight,
)
from sakuramoon.train.runtime import (
    SingleGpuBatchRuntime,
    SuccessfulTrainingObservation,
    run_single_gpu_training,
)
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    TrainableComposite,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def _set_path(payload: dict[str, Any], path: str, value: object) -> None:
    current = payload
    parts = path.split(".")
    for part in parts[:-1]:
        current = cast(dict[str, Any], current[part])
    current[parts[-1]] = value


def _config(
    repository_root: Path,
    *,
    overrides: dict[str, object] | None = None,
) -> RuntimeConfig:
    payload = tomllib.loads(
        (repository_root / "config/examples/all_options.example.toml").read_text()
    )
    values: dict[str, object] = {
        "storage.shared_mount_source": "server.example:/governed/export",
        "storage.minimum_free_gib": 8,
        "data.source.revision": "c" * 40,
        "data.manifest.path": "synthetic/train-manifest.jsonl",
        "data.manifest.sha256": "3" * 64,
        "data.cache.low_watermark_gib": 8,
        "data.cache.high_watermark_gib": 16,
        "data.cache.download_concurrency": 2,
        "data.cache.verified_shard_lookahead": 2,
        "data.cache.persistent_workers_per_rank": 2,
        "data.cache.ready_batches_per_rank": 2,
        "data.service.request_timeout_seconds": 30.0,
        "data.service.lease_channel_capacity": 2,
        "data.service.ack_channel_capacity": 2,
        "data.transport.connect_timeout_seconds": 10.0,
        "data.transport.read_timeout_seconds": 30.0,
        "data.transport.max_retries": 2,
        "data.transport.retry_backoff_seconds": 0.0,
        "data.transport.stream_chunk_bytes": 1_048_576,
        "data.validation.manifest_path": "synthetic/validation-manifest.jsonl",
        "data.validation.manifest_sha256": "4" * 64,
        "checkpoint.slots": 3,
        "kernels.attention_backend": "fa4_varlen",
        "stage.local_batch": 1,
        "stage.accumulation": 1,
        "stage.global_batch": 1,
        "stage.activation_checkpoint_mode": "none",
        "stage.planned_updates": 10,
        "stage.planned_valid_samples": 10,
        "stage.planned_equivalent_data_passes": 1.0,
        "stage.planned_dit_flops": 1.0,
        "stage.planned_wall_time_hours": 1.0,
        "profiling.schedule_updates": 10,
        "benchmark.profile_trace_updates": 5,
        "logging.flush_every_updates": 1,
        "logging.observer_queue_capacity": 2,
        "logging.observer_event_timeout_seconds": 30.0,
        "wandb.entity": "synthetic-entity",
        "wandb.queue_capacity": 2,
        "evaluation.fid.every_successful_updates": 10,
        "evaluation.fid.trend_samples": 100,
        "evaluation.fid.feature_extractor": "synthetic-locked-extractor",
        "evaluation.fid.feature_extractor_version": "synthetic-version",
        "evaluation.fid.preprocess_sha256": "6" * 64,
        "evaluation.fid.real_stats_sha256": "5" * 64,
        "evaluation.is.every_successful_updates": 10,
        "evaluation.is.trend_samples": 100,
        "evaluation.is.splits": 10,
        "evaluation.prompt_manifest_path": "synthetic/prompts.json",
        "evaluation.prompt_manifest_sha256": "7" * 64,
        "evaluation.gpu_index": 0,
        "evaluation.training_paused": True,
    }
    if overrides is not None:
        values.update(overrides)
    for path, value in values.items():
        _set_path(payload, path, value)
    global_batch = (
        cast(int, payload["stage"]["local_batch"])
        * cast(int, payload["stage"]["accumulation"])
        * cast(int, payload["stage"]["world_size"])
    )
    payload["stage"]["global_batch"] = global_batch
    payload["stage"]["planned_valid_samples"] = (
        global_batch * cast(int, payload["stage"]["planned_updates"])
    )
    return RuntimeConfig.model_validate(payload)


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _WorkerRecordingTokenizer:
    def __init__(self, record_root: Path) -> None:
        self.record_root = record_root

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        if text == SYSTEM_PREFIX:
            worker = get_worker_info()
            if worker is None:
                raise AssertionError("production tokenizer did not run in a worker")
            self.record_root.mkdir(parents=True, exist_ok=True)
            marker = self.record_root / f"worker-{worker.id}-pid-{os.getpid()}"
            marker.touch(exist_ok=True)
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [] if not text else [300]


def _worker_markers(record_root: Path) -> dict[int, int]:
    observed: dict[int, int] = {}
    for marker in record_root.glob("worker-*-pid-*"):
        _, worker_id, _, worker_pid = marker.name.split("-")
        worker_identity = int(worker_id)
        process_identity = int(worker_pid)
        previous = observed.get(worker_identity)
        if previous is not None and previous != process_identity:
            raise AssertionError("a persistent worker id changed PID")
        observed[worker_identity] = process_identity
    return observed


def _wait_for_worker_markers(record_root: Path, *, expected: int) -> None:
    expected_ids = set(range(expected))
    deadline = time.monotonic() + 30.0
    while True:
        observed = _worker_markers(record_root)
        if set(observed) == expected_ids:
            pids = set(observed.values())
            if len(pids) != expected or os.getpid() in pids:
                raise AssertionError("worker markers do not identify distinct children")
            return
        if not set(observed) <= expected_ids:
            raise AssertionError(f"unexpected worker ids observed: {observed}")
        if time.monotonic() >= deadline:
            raise AssertionError(
                f"expected {expected} worker markers before stream close, "
                f"observed {observed}"
            )
        time.sleep(0.01)


def _observe_rejection(_reason: str) -> None:
    return None


class _TarTransport:
    stream_chunk_bytes = 4096

    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies

    def download(
        self,
        manifest: DatasetManifest,
        shard: ShardRecord,
        output: _Writer,
    ) -> None:
        assert manifest.shard(shard.path) == shard
        output.write(self.bodies[shard.path])


class _BoundedClient:
    def __init__(self, client: DataServiceClient, lease_budget: int) -> None:
        self.client = client
        self.identity = client.identity
        self.remaining = lease_budget
        self.leased_worker_ids: set[int] = set()

    def health(self) -> bool:
        return self.client.health()

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if self.remaining == 0:
            return None
        descriptor = self.client.lease(worker_id)
        if descriptor is not None:
            self.remaining -= 1
            self.leased_worker_ids.add(descriptor.worker_id)
        return descriptor

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self.client.acknowledge(descriptor)


def _tar_bytes(sample_id: int) -> bytes:
    metadata = json.dumps(
        {
            "captions": {"nl2": "", "nl3": ""},
            "dropout": {"candidate_tags": []},
            "id": sample_id,
            "image": {"height": 512, "width": 640},
            "multicaptions": {"vibes": ""},
            "nsfw": "safe",
            "tags": {
                "artist": [],
                "character": [],
                "copyright": [],
                "general": [],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    image = io.BytesIO()
    Image.new("RGB", (640, 512), color=(10, 20, 30)).save(image, format="JPEG")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as archive:
        for extension, payload in (("json", metadata), ("jpg", image.getvalue())):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return body.getvalue()


def _dataset() -> tuple[DatasetManifest, dict[str, bytes]]:
    source = DatasetSourceIdentity(
        repo_id="leafmoone/webdataset_danbooru",
        revision="a" * 40,
        license_id="test-license",
        access_terms="test-terms",
    )
    bodies = {
        f"release/{index:06d}.tar": _tar_bytes(9000 + index) for index in range(4)
    }
    records = tuple(
        ShardRecord(
            path=path,
            release="1_2024",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            samples=1,
        )
        for path, body in bodies.items()
    )
    return DatasetManifest.from_shards(source, records), bodies


def _validation_payload() -> bytes:
    rows = (
        json.dumps(
            {
                "aspect_bucket": "square",
                "caption_available": False,
                "id": sample_id,
                "release": "1_2024",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for sample_id in range(1, VALIDATION_SAMPLE_COUNT + 1)
    )
    return ("\n".join(rows) + "\n").encode()


def _run_data_service(
    root: Path,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
    identity: DataServiceSessionIdentity,
    stop: Any,
    ready: Any,
) -> None:
    total_bytes = sum(len(body) for body in bodies.values())
    cache = ShardCache(
        (root / "cache").absolute(),
        manifest,
        cast(ModelScopeDatasetTransport, _TarTransport(bodies)),
        CacheQuota(total_bytes, total_bytes * 2),
    )
    service = DataSupplyService(
        manifest,
        cache,
        root / "mainset.json",
        Path("/run/sakuramoon/data-service.lock"),
        identity,
        DataServiceLimits(2, 3, 2, 2),
    )
    DataServiceServer(
        service,
        Path("/run/sakuramoon/data-service.sock"),
        request_timeout_seconds=5.0,
    ).serve(cast(threading.Event, stop), ready_callback=ready.set)


def _start_data_service(
    root: Path,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
) -> tuple[Any, Any, DataServiceSessionIdentity]:
    socket_path = Path("/run/sakuramoon/data-service.sock")
    if socket_path.exists() or socket_path.is_symlink():
        raise AssertionError("governed data-service socket is already occupied")
    identity = DataServiceSessionIdentity(manifest_sha256(manifest), 2)
    context = cast(Any, mp.get_context("spawn"))
    stop = context.Event()
    ready = context.Event()
    process = context.Process(
        target=_run_data_service,
        args=(root, manifest, bodies, identity, stop, ready),
    )
    process.start()
    deadline = time.monotonic() + 30.0
    while not ready.wait(timeout=0.25):
        if not process.is_alive():
            process.join(timeout=1.0)
            raise AssertionError(
                "data service exited before readiness: "
                f"exitcode={process.exitcode}"
            )
        if time.monotonic() >= deadline:
            stop.set()
            process.join(timeout=5.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            raise AssertionError(
                "data service did not reach readiness within 30 seconds: "
                f"exitcode={process.exitcode}"
            )
    return process, stop, identity


def _stop_data_service(process: Any, stop: Any) -> None:
    stop.set()
    process.join(timeout=15.0)
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
    assert process.exitcode == 0
    assert not Path("/run/sakuramoon/data-service.sock").exists()


def _composite() -> TrainableComposite:
    return TrainableComposite(
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


def _optimizer(composite: TrainableComposite):
    return build_adamw8bit(
        composite,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=8025,
    )


def _fresh_resume_worker(
    checkpoint: Path,
    repository_root: Path,
    config: RuntimeConfig,
    marker_root: Path,
    result_path: Path,
) -> None:
    from sakuramoon.checkpoint.schema import manifest_from_dict

    composite = _composite()
    optimizer = _optimizer(composite)
    qwen = load_local_qwen(repository_root, torch.device("cuda", 0))
    vae = load_local_mage_vae(repository_root, torch.device("cuda", 0))
    manifest = manifest_from_dict(
        json.loads((checkpoint / "manifest.json").read_bytes())
    )
    restored = restore_single_gpu_checkpoint(
        checkpoint,
        composite,
        optimizer,
        manifest.identity,
    )
    runtime = SingleGpuBatchRuntime(
        qwen=qwen.encoder,
        vae=vae,
        composite=composite,
        device=torch.device("cuda", 0),
        generator=torch.cuda.default_generators[0],
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        noise_observation_boundary=0.95,
        growth_alpha=restored.state.growth.alpha,
    )
    client = _BoundedClient(
        DataServiceClient(
            Path(config.data.service.socket_path),
            DataServiceSessionIdentity(config.data.manifest.sha256, 2),
            request_timeout_seconds=5.0,
        ),
        lease_budget=4,
    )
    factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=repository_root,
        tokenizer=_WorkerRecordingTokenizer(marker_root),
        framing=FramingContract(34, 5, 0),
        rejection_observer=_observe_rejection,
        pass_index=0,
    )
    stream = factory.batches(client)
    try:
        batch = next(stream)
        _wait_for_worker_markers(marker_root, expected=2)
        assert client.leased_worker_ids == {0, 1}
        step = SingleGpuStep(
            composite,
            optimizer,
            accumulation_steps=1,
            state=restored.state.trainer,
        )
        step.backward(runtime.measure(batch).per_sample_loss)
        next_update = step.finish_update().state
    finally:
        stream.close()
    result_path.write_text(
        json.dumps(
            {
                "checkpoint_id": manifest.identity.checkpoint_id,
                "restored_update": restored.state.trainer.successful_updates,
                "next_update": next_update.successful_updates,
            },
            sort_keys=True,
        )
    )


def test_real_qwen_vae_dit_loss_backward_and_update() -> None:
    repository_root = Path(__file__).parents[3]
    config = _config(repository_root)
    device = torch.device("cuda", 0)
    torch.cuda.default_generators[0].manual_seed(8024)
    qwen = load_local_qwen(repository_root, device)
    vae = load_local_mage_vae(repository_root, device)
    composite = _composite()
    optimizer = build_adamw8bit(
        composite,
        lr=2e-5,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.01,
        sensitive_weight_decay=0.0,
        sr_seed=8025,
    )
    runtime = SingleGpuBatchRuntime(
        qwen=qwen.encoder,
        vae=vae,
        composite=composite,
        device=device,
        generator=torch.cuda.default_generators[0],
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        noise_observation_boundary=0.95,
        growth_alpha=0.0,
    )

    workload = build_single_gpu_preflight_workload(
        config,
        runtime=runtime,
        trainable_module=composite,
        optimizer=optimizer,
    )
    observed: list[str] = []
    for name, check in workload._checks:  # pyright: ignore[reportPrivateUsage]
        check()
        observed.append(name)

    assert observed == [
        "image_shapes",
        "text_shapes",
        "zero_update_loss",
        "optimizer_step",
        "sample",
    ]
    assert all(parameter.grad is None for parameter in composite.parameters())


def test_real_service_preflight_training_checkpoint_and_fresh_resume(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).parents[3]
    manifest, bodies = _dataset()
    validation_path = tmp_path / "validation-manifest.jsonl"
    validation_payload = _validation_payload()
    validation_path.write_bytes(validation_payload)
    relative_root = tmp_path.relative_to(repository_root)
    service_root = tmp_path / "service"
    service_root.mkdir()
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    config = _config(
        repository_root,
        overrides={
            "paths.run_dir": str(relative_root / "run"),
            "paths.cache_dir": str(relative_root / "cache"),
            "paths.checkpoint_dir": str(relative_root / "checkpoints"),
            "paths.artifact_dir": str(relative_root / "artifacts"),
            "storage.shared_mount_source": (
                "cs1.vast1.bz1.paratera.com:"
                "/cs1/fs1/pvc-8eb5b2a2-c80d-4c40-b28a-800fbab13752"
            ),
            "storage.minimum_free_gib": 1,
            "data.source.revision": "a" * 40,
            "data.manifest.path": str(relative_root / "train-manifest.jsonl"),
            "data.manifest.sha256": manifest_sha256(manifest),
            "data.cache.low_watermark_gib": 0,
            "data.cache.high_watermark_gib": 1,
            "data.service.mainset_path": str(relative_root / "service/mainset.json"),
            "data.validation.manifest_path": str(validation_path),
            "data.validation.manifest_sha256": hashlib.sha256(
                validation_payload
            ).hexdigest(),
            "stage.planned_updates": 1,
            "sampling.profile": "preview",
        },
    )
    resolved_payload = resolved_config_bytes(config)
    resolved_path = tmp_path / "resolved.toml"
    resolved_path.write_bytes(resolved_payload)
    loaded = LoadedConfig(
        config,
        (),
        resolved_payload.decode("utf-8"),
        resolved_config_sha256(config),
    )
    device = torch.device("cuda", 0)
    torch.cuda.default_generators[0].manual_seed(8050)
    qwen_runtime = load_local_qwen(repository_root, device)
    vae = load_local_mage_vae(repository_root, device)
    composite = _composite()
    optimizer = _optimizer(composite)
    runtime = SingleGpuBatchRuntime(
        qwen=qwen_runtime.encoder,
        vae=vae,
        composite=composite,
        device=device,
        generator=torch.cuda.default_generators[0],
        p_mean=-0.8,
        p_std=0.8,
        noise_scale=1.0,
        t_eps=0.05,
        noise_observation_boundary=0.95,
        growth_alpha=1.0,
    )
    initial_identity = CheckpointIdentity(
        "t050-initial",
        0,
        loaded.resolved_sha256,
        "b" * 64,
        optimizer.audit.schema_sha256,
    )
    initial_state = RawCheckpointState(
        trainer=SingleGpuUpdateState.initial(),
        growth=GrowthCheckpointState(
            BASE_SLOT_IDS,
            1.0,
            "S0",
            1,
            256,
            None,
            None,
        ),
        stage_budget=StageBudgetCheckpointState(0, 1),
        checkpoint_cadence=CheckpointCadence(0, 0.0),
    )
    initial = save_raw_checkpoint(
        checkpoint_root,
        initial_identity,
        composite,
        optimizer,
        initial_state,
        resolved_config=resolved_payload,
    )
    restored = restore_single_gpu_checkpoint(
        initial.path,
        composite,
        optimizer,
        initial_identity,
    )

    service_process, service_stop, service_identity = _start_data_service(
        service_root,
        manifest,
        bodies,
    )
    main_markers = tmp_path / "main-worker-pids"
    durable_checkpoint: Path | None = None
    observations: list[int] = []
    try:
        client = _BoundedClient(
            DataServiceClient(
                Path(config.data.service.socket_path),
                service_identity,
                request_timeout_seconds=5.0,
            ),
            lease_budget=len(manifest.shards),
        )
        factory = ProductionPipelineFactory.from_config(
            config,
            repository_root=repository_root,
            tokenizer=_WorkerRecordingTokenizer(main_markers),
            framing=FramingContract(34, 5, 0),
            rejection_observer=_observe_rejection,
            pass_index=0,
        )
        stream = factory.batches(client)
        publisher = ProductionSingleGpuCheckpointPublisher(
            checkpoint_root=checkpoint_root,
            resolved_config=resolved_payload,
            module=composite,
            optimizer=optimizer,
            restored_checkpoint=restored,
            accepted_checkpoint_ids=frozenset(),
        )
        workload = build_single_gpu_preflight_workload(
            config,
            runtime=runtime,
            trainable_module=composite,
            optimizer=optimizer,
        )
        plan = build_single_gpu_preflight_checks(
            loaded,
            repository_root=repository_root,
            resolved_config_path=resolved_path,
            data_client=client,
            batches=stream,
            runtime=runtime,
            qwen=qwen_runtime.encoder,
            vae=vae,
            trainable_module=composite,
            optimizer=optimizer,
            restored_checkpoint=restored,
            workload=workload,
            checkpoint_publisher=publisher,
        )
        accepted = run_single_gpu_preflight(plan, tmp_path / "preflight-report.json")

        def observe(value: SuccessfulTrainingObservation) -> None:
            observations.append(value.loop.update.state.successful_updates)

        result = run_single_gpu_training(
            config,
            preflight=accepted,
            runtime=runtime,
            module=composite,
            optimizer=optimizer,
            batches=stream,
            scheduler_step=lambda _update: None,
            checkpoint_publisher=publisher,
            diagnostic_root=tmp_path / "diagnostics",
            failure_id=lambda phase, state: f"{phase}-{state.attempted_updates}",
            restored_checkpoint=restored,
            phase_timer=PhaseTimer(device=device),
            successful_update_observer=observe,
            forced_checkpoint=lambda update: (
                CheckpointReason.STAGE_FINALIZE if update == 1 else None
            ),
            clock=lambda: 1.0,
        )
        assert result.state == SingleGpuUpdateState(1, 1, 1)
        assert observations == [1]
        main_worker_markers = _worker_markers(main_markers)
        assert set(main_worker_markers) == {0, 1}
        assert len(set(main_worker_markers.values())) == 2
        assert not any("preflight-" in path.name for path in checkpoint_root.iterdir())
        candidates = tuple(
            path
            for path in discover_complete_checkpoints(checkpoint_root)
            if path.name.startswith("ckpt_1_")
        )
        assert len(candidates) == 1
        durable_checkpoint = candidates[0]
        durable_manifest, durable_state = read_raw_checkpoint_state(durable_checkpoint)
        assert durable_manifest.identity.update == 1
        assert durable_state.trainer == SingleGpuUpdateState(1, 1, 1)
        assert durable_state.checkpoint_cadence == CheckpointCadence(1, 1.0)
    finally:
        _stop_data_service(service_process, service_stop)

    assert durable_checkpoint is not None
    del (
        accepted,
        client,
        factory,
        optimizer,
        plan,
        publisher,
        qwen_runtime,
        restored,
        result,
        runtime,
        stream,
        vae,
        workload,
        composite,
    )
    gc.collect()
    torch.cuda.empty_cache()

    replay_process, replay_stop, _replay_identity = _start_data_service(
        service_root,
        manifest,
        bodies,
    )
    fresh_markers = tmp_path / "fresh-worker-pids"
    fresh_result_path = tmp_path / "fresh-resume-result.json"
    context = mp.get_context("spawn")
    fresh_process = context.Process(
        target=_fresh_resume_worker,
        args=(
            durable_checkpoint,
            repository_root,
            config,
            fresh_markers,
            fresh_result_path,
        ),
    )
    try:
        fresh_process.start()
        fresh_process.join(timeout=600.0)
        if fresh_process.is_alive():
            fresh_process.terminate()
            fresh_process.join(timeout=10.0)
            pytest.fail("fresh checkpoint resume process timed out")
    finally:
        _stop_data_service(replay_process, replay_stop)
    assert fresh_process.exitcode == 0
    fresh_result = json.loads(fresh_result_path.read_bytes())
    assert fresh_result["checkpoint_id"] == durable_manifest.identity.checkpoint_id
    assert fresh_result["restored_update"] == 1
    assert fresh_result["next_update"] == 2
    fresh_worker_markers = _worker_markers(fresh_markers)
    assert set(fresh_worker_markers) == {0, 1}
    assert len(set(fresh_worker_markers.values())) == 2
    service_state = json.loads((service_root / "mainset.json").read_bytes())
    assert service_state["replayed_shards"] >= 1
