"""Real single-GPU vertical engineering smoke with synthetic local shards."""

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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

import torch
from PIL import Image
from torch.utils.data import get_worker_info

from sakuramoon.checkpoint.load import (
    load_raw_checkpoint,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.policy import CheckpointCadence
from sakuramoon.checkpoint.save import save_raw_checkpoint
from sakuramoon.checkpoint.schema import (
    CheckpointIdentity,
    GrowthCheckpointState,
    RawCheckpointState,
    StageBudgetCheckpointState,
)
from sakuramoon.conditioning.style_resampler import StyleResampler
from sakuramoon.conditioning.text_mixer import TextConditioner
from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
from sakuramoon.data.cache import CacheQuota, ShardCache
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    NlDropoutProbabilities,
)
from sakuramoon.data.client import DataServiceClient
from sakuramoon.data.collate import TrainingBatch, iter_service_batches
from sakuramoon.data.manifest import (
    DATASET_REPO_ID,
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
    manifest_sha256,
)
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.production import (
    adapt_modelscope_metadata,
    parse_modelscope_caption_fields,
)
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    SYSTEM_PREFIX,
    FramingContract,
)
from sakuramoon.data.service import (
    DataServiceLimits,
    DataServiceServer,
    DataSupplyService,
)
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)
from sakuramoon.data.validation import load_validation_manifest_ids
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.engineering_smoke.config import (
    EngineeringSmokeConfig,
    LoadedEngineeringSmokeConfig,
    is_sha256,
    load_engineering_smoke_config,
    require_single_gpu_environment,
)
from sakuramoon.model.dit import DenseDiT
from sakuramoon.model.growth import BASE_SLOT_IDS
from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit, build_adamw8bit
from sakuramoon.telemetry.timers import PhaseTimer
from sakuramoon.train.runtime import SingleGpuBatchRuntime
from sakuramoon.train.step import (
    SingleGpuStep,
    SingleGpuUpdateState,
    TrainableComposite,
)


class EngineeringSmokeError(RuntimeError):
    """The bounded engineering smoke could not preserve its fixed contract."""


class _Writer(Protocol):
    def write(self, payload: bytes, /) -> int: ...


class _BatchStream(Protocol):
    def __iter__(self) -> _BatchStream: ...

    def __next__(self) -> TrainingBatch: ...

    def close(self) -> None: ...


class _LocalTarTransport:
    def __init__(self, bodies: dict[str, bytes], *, stream_chunk_bytes: int) -> None:
        self.bodies = bodies
        self.stream_chunk_bytes = stream_chunk_bytes

    def download(
        self,
        manifest: DatasetManifest,
        shard: ShardRecord,
        output: _Writer,
    ) -> None:
        if manifest.shard(shard.path) != shard:
            raise EngineeringSmokeError("synthetic transport manifest binding changed")
        body = self.bodies[shard.path]
        for offset in range(0, len(body), self.stream_chunk_bytes):
            output.write(body[offset : offset + self.stream_chunk_bytes])


class _RecordingTokenizer:
    def __init__(self, delegate: Any, marker_root: Path) -> None:
        self.delegate = delegate
        self.marker_root = marker_root

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        worker = get_worker_info()
        if text == SYSTEM_PREFIX:
            if worker is None:
                raise EngineeringSmokeError(
                    "engineering tokenizer did not execute in a DataLoader worker"
                )
            self.marker_root.mkdir(parents=True, exist_ok=True)
            (self.marker_root / f"worker-{worker.id}-pid-{os.getpid()}").touch(
                exist_ok=True
            )
        encoded = cast(
            object,
            self.delegate.encode(
                text,
                add_special_tokens=add_special_tokens,
            ),
        )
        if type(encoded) is not list:
            raise EngineeringSmokeError("Qwen tokenizer returned an invalid token list")
        items = cast(list[object], encoded)
        if any(type(item) is not int for item in items):
            raise EngineeringSmokeError("Qwen tokenizer returned an invalid token list")
        return cast(list[int], items)


class _PreleasedClient:
    def __init__(
        self,
        delegate: DataServiceClient,
        descriptor: ShardLeaseDescriptor,
    ) -> None:
        self.delegate = delegate
        self.descriptor: ShardLeaseDescriptor | None = descriptor
        self.identity = delegate.identity

    def health(self) -> bool:
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if worker_id == 0 and self.descriptor is not None:
            descriptor, self.descriptor = self.descriptor, None
            return descriptor
        return self.delegate.lease(worker_id)

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self.delegate.acknowledge(descriptor)


@dataclass(frozen=True, slots=True)
class _ServiceHandle:
    process: Any
    stop: Any
    identity: DataServiceSessionIdentity


@dataclass(frozen=True, slots=True)
class EngineeringSmokeResult:
    report_path: Path
    checkpoint_path: Path
    resolved_config_sha256: str
    initial_successful_update: int
    fresh_process_successful_update: int


_METADATA_FIELDS = MetadataFieldMapping(
    id_field="id",
    width_field="width",
    height_field="height",
    caption_available_field="caption_available",
)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_new_bytes(path: Path, payload: bytes) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f"engineering artifact already exists: {path.name}")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(path.parent)


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    _write_new_bytes(
        path,
        (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8"),
    )


def _repository_root(path: Path) -> Path:
    try:
        root = path.resolve(strict=True)
        if path.is_symlink() or not root.is_dir():
            raise OSError
    except OSError:
        raise EngineeringSmokeError("repository root must be a real directory") from None
    return root


def _create_output_root(repository_root: Path, configured: str) -> Path:
    relative = Path(configured)
    current = repository_root
    try:
        for part in relative.parts[:-1]:
            current /= part
            if current.is_symlink():
                raise OSError
            current.mkdir(exist_ok=True)
            if not current.is_dir():
                raise OSError
        output = current / relative.name
        if output.exists() or output.is_symlink():
            raise FileExistsError
        output.mkdir()
        _fsync_directory(output.parent)
        return output.resolve(strict=True)
    except FileExistsError:
        raise EngineeringSmokeError(
            "engineering output root already exists; no-clobber is mandatory"
        ) from None
    except OSError:
        raise EngineeringSmokeError("engineering output root is invalid") from None


def _existing_output_root(repository_root: Path, configured: str) -> Path:
    candidate = repository_root / configured
    try:
        output = candidate.resolve(strict=True)
        output.relative_to(repository_root)
        if candidate.is_symlink() or not output.is_dir():
            raise OSError
    except (OSError, ValueError):
        raise EngineeringSmokeError("engineering output root identity changed") from None
    return output


def _tar_bytes(config: EngineeringSmokeConfig, sample_id: int) -> bytes:
    metadata = json.dumps(
        {
            "captions": {
                "nl2": "A bounded SakuraMoon engineering smoke sample.",
                "nl3": "",
            },
            "dropout": {"candidate_tags": []},
            "id": sample_id,
            "image": {
                "height": config.data.source_height,
                "width": config.data.source_width,
            },
            "multicaptions": {"vibes": "quiet technical illustration"},
            "nsfw": "safe",
            "tags": {
                "artist": ["engineering_style"],
                "character": [],
                "copyright": [],
                "general": ["single_gpu", "dense_sdpa"],
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    image = io.BytesIO()
    color = (sample_id % 251, (sample_id * 3) % 251, (sample_id * 7) % 251)
    Image.new(
        "RGB",
        (config.data.source_width, config.data.source_height),
        color=color,
    ).save(image, format="JPEG")
    body = io.BytesIO()
    with tarfile.open(fileobj=body, mode="w") as archive:
        for extension, payload in (("json", metadata), ("jpg", image.getvalue())):
            info = tarfile.TarInfo(f"{sample_id:06d}.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return body.getvalue()


def _dataset(
    config: EngineeringSmokeConfig,
) -> tuple[DatasetManifest, dict[str, bytes]]:
    bodies = {
        f"engineering/{index:06d}.tar": _tar_bytes(config, 9001 + index)
        for index in range(config.data.shard_count)
    }
    source_digest = hashlib.sha256()
    for path, body in sorted(bodies.items()):
        source_digest.update(path.encode("ascii"))
        source_digest.update(body)
    source = DatasetSourceIdentity(
        repo_id=DATASET_REPO_ID,
        revision=source_digest.hexdigest()[:40],
        license_id="engineering-generated-local-tar",
        access_terms="synthetic-engineering-evidence-only",
    )
    records = tuple(
        ShardRecord(
            path=path,
            release="engineering-smoke",
            bytes=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            samples=config.data.samples_per_shard,
        )
        for path, body in sorted(bodies.items())
    )
    return DatasetManifest.from_shards(source, records), bodies


def _validation_payload(config: EngineeringSmokeConfig) -> bytes:
    rows = (
        json.dumps(
            {
                "aspect_bucket": "engineering-square",
                "caption_available": False,
                "id": sample_id,
                "release": "engineering-validation",
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        for sample_id in range(1, config.data.validation_sample_count + 1)
    )
    return ("\n".join(rows) + "\n").encode("utf-8")


def _run_data_service(
    output_root: Path,
    config: EngineeringSmokeConfig,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
    identity: DataServiceSessionIdentity,
    stop: Any,
    ready: Any,
) -> None:
    cache = ShardCache(
        (output_root / "service" / "cache").absolute(),
        manifest,
        cast(
            ModelScopeDatasetTransport,
            _LocalTarTransport(
                bodies,
                stream_chunk_bytes=config.data.transport_chunk_bytes,
            ),
        ),
        CacheQuota(config.data.cache_low_bytes, config.data.cache_high_bytes),
    )
    service = DataSupplyService(
        manifest,
        cache,
        output_root / "service" / "mainset.json",
        Path(config.service.ownership_lock_path),
        identity,
        DataServiceLimits(
            config.service.download_concurrency,
            config.service.verified_shard_lookahead,
            config.service.lease_channel_capacity,
            config.service.ack_channel_capacity,
        ),
    )
    DataServiceServer(
        service,
        Path(config.service.socket_path),
        request_timeout_seconds=config.service.request_timeout_seconds,
    ).serve(cast(threading.Event, stop), ready_callback=ready.set)


def _start_data_service(
    output_root: Path,
    config: EngineeringSmokeConfig,
    manifest: DatasetManifest,
    bodies: dict[str, bytes],
) -> _ServiceHandle:
    socket_path = Path(config.service.socket_path)
    if socket_path.exists() or socket_path.is_symlink():
        raise EngineeringSmokeError(
            "engineering data-service socket is already occupied"
        )
    identity = DataServiceSessionIdentity(
        manifest_sha256(manifest),
        config.data.persistent_workers,
    )
    context = cast(Any, mp.get_context("spawn"))
    stop = context.Event()
    ready = context.Event()
    process = context.Process(
        target=_run_data_service,
        args=(output_root, config, manifest, bodies, identity, stop, ready),
    )
    process.start()
    deadline = time.monotonic() + config.service.startup_timeout_seconds
    while not ready.wait(timeout=0.25):
        if not process.is_alive():
            process.join(timeout=1.0)
            raise EngineeringSmokeError(
                f"engineering data service exited before readiness: {process.exitcode}"
            )
        if time.monotonic() >= deadline:
            stop.set()
            process.join(timeout=config.service.shutdown_timeout_seconds)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)
            raise EngineeringSmokeError("engineering data service readiness timed out")
    return _ServiceHandle(process, stop, identity)


def _stop_data_service(
    handle: _ServiceHandle,
    config: EngineeringSmokeConfig,
) -> None:
    handle.stop.set()
    handle.process.join(timeout=config.service.shutdown_timeout_seconds)
    if handle.process.is_alive():
        handle.process.terminate()
        handle.process.join(timeout=5.0)
        raise EngineeringSmokeError("engineering data service required termination")
    if handle.process.exitcode != 0:
        raise EngineeringSmokeError(
            f"engineering data service exited with code {handle.process.exitcode}"
        )
    if Path(config.service.socket_path).exists():
        raise EngineeringSmokeError("engineering data-service socket was not cleaned")


def _stop_preserving(
    handle: _ServiceHandle,
    config: EngineeringSmokeConfig,
    primary: BaseException | None,
) -> None:
    try:
        _stop_data_service(handle, config)
    except BaseException as cleanup_error:
        if primary is not None:
            raise BaseExceptionGroup(
                "engineering smoke and data-service cleanup both failed",
                [primary, cleanup_error],
            ) from None
        raise


def _dropout_probabilities(config: EngineeringSmokeConfig) -> CaptionDropoutProbabilities:
    dropout = config.caption.dropout
    return CaptionDropoutProbabilities(
        nsfw=dropout.nsfw,
        character=dropout.character,
        copyright=dropout.copyright,
        general=dropout.general,
        artist=dropout.artist,
        candidate_source=dropout.candidate_source,
        nl=NlDropoutProbabilities(
            long_names=dropout.nl.long_names,
            long_no_names=dropout.nl.long_no_names,
            short_vibes=dropout.nl.short_vibes,
            nl2=dropout.nl.nl2,
            nl3=dropout.nl.nl3,
        ),
    )


def _buckets(config: EngineeringSmokeConfig):
    source = DataBucketsConfig(
        base_area_px=config.data.base_area_px,
        quantum_px=config.data.quantum_px,
        min_short_edge_px=config.data.min_short_edge_px,
        max_aspect_ratio=config.data.max_aspect_ratio,
        shape_count=config.data.shape_count,
        transpose_closed=config.data.transpose_closed,
    )
    return scale_buckets(generate_base_buckets(source), config.stage.resolution)


def _reject_sample(_reason: str) -> None:
    return None


def _open_batch_stream(
    output_root: Path,
    config: EngineeringSmokeConfig,
    identity: DataServiceSessionIdentity,
    tokenizer: Any,
    marker_root: Path,
) -> _BatchStream:
    validation_path = output_root / "validation-manifest.jsonl"
    validation_payload = validation_path.read_bytes()
    validation_ids = load_validation_manifest_ids(
        validation_path,
        expected_sha256=hashlib.sha256(validation_payload).hexdigest(),
        expected_count=config.data.validation_sample_count,
    )
    client = DataServiceClient(
        Path(config.service.socket_path),
        identity,
        request_timeout_seconds=config.service.request_timeout_seconds,
    )
    if client.health():
        raise EngineeringSmokeError("engineering data service has no lease")
    descriptor = client.lease(0)
    if descriptor is None:
        raise EngineeringSmokeError("engineering data service returned no first lease")
    padding_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(padding_token_id) is not int:
        raise EngineeringSmokeError("Qwen tokenizer padding identity is unavailable")
    pipeline = WebDatasetPipeline(
        shard_paths=(descriptor.local_path,),
        shard_records=(descriptor.record,),
        metadata_adapter=adapt_modelscope_metadata,
        metadata_fields=_METADATA_FIELDS,
        validation_ids=validation_ids,
        buckets=_buckets(config),
        min_crop_retention=config.data.min_crop_retention,
        probabilities=_dropout_probabilities(config),
        tokenizer=_RecordingTokenizer(tokenizer, marker_root),
        framing=FramingContract(
            EXPECTED_PREFIX_TOKENS,
            EXPECTED_SUFFIX_TOKENS,
            padding_token_id,
        ),
        caption_fields_parser=parse_modelscope_caption_fields,
        rejection_observer=_reject_sample,
        base_seed=config.run.seed,
        stage=config.stage.name,
        pass_index=config.data.pass_index,
    )
    stream = iter_service_batches(
        pipeline,
        _PreleasedClient(client, descriptor),
        batch_size=config.data.local_batch,
        worker_count=config.data.persistent_workers,
        ready_batches=config.data.ready_batches,
        pin_memory=config.data.pin_memory,
        drop_last=config.data.drop_last,
    )
    return cast(_BatchStream, stream)


def _worker_markers(marker_root: Path) -> dict[int, int]:
    observed: dict[int, int] = {}
    for marker in marker_root.glob("worker-*-pid-*"):
        _, raw_worker, _, raw_pid = marker.name.split("-")
        worker = int(raw_worker)
        pid = int(raw_pid)
        previous = observed.get(worker)
        if previous is not None and previous != pid:
            raise EngineeringSmokeError("a persistent worker identity changed")
        observed[worker] = pid
    return observed


def _wait_for_workers(
    marker_root: Path,
    config: EngineeringSmokeConfig,
) -> dict[int, int]:
    expected = set(range(config.data.persistent_workers))
    deadline = time.monotonic() + config.service.request_timeout_seconds
    while True:
        observed = _worker_markers(marker_root)
        if set(observed) == expected:
            if len(set(observed.values())) != len(expected) or os.getpid() in observed.values():
                raise EngineeringSmokeError("worker markers do not name distinct children")
            return observed
        if not set(observed) <= expected:
            raise EngineeringSmokeError("unexpected DataLoader worker identity")
        if time.monotonic() >= deadline:
            raise EngineeringSmokeError("persistent worker marker wait timed out")
        time.sleep(0.01)


def _dtype(name: str) -> torch.dtype:
    try:
        return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]
    except KeyError:
        raise EngineeringSmokeError("engineering model dtype is unsupported") from None


def _build_composite(
    config: EngineeringSmokeConfig,
    device: torch.device,
) -> TrainableComposite:
    model = config.model
    dit = model.dit
    rope = model.rope
    condition = model.condition
    head = model.head
    text = model.text
    style = model.style
    with torch.device(device):
        composite = TrainableComposite(
            dit=DenseDiT(
                depth=config.stage.depth,
                input_channels=config.assets.vae.latent_channels,
                hidden_size=dit.hidden_size,
                intermediate_size=dit.intermediate_size,
                q_heads=dit.q_heads,
                kv_heads=dit.kv_heads,
                head_dim=dit.head_dim,
                rope_nope_dim=rope.nope_dim,
                rope_y_dim=rope.y_dim,
                rope_x_dim=rope.x_dim,
                rope_position_scale=rope.position_scale,
                rope_theta=rope.theta,
                norm_eps=dit.norm_eps,
                timestep_dim=condition.timestep_dim,
                size_dim=condition.size_dim,
                aspect_dim=condition.aspect_dim,
                condition_hidden_size=condition.hidden_dim,
                stable_slot_count=dit.stable_slot_count,
                modulation_chunks=condition.block_modulation_chunks,
                final_modulation_size=head.final_modulation_size,
                out_channels=head.out_channels,
                modality_init_std=model.packing.modality_init_std,
                linear_dtype=_dtype(dit.activation_dtype),
                sensitive_dtype=_dtype(dit.norm_accumulation),
                projection_bias=dit.projection_bias,
                attention_dropout=dit.attention_dropout,
                mlp_dropout=dit.mlp_dropout,
                output_weight_zero_init=head.weight_zero_init,
                output_bias_zero_init=head.bias_zero_init,
            ),
            text=TextConditioner(
                input_size=text.input_size,
                adapter_size=text.adapter_size,
                output_size=text.output_size,
                groups=text.groups,
                attention_heads=text.attention_heads,
                norm_eps=text.norm_eps,
                mix_gate_init=text.mix_gate_init,
                layer_scale_init=text.layer_scale_init,
                projection_bias=text.projection_bias,
                linear_dtype=_dtype(text.linear_dtype),
                sensitive_dtype=_dtype(text.sensitive_dtype),
            ),
            style=StyleResampler(
                input_size=style.input_size,
                hidden_size=style.hidden_size,
                intermediate_size=style.mlp_intermediate_size,
                output_size=style.output_size,
                query_count=style.query_count,
                attention_heads=style.attention_heads,
                norm_eps=style.norm_eps,
                init_std=style.init_std,
                projection_bias=style.projection_bias,
                linear_dtype=_dtype(style.linear_dtype),
                sensitive_dtype=_dtype(style.sensitive_dtype),
            ),
        )
    composite.train()
    artifact = composite.dit.artifact_config()
    if artifact.get("attention_backend") != "dense_sdpa":
        raise EngineeringSmokeError("assembled DiT is not dense SDPA")
    if composite.dit.model_metadata().get("depth") != 16:
        raise EngineeringSmokeError("assembled DiT is not 16 layers")
    return composite


def _build_optimizer(
    config: EngineeringSmokeConfig,
    composite: TrainableComposite,
) -> IsolatedAdamW8bit:
    optimizer = config.optimizer
    return build_adamw8bit(
        composite,
        lr=optimizer.lr,
        betas=optimizer.betas,
        eps=optimizer.eps,
        block_size=optimizer.block_size,
        bf16_stochastic_round=optimizer.bf16_stochastic_round,
        matrix_weight_decay=optimizer.matrix_weight_decay,
        sensitive_weight_decay=optimizer.sensitive_weight_decay,
        sr_seed=config.run.optimizer_sr_seed,
    )


def _build_runtime(
    config: EngineeringSmokeConfig,
    *,
    qwen: Any,
    vae: Any,
    composite: TrainableComposite,
    device: torch.device,
) -> SingleGpuBatchRuntime:
    return SingleGpuBatchRuntime(
        qwen=qwen,
        vae=vae,
        composite=composite,
        device=device,
        generator=torch.cuda.default_generators[config.device.index],
        p_mean=config.timestep.p_mean,
        p_std=config.timestep.p_std,
        noise_scale=config.timestep.noise_scale,
        t_eps=config.timestep.t_eps,
        noise_observation_boundary=config.measurement.noise_observation_boundary,
        growth_alpha=config.stage.growth_alpha,
    )


def _run_one_update(
    config: EngineeringSmokeConfig,
    *,
    stream: Any,
    marker_root: Path,
    runtime: SingleGpuBatchRuntime,
    composite: TrainableComposite,
    optimizer: IsolatedAdamW8bit,
    state: SingleGpuUpdateState,
) -> tuple[SingleGpuUpdateState, dict[str, object]]:
    timer = PhaseTimer(device=runtime.device)
    batch = cast(TrainingBatch, next(stream))
    workers = _wait_for_workers(marker_root, config)
    measurement = runtime.measure(batch, phase_timer=timer)
    step = SingleGpuStep(
        composite,
        optimizer,
        accumulation_steps=config.stage.accumulation,
        state=state,
    )
    with timer.record("backward"):
        step.backward(measurement.per_sample_loss)
    result = step.finish_update(phase_timer=timer)
    torch.cuda.synchronize(runtime.device)
    durations = timer.collect_ready()
    required_phases = {
        "h2d",
        "qwen",
        "vae",
        "conditioning",
        "dit_forward",
        "loss",
        "backward",
        "clip",
        "optimizer",
        "zero_grad",
    }
    if timer.pending_cuda_pairs or not required_phases <= set(durations):
        raise EngineeringSmokeError("engineering update phase coverage is incomplete")
    optimizer_states = optimizer.audit_state()
    initialized = tuple(item for item in optimizer_states if item.initialized)
    if not initialized or any(item.step != result.state.successful_updates for item in initialized):
        raise EngineeringSmokeError("TorchAO optimizer state did not advance exactly once")
    if not torch.isfinite(result.mean_loss) or not torch.isfinite(result.clip.pre_clip_norm):
        raise EngineeringSmokeError("engineering update produced nonfinite evidence")
    return result.state, {
        "attention_backend": "dense_sdpa_reference",
        "clip_coefficient": float(result.clip.coefficient.item()),
        "effective_samples": result.effective_samples,
        "jlt_timesteps": [float(item) for item in measurement.timesteps.detach().cpu()],
        "mean_loss": float(result.mean_loss.item()),
        "optimizer_initialized_parameters": len(initialized),
        "optimizer_quantized_parameters": sum(
            item.state_class == "OptimState8bit" for item in initialized
        ),
        "post_clip_norm": float(result.clip.post_clip_norm.item()),
        "pre_clip_norm": float(result.clip.pre_clip_norm.item()),
        "recorded_phases": sorted(durations),
        "sample_ids": list(measurement.sample_ids),
        "shape_keys": list(measurement.shape_keys),
        "successful_update": result.state.successful_updates,
        "torchao_optimizer": "AdamW8bit",
        "worker_pids": {str(worker): pid for worker, pid in sorted(workers.items())},
    }


def _initial_checkpoint_state(
    config: EngineeringSmokeConfig,
    trainer: SingleGpuUpdateState,
) -> RawCheckpointState:
    return RawCheckpointState(
        trainer=trainer,
        growth=GrowthCheckpointState(
            BASE_SLOT_IDS,
            config.stage.growth_alpha,
            config.stage.name,
            config.device.world_size,
            config.stage.resolution,
            None,
            None,
        ),
        stage_budget=StageBudgetCheckpointState(
            0,
            config.run.total_successful_updates,
        ),
        checkpoint_cadence=CheckpointCadence(
            trainer.successful_updates,
            0.0,
        ),
    )


def _fresh_resume_worker(
    repository_root: Path,
    config_path: Path,
    config_root: Path,
    checkpoint_path: Path,
    result_path: Path,
) -> None:
    loaded = load_engineering_smoke_config(config_path, config_root=config_root)
    config = loaded.config
    require_single_gpu_environment(config)
    output_root = _existing_output_root(repository_root, config.run.output_root)
    manifest, _bodies = _dataset(config)
    identity = DataServiceSessionIdentity(
        manifest_sha256(manifest), config.data.persistent_workers
    )
    device = torch.device("cuda", config.device.index)
    torch.cuda.set_device(device)
    qwen_runtime = load_local_qwen(repository_root, device)
    vae = load_local_mage_vae(repository_root, device)
    composite = _build_composite(config, device)
    optimizer = _build_optimizer(config, composite)
    checkpoint_manifest, checkpoint_state = read_raw_checkpoint_state(checkpoint_path)
    if checkpoint_manifest.identity.config_sha256 != loaded.resolved_sha256:
        raise EngineeringSmokeError("fresh process config differs from checkpoint")
    restored = load_raw_checkpoint(
        checkpoint_path,
        composite,
        optimizer,
        checkpoint_manifest.identity,
    )
    if restored != checkpoint_state:
        raise EngineeringSmokeError("fresh process restored state differs from raw sidecar")
    runtime = _build_runtime(
        config,
        qwen=qwen_runtime.encoder,
        vae=vae,
        composite=composite,
        device=device,
    )
    marker_root = output_root / "fresh-worker-pids"
    stream = _open_batch_stream(
        output_root,
        config,
        identity,
        qwen_runtime.tokenizer,
        marker_root,
    )
    try:
        state, update = _run_one_update(
            config,
            stream=stream,
            marker_root=marker_root,
            runtime=runtime,
            composite=composite,
            optimizer=optimizer,
            state=restored.trainer,
        )
    finally:
        stream.close()
    _write_new_json(
        result_path,
        {
            "checkpoint_id": checkpoint_manifest.identity.checkpoint_id,
            "config_sha256": loaded.resolved_sha256,
            "next_successful_update": state.successful_updates,
            "restored_successful_update": restored.trainer.successful_updates,
            "update": update,
        },
    )


def _run_initial_update(
    repository_root: Path,
    output_root: Path,
    loaded: LoadedEngineeringSmokeConfig,
    identity: DataServiceSessionIdentity,
) -> tuple[
    TrainableComposite,
    IsolatedAdamW8bit,
    SingleGpuUpdateState,
    dict[str, object],
]:
    config = loaded.config
    device = torch.device("cuda", config.device.index)
    torch.cuda.set_device(device)
    torch.manual_seed(config.run.seed)  # pyright: ignore[reportUnknownMemberType]
    torch.cuda.default_generators[config.device.index].manual_seed(config.run.seed)
    torch.cuda.reset_peak_memory_stats(device)
    qwen_runtime = load_local_qwen(repository_root, device)
    vae = load_local_mage_vae(repository_root, device)
    composite = _build_composite(config, device)
    optimizer = _build_optimizer(config, composite)
    runtime = _build_runtime(
        config,
        qwen=qwen_runtime.encoder,
        vae=vae,
        composite=composite,
        device=device,
    )
    marker_root = output_root / "initial-worker-pids"
    stream = _open_batch_stream(
        output_root,
        config,
        identity,
        qwen_runtime.tokenizer,
        marker_root,
    )
    try:
        state = SingleGpuUpdateState.initial()
        updates: list[dict[str, object]] = []
        for _index in range(config.run.initial_successful_updates):
            state, update = _run_one_update(
                config,
                stream=stream,
                marker_root=marker_root,
                runtime=runtime,
                composite=composite,
                optimizer=optimizer,
                state=state,
            )
            updates.append(update)
    finally:
        stream.close()
    return composite, optimizer, state, {
        "peak_memory_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "updates": updates,
    }


def _release_cuda_resources() -> None:
    gc.collect()
    torch.cuda.empty_cache()


def run_s000_engineering_smoke(
    config_path: Path,
    *,
    config_root: Path,
    repository_root: Path,
) -> EngineeringSmokeResult:
    """Run the bounded synthetic vertical path without opening production training."""

    root = _repository_root(repository_root)
    loaded = load_engineering_smoke_config(config_path, config_root=config_root)
    config = loaded.config
    require_single_gpu_environment(config)
    output_root = _create_output_root(root, config.run.output_root)
    (output_root / "service").mkdir()
    (output_root / "checkpoints").mkdir()
    _fsync_directory(output_root)
    validation_payload = _validation_payload(config)
    _write_new_bytes(output_root / "validation-manifest.jsonl", validation_payload)
    manifest, bodies = _dataset(config)
    dependency_path = root / config.run.dependency_lock_path
    if dependency_path.is_symlink() or not dependency_path.is_file():
        raise EngineeringSmokeError("dependency lock identity is unavailable")
    dependency_sha256 = hashlib.sha256(dependency_path.read_bytes()).hexdigest()
    if not is_sha256(dependency_sha256):
        raise EngineeringSmokeError("dependency lock digest is invalid")

    initial_service = _start_data_service(output_root, config, manifest, bodies)
    initial_error: BaseException | None = None
    composite: TrainableComposite | None = None
    optimizer: IsolatedAdamW8bit | None = None
    try:
        composite, optimizer, trainer_state, initial_update = _run_initial_update(
            root,
            output_root,
            loaded,
            initial_service.identity,
        )
        checkpoint_identity = CheckpointIdentity(
            config.run.run_id,
            trainer_state.successful_updates,
            loaded.resolved_sha256,
            dependency_sha256,
            optimizer.audit.schema_sha256,
        )
        saved = save_raw_checkpoint(
            output_root / "checkpoints",
            checkpoint_identity,
            composite,
            optimizer,
            _initial_checkpoint_state(config, trainer_state),
            resolved_config=loaded.resolved_toml.encode("utf-8"),
        )
        checkpoint_manifest, checkpoint_state = read_raw_checkpoint_state(saved.path)
        if (
            checkpoint_manifest.identity != checkpoint_identity
            or checkpoint_state.trainer != trainer_state
            or (saved.path / "COMPLETE").read_bytes() != b"complete\n"
        ):
            raise EngineeringSmokeError("raw COMPLETE checkpoint verification failed")
    except BaseException as error:
        initial_error = error
        raise
    finally:
        _stop_preserving(initial_service, config, initial_error)

    assert composite is not None and optimizer is not None
    checkpoint_path = saved.path
    initial_successful_update = trainer_state.successful_updates
    del composite, optimizer
    _release_cuda_resources()

    replay_service = _start_data_service(output_root, config, manifest, bodies)
    fresh_result_path = output_root / "fresh-resume-result.json"
    context = cast(Any, mp.get_context("spawn"))
    fresh = context.Process(
        target=_fresh_resume_worker,
        args=(
            root,
            config_path,
            config_root,
            checkpoint_path,
            fresh_result_path,
        ),
    )
    replay_error: BaseException | None = None
    try:
        fresh.start()
        fresh.join(timeout=config.service.fresh_process_timeout_seconds)
        if fresh.is_alive():
            fresh.terminate()
            fresh.join(timeout=10.0)
            raise EngineeringSmokeError("fresh-process next-step timed out")
        if fresh.exitcode != 0:
            raise EngineeringSmokeError(
                f"fresh-process next-step exited with code {fresh.exitcode}"
            )
    except BaseException as error:
        replay_error = error
        raise
    finally:
        _stop_preserving(replay_service, config, replay_error)

    fresh_result = json.loads(fresh_result_path.read_bytes())
    if not isinstance(fresh_result, dict):
        raise EngineeringSmokeError("fresh-process result is not an object")
    fresh_document = cast(dict[str, object], fresh_result)
    next_update = fresh_document.get("next_successful_update")
    if (
        fresh_document.get("checkpoint_id") != checkpoint_manifest.identity.checkpoint_id
        or fresh_document.get("config_sha256") != loaded.resolved_sha256
        or fresh_document.get("restored_successful_update") != initial_successful_update
        or next_update != initial_successful_update + 1
        or next_update != config.run.total_successful_updates
    ):
        raise EngineeringSmokeError("fresh-process next-step identity is inconsistent")
    mainset = json.loads((output_root / "service" / "mainset.json").read_bytes())
    if not isinstance(mainset, dict):
        raise EngineeringSmokeError("engineering service state is not an object")
    mainset_document = cast(dict[str, object], mainset)
    replayed_shards = mainset_document.get("replayed_shards")
    if type(replayed_shards) is not int or replayed_shards < 1:
        raise EngineeringSmokeError("fresh service did not record shard replay")

    report_path = output_root / "engineering-report.json"
    _write_new_json(
        report_path,
        {
            "attention_backend": "dense_sdpa_reference",
            "checkpoint": {
                "complete_marker": True,
                "id": checkpoint_manifest.identity.checkpoint_id,
                "kind": checkpoint_manifest.kind.value,
                "payload_bytes": saved.payload_bytes,
                "successful_update": checkpoint_manifest.identity.update,
            },
            "classification": config.evidence.classification,
            "config_input_path": loaded.input_path,
            "config_input_sha256": loaded.input_sha256,
            "data": {
                "manifest_sha256": manifest_sha256(manifest),
                "replayed_shards": replayed_shards,
                "service_boundary": "DataServiceServer/DataServiceClient leases",
                "shards": len(manifest.shards),
                "source": config.data.source_kind,
                "workers": config.data.persistent_workers,
            },
            "dependency_lock_sha256": dependency_sha256,
            "dit_depth": config.stage.depth,
            "formal_blockers": [
                "APPROVED-LONG-RUN-RESOURCES",
                "FORMAL-EVALUATOR-IDENTITIES",
                "FROZEN-PRODUCTION-S000-CONFIG",
                "PRODUCTION-COMPLETE-CHECKPOINT",
                "FOUR-GPU-AVAILABLE",
            ],
            "formal_s000": False,
            "fresh_process": fresh_document,
            "growth_alpha": config.stage.growth_alpha,
            "initial_process": initial_update,
            "noise_observation_boundary": config.measurement.noise_observation_boundary,
            "production_capacity_claim": False,
            "production_cli_unlocked": False,
            "production_quality_claim": False,
            "resolved_config_sha256": loaded.resolved_sha256,
            "run_id": config.run.run_id,
            "schema_version": 1,
            "successful_updates_total": next_update,
            "task_id": "S000",
        },
    )
    return EngineeringSmokeResult(
        report_path=report_path,
        checkpoint_path=checkpoint_path,
        resolved_config_sha256=loaded.resolved_sha256,
        initial_successful_update=initial_successful_update,
        fresh_process_successful_update=cast(int, next_update),
    )


__all__ = [
    "EngineeringSmokeError",
    "EngineeringSmokeResult",
    "run_s000_engineering_smoke",
]
