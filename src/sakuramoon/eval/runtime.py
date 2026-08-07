"""Periodic in-process FID and Inception Score evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import math
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import tomli_w
import torch
from accelerate import Accelerator
from PIL import Image
from torch.nn import functional
from torchvision.models import (  # pyright: ignore[reportMissingTypeStubs]
    Inception_V3_Weights,
    inception_v3,
)
from torchvision.transforms import (  # pyright: ignore[reportMissingTypeStubs]
    functional as vision_functional,
)

from sakuramoon.config.schema import EvaluationEnabledConfig, RuntimeConfig
from sakuramoon.data.caption import CaptionDropoutHits, CaptionPlan
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
    SerializedCaption,
    TokenEncoder,
    serialize_caption,
)
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.qwen import QwenRuntime
from sakuramoon.eval.metrics import (
    FeatureStats,
    frechet_inception_distance,
    inception_score,
)
from sakuramoon.eval.spec import PromptCase, PromptManifest
from sakuramoon.objective.flow import guided_velocity
from sakuramoon.sampling.sampler import sample_profile
from sakuramoon.storage import repository_directory
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs


class EvaluationError(RuntimeError):
    pass


class _FidExtractor(Protocol):
    def __call__(self, images: torch.Tensor) -> list[torch.Tensor]: ...


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    update: int
    fid: float
    inception_score_mean: float
    inception_score_std: float
    sample_count: int
    result_path: Path


_NO_DROPOUT = CaptionDropoutHits(
    all_condition=False,
    nsfw=False,
    character=False,
    copyright=False,
    general=False,
    artist=False,
    candidate_source=False,
    long_names=False,
    long_no_names=False,
    short_vibes=False,
    nl2=False,
    nl3=False,
)
_ALL_DROPPED = dataclasses.replace(_NO_DROPOUT, all_condition=True)


def _conditional_plan(case: PromptCase) -> CaptionPlan:
    if case.caption_plan is not None:
        return case.caption_plan
    return CaptionPlan(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        nl_text=case.prompt,
        selected_nl="long_names",
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


def _unconditional_plan() -> CaptionPlan:
    return CaptionPlan(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=True,
        dropout_hits=_ALL_DROPPED,
    )


def _index_tensor(
    values: tuple[tuple[int, ...], ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(item) for item in values), default=0)
    indices = torch.full((len(values), width), -1, dtype=torch.long, device=device)
    mask = torch.zeros((len(values), width), dtype=torch.bool, device=device)
    for row, item in enumerate(values):
        if item:
            indices[row, : len(item)] = torch.tensor(item, device=device)
            mask[row, : len(item)] = True
    return indices, mask


def _conditioning_inputs(
    cases: tuple[PromptCase, ...], tokenizer: object, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    encoder = cast(TokenEncoder, tokenizer)
    padding_token_id = getattr(encoder, "pad_token_id", None)
    if type(padding_token_id) is not int or not callable(getattr(encoder, "encode", None)):
        raise EvaluationError("Qwen tokenizer is unavailable")
    framing = FramingContract(
        EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, padding_token_id
    )
    serialized: list[SerializedCaption] = []
    for case in cases:
        caption = serialize_caption(_conditional_plan(case), encoder, framing)
        serialized.append(caption)
    unconditional = serialize_caption(_unconditional_plan(), encoder, framing)
    serialized.extend(unconditional for _ in cases)
    captions = tuple(serialized)
    dense_length = max(item.dense_length for item in captions)
    input_ids = torch.full(
        (len(captions), dense_length),
        padding_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(captions), dense_length), dtype=torch.bool, device=device
    )
    for row, item in enumerate(captions):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids, device=device)
        attention_mask[row, :length] = True
    main_indices, main_mask = _index_tensor(
        tuple(item.main_token_indices for item in captions), device
    )
    artist_indices, artist_mask = _index_tensor(
        tuple(item.artist_token_indices for item in captions), device
    )
    use_null = torch.tensor(
        tuple(item.use_null_style for item in captions),
        dtype=torch.bool,
        device=device,
    )
    active_style = torch.tensor(
        tuple(index for index, item in enumerate(captions) if not item.use_null_style),
        dtype=torch.long,
        device=device,
    )
    return (
        input_ids,
        attention_mask,
        main_indices,
        main_mask,
        tuple(len(item.main_token_indices) for item in captions),
        artist_indices,
        artist_mask,
        use_null,
        active_style,
    )


def _initial_noise(
    cases: tuple[PromptCase, ...], device: torch.device
) -> torch.Tensor:
    images: list[torch.Tensor] = []
    for case in cases:
        generator = torch.Generator(device=device)
        generator.manual_seed(case.seed)
        images.append(
            torch.randn(
                (128, case.height // 16, case.width // 16),
                generator=generator,
                device=device,
                dtype=torch.float32,
            )
        )
    return torch.stack(images)


@torch.inference_mode()
def _generate(
    cases: tuple[PromptCase, ...],
    *,
    config: RuntimeConfig,
    evaluation: EvaluationEnabledConfig,
    composite: TrainableComposite,
    qwen: QwenRuntime,
    vae: FrozenMageVAE,
    device: torch.device,
    growth_alpha: float,
) -> torch.Tensor:
    height, width = cases[0].height, cases[0].width
    if any((item.height, item.width) != (height, width) for item in cases):
        raise EvaluationError("evaluation batch contains mixed image shapes")
    (
        input_ids,
        attention_mask,
        main_indices,
        main_mask,
        main_lengths,
        artist_indices,
        artist_mask,
        use_null,
        active_style,
    ) = _conditioning_inputs(cases, qwen.tokenizer, device)
    qwen_states = qwen.encoder(input_ids, attention_mask).hidden_states
    branch_count = len(cases) * 2
    size_scale = torch.full(
        (branch_count,),
        0.5 * math.log2((height * width) / float(512 * 512)),
        dtype=torch.float32,
        device=device,
    )
    aspect = torch.full(
        (branch_count,),
        math.log2(width / float(height)),
        dtype=torch.float32,
        device=device,
    )
    placeholders = tuple(
        torch.empty(
            128,
            height // 16,
            width // 16,
            dtype=torch.bfloat16,
            device=device,
        )
        for _ in range(branch_count)
    )
    inputs = TrainableCompositeInputs(
        qwen_states=qwen_states,
        main_token_indices=main_indices,
        main_mask=main_mask,
        main_token_lengths=main_lengths,
        artist_token_indices=artist_indices,
        artist_mask=artist_mask,
        use_null_style=use_null,
        active_style_sample_indices=active_style,
        latents=placeholders,
        timestep=torch.zeros(branch_count, dtype=torch.float32, device=device),
        size_scale=size_scale,
        aspect=aspect,
        growth_alpha=growth_alpha,
    )
    conditioning = composite.forward_conditioning(inputs)
    noise = _initial_noise(cases, device)

    def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((state, state), dim=0).to(torch.bfloat16)
        step_inputs = dataclasses.replace(
            inputs,
            latents=tuple(branches.unbind(0)),
            timestep=torch.cat((timestep, timestep), dim=0),
        )
        predicted = composite.forward_dit(step_inputs, conditioning)
        conditional = torch.stack(predicted[: len(cases)])
        unconditional = torch.stack(predicted[len(cases) :])
        return guided_velocity(
            conditional,
            unconditional,
            state,
            timestep,
            t_eps=config.timestep.t_eps,
            guidance_scale=config.cfg.scale,
        )

    sampled = sample_profile(velocity, noise, profile=evaluation.sampling_profile)
    decoded = vae.decode(sampled.state.to(torch.bfloat16))
    if not bool(torch.isfinite(decoded).all().item()):
        raise EvaluationError("VAE produced nonfinite validation images")
    return (
        decoded.float()
        .add(1.0)
        .mul(127.5)
        .round()
        .clamp(0.0, 255.0)
        .to(dtype=torch.uint8)
    )


class _InceptionModels:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        print(f"[eval] Inception 推理设备: {device}", flush=True)
        print("[eval] 加载标准 FID Inception 权重", flush=True)
        from pytorch_fid.inception import (  # pyright: ignore[reportMissingTypeStubs]
            InceptionV3,
        )

        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.fid = cast(_FidExtractor, InceptionV3([block]).eval().to(device))
        print("[eval] 加载 Inception Score 分类权重", flush=True)
        self.is_weights = Inception_V3_Weights.DEFAULT
        self.classifier = inception_v3(weights=self.is_weights).eval().to(device)

    @torch.inference_mode()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        values = images.float().div(255.0).to(self.device, non_blocking=True)
        result = self.fid(values)[0]
        return result.flatten(1)

    @torch.inference_mode()
    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        values = self.is_weights.transforms()(images.float().div(255.0)).to(
            self.device, non_blocking=True
        )
        logits = self.classifier(values)
        if not isinstance(logits, torch.Tensor):
            raise EvaluationError("Inception classifier returned an invalid result")
        return logits.softmax(dim=1)


def _validation_images(root: Path, count: int, batch_size: int) -> list[torch.Tensor]:
    batches: list[torch.Tensor] = []
    current: list[torch.Tensor] = []
    observed = 0
    archives = sorted(root.rglob("*.tar"))
    if not archives:
        raise EvaluationError(f"validation tar files are absent: {root}")
    for archive in archives:
        with tarfile.open(archive, "r:*") as handle:
            for member in handle:
                if observed >= count:
                    break
                if not member.isfile() or Path(member.name).suffix.casefold() not in {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".webp",
                }:
                    continue
                source = handle.extractfile(member)
                if source is None:
                    continue
                try:
                    with Image.open(io.BytesIO(source.read())) as image:
                        tensor = vision_functional.pil_to_tensor(image.convert("RGB"))
                except (OSError, ValueError):
                    continue
                tensor = functional.interpolate(
                    tensor.unsqueeze(0).float(),
                    size=(299, 299),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).to(torch.uint8)
                current.append(tensor)
                observed += 1
                if len(current) == batch_size:
                    batches.append(torch.stack(current))
                    current = []
            if observed >= count:
                break
    if current:
        batches.append(torch.stack(current))
    if observed != count:
        raise EvaluationError(f"validation images are incomplete: {observed}/{count}")
    return batches


def _stage_cases(
    path: Path,
    count: int,
    *,
    resolution: int,
) -> tuple[PromptCase, ...]:
    if type(resolution) is not int or resolution <= 0 or resolution % 16:
        raise EvaluationError("evaluation resolution must be a positive multiple of 16")
    print(f"[eval] 读取验证提示词: {path}", flush=True)
    manifest = PromptManifest.from_canonical_bytes(path.read_bytes())
    selected = manifest.cases[:count]
    if len(selected) != count:
        raise EvaluationError(f"validation prompts are incomplete: {len(selected)}/{count}")
    print(f"[eval] 统一 1:1 生成分辨率: {resolution}x{resolution}", flush=True)
    return tuple(
        dataclasses.replace(
            case,
            height=resolution,
            width=resolution,
        )
        for case in selected
    )


def _rank_case_bounds(
    count: int,
    batch_size: int,
    *,
    rank: int,
    world_size: int,
) -> tuple[int, int]:
    batch_count = (count + batch_size - 1) // batch_size
    if world_size > batch_count:
        raise EvaluationError("evaluation has fewer batches than distributed ranks")
    start_batch = batch_count * rank // world_size
    end_batch = batch_count * (rank + 1) // world_size
    return start_batch * batch_size, min(end_batch * batch_size, count)


def _write_toml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(tomli_w.dumps(payload).encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class TrainingEvaluator:
    def __init__(
        self,
        config: RuntimeConfig,
        *,
        repository_root: Path,
        composite: TrainableComposite,
        qwen: QwenRuntime,
        vae: FrozenMageVAE,
        device: torch.device,
        growth_alpha: float,
        accelerator: Accelerator | None = None,
    ) -> None:
        if config.evaluation.enabled is not True:
            raise ValueError("TrainingEvaluator requires evaluation.enabled=true")
        self.config = config
        self.evaluation = config.evaluation
        self.root = repository_root
        self.composite = composite
        self.qwen = qwen
        self.vae = vae
        self.device = device
        self.growth_alpha = growth_alpha
        self.accelerator = accelerator
        self.rank = 0 if accelerator is None else accelerator.process_index
        self.world_size = 1 if accelerator is None else accelerator.num_processes
        self.is_main_process = (
            True if accelerator is None else accelerator.is_main_process
        )
        self.output = repository_directory(repository_root, self.evaluation.output_dir)
        self._inception: _InceptionModels | None = None
        self._real_stats: FeatureStats | None = None

    def due(self, update: int) -> bool:
        return update > 0 and update % self.evaluation.every_updates == 0

    def _models(self) -> _InceptionModels:
        if self._inception is None:
            self._inception = _InceptionModels(self.device)
        return self._inception

    def _gather(self, values: torch.Tensor) -> torch.Tensor:
        if self.accelerator is None:
            return values
        return cast(torch.Tensor, self.accelerator.gather(values))

    def _checkpoint_fingerprint(self, update: int) -> str | None:
        root = repository_directory(self.root, self.config.paths.checkpoint_dir)
        candidates = tuple(
            path
            for path in root.glob(f"ckpt_{update}_*")
            if path.is_dir()
            and not path.is_symlink()
            and (path / "COMPLETE").is_file()
            and not (path / "COMPLETE").is_symlink()
            and (path / "manifest.json").is_file()
            and not (path / "manifest.json").is_symlink()
        )
        if len(candidates) != 1:
            print(
                f"[eval] update={update} 检查点不唯一，禁用生成特征缓存",
                flush=True,
            )
            return None
        checkpoint = candidates[0]
        if (checkpoint / "COMPLETE").read_bytes() != b"complete\n":
            print(
                f"[eval] update={update} 检查点不完整，禁用生成特征缓存",
                flush=True,
            )
            return None
        return hashlib.sha256((checkpoint / "manifest.json").read_bytes()).hexdigest()

    def _generated_cache_path(self, update: int, fingerprint: str) -> Path:
        return self.output / "cache" / (
            f"generated-step-{update}-{fingerprint[:16]}.pt"
        )

    def _load_generated_cache(
        self,
        update: int,
        *,
        fingerprint: str,
        prompt_sha256: str,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        path = self._generated_cache_path(update, fingerprint)
        if not path.is_file() or path.is_symlink():
            return None
        try:
            raw = cast(
                object,
                torch.load(path, map_location="cpu", weights_only=True),
            )
        except (OSError, RuntimeError, ValueError):
            return None
        if type(raw) is not dict:
            return None
        document = cast(dict[str, object], raw)
        expected: dict[str, object] = {
            "schema_version": 1,
            "run_id": self.config.run.run_id,
            "update": update,
            "checkpoint_fingerprint": fingerprint,
            "prompt_sha256": prompt_sha256,
            "sample_count": self.evaluation.sample_count,
            "resolution": self.config.stage.resolution,
            "sampling_profile": self.evaluation.sampling_profile,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            return None
        features = document.get("features")
        probabilities = document.get("probabilities")
        sample_count = self.evaluation.sample_count
        if (
            not isinstance(features, torch.Tensor)
            or features.shape != (sample_count, 2048)
            or not torch.is_floating_point(features)
            or not isinstance(probabilities, torch.Tensor)
            or probabilities.shape != (sample_count, 1000)
            or not torch.is_floating_point(probabilities)
            or not bool(torch.isfinite(features).all().item())
            or not bool(torch.isfinite(probabilities).all().item())
        ):
            return None
        print(f"[eval] 复用生成特征缓存: {path}", flush=True)
        return (
            features.to(self.device, non_blocking=True),
            probabilities.to(self.device, non_blocking=True),
        )

    def _save_generated_cache(
        self,
        update: int,
        *,
        fingerprint: str,
        prompt_sha256: str,
        features: torch.Tensor,
        probabilities: torch.Tensor,
    ) -> None:
        path = self._generated_cache_path(update, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                {
                    "schema_version": 1,
                    "run_id": self.config.run.run_id,
                    "update": update,
                    "checkpoint_fingerprint": fingerprint,
                    "prompt_sha256": prompt_sha256,
                    "sample_count": self.evaluation.sample_count,
                    "resolution": self.config.stage.resolution,
                    "sampling_profile": self.evaluation.sampling_profile,
                    "features": features.detach().to("cpu").contiguous(),
                    "probabilities": probabilities.detach().to("cpu").contiguous(),
                },
                temporary,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[eval] 已保存生成特征缓存: {path}", flush=True)

    def _real(self) -> FeatureStats:
        if self._real_stats is not None:
            return self._real_stats
        cache = self.output / f"real-stats-{self.evaluation.sample_count}.pt"
        if cache.is_file() and not cache.is_symlink():
            document = cast(
                object, torch.load(cache, map_location="cpu", weights_only=True)
            )
            if isinstance(document, dict):
                fields = cast(dict[str, object], document)
                count = fields.get("count")
                mean = fields.get("mean")
                covariance = fields.get("covariance")
                if (
                    type(count) is int
                    and count == self.evaluation.sample_count
                    and isinstance(mean, torch.Tensor)
                    and isinstance(covariance, torch.Tensor)
                ):
                    self._real_stats = FeatureStats(count, mean, covariance)
                    print(f"[eval] 使用真实集统计缓存: {cache}", flush=True)
                    return self._real_stats
        print("[eval] 首次计算真实验证集 Inception 统计", flush=True)
        root = self.root / self.evaluation.validation_shard_root
        features = [
            self._models().features(batch)
            for batch in _validation_images(
                root, self.evaluation.sample_count, self.evaluation.batch_size
            )
        ]
        computed = FeatureStats.from_features(
            torch.cat(features),
            device=self.device,
        )
        self._real_stats = FeatureStats(
            computed.count,
            computed.mean.cpu(),
            computed.covariance.cpu(),
        )
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        torch.save(
            {
                "count": self._real_stats.count,
                "mean": self._real_stats.mean,
                "covariance": self._real_stats.covariance,
            },
            temporary,
        )
        os.replace(temporary, cache)
        return self._real_stats

    def evaluate(self, update: int) -> EvaluationResult | None:
        if not self.due(update):
            raise ValueError("evaluation update is not due")
        print(
            f"[eval] update={update} rank={self.rank}/{self.world_size} "
            "开始计算 FID/IS",
            flush=True,
        )
        prompt_path = self.root / self.evaluation.prompt_path
        prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        cases = _stage_cases(
            prompt_path,
            self.evaluation.sample_count,
            resolution=self.config.stage.resolution,
        )
        fingerprint = self._checkpoint_fingerprint(update)
        cached = (
            self._load_generated_cache(
                update,
                fingerprint=fingerprint,
                prompt_sha256=prompt_sha256,
            )
            if self.is_main_process and fingerprint is not None
            else None
        )
        cache_flags = self._gather(
            torch.tensor(
                [int(cached is not None)],
                dtype=torch.int64,
                device=self.device,
            )
        )
        if bool(cache_flags[0].item()):
            if not self.is_main_process:
                return None
            if cached is None:
                raise EvaluationError("rank zero generated cache state is inconsistent")
            feature_values, probability_values = cached
        else:
            local_start, local_end = _rank_case_bounds(
                len(cases),
                self.evaluation.batch_size,
                rank=self.rank,
                world_size=self.world_size,
            )
            local_cases = cases[local_start:local_end]
            groups: dict[tuple[int, int], list[PromptCase]] = {}
            for case in local_cases:
                groups.setdefault((case.height, case.width), []).append(case)
            generated_features: list[torch.Tensor] = []
            generated_probabilities: list[torch.Tensor] = []
            local_feature_values = torch.empty(
                (0, 2048), dtype=torch.float32, device=self.device
            )
            local_probability_values = torch.empty(
                (0, 1000), dtype=torch.float32, device=self.device
            )
            local_error: Exception | None = None
            was_training = self.composite.training
            self.composite.eval()
            try:
                completed = 0
                for group in groups.values():
                    for start in range(0, len(group), self.evaluation.batch_size):
                        batch_cases = tuple(
                            group[start : start + self.evaluation.batch_size]
                        )
                        images = _generate(
                            batch_cases,
                            config=self.config,
                            evaluation=self.evaluation,
                            composite=self.composite,
                            qwen=self.qwen,
                            vae=self.vae,
                            device=self.device,
                            growth_alpha=self.growth_alpha,
                        )
                        generated_features.append(self._models().features(images))
                        generated_probabilities.append(
                            self._models().probabilities(images)
                        )
                        completed += len(batch_cases)
                        print(
                            f"[eval] rank={self.rank} 已处理 "
                            f"{completed}/{len(local_cases)}",
                            flush=True,
                        )
                local_feature_values = torch.cat(generated_features)
                local_probability_values = torch.cat(generated_probabilities)
                if local_feature_values.shape != (len(local_cases), 2048):
                    raise EvaluationError("generated FID features have an invalid shape")
                if local_probability_values.shape != (len(local_cases), 1000):
                    raise EvaluationError(
                        "generated IS probabilities have an invalid shape"
                    )
            except Exception as error:  # noqa: BLE001 - synchronize rank failures
                local_error = error
            finally:
                self.composite.train(was_training)

            failure_flags = self._gather(
                torch.tensor(
                    [int(local_error is not None)],
                    dtype=torch.int64,
                    device=self.device,
                )
            )
            if bool(failure_flags.any().item()):
                if local_error is not None:
                    raise local_error
                raise EvaluationError("evaluation generation failed on another rank")

            rank_bounds = tuple(
                _rank_case_bounds(
                    len(cases),
                    self.evaluation.batch_size,
                    rank=process_index,
                    world_size=self.world_size,
                )
                for process_index in range(self.world_size)
            )
            rank_counts = tuple(end - start for start, end in rank_bounds)
            max_rank_count = max(rank_counts)
            padded_features = functional.pad(
                local_feature_values,
                (0, 0, 0, max_rank_count - len(local_cases)),
            )
            padded_probabilities = functional.pad(
                local_probability_values,
                (0, 0, 0, max_rank_count - len(local_cases)),
            )
            gathered_features = self._gather(padded_features)
            gathered_probabilities = self._gather(padded_probabilities)
            if not self.is_main_process:
                return None
            feature_chunks = gathered_features.reshape(
                self.world_size, max_rank_count, 2048
            )
            probability_chunks = gathered_probabilities.reshape(
                self.world_size, max_rank_count, 1000
            )
            feature_values = torch.cat(
                tuple(
                    feature_chunks[index, :count]
                    for index, count in enumerate(rank_counts)
                )
            )
            probability_values = torch.cat(
                tuple(
                    probability_chunks[index, :count]
                    for index, count in enumerate(rank_counts)
                )
            )
            if fingerprint is not None:
                self._save_generated_cache(
                    update,
                    fingerprint=fingerprint,
                    prompt_sha256=prompt_sha256,
                    features=feature_values,
                    probabilities=probability_values,
                )
        real = self._real()
        generated = FeatureStats.from_features(
            feature_values,
            device=self.device,
        )
        score = inception_score(
            probability_values,
            splits=self.evaluation.is_splits,
            device=self.device,
        )
        print(
            f"[eval] FID 矩阵运算设备: {self.device}; "
            f"样本空间: {generated.count}x{generated.count}",
            flush=True,
        )
        fid = frechet_inception_distance(generated, real, device=self.device)
        payload: dict[str, object] = {
            "schema_version": 1,
            "update": update,
            "fid": fid,
            "inception_score_mean": score.mean,
            "inception_score_std": score.std,
            "sample_count": score.sample_count,
            "is_splits": score.splits,
            "sampling_profile": self.evaluation.sampling_profile,
        }
        result_path = self.output / f"step-{update}.toml"
        _write_toml(result_path, payload)
        _write_toml(self.output / "latest.toml", payload)
        print(
            f"[eval] 完成: FID={fid:.4f}, IS={score.mean:.4f}±{score.std:.4f}",
            flush=True,
        )
        return EvaluationResult(
            update,
            fid,
            score.mean,
            score.std,
            score.sample_count,
            result_path,
        )


__all__ = ["EvaluationError", "EvaluationResult", "TrainingEvaluator"]
