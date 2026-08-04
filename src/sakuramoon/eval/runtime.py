"""Periodic in-process FID and Inception Score evaluation."""

from __future__ import annotations

import dataclasses
import io
import math
import os
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import tomli_w
import torch
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
        .to(device="cpu", dtype=torch.uint8)
    )


class _InceptionModels:
    def __init__(self) -> None:
        print("[eval] 加载标准 FID Inception 权重", flush=True)
        from pytorch_fid.inception import (  # pyright: ignore[reportMissingTypeStubs]
            InceptionV3,
        )

        block = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
        self.fid = cast(_FidExtractor, InceptionV3([block]).eval().to("cpu"))
        print("[eval] 加载 Inception Score 分类权重", flush=True)
        self.is_weights = Inception_V3_Weights.DEFAULT
        self.classifier = inception_v3(weights=self.is_weights).eval().to("cpu")

    @torch.inference_mode()
    def features(self, images: torch.Tensor) -> torch.Tensor:
        values = images.float().div(255.0)
        result = self.fid(values)[0]
        return result.flatten(1).cpu()

    @torch.inference_mode()
    def probabilities(self, images: torch.Tensor) -> torch.Tensor:
        values = self.is_weights.transforms()(images.float().div(255.0))
        logits = self.classifier(values)
        if not isinstance(logits, torch.Tensor):
            raise EvaluationError("Inception classifier returned an invalid result")
        return logits.softmax(dim=1).cpu()


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


def _stage_cases(config: RuntimeConfig, path: Path, count: int) -> tuple[PromptCase, ...]:
    print(f"[eval] 读取验证提示词: {path}", flush=True)
    manifest = PromptManifest.from_canonical_bytes(path.read_bytes())
    selected = manifest.cases[:count]
    if len(selected) != count:
        raise EvaluationError(f"validation prompts are incomplete: {len(selected)}/{count}")
    scale = config.stage.resolution / 512.0
    return tuple(
        dataclasses.replace(
            case,
            height=max(16, round(case.height * scale / 16) * 16),
            width=max(16, round(case.width * scale / 16) * 16),
        )
        for case in selected
    )


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
        self.output = repository_directory(repository_root, self.evaluation.output_dir)
        self._inception: _InceptionModels | None = None
        self._real_stats: FeatureStats | None = None

    def due(self, update: int) -> bool:
        return update > 0 and update % self.evaluation.every_updates == 0

    def _models(self) -> _InceptionModels:
        if self._inception is None:
            self._inception = _InceptionModels()
        return self._inception

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
        self._real_stats = FeatureStats.from_features(torch.cat(features))
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

    def evaluate(self, update: int) -> EvaluationResult:
        if not self.due(update):
            raise ValueError("evaluation update is not due")
        print(f"[eval] update={update} 开始计算 FID/IS", flush=True)
        real = self._real()
        prompt_path = self.root / self.evaluation.prompt_path
        cases = _stage_cases(self.config, prompt_path, self.evaluation.sample_count)
        groups: dict[tuple[int, int], list[PromptCase]] = {}
        for case in cases:
            groups.setdefault((case.height, case.width), []).append(case)
        generated_features: list[torch.Tensor] = []
        generated_probabilities: list[torch.Tensor] = []
        was_training = self.composite.training
        self.composite.eval()
        try:
            completed = 0
            for group in groups.values():
                for start in range(0, len(group), self.evaluation.batch_size):
                    batch_cases = tuple(group[start : start + self.evaluation.batch_size])
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
                    generated_probabilities.append(self._models().probabilities(images))
                    completed += len(batch_cases)
                    print(
                        f"[eval] 已处理 {completed}/{self.evaluation.sample_count}",
                        flush=True,
                    )
        finally:
            self.composite.train(was_training)
        generated = FeatureStats.from_features(torch.cat(generated_features))
        score = inception_score(
            torch.cat(generated_probabilities), splits=self.evaluation.is_splits
        )
        fid = frechet_inception_distance(generated, real)
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
