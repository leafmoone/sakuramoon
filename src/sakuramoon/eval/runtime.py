"""Periodic in-process FID, IS, KID, and CMMD evaluation."""

from __future__ import annotations

import dataclasses
import hashlib
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Iterator
from typing import cast

import tomli_w
import torch
from accelerate import Accelerator
from torch.nn import functional

from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.config.schema import EvaluationEnabledConfig, RuntimeConfig
from sakuramoon.data.caption import CaptionPlan, empty_caption_dropout_hits
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
    SerializedCaption,
    TokenEncoder,
    serialize_caption,
)
from sakuramoon.distributed import DistributedProgress
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.qwen import QwenRuntime
from sakuramoon.eval.features import (
    CLIP_EMBEDDING_DIM,
    CLIP_MODEL_ID,
    FEATURE_CACHE_SCHEMA_VERSION,
    INCEPTION_FEATURE_DIM,
    INCEPTION_LOGIT_DIM,
    REAL_PREPROCESSING_ID,
    EvaluationFeatureModels,
    ImageFeatureBatch,
    iter_validation_image_batches,
    validation_dataset_fingerprint,
)
from sakuramoon.eval.metrics import (
    FeatureStats,
    clip_maximum_mean_discrepancy,
    frechet_inception_distance,
    inception_score,
    kernel_inception_distance,
)
from sakuramoon.eval.spec import PromptCase, PromptManifest
from sakuramoon.objective.flow import guided_velocity
from sakuramoon.sampling.sampler import sample_profile
from sakuramoon.storage import repository_directory
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs


class EvaluationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    update: int
    fid: float
    inception_score_mean: float
    inception_score_std: float
    kid_mean: float
    kid_std: float
    cmmd: float
    sample_count: int
    real_sample_count: int
    result_path: Path


_NO_DROPOUT = empty_caption_dropout_hits()
_ALL_DROPPED = empty_caption_dropout_hits(all_condition=True)


@contextmanager
def _eager_composite_calls(composite: torch.nn.Module) -> Iterator[None]:
    """Temporarily bypass regional compiled call slots during evaluation."""

    compiled: list[tuple[torch.nn.Module, object]] = []
    for module in composite.modules():
        call_impl = getattr(module, "_compiled_call_impl", None)
        if call_impl is not None:
            compiled.append((module, call_impl))
            setattr(module, "_compiled_call_impl", None)
    try:
        yield
    finally:
        for module, call_impl in compiled:
            setattr(module, "_compiled_call_impl", call_impl)


def _conditional_plan(case: PromptCase) -> CaptionPlan:
    if case.caption_plan is not None:
        return case.caption_plan
    return CaptionPlan(
        tags=(),
        condition=None,
        nl_text=case.prompt,
        selected_nl="long_names",
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


def _unconditional_plan() -> CaptionPlan:
    return CaptionPlan(
        tags=(),
        condition=None,
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
    if type(padding_token_id) is not int or not callable(
        getattr(encoder, "encode", None)
    ):
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
    condition_indices, condition_mask = _index_tensor(
        tuple(item.condition_token_indices for item in captions), device
    )
    use_null = torch.tensor(
        tuple(item.use_null_condition for item in captions),
        dtype=torch.bool,
        device=device,
    )
    active_condition = torch.tensor(
        tuple(
            index
            for index, item in enumerate(captions)
            if not item.use_null_condition
        ),
        dtype=torch.long,
        device=device,
    )
    return (
        input_ids,
        attention_mask,
        main_indices,
        main_mask,
        tuple(len(item.main_token_indices) for item in captions),
        condition_indices,
        condition_mask,
        use_null,
        active_condition,
    )


def _initial_noise(cases: tuple[PromptCase, ...], device: torch.device) -> torch.Tensor:
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
        condition_indices,
        condition_mask,
        use_null,
        active_condition,
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
    coordinate_map = image_coordinates(
        height // 16,
        width // 16,
        device=device,
    )
    inputs = TrainableCompositeInputs(
        qwen_states=qwen_states,
        main_token_indices=main_indices,
        main_mask=main_mask,
        main_token_lengths=main_lengths,
        condition_token_indices=condition_indices,
        condition_mask=condition_mask,
        use_null_condition=use_null,
        active_condition_sample_indices=active_condition,
        latents=placeholders,
        image_coordinates=(coordinate_map,) * branch_count,
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

    with _eager_composite_calls(composite):
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
        raise EvaluationError(
            f"validation prompts are incomplete: {len(selected)}/{count}"
        )
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
        self.progress = DistributedProgress.from_default_store(
            namespace=f"evaluation/{config.run.run_id}",
            rank=self.rank,
            world_size=self.world_size,
        )
        self.output = repository_directory(repository_root, self.evaluation.output_dir)
        self._feature_models: EvaluationFeatureModels | None = None
        self._real_features: ImageFeatureBatch | None = None

    def set_growth_alpha(self, value: float) -> None:
        """Select the canonical alpha for the update being evaluated."""

        if type(value) is not float or not 0.0 <= value <= 1.0:
            raise ValueError("growth_alpha must be a float in [0,1]")
        self.growth_alpha = value

    def due(self, update: int) -> bool:
        return update > 0 and update % self.evaluation.every_updates == 0

    def _models(self) -> EvaluationFeatureModels:
        if self._feature_models is None:
            self._feature_models = EvaluationFeatureModels(self.root, self.device)
        return self._feature_models

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
        return (
            self.output
            / "cache"
            / (
                f"generated-features-v{FEATURE_CACHE_SCHEMA_VERSION}-step-{update}-"
                f"{fingerprint[:16]}.pt"
            )
        )

    def _load_generated_cache(
        self,
        update: int,
        *,
        fingerprint: str,
        prompt_sha256: str,
    ) -> ImageFeatureBatch | None:
        path = self._generated_cache_path(update, fingerprint)
        if not path.exists():
            return None
        if not path.is_file() or path.is_symlink():
            raise EvaluationError("generated feature cache is not a regular file")
        try:
            raw = cast(
                object,
                torch.load(path, map_location="cpu", weights_only=True),
            )
        except (OSError, RuntimeError, ValueError) as error:
            raise EvaluationError("generated feature cache cannot be loaded") from error
        if type(raw) is not dict:
            raise EvaluationError("generated feature cache document is invalid")
        document = cast(dict[str, object], raw)
        expected: dict[str, object] = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "run_id": self.config.run.run_id,
            "update": update,
            "checkpoint_fingerprint": fingerprint,
            "prompt_sha256": prompt_sha256,
            "sample_count": self.evaluation.sample_count,
            "resolution": self.config.stage.resolution,
            "sampling_profile": self.evaluation.sampling_profile,
            "clip_model_id": CLIP_MODEL_ID,
            "preprocessing": REAL_PREPROCESSING_ID,
        }
        if any(document.get(key) != value for key, value in expected.items()):
            raise EvaluationError("generated feature cache metadata differs")
        inception = document.get("inception_features")
        clip = document.get("clip_features")
        logits = document.get("inception_logits")
        if not all(
            isinstance(value, torch.Tensor) for value in (inception, clip, logits)
        ):
            raise EvaluationError("generated feature cache tensors are missing")
        bundle = ImageFeatureBatch(
            cast(torch.Tensor, inception),
            cast(torch.Tensor, clip),
            cast(torch.Tensor, logits),
        )
        if bundle.count != self.evaluation.sample_count:
            raise EvaluationError("generated feature cache count differs")
        print(f"[eval] reusing generated feature cache: {path}", flush=True)
        return bundle.to(self.device)

    def _save_generated_cache(
        self,
        update: int,
        *,
        fingerprint: str,
        prompt_sha256: str,
        features: ImageFeatureBatch,
    ) -> None:
        path = self._generated_cache_path(update, fingerprint)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        cpu_features = features.cpu()
        try:
            torch.save(
                {
                    "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
                    "run_id": self.config.run.run_id,
                    "update": update,
                    "checkpoint_fingerprint": fingerprint,
                    "prompt_sha256": prompt_sha256,
                    "sample_count": self.evaluation.sample_count,
                    "resolution": self.config.stage.resolution,
                    "sampling_profile": self.evaluation.sampling_profile,
                    "clip_model_id": CLIP_MODEL_ID,
                    "preprocessing": REAL_PREPROCESSING_ID,
                    "inception_features": cpu_features.inception_features,
                    "clip_features": cpu_features.clip_features,
                    "inception_logits": cpu_features.inception_logits,
                },
                temporary,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        print(f"[eval] saved generated feature cache: {path}", flush=True)

    def _real(self) -> ImageFeatureBatch:
        if self._real_features is not None:
            return self._real_features
        shard_root = self.root / self.evaluation.validation_shard_root
        selection_path = self.root / self.config.data.validation.selection_path
        dataset_fingerprint = validation_dataset_fingerprint(shard_root, selection_path)
        real_sample_count = self.evaluation.resolved_real_sample_count
        cache = (
            self.output
            / "cache"
            / (
                f"real-features-v{FEATURE_CACHE_SCHEMA_VERSION}-n"
                f"{real_sample_count}-r{self.config.stage.resolution}-"
                f"{dataset_fingerprint[:16]}.pt"
            )
        )
        expected: dict[str, object] = {
            "schema_version": FEATURE_CACHE_SCHEMA_VERSION,
            "dataset_fingerprint": dataset_fingerprint,
            "sample_count": real_sample_count,
            "resolution": self.config.stage.resolution,
            "clip_model_id": CLIP_MODEL_ID,
            "preprocessing": REAL_PREPROCESSING_ID,
        }
        if cache.exists():
            if not cache.is_file() or cache.is_symlink():
                raise EvaluationError("real feature cache is not a regular file")
            try:
                raw = cast(
                    object,
                    torch.load(cache, map_location="cpu", weights_only=True),
                )
            except (OSError, RuntimeError, ValueError) as error:
                raise EvaluationError("real feature cache cannot be loaded") from error
            if type(raw) is not dict:
                raise EvaluationError("real feature cache document is invalid")
            document = cast(dict[str, object], raw)
            if any(document.get(key) != value for key, value in expected.items()):
                raise EvaluationError("real feature cache metadata differs")
            inception = document.get("real_inception_features")
            clip = document.get("real_clip_features")
            logits = document.get("real_inception_logits")
            if not all(
                isinstance(value, torch.Tensor) for value in (inception, clip, logits)
            ):
                raise EvaluationError("real feature cache tensors are missing")
            loaded = ImageFeatureBatch(
                cast(torch.Tensor, inception),
                cast(torch.Tensor, clip),
                cast(torch.Tensor, logits),
            )
            if loaded.count != real_sample_count:
                raise EvaluationError("real feature cache count differs")
            self._real_features = loaded.to(self.device)
            print(f"[eval] using real feature cache: {cache}", flush=True)
            return self._real_features

        print("[eval] computing one-time real Inception/CLIP/logit cache", flush=True)
        feature_batches: list[ImageFeatureBatch] = []
        completed = 0
        for batch in iter_validation_image_batches(
            shard_root,
            real_sample_count,
            self.evaluation.batch_size,
            output_size=self.config.stage.resolution,
        ):
            feature_batches.append(self._models().extract(batch.images).cpu())
            completed += len(batch.sample_ids)
            print(
                f"[eval] real features {completed}/{real_sample_count}",
                flush=True,
            )
        computed = ImageFeatureBatch.concatenate(tuple(feature_batches)).to(
            self.device
        )
        if computed.count != real_sample_count:
            raise EvaluationError("computed real feature count differs")
        cpu_features = computed.cpu()
        cache.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
        try:
            torch.save(
                {
                    **expected,
                    "real_inception_features": cpu_features.inception_features,
                    "real_clip_features": cpu_features.clip_features,
                    "real_inception_logits": cpu_features.inception_logits,
                },
                temporary,
            )
            with temporary.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temporary, cache)
        finally:
            temporary.unlink(missing_ok=True)
        self._real_features = computed
        print(f"[eval] saved real feature cache: {cache}", flush=True)
        return self._real_features

    def _finalize(
        self,
        update: int,
        features: ImageFeatureBatch,
        *,
        fingerprint: str | None,
        prompt_sha256: str,
        save_generated_cache: bool,
    ) -> EvaluationResult:
        if features.count != self.evaluation.sample_count:
            raise EvaluationError("generated feature count differs before metrics")
        if save_generated_cache:
            if fingerprint is None:
                raise EvaluationError(
                    "generated cache publication requires a checkpoint fingerprint"
                )
            self._save_generated_cache(
                update,
                fingerprint=fingerprint,
                prompt_sha256=prompt_sha256,
                features=features,
            )
        real = self._real()
        generated = FeatureStats.from_features(
            features.inception_features,
            device=self.device,
        )
        real_stats = FeatureStats.from_features(
            real.inception_features,
            device=self.device,
        )
        probability_values = features.inception_logits.softmax(dim=1)
        score = inception_score(
            probability_values,
            splits=self.evaluation.is_splits,
            device=self.device,
        )
        fid_dimension = (
            generated.count
            if generated.centered_features is not None
            else generated.covariance.shape[0]
        )
        print(
            f"[eval] FID 矩阵运算设备: {self.device}; "
            f"特征空间: {fid_dimension}x{fid_dimension}",
            flush=True,
        )
        fid = frechet_inception_distance(generated, real_stats, device=self.device)
        kid_seed = ((self.config.run.seed << 32) ^ update) % (2**63)
        kid = kernel_inception_distance(
            features.inception_features,
            real.inception_features,
            subsets=self.evaluation.kid_subsets,
            subset_size=self.evaluation.kid_subset_size,
            seed=kid_seed,
            device=self.device,
        )
        cmmd = clip_maximum_mean_discrepancy(
            features.clip_features,
            real.clip_features,
            device=self.device,
        )
        payload: dict[str, object] = {
            "schema_version": 3,
            "update": update,
            "fid": fid,
            "inception_score_mean": score.mean,
            "inception_score_std": score.std,
            "kid_mean": kid.mean,
            "kid_std": kid.std,
            "kid_subsets": kid.subsets,
            "kid_subset_size": kid.subset_size,
            "kid_seed": kid_seed,
            "cmmd": cmmd,
            "cmmd_clip_model": CLIP_MODEL_ID,
            "sample_count": score.sample_count,
            "real_sample_count": real.count,
            "is_splits": score.splits,
            "sampling_profile": self.evaluation.sampling_profile,
        }
        result_path = self.output / f"step-{update}.toml"
        _write_toml(result_path, payload)
        _write_toml(self.output / "latest.toml", payload)
        print(
            f"[eval] complete: FID={fid:.4f}, "
            f"IS={score.mean:.4f}±{score.std:.4f}, "
            f"KID={kid.mean:.6f}±{kid.std:.6f}, CMMD={cmmd:.6f}",
            flush=True,
        )
        return EvaluationResult(
            update,
            fid,
            score.mean,
            score.std,
            kid.mean,
            kid.std,
            cmmd,
            score.sample_count,
            real.count,
            result_path,
        )

    def evaluate(self, update: int) -> EvaluationResult | None:
        if not self.due(update):
            raise ValueError("evaluation update is not due")
        print(
            f"[eval] update={update} rank={self.rank}/{self.world_size} "
            "开始计算 FID/IS/KID/CMMD",
            flush=True,
        )
        self.progress.run_all(
            f"evaluation/update-{update}/initialize-feature-models",
            self._models,
        )
        prompt_path = self.root / self.evaluation.prompt_path
        prompt_sha256 = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
        cases = _stage_cases(
            prompt_path,
            self.evaluation.sample_count,
            resolution=self.config.stage.resolution,
        )
        fingerprint = self._checkpoint_fingerprint(update)
        cached = self.progress.run_on_rank(
            f"evaluation/update-{update}/load-generated-cache",
            0,
            lambda: (
                self._load_generated_cache(
                    update,
                    fingerprint=fingerprint,
                    prompt_sha256=prompt_sha256,
                )
                if fingerprint is not None
                else None
            ),
        )
        cache_flags = self._gather(
            torch.tensor(
                [int(cached is not None)],
                dtype=torch.int64,
                device=self.device,
            )
        )
        if cache_flags.shape != (self.world_size,):
            raise EvaluationError("generated cache flags have an invalid shape")
        if self.world_size > 1 and bool(cache_flags[1:].any().item()):
            raise EvaluationError("only rank zero may own generated cache state")
        cache_hit = bool(cache_flags[0].item())
        feature_values: ImageFeatureBatch | None = None
        if cache_hit:
            if self.is_main_process:
                if cached is None:
                    raise EvaluationError(
                        "rank zero generated cache state is inconsistent"
                    )
                feature_values = cached
        else:
            local_start, local_end = _rank_case_bounds(
                len(cases),
                self.evaluation.batch_size,
                rank=self.rank,
                world_size=self.world_size,
            )
            local_cases = cases[local_start:local_end]

            def generate_local_features() -> ImageFeatureBatch:
                groups: dict[tuple[int, int], list[PromptCase]] = {}
                for case in local_cases:
                    groups.setdefault((case.height, case.width), []).append(case)
                generated_features: list[ImageFeatureBatch] = []
                was_training = self.composite.training
                self.composite.eval()
                try:
                    completed = 0
                    for group in groups.values():
                        for start in range(
                            0,
                            len(group),
                            self.evaluation.batch_size,
                        ):
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
                            generated_features.append(
                                self._models().extract(images)
                            )
                            completed += len(batch_cases)
                            print(
                                f"[eval] rank={self.rank} 已处理 "
                                f"{completed}/{len(local_cases)}",
                                flush=True,
                            )
                finally:
                    self.composite.train(was_training)
                result = ImageFeatureBatch.concatenate(
                    tuple(generated_features)
                )
                if result.count != len(local_cases):
                    raise EvaluationError(
                        "local generated feature count is invalid"
                    )
                return result

            local_feature_values = self.progress.run_all(
                f"evaluation/update-{update}/generate-local-features",
                generate_local_features,
            )
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
            padded_inception = functional.pad(
                local_feature_values.inception_features,
                (0, 0, 0, max_rank_count - len(local_cases)),
            )
            padded_clip = functional.pad(
                local_feature_values.clip_features,
                (0, 0, 0, max_rank_count - len(local_cases)),
            )
            padded_logits = functional.pad(
                local_feature_values.inception_logits,
                (0, 0, 0, max_rank_count - len(local_cases)),
            )
            gathered_inception = self._gather(padded_inception)
            gathered_clip = self._gather(padded_clip)
            gathered_logits = self._gather(padded_logits)
            if self.is_main_process:
                inception_chunks = gathered_inception.reshape(
                    self.world_size,
                    max_rank_count,
                    INCEPTION_FEATURE_DIM,
                )
                clip_chunks = gathered_clip.reshape(
                    self.world_size,
                    max_rank_count,
                    CLIP_EMBEDDING_DIM,
                )
                logit_chunks = gathered_logits.reshape(
                    self.world_size,
                    max_rank_count,
                    INCEPTION_LOGIT_DIM,
                )
                feature_values = ImageFeatureBatch(
                    torch.cat(
                        tuple(
                            inception_chunks[index, :count]
                            for index, count in enumerate(rank_counts)
                        )
                    ),
                    torch.cat(
                        tuple(
                            clip_chunks[index, :count]
                            for index, count in enumerate(rank_counts)
                        )
                    ),
                    torch.cat(
                        tuple(
                            logit_chunks[index, :count]
                            for index, count in enumerate(rank_counts)
                        )
                    ),
                )
                if feature_values.count != self.evaluation.sample_count:
                    raise EvaluationError(
                        "gathered generated feature count differs"
                    )

        def finalize_on_main() -> EvaluationResult:
            if feature_values is None:
                raise EvaluationError(
                    "rank zero generated features are unavailable"
                )
            return self._finalize(
                update,
                feature_values,
                fingerprint=fingerprint,
                prompt_sha256=prompt_sha256,
                save_generated_cache=not cache_hit and fingerprint is not None,
            )

        result = self.progress.run_on_rank(
            f"evaluation/update-{update}/finalize-metrics",
            0,
            finalize_on_main,
        )
        if self.is_main_process:
            if not isinstance(result, EvaluationResult):
                raise EvaluationError(
                    "rank zero evaluation result is unavailable"
                )
            return result
        if result is not None:
            raise EvaluationError("nonzero rank received an evaluation result")
        return None


__all__ = ["EvaluationError", "EvaluationResult", "TrainingEvaluator"]
