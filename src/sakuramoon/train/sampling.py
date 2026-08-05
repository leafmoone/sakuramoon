"""Periodic training-caption image sampling for the single-GPU trainer."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.caption import CaptionDropoutHits, CaptionPlan
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
    SerializedCaption,
    serialize_caption,
)
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.qwen import QwenRuntime
from sakuramoon.objective.flow import guided_velocity
from sakuramoon.sampling.sampler import sample_profile
from sakuramoon.storage import repository_directory
from sakuramoon.train.runtime import RuntimeMeasurement
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs


class TrainingSamplingError(RuntimeError):
    """A periodic training sample could not be generated or persisted."""


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


@dataclass(frozen=True, slots=True)
class TrainingSampleItem:
    ordinal: int
    sample_id: str
    caption: SerializedCaption
    height: int
    width: int
    seed: int


@dataclass(frozen=True, slots=True)
class TrainingSampleResult:
    update: int
    paths: tuple[Path, ...]
    captions: tuple[str, ...]


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


def _parse_shape_key(value: str) -> tuple[int, int]:
    parts = value.split("x")
    if len(parts) != 3:
        raise TrainingSamplingError("training sample shape key is invalid")
    try:
        height, width, dense_length = (int(item) for item in parts)
    except ValueError:
        raise TrainingSamplingError("training sample shape key is invalid") from None
    if min(height, width, dense_length) <= 0 or height % 16 or width % 16:
        raise TrainingSamplingError("training sample shape dimensions are invalid")
    return height, width


def _conditioning_inputs(
    captions: tuple[SerializedCaption, ...],
    *,
    tokenizer: object,
    device: torch.device,
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
    padding_token_id = getattr(tokenizer, "pad_token_id", None)
    if type(padding_token_id) is not int or padding_token_id < 0:
        raise TrainingSamplingError("Qwen padding token ID is unavailable")
    if not captions:
        raise TrainingSamplingError("training sample caption group is empty")
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
    for row, caption in enumerate(captions):
        length = len(caption.input_ids)
        if length > dense_length:
            raise TrainingSamplingError("serialized sample caption exceeds its bucket")
        input_ids[row, :length] = torch.tensor(
            caption.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = True

    def index_tensor(
        values: tuple[tuple[int, ...], ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        width = max((len(item) for item in values), default=0)
        indices = torch.full(
            (len(values), width), -1, dtype=torch.long, device=device
        )
        mask = torch.zeros((len(values), width), dtype=torch.bool, device=device)
        for row, item in enumerate(values):
            if item:
                indices[row, : len(item)] = torch.tensor(
                    item, dtype=torch.long, device=device
                )
                mask[row, : len(item)] = True
        return indices, mask

    main_indices, main_mask = index_tensor(
        tuple(item.main_token_indices for item in captions)
    )
    artist_indices, artist_mask = index_tensor(
        tuple(item.artist_token_indices for item in captions)
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


class TrainingSampler:
    """Generate images from captions that actually entered recent training updates."""

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
        if not isinstance(config, RuntimeConfig) or not repository_root.is_absolute():
            raise TypeError("training sampler requires resolved config and absolute root")
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("training sampler requires the active CUDA device")
        self.config = config
        self.repository_root = repository_root
        self.composite = composite
        self.qwen = qwen
        self.vae = vae
        self.device = device
        self.growth_alpha = growth_alpha
        self.output_root = (
            repository_directory(repository_root, config.paths.checkpoint_dir)
            / config.sampling.training.output_subdir
        )

    def due(self, update: int) -> bool:
        settings = self.config.sampling.training
        return (
            settings.enabled
            and update > 0
            and update % settings.every_updates == 0
        )

    def _candidates(
        self,
        measurements: tuple[RuntimeMeasurement, ...],
    ) -> list[tuple[str, SerializedCaption, int, int]]:
        result: list[tuple[str, SerializedCaption, int, int]] = []
        for measurement in measurements:
            captions = measurement.captions
            if not captions:
                continue
            if not (
                len(captions) == len(measurement.sample_ids)
                == len(measurement.shape_keys)
            ):
                raise TrainingSamplingError(
                    "training sample metadata does not match the measured batch"
                )
            for sample_id, caption, shape_key in zip(
                measurement.sample_ids,
                captions,
                measurement.shape_keys,
                strict=True,
            ):
                height, width = _parse_shape_key(shape_key)
                result.append((sample_id, caption, height, width))
        return result

    def _generate_group(
        self,
        items: tuple[TrainingSampleItem, ...],
    ) -> torch.Tensor:
        if not items:
            raise TrainingSamplingError("cannot generate an empty sample group")
        height, width = items[0].height, items[0].width
        if any((item.height, item.width) != (height, width) for item in items):
            raise TrainingSamplingError("sample group contains mixed image shapes")
        padding_token_id = getattr(self.qwen.tokenizer, "pad_token_id", None)
        if type(padding_token_id) is not int:
            raise TrainingSamplingError("Qwen tokenizer padding identity is unavailable")
        framing = FramingContract(
            EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, padding_token_id
        )
        conditional = tuple(item.caption for item in items)
        unconditional = serialize_caption(
            _unconditional_plan(), self.qwen.tokenizer, framing
        )
        captions = conditional + (unconditional,) * len(conditional)
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
        ) = _conditioning_inputs(
            captions, tokenizer=self.qwen.tokenizer, device=self.device
        )
        qwen_output = self.qwen.encoder(input_ids, attention_mask)
        qwen_states = getattr(qwen_output, "hidden_states", None)
        if not isinstance(qwen_states, torch.Tensor):
            raise TrainingSamplingError("Qwen output lacks hidden states")
        branch_count = len(items) * 2
        size_scale_value = 0.5 * math.log2((height * width) / float(512 * 512))
        aspect_value = math.log2(width / float(height))
        size_scale = torch.full(
            (branch_count,), size_scale_value, dtype=torch.float32, device=self.device
        )
        aspect = torch.full(
            (branch_count,), aspect_value, dtype=torch.float32, device=self.device
        )
        placeholders = tuple(
            torch.empty(
                128,
                height // 16,
                width // 16,
                dtype=torch.bfloat16,
                device=self.device,
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
            timestep=torch.zeros(branch_count, dtype=torch.float32, device=self.device),
            size_scale=size_scale,
            aspect=aspect,
            growth_alpha=self.growth_alpha,
        )
        conditioning = self.composite.forward_conditioning(inputs)
        noises: list[torch.Tensor] = []
        for item in items:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(item.seed)
            noises.append(
                torch.randn(
                    (128, height // 16, width // 16),
                    generator=generator,
                    device=self.device,
                    dtype=torch.float32,
                )
            )
        noise = torch.stack(noises)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            branches = torch.cat((state, state), dim=0).to(torch.bfloat16)
            step_inputs = dataclasses.replace(
                inputs,
                latents=tuple(branches.unbind(0)),
                timestep=torch.cat((timestep, timestep), dim=0),
            )
            predicted = self.composite.forward_dit(step_inputs, conditioning)
            conditional_pred = torch.stack(predicted[: len(items)])
            unconditional_pred = torch.stack(predicted[len(items) :])
            return guided_velocity(
                conditional_pred,
                unconditional_pred,
                state,
                timestep,
                t_eps=self.config.timestep.t_eps,
                guidance_scale=self.config.cfg.scale,
            )

        sampled = sample_profile(
            velocity, noise, profile=self.config.sampling.profile
        )
        decoded = self.vae.decode(sampled.state.to(torch.bfloat16))
        if not bool(torch.isfinite(decoded).all().item()):
            raise TrainingSamplingError("VAE produced nonfinite training samples")
        return (
            decoded.float()
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0.0, 255.0)
            .to(device="cpu", dtype=torch.uint8)
        )

    @staticmethod
    def _wandb_caption(item: TrainingSampleItem) -> str:
        dropout = json.dumps(
            item.caption.dropout_hits.as_mapping(),
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            f"sample_id={item.sample_id}\n"
            f"body={item.caption.body}\n"
            f"artist={item.caption.artist_text or '<null>'}\n"
            f"dropout={dropout}"
        )

    def sample(
        self,
        update: int,
        measurements: tuple[RuntimeMeasurement, ...],
    ) -> TrainingSampleResult | None:
        if not self.due(update):
            return None
        candidates = self._candidates(measurements)
        if not candidates:
            raise TrainingSamplingError(
                "no post-dropout training captions were available for sampling"
            )
        selector = random.Random(
            f"{self.config.run.seed}\0training-sample\0{update}"
        )
        selector.shuffle(candidates)
        selected = candidates[: self.config.sampling.training.image_count]
        items = tuple(
            TrainingSampleItem(
                ordinal=ordinal,
                sample_id=sample_id,
                caption=caption,
                height=height,
                width=width,
                seed=selector.randrange(2**63),
            )
            for ordinal, (sample_id, caption, height, width) in enumerate(selected)
        )
        grouped: dict[tuple[int, int], list[TrainingSampleItem]] = defaultdict(list)
        for item in items:
            grouped[(item.height, item.width)].append(item)

        step_root = self.output_root / f"step-{update}"
        step_root.mkdir(parents=True, exist_ok=True)
        paths_by_ordinal: dict[int, Path] = {}
        was_training = self.composite.training
        self.composite.eval()
        try:
            with torch.inference_mode():
                for group_items in grouped.values():
                    images = self._generate_group(tuple(group_items))
                    if images.shape[0] != len(group_items):
                        raise TrainingSamplingError(
                            "sample image count differs from prompts"
                        )
                    for item, image in zip(group_items, images, strict=True):
                        path = step_root / f"sample-{len(paths_by_ordinal):02d}.png"
                        array = image.permute(1, 2, 0).contiguous().numpy()
                        Image.fromarray(array).save(path)
                        paths_by_ordinal[item.ordinal] = path
        finally:
            self.composite.train(was_training)
            torch.cuda.empty_cache()

        if len(paths_by_ordinal) != len(items):
            raise TrainingSamplingError("saved sample count differs from prompts")
        paths = tuple(paths_by_ordinal[item.ordinal] for item in items)
        wandb_captions = tuple(self._wandb_caption(item) for item in items)
        records = [
            {
                "sample_id": item.sample_id,
                "path": path.relative_to(self.repository_root).as_posix(),
                "height": item.height,
                "width": item.width,
                "prompt": item.caption.text,
                "body": item.caption.body,
                "artist": item.caption.artist_text,
                "selected_nl": item.caption.selected_nl,
                "dropout_hits": item.caption.dropout_hits.as_mapping(),
            }
            for item, path in zip(items, paths, strict=True)
        ]
        metadata = {
            "schema_version": 1,
            "update": update,
            "profile": self.config.sampling.profile,
            "records": records,
        }
        metadata_path = step_root / "metadata.json"
        temporary = step_root / f".metadata.{update}.tmp"
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
        print(
            f"[sample] update={update} ??? {len(paths)} ?: {step_root}",
            flush=True,
        )
        return TrainingSampleResult(update, paths, wandb_captions)


__all__ = [
    "TrainingSampleResult",
    "TrainingSampler",
    "TrainingSamplingError",
]
