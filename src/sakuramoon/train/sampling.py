"""Periodic paired-condition sampling for the single-GPU trainer."""

from __future__ import annotations

import dataclasses
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
from PIL import Image

from sakuramoon.conditioning.rope import full_canvas_crop_coordinates
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    Tag,
    empty_caption_dropout_hits,
)
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
from sakuramoon.objective.flow import guided_velocity
from sakuramoon.sampling.sampler import sample_profile
from sakuramoon.storage import repository_directory
from sakuramoon.train.runtime import RuntimeMeasurement
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs

PromptLabel = Literal["A", "B"]
VariantName = Literal[
    "A-base",
    "B-base",
    "A-with-B",
    "B-with-A",
    "A-null",
    "B-null",
    "A-with-BA",
    "B-with-BA",
    "A-zoom",
    "B-zoom",
    "A-shift-zoom",
    "B-shift-zoom",
]
GeometryKind = Literal["canonical", "zoom", "shift_zoom"]
CoordinateType = Literal[
    "canonical_full_canvas",
    "zoom_full_canvas_crop",
    "shift_zoom_full_canvas_crop",
]

_VARIANT_DEFINITIONS: tuple[
    tuple[VariantName, PromptLabel, tuple[PromptLabel, ...], GeometryKind], ...
] = (
    ("A-base", "A", ("A",), "canonical"),
    ("B-base", "B", ("B",), "canonical"),
    ("A-with-B", "A", ("B",), "canonical"),
    ("B-with-A", "B", ("A",), "canonical"),
    ("A-null", "A", (), "canonical"),
    ("B-null", "B", (), "canonical"),
    ("A-with-BA", "A", ("B", "A"), "canonical"),
    ("B-with-BA", "B", ("B", "A"), "canonical"),
    ("A-zoom", "A", ("A",), "zoom"),
    ("B-zoom", "B", ("B",), "zoom"),
    ("A-shift-zoom", "A", ("A",), "shift_zoom"),
    ("B-shift-zoom", "B", ("B",), "shift_zoom"),
)
_VARIANT_NAMES = tuple(definition[0] for definition in _VARIANT_DEFINITIONS)
_VARIANT_COUNT = 12
_CFG_BRANCH_COUNT = 24
_ZOOM = 1.5


class TrainingSamplingError(RuntimeError):
    """A periodic training sample could not be generated or persisted."""


_NO_DROPOUT = empty_caption_dropout_hits()
_ALL_DROPPED = empty_caption_dropout_hits(all_condition=True)


@dataclass(frozen=True, slots=True)
class _PostDropoutPrompt:
    sample_id: str
    caption: SerializedCaption
    plan: CaptionPlan
    observed_height: int
    observed_width: int

    def __post_init__(self) -> None:
        if self.caption.plan != self.plan:
            raise TrainingSamplingError("serialized caption and retained plan disagree")


@dataclass(frozen=True, slots=True)
class _PromptPair:
    a: _PostDropoutPrompt
    b: _PostDropoutPrompt

    def __post_init__(self) -> None:
        if self.a.sample_id == self.b.sample_id:
            raise TrainingSamplingError("A and B must be different training samples")
        a_condition = self.a.plan.condition
        b_condition = self.b.plan.condition
        if a_condition is None or b_condition is None:
            raise TrainingSamplingError("A and B must both have conditions")
        if (
            a_condition.source != b_condition.source
            or a_condition.role != b_condition.role
        ):
            raise TrainingSamplingError("A and B condition protocols must match")
        a_tags = frozenset(tag.canonical for tag in a_condition.tags)
        b_tags = frozenset(tag.canonical for tag in b_condition.tags)
        if a_tags & b_tags:
            raise TrainingSamplingError("A and B conditions must not overlap")


@dataclass(frozen=True, slots=True)
class TrainingSampleItem:
    ordinal: int
    variant: VariantName
    main_source: PromptLabel
    condition_sources: tuple[PromptLabel, ...]
    sample_id: str
    caption: SerializedCaption
    plan: CaptionPlan
    height: int
    width: int
    zoom: float
    virtual_canvas_size: tuple[int, int]
    crop_box: tuple[int, int, int, int]
    coordinate_type: CoordinateType


@dataclass(frozen=True, slots=True)
class TrainingSampleResult:
    update: int
    paths: tuple[Path, ...]
    captions: tuple[str, ...]


def _unconditional_plan() -> CaptionPlan:
    return CaptionPlan(
        tags=(),
        condition=None,
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
        indices = torch.full((len(values), width), -1, dtype=torch.long, device=device)
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
    condition_indices, condition_mask = index_tensor(
        tuple(item.condition_token_indices for item in captions)
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


def _select_prompt_pair(
    candidates: tuple[_PostDropoutPrompt, ...],
    selector: random.Random,
) -> _PromptPair:
    if type(candidates) is not tuple or not candidates:
        raise TrainingSamplingError("post-dropout candidate batch is empty")
    valid_pairs: list[_PromptPair] = []
    for first_index, first in enumerate(candidates):
        first_condition = first.plan.condition
        if first_condition is None:
            continue
        first_tags = frozenset(tag.canonical for tag in first_condition.tags)
        for second in candidates[first_index + 1 :]:
            second_condition = second.plan.condition
            if first.sample_id == second.sample_id or second_condition is None:
                continue
            second_tags = frozenset(tag.canonical for tag in second_condition.tags)
            if (
                first_condition.source == second_condition.source
                and first_condition.role == second_condition.role
                and first_tags.isdisjoint(second_tags)
            ):
                valid_pairs.append(_PromptPair(first, second))
    if not valid_pairs:
        condition_candidates = sum(
            candidate.plan.condition is not None for candidate in candidates
        )
        raise TrainingSamplingError(
            "no valid non-overlapping condition pair exists: "
            f"candidates={len(candidates)} "
            f"condition_candidates={condition_candidates}"
        )
    selected = valid_pairs[selector.randrange(len(valid_pairs))]
    if selector.getrandbits(1):
        return _PromptPair(selected.b, selected.a)
    return selected


def _variant_geometry(
    resolution: int,
    kind: GeometryKind,
) -> tuple[float, tuple[int, int], tuple[int, int, int, int], CoordinateType]:
    if type(resolution) is not int or resolution <= 0 or resolution % 16:
        raise TrainingSamplingError(
            "stage resolution must be a positive multiple of 16"
        )
    if resolution % 8:
        raise TrainingSamplingError(
            "stage resolution must support exact shift-zoom quarters"
        )
    if kind == "canonical":
        return (
            1.0,
            (resolution, resolution),
            (0, 0, resolution, resolution),
            "canonical_full_canvas",
        )
    virtual = 3 * resolution // 2
    available = virtual - resolution
    centered = available // 2
    if kind == "zoom":
        left = centered
        top = centered
        coordinate_type: CoordinateType = "zoom_full_canvas_crop"
    elif kind == "shift_zoom":
        left = centered + available // 4
        top = centered + available // 4
        coordinate_type = "shift_zoom_full_canvas_crop"
    else:
        raise TrainingSamplingError("unknown training sample geometry")
    return (
        _ZOOM,
        (virtual, virtual),
        (left, top, left + resolution, top + resolution),
        coordinate_type,
    )


def _build_variant_items(
    pair: _PromptPair,
    *,
    tokenizer: TokenEncoder,
    framing: FramingContract,
    resolution: int,
) -> tuple[TrainingSampleItem, ...]:
    sources: dict[PromptLabel, _PostDropoutPrompt] = {"A": pair.a, "B": pair.b}
    items: list[TrainingSampleItem] = []
    for ordinal, (variant, main_label, condition_labels, geometry) in enumerate(
        _VARIANT_DEFINITIONS
    ):
        main = sources[main_label]
        source_conditions = tuple(
            sources[condition_label].plan.condition
            for condition_label in condition_labels
        )
        if any(condition is None for condition in source_conditions):
            raise TrainingSamplingError("variant condition is unavailable")
        typed_conditions = tuple(
            condition
            for condition in source_conditions
            if condition is not None
        )
        protocols = {
            (condition.source, condition.role) for condition in typed_conditions
        }
        if len(protocols) > 1:
            raise TrainingSamplingError("variant condition protocols differ")
        condition = (
            ConditionRequest(
                source=typed_conditions[0].source,
                role=typed_conditions[0].role,
                tags=tuple(
                    tag for item in typed_conditions for tag in item.tags
                ),
            )
            if typed_conditions
            else None
        )
        requested_plan = dataclasses.replace(main.plan, condition=condition)
        caption = serialize_caption(requested_plan, tokenizer, framing)
        if caption.plan != requested_plan:
            raise TrainingSamplingError(
                f"variant {variant} cannot preserve its complete structured plan"
            )
        zoom, virtual_canvas, crop_box, coordinate_type = _variant_geometry(
            resolution, geometry
        )
        items.append(
            TrainingSampleItem(
                ordinal=ordinal,
                variant=variant,
                main_source=main_label,
                condition_sources=condition_labels,
                sample_id=main.sample_id,
                caption=caption,
                plan=requested_plan,
                height=resolution,
                width=resolution,
                zoom=zoom,
                virtual_canvas_size=virtual_canvas,
                crop_box=crop_box,
                coordinate_type=coordinate_type,
            )
        )
    result = tuple(items)
    _require_variant_batch(result)
    return result


def _require_variant_batch(
    items: tuple[TrainingSampleItem, ...],
) -> tuple[int, int]:
    if type(items) is not tuple or len(items) != _VARIANT_COUNT:
        raise TrainingSamplingError("training sampler requires exactly 12 variants")
    if tuple(item.variant for item in items) != _VARIANT_NAMES:
        raise TrainingSamplingError(
            "training sample variants are incomplete or reordered"
        )
    height, width = items[0].height, items[0].width
    if any((item.height, item.width) != (height, width) for item in items):
        raise TrainingSamplingError("training sample variants have mixed output sizes")
    return height, width


def _coordinate_maps(
    items: tuple[TrainingSampleItem, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    _require_variant_batch(items)
    maps = tuple(
        full_canvas_crop_coordinates(
            item.height // 16,
            item.width // 16,
            full_height=item.virtual_canvas_size[0],
            full_width=item.virtual_canvas_size[1],
            crop_box=item.crop_box,
            device=device,
        )
        for item in items
    )
    return maps


def _shared_initial_noise(
    *,
    height: int,
    width: int,
    shared_seed: int,
    device: torch.device,
) -> torch.Tensor:
    if type(shared_seed) is not int or not 0 <= shared_seed < 2**63:
        raise TrainingSamplingError("shared sample seed must be a 63-bit integer")
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise TrainingSamplingError("shared noise canvas is invalid")
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_seed)
    base_noise = torch.randn(
        (1, 128, height // 16, width // 16),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return base_noise.repeat(_VARIANT_COUNT, 1, 1, 1)


def _tag_metadata(tag: Tag) -> dict[str, str]:
    return {"text": tag.text, "canonical": tag.canonical}


def _caption_tag_metadata(tag: CaptionTag) -> dict[str, str]:
    return {"source": tag.source, **_tag_metadata(tag.tag)}


def _plan_metadata(plan: CaptionPlan) -> dict[str, object]:
    return {
        "tags": [_caption_tag_metadata(tag) for tag in plan.tags],
        "condition": (
            None
            if plan.condition is None
            else {
                "source": plan.condition.source,
                "role": plan.condition.role,
                "tags": [_tag_metadata(tag) for tag in plan.condition.tags],
            }
        ),
        "nl_text": plan.nl_text,
        "selected_nl": plan.selected_nl,
        "all_condition_dropped": plan.all_condition_dropped,
        "dropout_hits": plan.dropout_hits.as_mapping(),
    }


def _prompt_metadata(prompt: _PostDropoutPrompt) -> dict[str, object]:
    return {
        "sample_id": prompt.sample_id,
        "observed_training_canvas_size": {
            "height": prompt.observed_height,
            "width": prompt.observed_width,
        },
        "plan": _plan_metadata(prompt.plan),
        "serialized": {
            "body": prompt.caption.body,
            "condition_text": prompt.caption.condition_text,
            "condition_tokens": prompt.caption.condition_tokens,
            "dense_length": prompt.caption.dense_length,
            "truncated": prompt.caption.truncated,
        },
    }


def _source_for_label(pair: _PromptPair, label: PromptLabel) -> _PostDropoutPrompt:
    return pair.a if label == "A" else pair.b


def _variant_metadata(
    item: TrainingSampleItem,
    path: Path,
    *,
    pair: _PromptPair,
    shared_seed: int,
    update: int,
    repository_root: Path,
) -> dict[str, object]:
    main = _source_for_label(pair, item.main_source)
    condition_sources = tuple(
        _source_for_label(pair, label) for label in item.condition_sources
    )
    left, top, right, bottom = item.crop_box
    return {
        "update": update,
        "A_sample_id": pair.a.sample_id,
        "B_sample_id": pair.b.sample_id,
        "variant": item.variant,
        "path": path.relative_to(repository_root).as_posix(),
        "shared_seed": shared_seed,
        "main_source": {
            "label": item.main_source,
            "sample_id": main.sample_id,
        },
        "condition_sources": [label for label in item.condition_sources],
        "condition_source_details": [
            {
                "label": label,
                "sample_id": source.sample_id,
                "condition": _plan_metadata(source.plan)["condition"],
            }
            for label, source in zip(
                item.condition_sources,
                condition_sources,
                strict=True,
            )
        ],
        "resolved_plan": _plan_metadata(item.plan),
        "output_size": {"height": item.height, "width": item.width},
        "zoom": item.zoom,
        "virtual_canvas_size": {
            "height": item.virtual_canvas_size[0],
            "width": item.virtual_canvas_size[1],
        },
        "crop_box": {
            "left": left,
            "top": top,
            "right": right,
            "bottom": bottom,
        },
        "coordinate_type": item.coordinate_type,
    }


class TrainingSampler:
    """Generate one strict paired-condition diagnostic batch per due update."""

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
        if not repository_root.is_absolute():
            raise ValueError("training sampler repository root must be absolute")
        if config.stage.resolution % 16 or config.stage.resolution % 8:
            raise ValueError("training sampler stage resolution has invalid geometry")
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
        return settings.enabled and update > 0 and update % settings.every_updates == 0

    def _candidates(
        self,
        measurements: tuple[RuntimeMeasurement, ...],
    ) -> tuple[_PostDropoutPrompt, ...]:
        if type(measurements) is not tuple or not measurements:
            raise TrainingSamplingError(
                "training sampler requires measured microbatches"
            )
        result: list[_PostDropoutPrompt] = []
        for measurement in measurements:
            if not (
                len(measurement.captions)
                == len(measurement.caption_plans)
                == len(measurement.sample_ids)
                == len(measurement.shape_keys)
            ):
                raise TrainingSamplingError(
                    "training sample metadata does not match the measured batch"
                )
            for sample_id, caption, plan, shape_key in zip(
                measurement.sample_ids,
                measurement.captions,
                measurement.caption_plans,
                measurement.shape_keys,
                strict=True,
            ):
                height, width = _parse_shape_key(shape_key)
                result.append(
                    _PostDropoutPrompt(
                        sample_id=sample_id,
                        caption=caption,
                        plan=plan,
                        observed_height=height,
                        observed_width=width,
                    )
                )
        if not result:
            raise TrainingSamplingError(
                "no post-dropout training captions were available for sampling"
            )
        return tuple(result)

    def _generate_batch(
        self,
        items: tuple[TrainingSampleItem, ...],
        *,
        shared_seed: int,
        framing: FramingContract,
    ) -> torch.Tensor:
        height, width = _require_variant_batch(items)
        conditional = tuple(item.caption for item in items)
        unconditional = serialize_caption(
            _unconditional_plan(), self.qwen.tokenizer, framing
        )
        captions = conditional + (unconditional,) * _VARIANT_COUNT
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
        ) = _conditioning_inputs(
            captions, tokenizer=self.qwen.tokenizer, device=self.device
        )
        qwen_output = self.qwen.encoder(input_ids, attention_mask)
        qwen_states = getattr(qwen_output, "hidden_states", None)
        if not isinstance(qwen_states, torch.Tensor):
            raise TrainingSamplingError("Qwen output lacks hidden states")
        size_scale_value = 0.5 * math.log2((height * width) / float(512 * 512))
        aspect_value = math.log2(width / float(height))
        size_scale = torch.full(
            (_CFG_BRANCH_COUNT,),
            size_scale_value,
            dtype=torch.float32,
            device=self.device,
        )
        aspect = torch.full(
            (_CFG_BRANCH_COUNT,),
            aspect_value,
            dtype=torch.float32,
            device=self.device,
        )
        placeholders = tuple(
            torch.empty(
                128,
                height // 16,
                width // 16,
                dtype=torch.bfloat16,
                device=self.device,
            )
            for _ in range(_CFG_BRANCH_COUNT)
        )
        conditional_coordinates = _coordinate_maps(items, device=self.device)
        branch_coordinates = conditional_coordinates + conditional_coordinates
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
            image_coordinates=branch_coordinates,
            timestep=torch.zeros(
                _CFG_BRANCH_COUNT, dtype=torch.float32, device=self.device
            ),
            size_scale=size_scale,
            aspect=aspect,
            growth_alpha=self.growth_alpha,
        )
        conditioning = self.composite.forward_conditioning(inputs)
        noise = _shared_initial_noise(
            height=height,
            width=width,
            shared_seed=shared_seed,
            device=self.device,
        )

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            if state.shape != noise.shape or timestep.shape != (_VARIANT_COUNT,):
                raise TrainingSamplingError(
                    "sampler state or timestep batch is invalid"
                )
            branches = torch.cat((state, state), dim=0).to(torch.bfloat16)
            step_inputs = dataclasses.replace(
                inputs,
                latents=tuple(branches.unbind(0)),
                timestep=torch.cat((timestep, timestep), dim=0),
            )
            predicted = self.composite.forward_dit(step_inputs, conditioning)
            if len(predicted) != _CFG_BRANCH_COUNT:
                raise TrainingSamplingError("DiT did not return 24 CFG predictions")
            conditional_pred = torch.stack(predicted[:_VARIANT_COUNT])
            unconditional_pred = torch.stack(predicted[_VARIANT_COUNT:])
            return guided_velocity(
                conditional_pred,
                unconditional_pred,
                state,
                timestep,
                t_eps=self.config.timestep.t_eps,
                guidance_scale=self.config.cfg.scale,
            )

        sampled = sample_profile(
            velocity,
            noise,
            profile=self.config.sampling.profile,
        )
        if sampled.state.shape != noise.shape or sampled.state.dtype != torch.float32:
            raise TrainingSamplingError(
                "sample profile returned an invalid state batch"
            )
        decoded = self.vae.decode(sampled.state.to(torch.bfloat16))
        if decoded.shape != (_VARIANT_COUNT, 3, height, width):
            raise TrainingSamplingError("VAE returned an invalid 12-image batch")
        if not bool(torch.isfinite(decoded).all().item()):
            raise TrainingSamplingError("VAE produced nonfinite training samples")
        images = (
            decoded.float()
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0.0, 255.0)
            .to(device="cpu", dtype=torch.uint8)
        )
        return images

    @staticmethod
    def _wandb_caption(
        item: TrainingSampleItem,
        *,
        pair: _PromptPair,
        shared_seed: int,
    ) -> str:
        return (
            f"variant={item.variant}\n"
            f"A_sample_id={pair.a.sample_id}\n"
            f"B_sample_id={pair.b.sample_id}\n"
            f"shared_seed={shared_seed}\n"
            f"main_source={item.main_source}\n"
            f"condition_sources={','.join(item.condition_sources) or '<null>'}\n"
            f"body={item.caption.body}\n"
            f"condition={item.caption.condition_text or '<null>'}"
        )

    def sample(
        self,
        update: int,
        measurements: tuple[RuntimeMeasurement, ...],
    ) -> TrainingSampleResult:
        if not self.due(update):
            raise TrainingSamplingError(
                "training sampler was called for a non-due update"
            )
        candidates = self._candidates(measurements)
        selector = random.Random(
            f"{self.config.run.seed}\0training-sample-pair\0{update}"
        )
        pair = _select_prompt_pair(candidates, selector)
        shared_seed = selector.randrange(2**63)
        padding_token_id = getattr(self.qwen.tokenizer, "pad_token_id", None)
        if type(padding_token_id) is not int or padding_token_id < 0:
            raise TrainingSamplingError(
                "Qwen tokenizer padding identity is unavailable"
            )
        framing = FramingContract(
            EXPECTED_PREFIX_TOKENS,
            EXPECTED_SUFFIX_TOKENS,
            padding_token_id,
        )
        items = _build_variant_items(
            pair,
            tokenizer=self.qwen.tokenizer,
            framing=framing,
            resolution=self.config.stage.resolution,
        )

        step_root = self.output_root / f"step-{update}"
        step_root.mkdir(parents=True, exist_ok=False)
        was_training = self.composite.training
        self.composite.eval()
        try:
            with torch.inference_mode():
                images = self._generate_batch(
                    items,
                    shared_seed=shared_seed,
                    framing=framing,
                )
                paths = tuple(
                    step_root / f"{item.ordinal + 1:02d}-{item.variant}.png"
                    for item in items
                )
                for item, image, path in zip(items, images, paths, strict=True):
                    array = image.permute(1, 2, 0).contiguous().numpy()
                    Image.fromarray(array).save(path)
        finally:
            self.composite.train(was_training)
            torch.cuda.empty_cache()

        if len(paths) != _VARIANT_COUNT or not all(path.is_file() for path in paths):
            raise TrainingSamplingError("saved sample files are incomplete")
        wandb_captions = tuple(
            self._wandb_caption(item, pair=pair, shared_seed=shared_seed)
            for item in items
        )
        records = [
            _variant_metadata(
                item,
                path,
                pair=pair,
                shared_seed=shared_seed,
                update=update,
                repository_root=self.repository_root,
            )
            for item, path in zip(items, paths, strict=True)
        ]
        metadata = {
            "schema_version": 2,
            "update": update,
            "profile": self.config.sampling.profile,
            "A_sample_id": pair.a.sample_id,
            "B_sample_id": pair.b.sample_id,
            "shared_seed": shared_seed,
            "state_count": _VARIANT_COUNT,
            "cfg_branch_count": _CFG_BRANCH_COUNT,
            "initial_noise": "single_base_noise_repeated_12",
            "cfg_coordinate_sharing": True,
            "output_size": {
                "height": self.config.stage.resolution,
                "width": self.config.stage.resolution,
            },
            "condition_sources": {
                "A": _prompt_metadata(pair.a),
                "B": _prompt_metadata(pair.b),
            },
            "records": records,
        }
        metadata_path = step_root / "metadata.json"
        temporary = step_root / f".metadata.{update}.tmp"
        temporary.write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(metadata_path)
        if not metadata_path.is_file():
            raise TrainingSamplingError("training sample metadata was not committed")
        print(
            f"[sample] update={update} saved {_VARIANT_COUNT} variants: {step_root}",
            flush=True,
        )
        return TrainingSampleResult(update, paths, wandb_captions)


__all__ = [
    "TrainingSampleResult",
    "TrainingSampler",
    "TrainingSamplingError",
]
