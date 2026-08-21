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

from sakuramoon.conditioning.condition_tokens import ConditionTokenOutput
from sakuramoon.conditioning.rope import full_canvas_crop_coordinates
from sakuramoon.conditioning.text_mixer import TextConditioningOutput
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
from sakuramoon.train.condition_diagnostics import (
    FixedConditionPair,
    TrainingSamplingError,
    condition_representation_diagnostics,
    global_path_diagnostics,
    load_fixed_condition_pairs,
)
from sakuramoon.train.condition_diagnostics import (
    metric_float as _metric_float,
)
from sakuramoon.train.condition_diagnostics import (
    tensor_rms as _tensor_rms,
)
from sakuramoon.train.fixed_sample_prompts import (
    FIXED_NEUTRAL_PROMPTS,
    FIXED_NEUTRAL_SHARED_SEED,
    fixed_neutral_provenance,
)
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
    "A-zoom-mild",
    "A-zoom-strong",
    "A-shift-zoom-mild",
    "A-shift-zoom-strong",
]
GeometryKind = Literal["canonical", "zoom", "shift_zoom"]
CoordinateType = Literal[
    "canonical_full_canvas",
    "zoom_full_canvas_crop",
    "shift_zoom_full_canvas_crop",
]

_VARIANT_DEFINITIONS: tuple[
    tuple[
        VariantName,
        PromptLabel,
        tuple[PromptLabel, ...],
        GeometryKind,
        float,
    ],
    ...,
] = (
    ("A-base", "A", ("A",), "canonical", 1.0),
    ("B-base", "B", ("B",), "canonical", 1.0),
    ("A-with-B", "A", ("B",), "canonical", 1.0),
    ("B-with-A", "B", ("A",), "canonical", 1.0),
    ("A-null", "A", (), "canonical", 1.0),
    ("B-null", "B", (), "canonical", 1.0),
    ("A-with-BA", "A", ("B", "A"), "canonical", 1.0),
    ("B-with-BA", "B", ("B", "A"), "canonical", 1.0),
    ("A-zoom-mild", "A", ("A",), "zoom", 1.10),
    ("A-zoom-strong", "A", ("A",), "zoom", 1.50),
    ("A-shift-zoom-mild", "A", ("A",), "shift_zoom", 1.10),
    ("A-shift-zoom-strong", "A", ("A",), "shift_zoom", 1.50),
)
_VARIANT_NAMES = tuple(definition[0] for definition in _VARIANT_DEFINITIONS)
_VARIANT_COUNT = 12
_CFG_BRANCH_COUNT = 24
_GEOMETRY_PROTOCOL = "tiered-zoom-v1"
_TOTAL_VARIANT_COUNT = 24
_TOTAL_CFG_BRANCH_COUNT = 48
_DIAGNOSTIC_ITEM_INDICES = (0, 2, 4)
_DIAGNOSTIC_TIMESTEPS = (0.2, 0.5, 0.8)


def _condition_representation_diagnostics(
    tokens: torch.Tensor,
) -> dict[str, float]:
    return condition_representation_diagnostics(
        tokens,
        expected_batch=_CFG_BRANCH_COUNT,
        a_index=0,
        b_index=1,
        null_index=4,
    )


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
    diagnostics: dict[str, float]


def _unconditional_plan() -> CaptionPlan:
    return CaptionPlan(
        tags=(),
        condition=None,
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=True,
        dropout_hits=_ALL_DROPPED,
    )


def _fixed_neutral_prompt_pair(
    *,
    tokenizer: TokenEncoder,
    framing: FramingContract,
    resolution: int,
) -> _PromptPair:
    """Build the immutable Hiten/WLOP prompt pair for the fixed gallery."""
    prompts: list[_PostDropoutPrompt] = []
    for record in FIXED_NEUTRAL_PROMPTS:
        plan = record.caption_plan()
        caption = serialize_caption(plan, tokenizer, framing)
        if caption.plan.condition is None:
            raise TrainingSamplingError(
                f"fixed neutral prompt {record.sample_id} lost its condition"
            )
        prompts.append(
            _PostDropoutPrompt(
                sample_id=record.sample_id,
                caption=caption,
                # The serializer may drop complete trailing body tags to honor
                # the token budget; retain the governed serialized plan.
                plan=caption.plan,
                observed_height=resolution,
                observed_width=resolution,
            )
        )
    if len(prompts) != 2:
        raise TrainingSamplingError("fixed neutral prompt cohort is incomplete")
    return _PromptPair(prompts[0], prompts[1])


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
    requested_zoom: float,
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
        if requested_zoom != 1.0:
            raise TrainingSamplingError("canonical geometry must use unit zoom")
        return (
            1.0,
            (resolution, resolution),
            (0, 0, resolution, resolution),
            "canonical_full_canvas",
        )
    if (
        type(requested_zoom) is not float
        or not math.isfinite(requested_zoom)
        or requested_zoom <= 1.0
    ):
        raise TrainingSamplingError(
            "spatial geometry zoom must be finite and above one"
        )
    virtual = math.floor(resolution * requested_zoom + 0.5)
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
        virtual / resolution,
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
    for ordinal, (
        variant,
        main_label,
        condition_labels,
        geometry,
        requested_zoom,
    ) in enumerate(_VARIANT_DEFINITIONS):
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
            resolution, geometry, requested_zoom
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


def _require_image_batch(
    items: tuple[TrainingSampleItem, ...],
) -> tuple[int, int]:
    if type(items) is not tuple or not items:
        raise TrainingSamplingError("training sampler requires a nonempty image batch")
    height, width = items[0].height, items[0].width
    if any((item.height, item.width) != (height, width) for item in items):
        raise TrainingSamplingError("training sample variants have mixed output sizes")
    return height, width


def _require_variant_batch(
    items: tuple[TrainingSampleItem, ...],
) -> tuple[int, int]:
    if len(items) != _VARIANT_COUNT:
        raise TrainingSamplingError("training sampler requires exactly 12 variants")
    if tuple(item.variant for item in items) != _VARIANT_NAMES:
        raise TrainingSamplingError(
            "training sample variants are incomplete or reordered"
        )
    return _require_image_batch(items)


def _coordinate_maps(
    items: tuple[TrainingSampleItem, ...],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    _require_image_batch(items)
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
    count: int = _VARIANT_COUNT,
    device: torch.device,
) -> torch.Tensor:
    if type(shared_seed) is not int or not 0 <= shared_seed < 2**63:
        raise TrainingSamplingError("shared sample seed must be a 63-bit integer")
    if height <= 0 or width <= 0 or height % 16 or width % 16:
        raise TrainingSamplingError("shared noise canvas is invalid")
    if type(count) is not int or count <= 0:
        raise TrainingSamplingError("shared noise batch count is invalid")
    generator = torch.Generator(device=device)
    generator.manual_seed(shared_seed)
    base_noise = torch.randn(
        (1, 128, height // 16, width // 16),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )
    return base_noise.repeat(count, 1, 1, 1)


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
    cohort: str = "dynamic",
    fixed_pair_label: str | None = None,
) -> dict[str, object]:
    main = _source_for_label(pair, item.main_source)
    condition_sources = tuple(
        _source_for_label(pair, label) for label in item.condition_sources
    )
    left, top, right, bottom = item.crop_box
    metadata = {
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
    metadata["cohort"] = cohort
    if fixed_pair_label is not None:
        metadata["fixed_pair"] = fixed_pair_label
    return metadata


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
        image_count = config.sampling.training.image_count
        if image_count not in (_VARIANT_COUNT, _TOTAL_VARIANT_COUNT):
            raise ValueError(
                "training sampling image_count must be "
                f"{_VARIANT_COUNT} (single dynamic cohort) or "
                f"{_TOTAL_VARIANT_COUNT} (dynamic plus fixed-neutral cohorts)"
            )
        self.fixed_condition_pairs = (
            load_fixed_condition_pairs(
                repository_root / config.evaluation.prompt_path
            )
            if config.evaluation.enabled
            else ()
        )
        self.output_root = (
            repository_directory(repository_root, config.paths.checkpoint_dir)
            / config.sampling.training.output_subdir
        )

    def set_growth_alpha(self, value: float) -> None:
        """Select the canonical alpha for the update being sampled."""

        if type(value) is not float or not 0.0 <= value <= 1.0:
            raise ValueError("growth_alpha must be a float in [0,1]")
        self.growth_alpha = value

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

    def _condition_diagnostics(
        self,
        inputs: TrainableCompositeInputs,
        conditioning: tuple[TextConditioningOutput, ConditionTokenOutput],
        noise: torch.Tensor,
    ) -> dict[str, float]:
        text, condition = conditioning
        diagnostics = _condition_representation_diagnostics(condition.tokens)
        active_text = text.tokens[:_VARIANT_COUNT][text.mask[:_VARIANT_COUNT]]
        diagnostics["text_token_rms"] = _metric_float(
            "text_token_rms", _tensor_rms(active_text)
        )

        dit = self.composite.dit
        conditional_text = text.tokens[:_VARIANT_COUNT]
        conditional_condition = condition.tokens[:_VARIANT_COUNT]
        text_after = dit.modality(conditional_text, "text")
        condition_after = dit.modality(conditional_condition, "condition")
        diagnostics["text_token_after_modality_rms"] = _metric_float(
            "text_token_after_modality_rms",
            _tensor_rms(text_after[text.mask[:_VARIANT_COUNT]]),
        )
        diagnostics["condition_token_after_modality_rms"] = _metric_float(
            "condition_token_after_modality_rms",
            0.5 * (
                _tensor_rms(condition_after[0])
                + _tensor_rms(condition_after[1])
            ),
        )
        diagnostics["null_condition_after_modality_rms"] = _metric_float(
            "null_condition_after_modality_rms",
            _tensor_rms(condition_after[4]),
        )

        global_output = dit.conditioner(
            torch.full(
                (_VARIANT_COUNT,),
                0.5,
                dtype=torch.float32,
                device=self.device,
            ),
            inputs.size_scale[:_VARIANT_COUNT],
            inputs.aspect[:_VARIANT_COUNT],
            conditional_condition,
            condition.active_mask[:_VARIANT_COUNT],
            dit.active_slot_ids,
        )
        diagnostics.update(
            global_path_diagnostics(
                global_output,
                dit.conditioner.condition_global_projection.weight,
                a_index=0,
                b_index=1,
                null_index=4,
            )
        )

        latent_batch = noise.to(dit.input_projection.weight.dtype)
        batch, channels, height, width = latent_batch.shape
        projected_image = dit.input_projection(
            latent_batch.permute(0, 2, 3, 1).reshape(
                batch * height * width, channels
            )
        ).reshape(batch, height * width, dit.hidden_size)
        diagnostics["image_token_rms"] = _metric_float(
            "image_token_rms", _tensor_rms(projected_image)
        )
        diagnostics["image_token_after_modality_rms"] = _metric_float(
            "image_token_after_modality_rms",
            _tensor_rms(dit.modality(projected_image, "image")),
        )

        indices = torch.tensor(
            _DIAGNOSTIC_ITEM_INDICES,
            dtype=torch.long,
            device=self.device,
        )
        diagnostic_text = dataclasses.replace(
            text,
            tokens=text.tokens.index_select(0, indices),
            mask=text.mask.index_select(0, indices),
            layer_weights=text.layer_weights.index_select(0, indices),
        )
        diagnostic_condition = dataclasses.replace(
            condition,
            tokens=condition.tokens.index_select(0, indices),
            mask=condition.mask.index_select(0, indices),
            active_mask=condition.active_mask.index_select(0, indices),
        )
        use_null = inputs.use_null_condition.index_select(0, indices)
        diagnostic_inputs = dataclasses.replace(
            inputs,
            qwen_states=inputs.qwen_states.index_select(0, indices),
            main_token_indices=inputs.main_token_indices.index_select(0, indices),
            main_mask=inputs.main_mask.index_select(0, indices),
            main_token_lengths=tuple(
                inputs.main_token_lengths[index]
                for index in _DIAGNOSTIC_ITEM_INDICES
            ),
            condition_token_indices=inputs.condition_token_indices.index_select(
                0, indices
            ),
            condition_mask=inputs.condition_mask.index_select(0, indices),
            use_null_condition=use_null,
            active_condition_sample_indices=torch.nonzero(
                ~use_null, as_tuple=False
            ).flatten(),
            latents=(latent_batch[0],) * 3,
            image_coordinates=tuple(
                inputs.image_coordinates[index]
                for index in _DIAGNOSTIC_ITEM_INDICES
            ),
            size_scale=inputs.size_scale.index_select(0, indices),
            aspect=inputs.aspect.index_select(0, indices),
        )
        for timestep_value in _DIAGNOSTIC_TIMESTEPS:
            timestep = torch.full(
                (3,),
                timestep_value,
                dtype=torch.float32,
                device=self.device,
            )
            predictions = torch.stack(
                self.composite.forward_dit(
                    dataclasses.replace(diagnostic_inputs, timestep=timestep),
                    (diagnostic_text, diagnostic_condition),
                )
            )
            denominator = _tensor_rms(predictions[0]).clamp_min(1e-12)
            suffix = f"t{round(timestep_value * 10):02d}"
            diagnostics[f"dit_swap_relative_rms_{suffix}"] = _metric_float(
                f"dit_swap_relative_rms_{suffix}",
                _tensor_rms(predictions[0] - predictions[1]) / denominator,
            )
            diagnostics[f"dit_null_relative_rms_{suffix}"] = _metric_float(
                f"dit_null_relative_rms_{suffix}",
                _tensor_rms(predictions[0] - predictions[2]) / denominator,
            )
        return diagnostics

    def _fixed_prompt_pair(
        self,
        pair: FixedConditionPair,
        *,
        framing: FramingContract,
    ) -> _PromptPair:
        prompts: list[_PostDropoutPrompt] = []
        for case in (pair.a, pair.b):
            plan = case.caption_plan
            if plan is None:
                raise TrainingSamplingError(
                    f"fixed condition pair {pair.label} lacks a caption plan"
                )
            caption = serialize_caption(plan, self.qwen.tokenizer, framing)
            if caption.plan != plan:
                raise TrainingSamplingError(
                    f"fixed condition pair {pair.label} cannot preserve its plan"
                )
            prompts.append(
                _PostDropoutPrompt(
                    sample_id=case.prompt_id,
                    caption=caption,
                    plan=plan,
                    observed_height=case.height,
                    observed_width=case.width,
                )
            )
        return _PromptPair(prompts[0], prompts[1])

    def _fixed_condition_diagnostics(
        self,
        *,
        framing: FramingContract,
    ) -> dict[str, float]:
        if not self.fixed_condition_pairs:
            return {}
        if len(self.fixed_condition_pairs) != 4:
            raise TrainingSamplingError(
                "fixed condition diagnostics require exactly four pairs"
            )
        resolution = self.config.stage.resolution
        items: list[TrainingSampleItem] = []
        noise_rows: list[torch.Tensor] = []
        for fixed_pair in self.fixed_condition_pairs:
            pair = self._fixed_prompt_pair(fixed_pair, framing=framing)
            variants = _build_variant_items(
                pair,
                tokenizer=self.qwen.tokenizer,
                framing=framing,
                resolution=resolution,
            )
            items.extend((variants[0], variants[2], variants[4]))
            generator = torch.Generator(device=self.device)
            generator.manual_seed(fixed_pair.a.seed % (2**63))
            base_noise = torch.randn(
                (128, resolution // 16, resolution // 16),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )
            noise_rows.extend((base_noise, base_noise, base_noise))

        captions = tuple(item.caption for item in items)
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
            raise TrainingSamplingError("Qwen output lacks fixed-cohort hidden states")
        batch = len(items)
        coordinates = tuple(
            full_canvas_crop_coordinates(
                resolution // 16,
                resolution // 16,
                full_height=resolution,
                full_width=resolution,
                crop_box=(0, 0, resolution, resolution),
                device=self.device,
            )
            for _ in items
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
            latents=tuple(
                noise.to(torch.bfloat16) for noise in noise_rows
            ),
            image_coordinates=coordinates,
            timestep=torch.zeros(batch, dtype=torch.float32, device=self.device),
            size_scale=torch.full(
                (batch,),
                0.5 * math.log2((resolution * resolution) / float(512 * 512)),
                dtype=torch.float32,
                device=self.device,
            ),
            aspect=torch.zeros(batch, dtype=torch.float32, device=self.device),
            growth_alpha=self.growth_alpha,
        )
        text, condition = self.composite.forward_conditioning(inputs)
        diagnostics: dict[str, float] = {}
        representation: list[dict[str, float]] = []
        for pair_index in range(4):
            offset = 3 * pair_index
            representation.append(
                condition_representation_diagnostics(
                    condition.tokens,
                    expected_batch=batch,
                    a_index=offset,
                    b_index=offset + 1,
                    null_index=offset + 2,
                )
            )
        for key in (
            "condition_A_B_cosine",
            "condition_A_B_delta_rms",
            "condition_A_null_delta_rms",
        ):
            diagnostics[f"fixed_{key}"] = sum(item[key] for item in representation) / 4

        dit = self.composite.dit
        global_output = dit.conditioner(
            torch.full((batch,), 0.5, dtype=torch.float32, device=self.device),
            inputs.size_scale,
            inputs.aspect,
            condition.tokens,
            condition.active_mask,
            dit.active_slot_ids,
        )
        global_values = [
            global_path_diagnostics(
                global_output,
                dit.conditioner.condition_global_projection.weight,
                a_index=3 * pair_index,
                b_index=3 * pair_index + 1,
                null_index=3 * pair_index + 2,
            )
            for pair_index in range(4)
        ]
        for key in (
            "global_base_rms",
            "global_condition_active_rms",
            "global_condition_to_base_ratio",
            "global_total_rms",
            "global_A_B_delta_rms",
        ):
            diagnostics[f"fixed_{key}"] = sum(item[key] for item in global_values) / 4
        diagnostics["fixed_condition_global_projection_weight_rms"] = global_values[
            0
        ]["condition_global_projection_weight_rms"]
        if all("global_A_B_cosine" in item for item in global_values):
            diagnostics["fixed_global_A_B_cosine"] = sum(
                item["global_A_B_cosine"] for item in global_values
            ) / 4

        group_indices = {
            "all": range(4),
            "style": range(2),
            "identity": range(2, 4),
        }
        for timestep_value in _DIAGNOSTIC_TIMESTEPS:
            timestep = torch.full(
                (batch,),
                timestep_value,
                dtype=torch.float32,
                device=self.device,
            )
            predictions = torch.stack(
                self.composite.forward_dit(
                    dataclasses.replace(inputs, timestep=timestep),
                    (text, condition),
                )
            )
            swap_values: list[float] = []
            null_values: list[float] = []
            for pair_index in range(4):
                offset = 3 * pair_index
                denominator = _tensor_rms(predictions[offset]).clamp_min(1e-12)
                swap_values.append(
                    _metric_float(
                        "fixed_dit_swap_relative_rms",
                        _tensor_rms(
                            predictions[offset] - predictions[offset + 1]
                        )
                        / denominator,
                    )
                )
                null_values.append(
                    _metric_float(
                        "fixed_dit_null_relative_rms",
                        _tensor_rms(
                            predictions[offset] - predictions[offset + 2]
                        )
                        / denominator,
                    )
                )
            suffix = f"t{round(timestep_value * 10):02d}"
            for group, pair_indices in group_indices.items():
                selected = tuple(pair_indices)
                diagnostics[f"fixed_{group}_dit_swap_relative_rms_{suffix}"] = (
                    sum(swap_values[index] for index in selected) / len(selected)
                )
                diagnostics[f"fixed_{group}_dit_null_relative_rms_{suffix}"] = (
                    sum(null_values[index] for index in selected) / len(selected)
                )
        return diagnostics

    def _generate_batch(
        self,
        items: tuple[TrainingSampleItem, ...],
        *,
        shared_seed: int,
        framing: FramingContract,
        include_fixed_diagnostics: bool = True,
    ) -> tuple[torch.Tensor, dict[str, float]]:
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
        diagnostics = self._condition_diagnostics(inputs, conditioning, noise)
        if include_fixed_diagnostics:
            diagnostics.update(self._fixed_condition_diagnostics(framing=framing))

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
        return images, diagnostics

    @staticmethod
    def _wandb_caption(
        item: TrainingSampleItem,
        *,
        pair: _PromptPair,
        shared_seed: int,
        cohort: str,
    ) -> str:
        return (
            f"cohort={cohort}\n"
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
        two_cohorts = (
            self.config.sampling.training.image_count == _TOTAL_VARIANT_COUNT
        )
        items = _build_variant_items(
            pair,
            tokenizer=self.qwen.tokenizer,
            framing=framing,
            resolution=self.config.stage.resolution,
        )
        fixed_pair: _PromptPair | None = None
        fixed_items: tuple[TrainingSampleItem, ...] = ()
        if two_cohorts:
            fixed_pair = _fixed_neutral_prompt_pair(
                tokenizer=self.qwen.tokenizer,
                framing=framing,
                resolution=self.config.stage.resolution,
            )
            fixed_items = _build_variant_items(
                fixed_pair,
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
                images, diagnostics = self._generate_batch(
                    items,
                    shared_seed=shared_seed,
                    framing=framing,
                )
                fixed_images: tuple[torch.Tensor, ...] = ()
                fixed_diagnostics: dict[str, float] = {}
                if two_cohorts:
                    fixed_images, fixed_diagnostics = self._generate_batch(
                        fixed_items,
                        shared_seed=FIXED_NEUTRAL_SHARED_SEED,
                        framing=framing,
                        include_fixed_diagnostics=False,
                    )
                dynamic_paths = tuple(
                    step_root / f"{item.ordinal + 1:02d}-{item.variant}.png"
                    for item in items
                )
                fixed_paths = tuple(
                    step_root
                    / f"{_VARIANT_COUNT + item.ordinal + 1:02d}-fixed-neutral-{item.variant}.png"
                    for item in fixed_items
                )
                for item, image, path in zip(
                    items, images, dynamic_paths, strict=True
                ):
                    array = image.permute(1, 2, 0).contiguous().numpy()
                    Image.fromarray(array).save(path)
                for item, image, path in zip(
                    fixed_items, fixed_images, fixed_paths, strict=True
                ):
                    array = image.permute(1, 2, 0).contiguous().numpy()
                    Image.fromarray(array).save(path)
        finally:
            self.composite.train(was_training)
            torch.cuda.empty_cache()

        paths = dynamic_paths + fixed_paths
        expected_variant_count = (
            _TOTAL_VARIANT_COUNT if two_cohorts else _VARIANT_COUNT
        )
        if len(paths) != expected_variant_count or not all(
            path.is_file() for path in paths
        ):
            raise TrainingSamplingError("saved sample files are incomplete")
        wandb_captions = tuple(
            self._wandb_caption(
                item,
                pair=pair,
                shared_seed=shared_seed,
                cohort="dynamic",
            )
            for item in items
        )
        if two_cohorts:
            assert fixed_pair is not None
            wandb_captions = wandb_captions + tuple(
                self._wandb_caption(
                    item,
                    pair=fixed_pair,
                    shared_seed=FIXED_NEUTRAL_SHARED_SEED,
                    cohort="fixed-neutral",
                )
                for item in fixed_items
            )
        records = [
            _variant_metadata(
                item,
                path,
                pair=pair,
                shared_seed=shared_seed,
                update=update,
                repository_root=self.repository_root,
                cohort="dynamic",
            )
            for item, path in zip(items, dynamic_paths, strict=True)
        ]
        if two_cohorts:
            assert fixed_pair is not None
            records.extend(
                _variant_metadata(
                    item,
                    path,
                    pair=fixed_pair,
                    shared_seed=FIXED_NEUTRAL_SHARED_SEED,
                    update=update,
                    repository_root=self.repository_root,
                    cohort="fixed-neutral",
                )
                for item, path in zip(fixed_items, fixed_paths, strict=True)
            )
        metadata = {
            "schema_version": 5,
            "geometry_protocol": _GEOMETRY_PROTOCOL,
            "update": update,
            "profile": self.config.sampling.profile,
            "A_sample_id": pair.a.sample_id,
            "B_sample_id": pair.b.sample_id,
            "shared_seed": shared_seed,
            "state_count": (
                _TOTAL_VARIANT_COUNT if two_cohorts else _VARIANT_COUNT
            ),
            "cfg_branch_count": (
                _TOTAL_CFG_BRANCH_COUNT if two_cohorts else _CFG_BRANCH_COUNT
            ),
            "cohort_state_count": _VARIANT_COUNT,
            "cohort_cfg_branch_count": _CFG_BRANCH_COUNT,
            "cohorts": (
                ["dynamic", "fixed-neutral"] if two_cohorts else ["dynamic"]
            ),
            "condition_diagnostics": diagnostics,
            **({"fixed_neutral_condition_diagnostics": fixed_diagnostics}
               if two_cohorts else {}),
            "initial_noise": (
                "one_base_noise_per_cohort_repeated_12"
                if two_cohorts
                else "single_base_noise_repeated_12"
            ),
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
        if two_cohorts:
            assert fixed_pair is not None
            metadata["fixed_neutral"] = {
                "shared_seed": FIXED_NEUTRAL_SHARED_SEED,
                "provenance": fixed_neutral_provenance(),
                "condition_sources": {
                    "A": _prompt_metadata(fixed_pair.a),
                    "B": _prompt_metadata(fixed_pair.b),
                },
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
            f"[sample] update={update} saved "
            f"{_TOTAL_VARIANT_COUNT if two_cohorts else _VARIANT_COUNT} variants "
            f"across {2 if two_cohorts else 1} cohort"
            f"{'s' if two_cohorts else ''}: {step_root}",
            flush=True,
        )
        return TrainingSampleResult(update, paths, wandb_captions, diagnostics)


__all__ = [
    "TrainingSampleResult",
    "TrainingSampler",
    "TrainingSamplingError",
]
