"""Governed validation-shard prompts for checkpoint evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.buckets import (
    BucketRejection,
    BucketShape,
    assign_bucket,
    generate_base_buckets,
    scale_buckets,
)
from sakuramoon.data.caption import (
    CaptionDropoutHits,
    CaptionError,
    CaptionFields,
    CaptionPlan,
)
from sakuramoon.data.validation import ValidationPromptSample, ValidationSelection
from sakuramoon.eval.spec import (
    PromptCase,
    PromptManifest,
    caption_plan_prompt_text,
)


class ValidationPromptPlanError(ValueError):
    """A selected validation sample cannot enter the governed evaluator plan."""

    def __init__(self, code: str, subject: str) -> None:
        self.code = code
        self.subject = subject
        super().__init__(f"{code}:{subject}")


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


def _caption_plan(
    fields: CaptionFields, prompt: str | None, *, subject: str
) -> CaptionPlan:
    try:
        available_nl = fields.nl.available()
    except CaptionError:
        raise ValidationPromptPlanError(
            "VALIDATION_CAPTION_FIELDS_INVALID", subject
        ) from None
    if prompt is None:
        selected_nl, nl_text = None, None
    else:
        matches = tuple(item for item in available_nl if item[1] == prompt)
        if not matches:
            raise ValidationPromptPlanError(
                "VALIDATION_CAPTION_NL_IDENTITY_MISMATCH", subject
            )
        selected_nl, nl_text = matches[0]
    return CaptionPlan(
        nsfw=fields.nsfw,
        character=fields.character,
        copyright=fields.copyright,
        general=fields.general,
        artists=fields.artists,
        nl_text=nl_text,
        selected_nl=selected_nl,
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


@dataclass(frozen=True, slots=True)
class ValidationPromptPlan:
    selection: ValidationSelection
    prompts: PromptManifest
    batchable_cases: int
    bucket_shapes: tuple[BucketShape, ...]

    def __post_init__(self) -> None:
        if (
            type(self.batchable_cases) is not int
            or self.batchable_cases <= 0
            or self.batchable_cases > len(self.prompts.cases)
            or not self.bucket_shapes
        ):
            raise ValueError("validation prompt plan is invalid")


def build_validation_prompt_plan(
    config: RuntimeConfig,
    selection: ValidationSelection,
    samples: tuple[ValidationPromptSample, ...],
) -> ValidationPromptPlan:
    """Assign decoded source dimensions to stage buckets and form full batches."""

    evaluation = config.evaluation
    if getattr(evaluation, "enabled", False) is not True:
        raise ValidationPromptPlanError("EVALUATION_DISABLED", "evaluation.enabled")
    batch_size = getattr(evaluation, "batch_size", None)
    if type(batch_size) is not int or batch_size <= 0:
        raise ValidationPromptPlanError(
            "EVALUATION_BATCH_SIZE_INVALID", "evaluation.batch_size"
        )
    if not samples:
        raise ValidationPromptPlanError(
            "VALIDATION_PROMPTS_EMPTY", selection.selection_id
        )

    base_buckets = generate_base_buckets(config.data.buckets)
    buckets = scale_buckets(base_buckets, config.stage.resolution)
    allowed = frozenset(buckets)
    maximum_area = max(shape.area for shape in buckets)
    grouped: dict[BucketShape, list[tuple[ValidationPromptSample, PromptCase]]] = {
        shape: [] for shape in buckets
    }
    seen_prompt_ids: set[str] = set()
    for sample in samples:
        if sample.prompt_id in seen_prompt_ids:
            raise ValidationPromptPlanError(
                "VALIDATION_PROMPT_ID_DUPLICATE", sample.prompt_id
            )
        seen_prompt_ids.add(sample.prompt_id)
        if sample.caption_fields is None:
            raise ValidationPromptPlanError(
                "VALIDATION_CAPTION_FIELDS_REQUIRED", sample.prompt_id
            )
        caption_plan = _caption_plan(
            sample.caption_fields,
            sample.prompt,
            subject=sample.prompt_id,
        )
        if not any(
            (
                caption_plan.nsfw,
                caption_plan.character,
                caption_plan.copyright,
                caption_plan.general,
                caption_plan.artists,
            )
        ) and caption_plan.nl_text is None:
            raise ValidationPromptPlanError(
                "VALIDATION_CAPTION_EMPTY", sample.prompt_id
            )
        assignment = assign_bucket(
            sample.width,
            sample.height,
            buckets,
            min_crop_retention=config.data.image.min_crop_retention,
        )
        if isinstance(assignment, BucketRejection):
            code = (
                "VALIDATION_PROMPT_NO_UPSCALE"
                if assignment.reason == "no_upscale"
                else "VALIDATION_PROMPT_RETENTION_REJECTED"
            )
            raise ValidationPromptPlanError(code, sample.prompt_id)
        canvas = assignment.bucket
        if canvas not in allowed or canvas.area > maximum_area:
            raise ValidationPromptPlanError(
                "VALIDATION_PROMPT_CANVAS_TOO_LARGE", sample.prompt_id
            )
        grouped[canvas].append(
            (
                sample,
                PromptCase(
                    prompt_id=sample.prompt_id,
                    prompt=caption_plan_prompt_text(caption_plan),
                    conditions=(),
                    seed=sample.seed,
                    height=canvas.height,
                    width=canvas.width,
                    caption_plan=caption_plan,
                ),
            )
        )

    complete_batches: list[PromptCase] = []
    remainders: list[PromptCase] = []
    for shape in buckets:
        members = sorted(
            grouped[shape],
            key=lambda item: (
                item[0].source_shard,
                item[0].member_key,
                item[0].prompt_id,
            ),
        )
        full_count = len(members) // batch_size * batch_size
        complete_batches.extend(case for _sample, case in members[:full_count])
        remainders.extend(case for _sample, case in members[full_count:])
    if not complete_batches:
        raise ValidationPromptPlanError(
            "VALIDATION_FULL_BATCH_UNAVAILABLE", f"batch_size={batch_size}"
        )
    prompts = PromptManifest((*complete_batches, *remainders))
    return ValidationPromptPlan(
        selection=selection,
        prompts=prompts,
        batchable_cases=len(complete_batches),
        bucket_shapes=buckets,
    )


__all__ = [
    "ValidationPromptPlan",
    "ValidationPromptPlanError",
    "build_validation_prompt_plan",
]
