from __future__ import annotations

# pyright: reportPrivateUsage=false
import dataclasses
from types import SimpleNamespace
from typing import cast

import pytest

from sakuramoon.config.schema import (
    DataBucketsConfig,
    EvaluationEnabledConfig,
    RuntimeConfig,
)
from sakuramoon.data.buckets import BucketAssignment, BucketShape
from sakuramoon.data.caption import CaptionFields, NlCandidates, Tag
from sakuramoon.data.manifest import (
    DatasetManifest,
    DatasetSourceIdentity,
    ShardRecord,
)
from sakuramoon.data.validation import (
    VALIDATION_SHARD_PATHS,
    ValidationPromptSample,
    ValidationSelection,
    select_validation_shards,
)
from sakuramoon.eval import generate as generate_module
from sakuramoon.eval import validation as validation_module
from sakuramoon.eval.generate import GenerationContractError
from sakuramoon.eval.validation import (
    ValidationPromptPlanError,
    build_validation_prompt_plan,
)


def _selection() -> ValidationSelection:
    paths = (*VALIDATION_SHARD_PATHS, "training-shard.tar")
    manifest = DatasetManifest.from_shards(
        DatasetSourceIdentity(
            repo_id="leafmoone/webdataset_danbooru", revision="master"
        ),
        tuple(
            ShardRecord(
                path=path,
                bytes=index,
                upstream_sha256=f"{index:064x}",
            )
            for index, path in enumerate(paths, start=1)
        ),
    )
    return select_validation_shards(manifest)


def _config() -> RuntimeConfig:
    return cast(
        RuntimeConfig,
        SimpleNamespace(
            evaluation=SimpleNamespace(enabled=True, batch_size=2),
            data=SimpleNamespace(
                buckets=DataBucketsConfig(
                    base_area_px=262144,
                    quantum_px=32,
                    min_short_edge_px=256,
                    max_aspect_ratio=4.0,
                    shape_count=17,
                    transpose_closed=True,
                ),
                image=SimpleNamespace(min_crop_retention=0.8),
            ),
            stage=SimpleNamespace(resolution=256),
        ),
    )


def _sample(
    index: int,
    selection: ValidationSelection,
    *,
    height: int,
    width: int,
    tags_only: bool = False,
) -> ValidationPromptSample:
    caption_fields = CaptionFields(
        nsfw=((Tag("safe", "safe"),) if tags_only else ()),
        character=(),
        copyright=(),
        general=(
            (Tag("1girl", "1girl"), Tag("blue_hair", "blue_hair"))
            if tags_only
            else ()
        ),
        artists=((Tag("artist_name", "artist_name"),) if tags_only else ()),
        candidate_tags=frozenset(),
        nl=NlCandidates(
            None,
            None,
            None,
            None if tags_only else f"validation caption {index}",
            None,
        ),
    )
    return ValidationPromptSample(
        prompt_id=f"validation-{index:032x}",
        sample_id=index + 1,
        source_shard=selection.shards[index % 2].path,
        member_key=f"sample-{index:05d}",
        prompt=None if tags_only else f"validation caption {index}",
        seed=index + 44,
        height=height,
        width=width,
        caption_fields=caption_fields,
    )


def test_validation_captions_become_bucketed_nl_only_full_batches(
) -> None:
    config = _config()
    selection = _selection()
    samples = tuple(
        _sample(
            index,
            selection,
            height=513 if index < 10 else 321,
            width=517 if index < 10 else 1281,
        )
        for index in range(20)
    )

    plan = build_validation_prompt_plan(config, selection, samples)

    assert plan.selection == selection
    assert plan.batchable_cases == 20
    assert all(case.conditions == () for case in plan.prompts.cases)
    assert all(case.caption_plan is not None for case in plan.prompts.cases)
    assert all(
        case.caption_plan is not None
        and case.caption_plan.nl_text is not None
        and case.caption_plan.nl_text.startswith("validation caption")
        for case in plan.prompts.cases
    )
    assert all(
        (case.height, case.width) not in {(513, 517), (321, 1281)}
        for case in plan.prompts.cases
    )
    batch_size = cast(EvaluationEnabledConfig, config.evaluation).batch_size
    for start in range(0, plan.batchable_cases, batch_size):
        batch = plan.prompts.cases[start : start + batch_size]
        assert len({(case.height, case.width) for case in batch}) == 1


def test_validation_passes_unrounded_source_dimensions_to_bucket_assignment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    selection = _selection()
    observed: list[tuple[int, int]] = []

    def capture(
        source_width: int,
        source_height: int,
        buckets: tuple[BucketShape, ...],
        *,
        min_crop_retention: float,
    ) -> BucketAssignment:
        observed.append((source_width, source_height))
        assert min_crop_retention == 0.8
        return BucketAssignment(
            source_width=source_width,
            source_height=source_height,
            bucket=buckets[0],
            resized_width=buckets[0].width,
            resized_height=buckets[0].height,
            crop_retention=1.0,
        )

    monkeypatch.setattr(validation_module, "assign_bucket", capture)
    plan = build_validation_prompt_plan(
        config,
        selection,
        tuple(
            _sample(index, selection, height=513, width=517)
            for index in range(2)
        ),
    )

    assert observed == [(517, 513), (517, 513)]
    assert plan.batchable_cases == 2


@pytest.mark.parametrize(
    ("height", "width", "code"),
    [
        (16, 16, "VALIDATION_PROMPT_NO_UPSCALE"),
        (160, 1600, "VALIDATION_PROMPT_RETENTION_REJECTED"),
    ],
)
def test_validation_bucket_rejections_fail_closed(
    height: int, width: int, code: str
) -> None:
    config = _config()
    selection = _selection()

    with pytest.raises(ValidationPromptPlanError) as captured:
        build_validation_prompt_plan(
            config,
            selection,
            (_sample(0, selection, height=height, width=width),),
        )

    assert captured.value.code == code


def test_validation_rejects_canvas_outside_the_stage_bucket_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config()
    selection = _selection()

    def oversized(*_args: object, **_kwargs: object) -> BucketAssignment:
        return BucketAssignment(
            source_width=4096,
            source_height=4096,
            bucket=BucketShape(4096, 4096),
            resized_width=4096,
            resized_height=4096,
            crop_retention=1.0,
        )

    monkeypatch.setattr(validation_module, "assign_bucket", oversized)
    with pytest.raises(ValidationPromptPlanError) as captured:
        build_validation_prompt_plan(
            config,
            selection,
            (_sample(0, selection, height=4096, width=4096),),
        )

    assert captured.value.code == "VALIDATION_PROMPT_CANVAS_TOO_LARGE"


def test_tags_only_metadata_becomes_typed_no_dropout_evaluator_prompts() -> None:
    config = _config()
    selection = _selection()

    plan = build_validation_prompt_plan(
        config,
        selection,
        tuple(
            _sample(
                index,
                selection,
                height=513,
                width=517,
                tags_only=True,
            )
            for index in range(2)
        ),
    )

    assert plan.batchable_cases == 2
    for case in plan.prompts.cases:
        assert case.conditions == ()
        assert case.caption_plan is not None
        assert tuple(tag.text for tag in case.caption_plan.general) == (
            "1girl",
            "blue_hair",
        )
        assert tuple(tag.text for tag in case.caption_plan.artists) == ("artist_name",)
        assert case.caption_plan.nl_text is None
        assert not any(case.caption_plan.dropout_hits.as_mapping().values())
        assert "1girl, blue_hair" in case.prompt
        assert case.prompt.endswith("artist_name")

    conditioned = dataclasses.replace(
        plan.prompts.cases[0], conditions=("ungoverned-condition",)
    )
    with pytest.raises(GenerationContractError, match="not governed"):
        generate_module._conditional_plan(conditioned)


def test_validation_prompt_plan_requires_typed_caption_fields() -> None:
    config = _config()
    selection = _selection()
    sample = _sample(0, selection, height=513, width=517)
    untyped = ValidationPromptSample(
        prompt_id=sample.prompt_id,
        sample_id=sample.sample_id,
        source_shard=sample.source_shard,
        member_key=sample.member_key,
        prompt=sample.prompt,
        seed=sample.seed,
        height=sample.height,
        width=sample.width,
    )

    with pytest.raises(ValidationPromptPlanError) as captured:
        build_validation_prompt_plan(config, selection, (untyped,))

    assert captured.value.code == "VALIDATION_CAPTION_FIELDS_REQUIRED"


def test_validation_prompt_plan_rejects_untyped_nl_fallback() -> None:
    config = _config()
    selection = _selection()
    sample = _sample(0, selection, height=513, width=517)
    mismatched = dataclasses.replace(sample, prompt="not present in typed metadata")

    with pytest.raises(ValidationPromptPlanError) as captured:
        build_validation_prompt_plan(config, selection, (mismatched,))

    assert captured.value.code == "VALIDATION_CAPTION_NL_IDENTITY_MISMATCH"
