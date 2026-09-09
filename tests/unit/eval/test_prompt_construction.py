# pyright: reportPrivateUsage=false

from __future__ import annotations

import json

import pytest
import torch

from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    render_caption_segments,
    serialize_caption,
)
from sakuramoon.eval.concepts import ConceptManifest, canonical_prompt_cases
from sakuramoon.eval.runtime import (
    EvaluationError,
    _conditional_plan,
    _conditioning_inputs,
    _unconditional_plan,
)
from sakuramoon.eval.spec import PromptCase, caption_plan_prompt_text

_NO_DROPOUT = empty_caption_dropout_hits()


class _Tokenizer:
    pad_token_id = 248044

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


def _framing() -> FramingContract:
    return FramingContract(34, 5, _Tokenizer.pad_token_id)


def _structured_plan() -> CaptionPlan:
    return CaptionPlan(
        tags=(CaptionTag("general", Tag("one_girl", "one_girl")),),
        condition=ConditionRequest(
            source="artist_text",
            role="style",
            tags=(Tag("hiten_(hitenkei)", "hiten_(hitenkei)"),),
        ),
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


def _structured_case(prompt_id: str = "p1") -> PromptCase:
    plan = _structured_plan()
    return PromptCase(
        prompt_id=prompt_id,
        prompt=caption_plan_prompt_text(plan),
        conditions=(),
        seed=7,
        height=256,
        width=256,
        caption_plan=plan,
    )


def test_structured_caption_plan_is_reused_verbatim() -> None:
    case = _structured_case()
    assert _conditional_plan(case) is case.caption_plan


def test_legacy_conditions_matching_the_structured_condition_are_accepted() -> None:
    case = _structured_case()
    matching = PromptCase(
        prompt_id=case.prompt_id,
        prompt=case.prompt,
        conditions=(case.caption_plan.condition.tags[0].text,),  # type: ignore[union-attr]
        seed=case.seed,
        height=case.height,
        width=case.width,
        caption_plan=case.caption_plan,
    )
    assert _conditional_plan(matching) is matching.caption_plan


def test_legacy_conditions_disagreeing_with_the_structured_condition_raise() -> None:
    case = _structured_case()
    disagreeing = PromptCase(
        prompt_id=case.prompt_id,
        prompt=case.prompt,
        conditions=("other_artist",),
        seed=case.seed,
        height=case.height,
        width=case.width,
        caption_plan=case.caption_plan,
    )
    with pytest.raises(EvaluationError, match="disagree"):
        _conditional_plan(disagreeing)


def test_legacy_conditions_without_a_structured_condition_raise() -> None:
    plan = CaptionPlan(
        tags=(CaptionTag("general", Tag("one_girl", "one_girl")),),
        condition=None,
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )
    case = PromptCase(
        prompt_id="p1",
        prompt=caption_plan_prompt_text(plan),
        conditions=("artist_a",),
        seed=7,
        height=256,
        width=256,
        caption_plan=plan,
    )
    with pytest.raises(EvaluationError, match="carries no condition"):
        _conditional_plan(case)


def test_string_conditions_without_a_plan_raise_explicitly() -> None:
    case = PromptCase(
        prompt_id="p9",
        prompt="a plain prompt",
        conditions=("artist_a",),
        seed=7,
        height=256,
        width=256,
    )
    with pytest.raises(EvaluationError, match="p9.*no longer accepted"):
        _conditional_plan(case)


def test_natural_language_prompt_passes_through_unchanged() -> None:
    raw = "a very_raw line\nwith, commas and  underscores"
    case = PromptCase(
        prompt_id="p1",
        prompt=raw,
        conditions=(),
        seed=7,
        height=256,
        width=256,
    )
    plan = _conditional_plan(case)
    assert plan.tags == ()
    assert plan.condition is None
    assert plan.nl_text == raw
    body, condition_text = render_caption_segments(plan)
    assert body == raw
    assert condition_text == ""
    serialized = serialize_caption(plan, _Tokenizer(), _framing())
    assert serialized.text == SYSTEM_PREFIX + raw + MAIN_SUFFIX


def test_long_prompt_dropping_all_content_raises_with_prompt_id() -> None:
    raw = "y" * 600
    case = PromptCase(
        prompt_id="p100",
        prompt=raw,
        conditions=(),
        seed=7,
        height=256,
        width=256,
    )
    with pytest.raises(EvaluationError, match="p100.*dropped all prompt content"):
        _conditioning_inputs((case,), _Tokenizer(), torch.device("cpu"))


def test_long_prompt_with_surviving_tags_is_not_blocked() -> None:
    plan = CaptionPlan(
        tags=(CaptionTag("general", Tag("one_girl", "one_girl")),),
        condition=None,
        nl_text="z" * 600,
        selected_nl="long_names",
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )
    case = PromptCase(
        prompt_id="p101",
        prompt=caption_plan_prompt_text(plan),
        conditions=(),
        seed=7,
        height=256,
        width=256,
        caption_plan=plan,
    )
    inputs = _conditioning_inputs((case,), _Tokenizer(), torch.device("cpu"))
    assert inputs[0].shape[0] == 2  # conditional plus unconditional
    serialized = serialize_caption(plan, _Tokenizer(), _framing())
    assert serialized.plan.nl_text is None  # truncated NL boundary
    assert serialized.plan.tags == plan.tags  # structured tags survive


def test_explicit_unconditional_evaluation_is_unaffected() -> None:
    case = PromptCase(
        prompt_id="p1",
        prompt="y" * 600,
        conditions=(),
        seed=7,
        height=256,
        width=256,
    )
    inputs = _conditioning_inputs(
        (case,),
        _Tokenizer(),
        torch.device("cpu"),
        plan_for=lambda _case: _unconditional_plan(),
    )
    assert bool(inputs[7][1].item())  # null condition is active


def _concept_manifest() -> ConceptManifest:
    document = {
        "schema_version": 1,
        "seed": 42,
        "concepts": [
            {
                "id": "A001",
                "type": "artist",
                "tier": "high",
                "stratum": 4,
                "tag": "hiten (hitenkei)",
                "count": 10,
                "meta_tag": "hiten_(hitenkei)",
                "actual_count": 10,
                "meta_status": "matched",
                "swap": "wlop",
                "swap_count": 9,
                "swap_delta": 1,
                "refs": [
                    {"id": 1, "fav": 1, "aesthetics": None},
                    {"id": 2, "fav": 2, "aesthetics": None},
                    {"id": 3, "fav": 3, "aesthetics": None},
                ],
                "replaced_from": None,
            }
        ],
    }
    return ConceptManifest.from_bytes(json.dumps(document).encode("utf-8"))


def test_concept_structured_tag_matches_the_training_serializer() -> None:
    cases = canonical_prompt_cases(
        _concept_manifest(), height=256, width=256
    )
    case = cases[0]
    plan = case.caption_plan
    assert plan is not None
    assert plan.condition is not None
    assert (plan.condition.source, plan.condition.role) == (
        "artist_text",
        "style",
    )
    tag = plan.condition.tags[0]
    assert tag.text == "hiten (hitenkei)"
    assert tag.canonical == "hiten_(hitenkei)"
    # Display text comes from the existing training serializer: the role
    # prefix plus underscore-to-space display, with the canonical identity
    # preserved on the tag.
    body, condition_text = render_caption_segments(plan)
    assert body == ""
    assert condition_text == "style reference: hiten (hitenkei)"
    # No secondary format drift: re-serializing the governed plan reproduces
    # the case prompt byte-for-byte.
    serialized = serialize_caption(plan, _Tokenizer(), _framing())
    assert serialized.text == case.prompt
    assert serialized.plan == plan
