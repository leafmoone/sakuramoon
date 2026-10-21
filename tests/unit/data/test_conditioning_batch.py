# pyright: reportPrivateUsage=false
"""Exact tensor tests for the shared conditioning batch primitives.

Expected tensors are constructed by hand per category (baseline
reference), never through the legacy entry helpers.  All tests run on
masked CPU with no model loading.
"""

from __future__ import annotations

import types
from typing import Any, cast

import pytest
import torch

from sakuramoon.data import conditioning_batch as batch
from sakuramoon.data.caption import (
    CaptionPlan,
    CaptionTag,
    ConditionRequest,
    NlBranch,
    Tag,
    empty_caption_dropout_hits,
)
from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
    SerializedCaption,
    serialize_caption,
)
from sakuramoon.eval.runtime import EvaluationError
from sakuramoon.eval.runtime import _conditioning_inputs as eval_inputs
from sakuramoon.eval.spec import PromptCase
from sakuramoon.inference import InferenceSettings, generate_image
from sakuramoon.sampling.profiles import SamplingProfile
from sakuramoon.train.sampling import _conditioning_inputs as train_inputs
from sakuramoon.train.tensor_metrics import TrainingSamplingError

DEVICE = torch.device("cpu")
PAD = 99


class _Tokenizer:
    def __init__(self, pad_token_id: int = PAD) -> None:
        self.pad_token_id = pad_token_id

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if text == SYSTEM_PREFIX:
            return list(range(100, 134))
        if text == MAIN_SUFFIX:
            return list(range(200, 205))
        return [1000 + index for index, _character in enumerate(text)]


def _framing() -> FramingContract:
    return FramingContract(34, 5, PAD)


def _plan(
    *,
    tags: tuple[CaptionTag, ...] = (),
    condition: ConditionRequest | None = None,
    nl_text: str | None = None,
    selected_nl: NlBranch | None = None,
    all_condition_dropped: bool = False,
) -> CaptionPlan:
    return CaptionPlan(
        tags=tags,
        condition=condition,
        nl_text=nl_text,
        selected_nl=selected_nl,
        all_condition_dropped=all_condition_dropped,
        dropout_hits=empty_caption_dropout_hits(all_condition=all_condition_dropped),
    )


def _serialize(plan: CaptionPlan) -> SerializedCaption:
    return serialize_caption(plan, _Tokenizer(), _framing())


def _caption(
    input_ids: tuple[int, ...],
    *,
    dense_length: int,
    main: tuple[int, ...] = (),
    cond: tuple[int, ...] = (),
    use_null: bool = False,
) -> SerializedCaption:
    return SerializedCaption(
        plan=_plan(all_condition_dropped=True),
        text="",
        input_ids=input_ids,
        attention_mask=(True,) * len(input_ids),
        main_token_indices=main,
        main_mask=(True,) * len(main),
        condition_token_indices=cond,
        condition_mask=(True,) * len(cond),
        use_null_condition=use_null,
        condition_source=None,
        condition_role=None,
        all_condition_dropped=True,
        dropout_hits=empty_caption_dropout_hits(all_condition=True),
        selected_nl=None,
        body="",
        condition_text="",
        condition_tokens=0,
        condition_bucket=64,
        dense_length=dense_length,
        truncated=False,
    )


def _reference_grid(
    rows: list[list[int]], dense: int
) -> tuple[torch.Tensor, torch.Tensor]:
    ids = torch.full((len(rows), dense), PAD, dtype=torch.long, device=DEVICE)
    mask = torch.zeros((len(rows), dense), dtype=torch.bool, device=DEVICE)
    for row, tokens in enumerate(rows):
        ids[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=DEVICE)
        mask[row, : len(tokens)] = True
    return ids, mask


def _reference_index_grid(rows: list[tuple[int, ...]]) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(row) for row in rows), default=0)
    indices = torch.full((len(rows), width), -1, dtype=torch.long, device=DEVICE)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=DEVICE)
    for row, tokens in enumerate(rows):
        if tokens:
            indices[row, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.long, device=DEVICE
            )
            mask[row, : len(tokens)] = True
    return indices, mask


def _reference_batch(
    id_rows: list[list[int]],
    dense: int,
    mains: list[tuple[int, ...]],
    conds: list[tuple[int, ...]],
    use_null_rows: list[bool],
) -> tuple[Any, ...]:
    ids, attention = _reference_grid(id_rows, dense)
    main_indices, main_mask = _reference_index_grid(mains)
    cond_indices, cond_mask = _reference_index_grid(conds)
    use_null = torch.tensor(use_null_rows, dtype=torch.bool, device=DEVICE)
    active = torch.tensor(
        [i for i, null in enumerate(use_null_rows) if not null],
        dtype=torch.long,
        device=DEVICE,
    )
    return (
        ids,
        attention,
        main_indices,
        main_mask,
        tuple(len(row) for row in mains),
        cond_indices,
        cond_mask,
        use_null,
        active,
    )


def _assert_batch(actual: tuple[Any, ...], expected: tuple[Any, ...]) -> None:
    assert len(actual) == 9
    for position, (got, want) in enumerate(zip(actual, expected, strict=True)):
        if isinstance(want, tuple):
            assert got == want, f"slot {position} main_lengths tuple"
            continue
        assert isinstance(got, torch.Tensor), f"slot {position} not a tensor"
        assert got.dtype == want.dtype, f"slot {position} dtype"
        assert got.device == want.device, f"slot {position} device"
        assert got.shape == want.shape, f"slot {position} shape"
        assert torch.equal(got, want), f"slot {position} content"


def test_uncond_category() -> None:
    caption = _serialize(_plan(all_condition_dropped=True))
    actual = batch.conditioning_batch(
        (caption,), pad_token_id=PAD, device=DEVICE
    )
    expected = _reference_batch(
        [list(range(100, 134)) + list(range(200, 205))],
        34 + 64,
        [tuple(range(39))],
        [()],
        [True],
    )
    _assert_batch(actual, expected)
    assert actual[8].shape == (0,)


def test_nl_category() -> None:
    plan = _plan(nl_text="abc", selected_nl="short_vibes")
    caption = _serialize(plan)
    actual = batch.conditioning_batch(
        (caption,), pad_token_id=PAD, device=DEVICE
    )
    expected = _reference_batch(
        [list(range(100, 134)) + [1000, 1001, 1002] + list(range(200, 205))],
        34 + 64,
        [tuple(range(42))],
        [()],
        [True],
    )
    _assert_batch(actual, expected)


def test_tags_category() -> None:
    plan = _plan(tags=(CaptionTag("general", Tag("long_hair", "long_hair")),))
    caption = _serialize(plan)
    body = "long hair"
    actual = batch.conditioning_batch(
        (caption,), pad_token_id=PAD, device=DEVICE
    )
    expected = _reference_batch(
        [
            list(range(100, 134))
            + [1000 + i for i in range(len(body))]
            + list(range(200, 205))
        ],
        34 + 64,
        [tuple(range(34 + len(body) + 5))],
        [()],
        [True],
    )
    _assert_batch(actual, expected)


def test_styleartist_category() -> None:
    plan = _plan(
        condition=ConditionRequest(
            source="artist_text", role="style", tags=(Tag("artist_name", "artist_name"),)
        ),
    )
    caption = _serialize(plan)
    condition_text = "style reference: artist name"
    main_length = 34 + 5
    actual = batch.conditioning_batch(
        (caption,), pad_token_id=PAD, device=DEVICE
    )
    expected = _reference_batch(
        [
            list(range(100, 134))
            + list(range(200, 205))
            + [1000 + i for i in range(len(condition_text))]
        ],
        34 + 64,
        [tuple(range(main_length))],
        [tuple(range(main_length, main_length + len(condition_text)))],
        [False],
    )
    _assert_batch(actual, expected)
    assert bool(actual[7][0]) is False
    assert torch.equal(actual[8], torch.tensor([0], dtype=torch.long))


def test_identity_category() -> None:
    plan = _plan(
        condition=ConditionRequest(
            source="character_text",
            role="identity",
            tags=(Tag("character_name", "character_name"),),
        ),
    )
    caption = _serialize(plan)
    condition_text = "character identity: character name"
    main_length = 34 + 5
    actual = batch.conditioning_batch(
        (caption,), pad_token_id=PAD, device=DEVICE
    )
    expected = _reference_batch(
        [
            list(range(100, 134))
            + list(range(200, 205))
            + [1000 + i for i in range(len(condition_text))]
        ],
        34 + 64,
        [tuple(range(main_length))],
        [tuple(range(main_length, main_length + len(condition_text)))],
        [False],
    )
    _assert_batch(actual, expected)


def test_mixed_variable_and_null_category() -> None:
    mixed = _serialize(
        _plan(
            tags=(CaptionTag("general", Tag("long_hair", "long_hair")),),
            condition=ConditionRequest(
                source="artist_text", role="style", tags=(Tag("artist_name", "artist_name"),)
            ),
            nl_text="soft lighting",
            selected_nl="short_vibes",
        )
    )
    nl = _serialize(_plan(nl_text="abc", selected_nl="short_vibes"))
    uncond = _serialize(_plan(all_condition_dropped=True))
    captions = (mixed, nl, uncond)
    actual = batch.conditioning_batch(captions, pad_token_id=PAD, device=DEVICE)

    mixed_body = "long hair\n\nsoft lighting"
    condition_text = "style reference: artist name"
    expected = _reference_batch(
        [
            list(range(100, 134))
            + [1000 + i for i in range(len(mixed_body))]
            + list(range(200, 205))
            + [1000 + i for i in range(len(condition_text))],
            list(range(100, 134)) + [1000, 1001, 1002] + list(range(200, 205)),
            list(range(100, 134)) + list(range(200, 205)),
        ],
        max(caption.dense_length for caption in (mixed, nl, uncond)),
        [
            tuple(range(34 + len(mixed_body) + 5)),
            tuple(range(42)),
            tuple(range(39)),
        ],
        [
            tuple(range(34 + len(mixed_body) + 5, 34 + len(mixed_body) + 5 + len(condition_text))),
            (),
            (),
        ],
        [False, True, True],
    )
    _assert_batch(actual, expected)
    assert actual[4] == (34 + len(mixed_body) + 5, 42, 39)
    assert torch.equal(actual[8], torch.tensor([0], dtype=torch.long))


def test_null_rows_leave_active_empty() -> None:
    captions = (
        _caption((1, 2), dense_length=4, use_null=True),
        _caption((3,), dense_length=4, use_null=True),
    )
    actual = batch.conditioning_batch(captions, pad_token_id=PAD, device=DEVICE)
    assert torch.equal(actual[7], torch.tensor([True, True], dtype=torch.bool))
    assert actual[8].dtype == torch.long
    assert actual[8].shape == (0,)


def test_empty_batch_raises_bare_value_error() -> None:
    with pytest.raises(ValueError, match="max"):
        batch.token_rows((), pad_token_id=PAD, device=DEVICE)
    with pytest.raises(ValueError):
        batch.conditioning_batch((), pad_token_id=PAD, device=DEVICE)
    empty = batch.index_rows((), device=DEVICE)
    assert empty[0].shape == (0, 0) and empty[1].shape == (0, 0)
    empty_flags = batch.condition_flags((), device=DEVICE)
    assert empty_flags[0].shape == (0,) and empty_flags[1].shape == (0,)


def test_long_row_without_padding() -> None:
    tokens = tuple(range(1000, 1200))
    caption = _caption(tokens, dense_length=200, main=tuple(range(200)))
    actual = batch.conditioning_batch((caption,), pad_token_id=PAD, device=DEVICE)
    assert actual[0].shape == (1, 200)
    assert not bool((actual[0] == PAD).any())
    assert bool(actual[1].all())
    assert actual[4] == (200,)


def test_dense_bucket_padding_identity() -> None:
    full = _caption(tuple(range(1000, 1064)), dense_length=64, main=tuple(range(64)))
    short = _caption((7,), dense_length=64, main=(0, 1))
    actual = batch.conditioning_batch((full, short), pad_token_id=PAD, device=DEVICE)
    ids, mask = actual[0], actual[1]
    assert ids.shape == (2, 64)
    assert bool((ids[1, 2:] == PAD).all())
    assert not bool(mask[1, 2:].any())
    assert bool(mask[0].all())
    main_indices, main_mask = actual[2], actual[3]
    assert torch.equal(main_indices[1, 2:], torch.full((62,), -1, dtype=torch.long))
    assert not bool(main_mask[1, 2:].any())
    cond_indices, cond_mask = actual[5], actual[6]
    assert cond_indices.shape == (2, 0) and cond_mask.shape == (2, 0)
    assert actual[4] == (64, 2)


def test_invalid_token_row_raises_torch_error_not_train_error() -> None:
    caption = _caption((1, 2, 3), dense_length=2, main=(0, 1))
    with pytest.raises(RuntimeError) as excinfo:
        batch.token_rows((caption,), pad_token_id=PAD, device=DEVICE)
    assert type(excinfo.value) is RuntimeError
    assert not isinstance(excinfo.value, TrainingSamplingError)


def test_index_none_element_default_dtype_keeps_old_inference_error() -> None:
    # Evaluation and inference relied on torch dtype inference: a None
    # element must keep the old RuntimeError, not an explicit-long TypeError.
    values = (cast("tuple[int, ...]", (None,)),)
    with pytest.raises(RuntimeError, match="Could not infer dtype"):
        batch.index_rows(values, device=DEVICE)


def test_index_none_element_explicit_long_keeps_old_train_error() -> None:
    values = (cast("tuple[int, ...]", (None,)),)
    with pytest.raises(TypeError):
        batch.index_rows(values, device=DEVICE, copy_dtype=torch.long)


def test_float_overflow_copy_semantics_per_caller() -> None:
    values = (cast("tuple[int, ...]", (1e40,)),)
    # Inferred-dtype callers (evaluation/inference) wrap-cast to the
    # int64 minimum exactly like the original assignment.
    indices, mask = batch.index_rows(values, device=DEVICE)
    assert torch.equal(
        indices, torch.full((1, 1), torch.iinfo(torch.int64).min, dtype=torch.long)
    )
    assert bool(mask[0, 0])
    # The training caller's explicit torch.long copy raises overflow instead.
    with pytest.raises(
        RuntimeError, match="value cannot be converted to type int64 without overflow"
    ):
        batch.index_rows(values, device=DEVICE, copy_dtype=torch.long)


# --- entry wrapper policies -------------------------------------------------


def test_train_wrapper_guard_order_and_messages() -> None:
    captions = (_caption((1, 2), dense_length=2, main=(0, 1)),)
    bad_pad = types.SimpleNamespace(pad_token_id="x")
    with pytest.raises(TrainingSamplingError, match="Qwen padding token ID is unavailable"):
        train_inputs(captions, tokenizer=bad_pad, device=DEVICE)
    negative = types.SimpleNamespace(pad_token_id=-1)
    with pytest.raises(TrainingSamplingError, match="Qwen padding token ID is unavailable"):
        train_inputs(captions, tokenizer=negative, device=DEVICE)
    with pytest.raises(TrainingSamplingError, match="training sample caption group is empty"):
        train_inputs((), tokenizer=_Tokenizer(), device=DEVICE)
    overflow = _caption((1, 2, 3, 4, 5, 6, 7), dense_length=5, main=(0, 1))
    with pytest.raises(TrainingSamplingError, match="exceeds its bucket"):
        train_inputs((overflow,), tokenizer=_Tokenizer(), device=DEVICE)


def test_train_overflow_keeps_original_row_fault_competition() -> None:
    # Row 0 carries a non-int ID (torch error at its own copy) and row 1
    # overflows its bucket.  The original loop faults at row 0 first; the
    # overflow guard must check at each row's copy position, never in a
    # pre-scan, or this race flips.
    bad = _caption(cast("tuple[int, ...]", (1, "x")), dense_length=3, main=(0, 1))
    overflow = _caption((1, 2, 3, 4, 5, 6, 7), dense_length=3)
    with pytest.raises(TypeError):
        train_inputs((bad, overflow), tokenizer=_Tokenizer(), device=DEVICE)


def test_train_wrapper_happy_path_matches_reference() -> None:
    mixed = _serialize(
        _plan(
            tags=(CaptionTag("general", Tag("long_hair", "long_hair")),),
            condition=ConditionRequest(
                source="artist_text", role="style", tags=(Tag("artist_name", "artist_name"),)
            ),
            nl_text="soft lighting",
            selected_nl="short_vibes",
        )
    )
    uncond = _serialize(_plan(all_condition_dropped=True))
    actual = train_inputs((mixed, uncond), tokenizer=_Tokenizer(), device=DEVICE)
    mixed_body = "long hair\n\nsoft lighting"
    condition_text = "style reference: artist name"
    expected = _reference_batch(
        [
            list(range(100, 134))
            + [1000 + i for i in range(len(mixed_body))]
            + list(range(200, 205))
            + [1000 + i for i in range(len(condition_text))],
            list(range(100, 134)) + list(range(200, 205)),
        ],
        max(mixed.dense_length, uncond.dense_length),
        [
            tuple(range(34 + len(mixed_body) + 5)),
            tuple(range(39)),
        ],
        [
            tuple(range(34 + len(mixed_body) + 5, 34 + len(mixed_body) + 5 + len(condition_text))),
            (),
        ],
        [False, True],
    )
    _assert_batch(actual, expected)
    assert type(actual[4]) is tuple


def test_eval_wrapper_tokenizer_and_content_dropped_errors() -> None:
    with pytest.raises(EvaluationError, match="Qwen tokenizer is unavailable"):
        eval_inputs((), types.SimpleNamespace(pad_token_id=None), DEVICE)
    case = PromptCase(
        prompt_id="p1",
        prompt="a" * 600,
        conditions=(),
        seed=1,
        height=128,
        width=128,
    )
    with pytest.raises(EvaluationError, match="serialization dropped all prompt content"):
        eval_inputs((case,), _Tokenizer(), DEVICE)
    with pytest.raises(ValueError):
        eval_inputs((), _Tokenizer(), DEVICE)


def _eval_case() -> PromptCase:
    condition_text = "style reference: artist name"
    return PromptCase(
        prompt_id="p1",
        prompt=SYSTEM_PREFIX + MAIN_SUFFIX + condition_text,
        conditions=(),
        seed=1,
        height=128,
        width=128,
        caption_plan=_plan(
            condition=ConditionRequest(
                source="artist_text", role="style", tags=(Tag("artist_name", "artist_name"),)
            ),
        ),
    )


def test_eval_wrapper_happy_path_two_rows() -> None:
    actual = eval_inputs((_eval_case(),), _Tokenizer(), DEVICE)
    condition_text = "style reference: artist name"
    expected = _reference_batch(
        [
            list(range(100, 134))
            + list(range(200, 205))
            + [1000 + i for i in range(len(condition_text))],
            list(range(100, 134)) + list(range(200, 205)),
        ],
        34 + 64,
        [
            tuple(range(39)),
            tuple(range(39)),
        ],
        [
            tuple(range(39, 39 + len(condition_text))),
            (),
        ],
        [False, True],
    )
    _assert_batch(actual, expected)


class _RecordingEncoder:
    def __init__(self, fail: str | None = None) -> None:
        self.calls = 0
        self.fail = fail
        self.hidden_states = torch.zeros(1, 1)

    def __call__(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Any:
        self.calls += 1
        if self.fail is not None:
            raise RuntimeError(self.fail)
        return self


class _Composite:
    training = False


def _settings() -> InferenceSettings:
    return InferenceSettings(
        SamplingProfile(name="test", solver="euler", steps=4, time_schedule="linear"),
        cfg_scale=1.0,
        noise_scale=1.0,
        t_eps=0.05,
    )


def _infer_kwargs(encoder: _RecordingEncoder, pad_token_id: Any = PAD) -> dict[str, Any]:
    return {
        "composite": _Composite(),
        "qwen": types.SimpleNamespace(
            tokenizer=types.SimpleNamespace(
                pad_token_id=pad_token_id,
                encode=_Tokenizer().encode,
            ),
            encoder=encoder,
        ),
        "vae": object(),
        "settings": _settings(),
        "width": 128,
        "height": 128,
        "seed": 1,
        "growth_alpha": 0.0,
        "device": DEVICE,
    }


def test_inference_truncation_fails_before_padding() -> None:
    plan = _plan(nl_text="a" * 600, selected_nl="short_vibes")
    encoder = _RecordingEncoder()
    with pytest.raises(ValueError, match="prompt exceeds the model's caption budget"):
        generate_image(plan, **_infer_kwargs(encoder))
    assert encoder.calls == 0


def test_inference_requires_integer_padding_token() -> None:
    encoder = _RecordingEncoder()
    with pytest.raises(ValueError, match="no integer padding token"):
        generate_image(_plan(all_condition_dropped=True), **_infer_kwargs(encoder, pad_token_id="x"))
    assert encoder.calls == 0


def test_inference_encoder_fault_precedes_index_materialization() -> None:
    encoder = _RecordingEncoder(fail="qwen-fault")
    with pytest.raises(RuntimeError, match="qwen-fault"):
        generate_image(_plan(all_condition_dropped=True), **_infer_kwargs(encoder))
    assert encoder.calls == 1
