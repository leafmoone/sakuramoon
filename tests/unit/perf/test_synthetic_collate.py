"""Unit tests for the deterministic synthetic batch source (CPU-safe).

Uses a tiny fake tokenizer that satisfies the Qwen framing contract
(34 prefix / 5 suffix tokens) so the REAL serialize_caption +
collate_samples path is exercised without loading the 2B Qwen model.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from sakuramoon.data.serialize import (
    MAIN_SUFFIX,
    SYSTEM_PREFIX,
    FramingContract,
)
from sakuramoon.perf.synthetic import SyntheticBatchSource

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


class _FakeTokenizer:
    """Deterministic TokenEncoder honoring the 34/5 framing contract."""

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise ValueError("fake tokenizer only encodes raw text")
        if text == SYSTEM_PREFIX:
            return list(range(1_000_000, 1_000_000 + 34))
        if text == MAIN_SUFFIX:
            return list(range(2_000_000, 2_000_000 + 5))
        if not text:
            return []
        # Deterministic pseudo-tokens: one per 3 characters, id in [1000, 999).
        count = max(1, (len(text) + 2) // 3)
        return [1000 + ((len(text) * 31 + index * 7) % 899) for index in range(count)]


def _source(seed: int = 7, **overrides: object) -> SyntheticBatchSource:
    params: dict[str, object] = {
        "seed": seed,
        "target_height": 64,
        "target_width": 64,
        "local_batch": 3,
        "tokenizer": _FakeTokenizer(),
        "framing": FramingContract(34, 5, 0),
    }
    params.update(overrides)
    return SyntheticBatchSource(**params)  # type: ignore[arg-type]


def test_validate_positive_ints() -> None:
    with pytest.raises(ValueError, match="local_batch"):
        _source(local_batch=0)
    with pytest.raises(ValueError, match="target_height"):
        _source(target_height=-64)


def test_generate_returns_production_batch_contract() -> None:
    batch = _source().generate(0)
    assert batch.images.shape == (3, 3, 64, 64)
    assert batch.images.dtype == torch.uint8
    assert batch.target_height == 64
    assert batch.target_width == 64
    assert batch.dense_length >= 98  # smallest supported Qwen dense bucket
    assert batch.input_ids.shape == (3, batch.dense_length)
    assert batch.input_ids.dtype == torch.int64
    assert batch.attention_mask.shape == (3, batch.dense_length)
    assert batch.sample_ids.tolist() == [0, 1, 2]
    assert all(audit.crop_box == (0, 0, 64, 64) for audit in batch.audits)
    assert all(audit.resized_height == 64 for audit in batch.audits)
    assert len(batch.captions) == 3
    assert all(
        len(caption.input_ids) <= caption.dense_length for caption in batch.captions
    )
    # Every caption's real length fits its dense bucket.
    for caption in batch.captions:
        assert len(caption.input_ids) <= batch.dense_length


def test_dense_buckets_vary_deterministically() -> None:
    lengths = set()
    for batch_index in range(30):
        batch = _source().generate(batch_index)
        for caption in batch.captions:
            lengths.add(caption.dense_length)
    assert len(lengths) >= 3, f"expected a real bucket mix, got {sorted(lengths)}"


def test_null_and_active_condition_paths() -> None:
    saw_active = False
    saw_null = False
    for batch_index in range(40):
        batch = _source().generate(batch_index)
        for caption in batch.captions:
            if caption.use_null_condition:
                saw_null = True
            elif caption.condition_source is not None:
                saw_active = True
        if saw_null and saw_active:
            break
    assert saw_active, "no active-condition caption in 40 batches"
    assert saw_null, "no null-condition caption in 40 batches"


def test_generation_is_deterministic() -> None:
    first = _source(seed=123).generate(4)
    second = _source(seed=123).generate(4)
    assert torch.equal(first.images, second.images)
    assert torch.equal(first.input_ids, second.input_ids)
    assert torch.equal(first.attention_mask, second.attention_mask)
    assert first.captions == second.captions
    other = _source(seed=124).generate(4)
    assert not torch.equal(first.images, other.images)


def test_pregenerate_and_validation() -> None:
    batches = _source(local_batch=2).pregenerate(3, pin=False)
    assert len(batches) == 3
    assert [batch.sample_ids[0].item() for batch in batches] == [0, 2, 4]
    with pytest.raises(ValueError, match="batch_count"):
        _source().pregenerate(0)
    with pytest.raises(ValueError, match="batch_index"):
        _source().generate(-1)
    with pytest.raises(ValueError, match="pregenerated"):
        list(_source().infinite(()))


def test_real_tokenizer_smoke_when_assets_present() -> None:
    """Load the REAL Qwen tokenizer (small, CPU-safe) and build one batch."""

    model_root = Path("/sakuramoon-runtime/model")
    qwen_root = model_root / "qwen_3.5_2B"
    if (
        not (qwen_root / "tokenizer.json").is_file()
        and not (qwen_root / "tokenizer_config.json").is_file()
    ):
        pytest.skip("local Qwen tokenizer assets are not present on this host")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(qwen_root), trust_remote_code=True)
    source = SyntheticBatchSource(
        seed=5,
        target_height=64,
        target_width=64,
        local_batch=2,
        tokenizer=tokenizer,
        framing=FramingContract(34, 5, int(tokenizer.pad_token_id)),
    )
    batch = source.generate(0)
    assert batch.images.shape == (2, 3, 64, 64)
    assert batch.dense_length >= 98
    for caption in batch.captions:
        assert 34 + 5 <= len(caption.input_ids) <= batch.dense_length
