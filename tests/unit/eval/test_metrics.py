# pyright: reportPrivateUsage=false

from __future__ import annotations

import pytest
import torch

from sakuramoon.config.schema import EvaluationEnabledConfig
from sakuramoon.data.serialize import MAIN_SUFFIX, SYSTEM_PREFIX
from sakuramoon.eval.metrics import (
    FeatureStats,
    frechet_inception_distance,
    inception_score,
)
from sakuramoon.eval.runtime import _conditioning_inputs
from sakuramoon.eval.spec import PromptCase


class _Tokenizer:
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        del add_special_tokens
        if text == SYSTEM_PREFIX:
            return [1] * 34
        if text == MAIN_SUFFIX:
            return [2] * 5
        return [3] * len(text.split())


def test_fid_is_zero_for_identical_feature_statistics() -> None:
    features = torch.tensor(
        [[0.0, 1.0], [1.0, 0.0], [0.5, 0.5], [0.25, 0.75]],
        dtype=torch.float32,
    )
    stats = FeatureStats.from_features(features)

    assert frechet_inception_distance(stats, stats) == pytest.approx(0.0, abs=1e-10)


def test_inception_score_and_default_schedule() -> None:
    probabilities = torch.tensor(
        [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]],
        dtype=torch.float32,
    )
    score = inception_score(probabilities, splits=2)
    config = EvaluationEnabledConfig.model_validate(
        {
            "enabled": True,
            "sample_count": 1000,
            "batch_size": 4,
            "is_splits": 10,
            "prompt_path": "data/prompts.json",
            "validation_shard_root": "data/validation",
            "output_dir": "output_model/evaluation",
            "sampling_profile": "preview",
        },
        strict=True,
    )

    assert score.mean > 1.0
    assert score.sample_count == 4
    assert config.every_updates == 1000


def test_evaluation_uses_normal_caption_boundary_truncation() -> None:
    case = PromptCase(
        prompt_id="long-prompt",
        prompt=" ".join(["word"] * 600),
        conditions=(),
        seed=1,
        height=256,
        width=256,
    )

    inputs = _conditioning_inputs((case,), _Tokenizer(), torch.device("cpu"))

    input_ids, attention_mask = inputs[:2]
    assert input_ids.shape == (2, 98)
    assert torch.equal(attention_mask.sum(dim=1), torch.tensor([39, 39]))
