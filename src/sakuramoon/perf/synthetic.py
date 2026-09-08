"""Deterministic in-memory synthetic TrainingBatch source (P0).

Every batch is built through the EXISTING production data helpers —
caption plan -> ``serialize_caption`` (real Qwen tokenizer) ->
``collate_samples`` — so the token index tensors, masks, dense buckets,
dropout counters and spatial audits are production-exact.  The only
synthetic inputs are the pixel values (deterministic uint8 RGB noise) and
the caption text (deterministic tag/NL templates).  No DataService lease,
no shard, no network.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import cast

import torch

from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    ConditionMode,
    NlCandidates,
    NlDropoutProbabilities,
    Tag,
    build_caption_plan,
)
from sakuramoon.data.collate import TrainingBatch, collate_samples
from sakuramoon.data.pipeline import ImageAudit, PipelineSample, RngIdentity
from sakuramoon.data.serialize import (
    FramingContract,
    SerializedCaption,
    TokenEncoder,
    serialize_caption,
)

_G1_DROPOUT = CaptionDropoutProbabilities(
    condition_route=0.0,
    condition_only=0.0,
    tag=0.1,
    candidate_source=0.3,
    nl=NlDropoutProbabilities(
        long_names=0.3,
        long_no_names=0.3,
        short_vibes=0.3,
        nl2=0.3,
        nl3=0.3,
    ),
)


def _tag(text: str) -> Tag:
    return Tag(text=text, canonical=text)


@dataclass(frozen=True, slots=True)
class SyntheticBatchSource:
    """One deterministic production-shaped batch per call.

    ``target_height``/``target_width`` are the canonical training bucket
    (G1: 256x256, the dominant production bucket).  Caption density varies
    deterministically with the sample index so that the supported Qwen
    dense-length buckets are all exercised.
    """

    seed: int
    target_height: int
    target_width: int
    local_batch: int
    tokenizer: object
    framing: FramingContract
    condition_mode: ConditionMode = "artist_or_character"
    probabilities: CaptionDropoutProbabilities = _G1_DROPOUT

    def __post_init__(self) -> None:
        for name in ("seed", "target_height", "target_width", "local_batch"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"SyntheticBatchSource.{name} must be a positive int")

    def _caption_fields(self, index: int) -> CaptionFields:
        general = tuple(
            _tag(f"synthetic_general_{index}_{j}") for j in range(4 + (index * 7) % 28)
        )
        character = tuple(_tag(f"character_{index}_{k}") for k in range(1 + index % 4))
        copyright_tags = tuple(
            _tag(f"copyright_{index}_{k}") for k in range(1 + index % 2)
        )
        artists = tuple(_tag(f"artist_{index}_{k}") for k in range(index % 2))
        return CaptionFields(
            nsfw=(),
            character=character,
            copyright=copyright_tags,
            general=general,
            artists=artists,
            candidate_tags=frozenset(),
            nl=NlCandidates(
                long_names=(
                    f"A synthetic long description with names, number {index}, "
                    "written in a calm deterministic style for benchmarking."
                ),
                long_no_names=(
                    f"A synthetic long description without names, number {index}, "
                    "describing lighting, composition, and a steady mood."
                ),
                short_vibes=f"soft light, number {index % 7}.",
                nl2=None,
                nl3=None,
            ),
            rating=(_tag("synthetic_rating"),),
            year=(),
            aesthetic=(),
            quality=(),
            anime_completeness=(),
            anime_classification=(),
        )

    def _caption(self, index: int) -> SerializedCaption:
        fields = self._caption_fields(index)
        plan = build_caption_plan(
            fields,
            self.probabilities,
            condition_mode=self.condition_mode,
            seed=index,
        )
        return serialize_caption(plan, cast(TokenEncoder, self.tokenizer), self.framing)

    def _image(self, index: int) -> torch.Tensor:
        generator = torch.Generator()
        generator.manual_seed(self.seed + index)
        return torch.randint(
            0,
            256,
            (3, self.target_height, self.target_width),
            generator=generator,
            dtype=torch.uint8,
        )

    def generate(self, batch_index: int, *, pin: bool = False) -> TrainingBatch:
        """Build batch ``batch_index`` (0-based) deterministically."""

        if type(batch_index) is not int or batch_index < 0:
            raise ValueError("batch_index must be a nonnegative int")
        samples: list[PipelineSample] = []
        for position in range(self.local_batch):
            index = batch_index * self.local_batch + position
            audit = ImageAudit(
                source_width=self.target_width,
                source_height=self.target_height,
                resized_width=self.target_width,
                resized_height=self.target_height,
                crop_box=(0, 0, self.target_width, self.target_height),
                crop_retention=1.0,
            )
            samples.append(
                PipelineSample(
                    sample_id=index,
                    source_shard="synthetic",
                    image=self._image(index),
                    target_height=self.target_height,
                    target_width=self.target_width,
                    caption=self._caption(index),
                    audit=audit,
                    rng=RngIdentity(
                        base_seed=self.seed,
                        stage="benchmark",
                        cycle_index=batch_index,
                        sample_id=index,
                        caption_seed=index,
                        crop_seed=index,
                    ),
                    padding_token_id=self.framing.padding_token_id,
                )
            )
        batch = collate_samples(tuple(samples))
        if pin and torch.cuda.is_available():
            batch = batch.pin_memory()
        return batch

    def pregenerate(
        self, batch_count: int, *, pin: bool = True
    ) -> tuple[TrainingBatch, ...]:
        """Build ``batch_count`` batches up front (excluded from timing)."""

        if type(batch_count) is not int or batch_count <= 0:
            raise ValueError("batch_count must be a positive int")
        return tuple(self.generate(index, pin=pin) for index in range(batch_count))

    def infinite(self, batches: tuple[TrainingBatch, ...]) -> Iterator[TrainingBatch]:
        """Cycle the pregenerated batches forever (retry-safe for the loop)."""

        if not batches:
            raise ValueError("infinite() requires pregenerated batches")
        position = 0
        while True:
            yield batches[position % len(batches)]
            position += 1


__all__ = ["SyntheticBatchSource"]
