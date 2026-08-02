from __future__ import annotations

from collections.abc import Iterator
from typing import cast

import pytest

import sakuramoon.data.production as production_module
from sakuramoon.data.collate import TrainingBatch
from sakuramoon.data.production import (
    ConfiguredDataLoader,
    ProductionBatchStreamIdentity,
    ProductionDataError,
    adapt_modelscope_metadata,
    parse_modelscope_caption_fields,
)


class _DepthIterator(Iterator[TrainingBatch]):
    def __init__(self, depth: int) -> None:
        self.depth = depth

    def __next__(self) -> TrainingBatch:
        raise StopIteration

    def ready_batch_depth_snapshot(self) -> int:
        return self.depth


def _stream_identity() -> ProductionBatchStreamIdentity:
    return ProductionBatchStreamIdentity(
        resolved_config_sha256="1" * 64,
        loader=ConfiguredDataLoader(1, 1, 1, False, True),
        manifest_id="2" * 64,
        service_session_sha256="3" * 64,
        factory_identity="4" * 64,
    )


def _real_row() -> dict[str, object]:
    return {
        "id": 71,
        "image": {"width": 832, "height": 1216},
        "captions": {"nl2": "A blue-haired character.", "nl3": ""},
        "multicaptions": {"vibes": "soft light"},
        "tags": {
            "character": ["alice"],
            "copyright": ["original"],
            "general": ["blue_hair", "dress"],
            "artist": ["artist_name"],
        },
        "dropout": {"candidate_tags": ["blue_hair"]},
        "nsfw": "safe",
    }


def test_governed_modelscope_adapter_and_caption_parser() -> None:
    raw = _real_row()

    adapted = adapt_modelscope_metadata(raw)
    fields = parse_modelscope_caption_fields(raw)

    assert adapted == {
        "id": 71,
        "width": 832,
        "height": 1216,
        "caption_available": True,
    }
    assert tuple(tag.text for tag in fields.nsfw) == ("safe",)
    assert tuple(tag.text for tag in fields.character) == ("alice",)
    assert tuple(tag.text for tag in fields.general) == ("blue_hair", "dress")
    assert tuple(tag.text for tag in fields.artists) == ("artist_name",)
    assert fields.candidate_tags == frozenset({"blue_hair"})
    assert fields.nl.long_names is None and fields.nl.long_no_names is None
    assert fields.nl.short_vibes == "soft light"
    assert fields.nl.nl2 == "A blue-haired character."
    assert fields.nl.nl3 is None


def test_governed_modelscope_parser_rejects_schema_drift() -> None:
    missing_nested = _real_row()
    missing_nested.pop("image")
    with pytest.raises(ProductionDataError, match="image must be an object"):
        adapt_modelscope_metadata(missing_nested)

    bad_tags = _real_row()
    tags = bad_tags["tags"]
    assert isinstance(tags, dict)
    tags["artist"] = "not-a-list"
    with pytest.raises(ProductionDataError, match=r"tags\.artist must be a list"):
        parse_modelscope_caption_fields(bad_tags)

    bad_candidate = _real_row()
    dropout = bad_candidate["dropout"]
    assert isinstance(dropout, dict)
    dropout["candidate_tags"] = ["valid", 2]
    with pytest.raises(ProductionDataError, match="only strings"):
        parse_modelscope_caption_fields(bad_candidate)


def test_accepted_stream_exposes_only_live_iterator_ready_batch_depth() -> None:
    iterator = _DepthIterator(2)
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        iterator, _stream_identity()
    )

    assert stream.ready_batch_depth_snapshot() == 2
    iterator.depth = 4
    assert stream.ready_batch_depth_snapshot() == 4
    stream.close()
    with pytest.raises(ProductionDataError, match="closed"):
        stream.ready_batch_depth_snapshot()


def test_accepted_stream_fails_when_iterator_has_no_ready_depth_source() -> None:
    stream = production_module._issue_batch_stream(  # pyright: ignore[reportPrivateUsage]
        cast(Iterator[TrainingBatch], iter(())),
        _stream_identity(),
    )

    with pytest.raises(ProductionDataError, match="unavailable"):
        stream.ready_batch_depth_snapshot()
    stream.close()
