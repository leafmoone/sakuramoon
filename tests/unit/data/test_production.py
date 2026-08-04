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
        loader=ConfiguredDataLoader(1, 1, 1, False, True),
        dataset_id="leafmoone/webdataset_danbooru@master",
        session_id="test-session",
    )


def _real_row() -> dict[str, object]:
    return {
        "id": 71,
        "image": {"width": 832, "height": 1216},
        "captions": {"nl2": "A blue-haired character.", "nl3": ""},
        "multicaptions": {
            "long_names": "Alice in a blue dress.",
            "long_no_names": None,
            "short": "A blue-haired character.",
            "vibes": "soft light",
        },
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
    }
    assert tuple(tag.text for tag in fields.nsfw) == ("safe",)
    assert tuple(tag.text for tag in fields.character) == ("alice",)
    assert tuple(tag.text for tag in fields.general) == ("blue_hair", "dress")
    assert tuple(tag.text for tag in fields.artists) == ("artist_name",)
    assert fields.candidate_tags == frozenset({"blue_hair"})
    assert fields.nl.long_names == "Alice in a blue dress."
    assert fields.nl.long_no_names is None
    assert fields.nl.short_vibes == "A blue-haired character.\n\nsoft light"
    assert fields.nl.nl2 == "A blue-haired character."
    assert fields.nl.nl3 is None


def test_modelscope_adapter_ignores_missing_declared_image_dimensions() -> None:
    raw = _real_row()
    raw["image"] = {"format": "webp", "width": None, "height": None}

    assert adapt_modelscope_metadata(raw) == {"id": 71}


@pytest.mark.parametrize("value", [None, ""])
def test_governed_modelscope_parser_accepts_empty_nsfw(value: object) -> None:
    raw = _real_row()
    raw["nsfw"] = value

    fields = parse_modelscope_caption_fields(raw)

    assert fields.nsfw == ()


def test_governed_modelscope_parser_rejects_schema_drift() -> None:
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

    bad_nsfw = _real_row()
    bad_nsfw["nsfw"] = 2
    with pytest.raises(ProductionDataError, match="nsfw must be text or null"):
        parse_modelscope_caption_fields(bad_nsfw)

    bad_nl = _real_row()
    multicaptions = bad_nl["multicaptions"]
    assert isinstance(multicaptions, dict)
    multicaptions["short"] = ["not", "text"]
    with pytest.raises(
        ProductionDataError, match=r"multicaptions\.short must be text or null"
    ):
        parse_modelscope_caption_fields(bad_nl)


@pytest.mark.parametrize(
    ("short", "vibes", "expected"),
    [
        (None, None, None),
        ("", "  ", None),
        ("short description", None, "short description"),
        (None, "quiet mood", "quiet mood"),
        (
            " short description ",
            " quiet mood ",
            "short description\n\nquiet mood",
        ),
    ],
)
def test_governed_modelscope_parser_combines_short_and_vibes(
    short: object, vibes: object, expected: str | None
) -> None:
    raw = _real_row()
    multicaptions = raw["multicaptions"]
    assert isinstance(multicaptions, dict)
    multicaptions["short"] = short
    multicaptions["vibes"] = vibes

    fields = parse_modelscope_caption_fields(raw)

    assert fields.nl.short_vibes == expected


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
