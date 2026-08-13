from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sakuramoon.train.benchmark import (
    BENCHMARK_TRACE_DIRECTORY_ENV,
    BenchmarkTrace,
)


def test_benchmark_trace_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(BENCHMARK_TRACE_DIRECTORY_ENV, raising=False)

    assert BenchmarkTrace.from_environment(0) is None


def test_benchmark_trace_is_exclusive_and_records_exact_samples(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(BENCHMARK_TRACE_DIRECTORY_ENV, str(tmp_path))
    trace = BenchmarkTrace.from_environment(1)

    assert trace is not None
    assert trace.path == tmp_path / "rank-1.jsonl"
    with pytest.raises(FileExistsError):
        BenchmarkTrace.from_environment(1)

    measurement = SimpleNamespace(
        sample_ids=("17", "23"),
        shape_keys=("256x256x64", "224x288x80"),
        image_tokens=512,
        text_tokens=144,
        dit_flops=987654321,
        captions=(
            SimpleNamespace(text="caption a"),
            SimpleNamespace(text="caption b"),
        ),
    )
    observation = SimpleNamespace(
        loop=SimpleNamespace(
            update=SimpleNamespace(
                state=SimpleNamespace(successful_updates=22001),
            ),
            update_wall_seconds=12.5,
        ),
        microbatches=(measurement,),
    )
    trace.append(observation)

    payload = json.loads(trace.path.read_text())
    assert payload == {
        "microbatches": [
            {
                "caption_sha256": [
                    hashlib.sha256(b"caption a").hexdigest(),
                    hashlib.sha256(b"caption b").hexdigest(),
                ],
                "dit_flops": 987654321,
                "image_tokens": 512,
                "sample_ids": ["17", "23"],
                "shape_keys": ["256x256x64", "224x288x80"],
                "text_tokens": 144,
            }
        ],
        "rank": 1,
        "schema_version": 1,
        "successful_update": 22001,
        "update_wall_seconds": 12.5,
    }


def test_benchmark_trace_rejects_noncontiguous_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(BENCHMARK_TRACE_DIRECTORY_ENV, str(tmp_path))
    trace = BenchmarkTrace.from_environment(0)
    assert trace is not None
    measurement = SimpleNamespace(
        sample_ids=("1",),
        shape_keys=("256x256x64",),
        image_tokens=256,
        text_tokens=64,
        dit_flops=100,
        captions=(SimpleNamespace(text="caption"),),
    )

    def observation(update: int) -> SimpleNamespace:
        return SimpleNamespace(
            loop=SimpleNamespace(
                update=SimpleNamespace(
                    state=SimpleNamespace(successful_updates=update),
                ),
                update_wall_seconds=1.0,
            ),
            microbatches=(measurement,),
        )

    trace.append(observation(10))
    with pytest.raises(RuntimeError, match="not contiguous"):
        trace.append(observation(12))
