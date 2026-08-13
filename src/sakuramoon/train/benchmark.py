"""Fail-closed, opt-in sample identity tracing for performance benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from sakuramoon.train.runtime import SuccessfulTrainingObservation

BENCHMARK_TRACE_DIRECTORY_ENV = "SAKURAMOON_BENCHMARK_TRACE_DIRECTORY"


class BenchmarkTrace:
    """Append exact per-rank update inputs without changing the training path."""

    def __init__(self, path: Path, rank: int) -> None:
        self.path = path
        self.rank = rank
        self._last_update: int | None = None

    @classmethod
    def from_environment(cls, rank: int) -> BenchmarkTrace | None:
        if type(rank) is not int or rank < 0:
            raise ValueError("benchmark trace rank is invalid")
        raw = os.environ.get(BENCHMARK_TRACE_DIRECTORY_ENV)
        if raw is None:
            return None
        if not raw:
            raise ValueError("benchmark trace directory environment is empty")
        directory = Path(raw)
        if not directory.is_absolute() or directory.is_symlink():
            raise ValueError(
                "benchmark trace directory must be an absolute real directory"
            )
        directory = directory.resolve(strict=True)
        if not directory.is_dir():
            raise ValueError("benchmark trace path is not a directory")
        path = directory / f"rank-{rank}.jsonl"
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW,
            0o600,
        )
        os.close(descriptor)
        return cls(path, rank)

    def append(self, observation: SuccessfulTrainingObservation) -> None:
        update = observation.loop.update.state.successful_updates
        if type(update) is not int or update <= 0:
            raise ValueError("benchmark trace update is invalid")
        if self._last_update is not None and update != self._last_update + 1:
            raise RuntimeError("benchmark trace updates are not contiguous")
        for item in observation.microbatches:
            if not (
                len(item.sample_ids)
                == len(item.shape_keys)
                == len(item.captions)
            ):
                raise ValueError(
                    "benchmark trace sample, shape, and caption counts differ"
                )
        payload = {
            "schema_version": 1,
            "rank": self.rank,
            "successful_update": update,
            "update_wall_seconds": observation.loop.update_wall_seconds,
            "microbatches": [
                {
                    "sample_ids": list(item.sample_ids),
                    "shape_keys": list(item.shape_keys),
                    "caption_sha256": [
                        hashlib.sha256(caption.text.encode("utf-8")).hexdigest()
                        for caption in item.captions
                    ],
                    "image_tokens": item.image_tokens,
                    "text_tokens": item.text_tokens,
                    "dit_flops": item.dit_flops,
                }
                for item in observation.microbatches
            ],
        }
        encoded = (
            json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.path,
            os.O_APPEND | os.O_WRONLY | os.O_NOFOLLOW,
        )
        try:
            offset = 0
            while offset < len(encoded):
                written = os.write(descriptor, encoded[offset:])
                if written <= 0:
                    raise OSError("benchmark trace write did not make progress")
                offset += written
        finally:
            os.close(descriptor)
        self._last_update = update


__all__ = ["BENCHMARK_TRACE_DIRECTORY_ENV", "BenchmarkTrace"]
