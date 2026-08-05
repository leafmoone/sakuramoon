"""DTK/HCU runtime kernel controls that do not alter checkpoint parameters."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from sakuramoon.storage import repository_directory


@dataclass(frozen=True, slots=True)
class TunableOpState:
    enabled: bool
    tuning: bool
    record_untuned: bool
    results_path: Path | None
    loaded_results: bool


def configure_tunableop(
    repository_root: Path,
    runtime_directory: str,
    *,
    enabled: bool,
    tuning: bool,
    record_untuned: bool,
    max_tuning_duration_ms: int,
) -> TunableOpState:
    """Apply a config-bound TunableOp policy through PyTorch's runtime API."""

    if (
        type(enabled) is not bool
        or type(tuning) is not bool
        or type(record_untuned) is not bool
        or type(max_tuning_duration_ms) is not int
        or not 1 <= max_tuning_duration_ms <= 1000
    ):
        raise ValueError("TunableOp configuration is invalid")
    tunable = getattr(torch.cuda, "tunable", None)
    if tunable is None:
        if enabled:
            raise RuntimeError("configured TunableOp is unavailable in this PyTorch build")
        return TunableOpState(False, False, False, None, False)

    tunable.enable(enabled)
    tunable.tuning_enable(enabled and tuning)
    tunable.record_untuned_enable(enabled and record_untuned)
    if not enabled:
        return TunableOpState(False, False, False, None, False)

    runtime = repository_directory(repository_root, runtime_directory)
    results_path = runtime / "tunableop_results.csv"
    tunable.set_filename(str(results_path), insert_device_ordinal=False)
    tunable.set_max_tuning_duration(max_tuning_duration_ms)
    tunable.write_file_on_exit(True)
    loaded_results = results_path.is_file() and bool(tunable.read_file(str(results_path)))
    observed = (
        bool(tunable.is_enabled()),
        bool(tunable.tuning_is_enabled()),
        bool(tunable.record_untuned_is_enabled()),
    )
    expected = (True, tuning, record_untuned)
    if observed != expected:
        raise RuntimeError(
            "TunableOp state differs from config; check overriding environment variables"
        )
    return TunableOpState(*observed, results_path, loaded_results)


__all__ = ["TunableOpState", "configure_tunableop"]
