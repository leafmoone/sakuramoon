from __future__ import annotations

import argparse
import math
import statistics
import time
from pathlib import Path

import torch

from sakuramoon.encoders.qwen import QWEN_DENSE_LENGTHS, load_local_qwen

_TRAINING_PROFILE = (173, 194, 240, 188, 124, 61, 30, 14)


def _profile_lengths(batch: int) -> tuple[int, ...]:
    total = sum(_TRAINING_PROFILE)
    exact = [count * batch / total for count in _TRAINING_PROFILE]
    allocated = [math.floor(value) for value in exact]
    missing = batch - sum(allocated)
    order = sorted(
        range(len(exact)),
        key=lambda index: exact[index] - allocated[index],
        reverse=True,
    )
    for index in order[:missing]:
        allocated[index] += 1
    return tuple(
        length
        for length, count in zip(
            QWEN_DENSE_LENGTHS, allocated, strict=True
        )
        for _ in range(count)
    )


def _inputs(
    dense_lengths: tuple[int, ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(dense_lengths)
    rows = len(dense_lengths)
    input_ids = (
        torch.arange(rows * maximum, device=device, dtype=torch.long)
        .reshape(rows, maximum)
        .remainder_(200_000)
    )
    attention_mask = torch.zeros(
        rows, maximum, device=device, dtype=torch.bool
    )
    for row, length in enumerate(dense_lengths):
        attention_mask[row, : max(1, length - 8)] = True
    return input_ids, attention_mask


def _timed(
    operation: object,
    *,
    repeats: int,
) -> tuple[list[float], float]:
    if not callable(operation):
        raise TypeError("operation must be callable")
    durations: list[float] = []
    peak_gib = 0.0
    for _ in range(repeats):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        started = time.perf_counter()
        output = operation()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - started)
        peak_gib = max(peak_gib, torch.cuda.max_memory_allocated() / 2**30)
        del output
    return durations, peak_gib


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.batch <= 0 or args.repeats <= 0:
        raise ValueError("batch and repeats must be positive")

    device = torch.device("cuda:0")
    qwen = load_local_qwen(args.root.resolve(), device).encoder

    correctness_lengths = QWEN_DENSE_LENGTHS
    small_ids, small_mask = _inputs(correctness_lengths, device)
    baseline = qwen(small_ids, small_mask).hidden_states
    grouped = qwen(
        small_ids,
        small_mask,
        dense_lengths=correctness_lengths,
        dense_group_size=2,
    ).hidden_states
    differences = []
    for row, _length in enumerate(correctness_lengths):
        differences.append(
            float(
                (baseline[row, small_mask[row]] - grouped[row, small_mask[row]])
                .abs()
                .max()
                .item()
            )
        )
    print("correctness_max_abs", max(differences), flush=True)
    del baseline, grouped, small_ids, small_mask
    torch.cuda.empty_cache()

    lengths = _profile_lengths(args.batch)
    input_ids, attention_mask = _inputs(lengths, device)

    plans: dict[str, int | None] = {
        "ungrouped": None,
        "sorted_chunks_64": 64,
        "sorted_chunks_32": 32,
    }

    # Compile/warm every shape before measured runs.
    for group_size in plans.values():
        output = (
            qwen(input_ids, attention_mask)
            if group_size is None
            else qwen(
                input_ids,
                attention_mask,
                dense_lengths=lengths,
                dense_group_size=group_size,
            )
        )
        del output
    torch.cuda.synchronize()

    measurements: dict[str, tuple[list[float], float]] = {}
    for name, group_size in plans.items():
        measurements[name] = _timed(
            (
                (lambda: qwen(input_ids, attention_mask))
                if group_size is None
                else (
                    lambda group_size=group_size: qwen(
                        input_ids,
                        attention_mask,
                        dense_lengths=lengths,
                        dense_group_size=group_size,
                    )
                )
            ),
            repeats=args.repeats,
        )
    baseline_mean = statistics.mean(measurements["ungrouped"][0])
    print("batch", args.batch, flush=True)
    print("dense_lengths", QWEN_DENSE_LENGTHS, flush=True)
    print(
        "profile_counts",
        {length: lengths.count(length) for length in QWEN_DENSE_LENGTHS},
        flush=True,
    )
    for name, (durations, peak_gib) in measurements.items():
        mean = statistics.mean(durations)
        print(f"{name}_seconds", durations, flush=True)
        print(f"{name}_mean_seconds", mean, flush=True)
        print(f"{name}_speedup", baseline_mean / mean, flush=True)
        print(f"{name}_peak_allocated_gib", peak_gib, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
