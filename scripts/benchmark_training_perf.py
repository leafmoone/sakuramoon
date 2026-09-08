#!/usr/bin/env python3
"""SakuraMoon P0 training-performance baseline runner.

Runs ONE canonical logical-update benchmark (synthetic in-memory data,
current production components) and writes JSON/Markdown artifacts under a
scratch output root.  Never touches production data services, checkpoints,
W&B, publishers, or evaluation.

Canonical timing baseline:

    warmup 5 updates (excluded) + measure 10 updates (reported)

The measured stage ALWAYS runs without a torch.profiler context.  The
profiler (``--profiler``) is a SEPARATE scratch stage AFTER the baseline,
capturing exactly ``--profiler-updates`` logical updates (default 1); it
does not append to the measured baseline.

Launch modes
------------
1 GPU (rank-local compute baseline):

    CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_training_perf.py \
        --config train_g1_cmuon_production.toml \
        --repository-root <repo> --output-root <scratch>/<sha>/<ts> \
        --gpus 1

2 GPU (real DDP; the script must be launched by accelerate):

    accelerate launch --multi_gpu --num_processes 2 --num_machines 1 \
        --mixed_precision no --dynamo_backend no \
        --main_process_port 29511 \
        scripts/benchmark_training_perf.py \
        --config train_g1_cmuon_production.toml \
        --repository-root <repo> --output-root <scratch>/<sha>/<ts> \
        --gpus 2

Environment requirements (DTK machine): ``source /opt/dtk/env.sh``,
``OMP_NUM_THREADS=32 MKL_NUM_THREADS=32``, and ``FA_SO_PATH`` pointing at
the DAS flash-attn 2 shared object (packed DiT attention).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

PLACEHOLDER_ENVIRONMENT = {
    "MODELSCOPE_API_TOKEN": "benchmark-placeholder-no-network",
    "WANDB_API_KEY": "benchmark-placeholder-no-network",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train_g1_cmuon_production.toml")
    parser.add_argument("--config-root", default="config")
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--gpus", type=int, choices=(1, 2), default=2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measure-updates", type=int, default=10)
    parser.add_argument("--profiler", action="store_true")
    parser.add_argument(
        "--profiler-updates",
        type=int,
        default=1,
        help="logical updates captured by the SEPARATE post-baseline "
        "profiler scratch stage (default 1; 0 is rejected)",
    )
    parser.add_argument("--label", default="")
    return parser.parse_args(argv)


def _stat_row(name: str, block: dict[str, float]) -> list[str]:
    return [
        (
            f"| {name} | {block['mean']:.3f} | {block['p50']:.3f} "
            f"| {block['p90']:.3f} | {block['p95']:.3f} "
            f"| {block['min']:.3f} | {block['max']:.3f} | {block['stddev']:.3f} |"
        )
    ]


def _render_markdown(
    label: str,
    summary: dict,
    fingerprint: dict,
    config_identity: dict,
    snapshot_markdown: str | None,
    limitations: list[str],
) -> str:
    ranks = summary["per_rank_step_seconds"]
    lines = [
        f"# {label} — logical-update baseline",
        "",
        f"Generated: {datetime.now(UTC).isoformat()}",
        "",
        "## Identity",
        "",
        f"- git sha: `{fingerprint['git_sha']}`",
        f"- hostname: {fingerprint['hostname']}",
        (
            f"- devices: {fingerprint['device_name']} x {fingerprint['device_count']} "
            f"({fingerprint.get('device_total_memory_bytes')} bytes each)"
        ),
        f"- torch: {fingerprint['torch_version']} (HIP {fingerprint.get('torch_hip_version')})",
        f"- dtk: {fingerprint['dtk_identity']}",
        f"- python: {fingerprint['python_version']}",
        (
            f"- transformers: {fingerprint.get('transformers_version')} | "
            f"flash_attn: {fingerprint.get('flash_attn_version')} | "
            f"fla: {fingerprint.get('fla_version')} | triton: {fingerprint.get('triton_version')}"
        ),
        "",
        "## Config identity",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in config_identity.items()]
    lines += [
        "",
        "## GLOBAL STEP — per-update slowest-rank wall distribution",
        "",
        "All ranks aligned by logical-update id; per-update global wall = the",
        "slowest rank (the real distributed logical-update critical path).",
        "",
        "| stat | mean | p50 | p90 | p95 | min | max | stddev |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    lines += _stat_row("global", summary["global_step_seconds"])
    lines += [
        "",
        "## THROUGHPUT — summed cross-rank volumes / summed aligned global walls",
        "",
    ]
    lines += [
        f"- global samples/s: {summary['global_samples_per_second']:.2f}",
        f"- image tokens/s: {summary['image_tokens_per_second']:.2f}",
        f"- text tokens/s: {summary['text_tokens_per_second']:.2f}",
        (
            "Volumes are the ACTUAL per-update sums across ranks (rank token "
            "counts are not assumed equal); denominator is the sum of the "
            "aligned per-update global walls."
        ),
        "",
        "## PER-RANK STEP — rank-local distributions",
        "",
        "| stat | " + " | ".join(f"rank {r}" for r in ranks) + " |",
        "| --- | " + " | ".join(" --- " for _ in ranks) + " |",
    ]
    for key in ("mean", "p50", "p95", "stddev"):
        lines.append(
            f"| {key} | " + " | ".join(f"{ranks[r][key]:.3f}" for r in ranks) + " |"
        )
    lines += [
        "",
        f"- rank step skew: {summary.get('rank_step_skew_pct', 0.0):.6f} %",
        "",
    ]

    lines += [
        "## MEMORY — true per-update peak counters (measured window)",
        "",
        "Peak counters are the max over measured updates of the per-update",
        "``torch.cuda.max_memory_*`` (each update's peak window is reset",
        "immediately before that update; no extra synchronization).  Final",
        "counters are the current post-update values of the last measured",
        "update — they are NOT peaks.",
        "",
        (
            "| rank | peak allocated (GiB) | peak reserved (GiB) "
            "| final allocated (GiB) | final reserved (GiB) |"
        ),
        "| --- | --- | --- | --- | --- |",
    ]
    gib = 2**30
    memory = summary["memory"]
    for rank in sorted(memory["per_rank"]):
        row = memory["per_rank"][rank]
        lines.append(
            f"| {rank} | {row['peak_allocated_bytes'] / gib:.2f} "
            f"| {row['peak_reserved_bytes'] / gib:.2f} "
            f"| {row['final_allocated_bytes'] / gib:.2f} "
            f"| {row['final_reserved_bytes'] / gib:.2f} |"
        )
    max_row = memory["max_across_ranks"]
    lines += [
        "",
        f"- max across ranks — peak allocated: {max_row['peak_allocated_bytes'] / gib:.2f} GiB",
        f"- max across ranks — peak reserved: {max_row['peak_reserved_bytes'] / gib:.2f} GiB",
        "",
        "## Phase seconds (pooled rank-local component statistics)",
        "",
        "Phase means are pooled rank-local component costs.  A phase's",
        "``share_of_step`` uses the MEAN aligned global logical-update wall as",
        "the denominator; phase means do NOT decompose one exact global",
        "critical path when overlap exists.",
        "",
        "| phase | mean | p50 | p95 | share of mean global step |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in sorted(summary["phase_seconds"]):
        block = summary["phase_seconds"][name]
        share = (
            f"{block['share_of_step']:.3f}"
            if block["share_of_step"] is not None
            else "n/a"
        )
        lines.append(
            f"| {name} | {block['mean']:.3f} | {block['p50']:.3f} "
            f"| {block['p95']:.3f} | {share} |"
        )
    dit = summary["dit"]
    lines += [
        "",
        "## DIT TFLOP/S — DiT forward matmul only (NOT MFU)",
        "",
        "| rank | DiT fwd matmul TFLOP/s | DiT forward seconds (window) |",
        "| --- | --- | --- |",
    ]
    for rank in sorted(dit["per_rank"]):
        row = dit["per_rank"][rank]
        lines.append(
            f"| {rank} | {row['dit_forward_matmul_tflops_per_second']:.1f} "
            f"| {row['dit_forward_seconds']:.3f} |"
        )
    if snapshot_markdown is not None:
        lines += ["", "## PROFILER — separate post-baseline capture", ""]
        lines.append(snapshot_markdown)
    lines += ["", "## Limitations", ""]
    lines += [f"- {item}" for item in limitations]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    for key, value in PLACEHOLDER_ENVIRONMENT.items():
        os.environ.setdefault(key, value)

    import torch

    repository_root = Path(args.repository_root).resolve()
    config_root = (
        Path(args.config_root)
        if Path(args.config_root).is_absolute()
        else repository_root / Path(args.config_root)
    )
    output_root = Path(args.output_root).resolve()

    from sakuramoon.config import load_config
    from sakuramoon.perf.fingerprint import collect_fingerprint
    from sakuramoon.perf.harness import (
        assemble_benchmark,
        git_head_sha,
        load_rank_samples,
        run_benchmark_stage,
        write_rank_samples,
    )
    from sakuramoon.perf.profiler import (
        capture_profiler_window,
        profiler_device_trace_available,
        render_snapshot_markdown,
        unavailable_snapshot,
        validate_profiler_updates,
        write_snapshot_json,
    )
    from sakuramoon.perf.summary import rank_step_skew_pct, summarize
    from sakuramoon.train.step import SingleGpuUpdateState

    try:
        profiler_updates = validate_profiler_updates(args.profiler_updates)
    except ValueError as exc:
        print(f"[bench] invalid --profiler-updates: {exc}", file=sys.stderr)
        return 2

    config_path = (
        Path(args.config)
        if Path(args.config).is_absolute()
        else config_root / args.config
    )
    try:
        loaded = load_config(config_path=config_path, config_root=config_root)
    except Exception as exc:  # noqa: BLE001 - a clean operator error is required here
        print(f"[bench] config load failed: {exc}", file=sys.stderr)
        return 2

    if args.gpus == 2 and os.environ.get("LOCAL_RANK") is None:
        print(
            "[bench] --gpus 2 requires an accelerate launch (LOCAL_RANK unset)",
            file=sys.stderr,
        )
        return 2

    label = args.label or ("baseline" + ("-" + args.config.replace(".toml", "")))
    print(
        f"[bench] {label}: git {git_head_sha(repository_root)} | "
        f"gpus={args.gpus} seed={args.seed} warmup={args.warmup_updates} "
        f"measure={args.measure_updates} "
        f"profiler={'on(' + str(profiler_updates) + ' update(s), separate stage)' if args.profiler else 'off'}",
        flush=True,
    )

    try:
        assembly = assemble_benchmark(
            repository_root=repository_root,
            config=loaded,
            seed=args.seed,
            warmup_updates=args.warmup_updates,
            measure_updates=args.measure_updates,
            single_rank=(args.gpus == 1),
        )
    except Exception as exc:  # noqa: BLE001 - a clean operator error is required here
        print(f"[bench] assembly failed: {exc}", file=sys.stderr)
        return 2

    fingerprint = collect_fingerprint(
        repository_root=repository_root,
        git_sha=git_head_sha(repository_root),
        config=assembly.config,
        device_index=int(assembly.device.index or 0),
    )

    world = assembly.world_size
    rank = assembly.rank
    device = assembly.device
    output_root.mkdir(parents=True, exist_ok=True)
    if rank == 0:
        (output_root / "runtime-fingerprint.json").write_text(
            json.dumps(
                fingerprint.to_dict(), ensure_ascii=True, indent=2, sort_keys=True
            )
            + "\n",
            encoding="utf-8",
        )

    diagnostic_root = output_root / f"diagnostics-rank{rank}"
    diagnostic_root.mkdir(parents=True, exist_ok=True)

    # Stage 1: warmup (load/compile/lazy state).  No per-update peak reset:
    # warmup memory must not enter measured peaks.
    warm_state = run_benchmark_stage(
        assembly,
        state=SingleGpuUpdateState.initial(),
        target_successful_updates=args.warmup_updates,
        diagnostic_root=diagnostic_root,
    )

    # Stage 2: the canonical measured baseline.  NEVER under a profiler
    # context; per-update peak-memory reset makes each sample's peak the
    # true peak of exactly that logical update.
    measured: list = []
    print(
        f"[bench rank{rank}] measuring {args.measure_updates} logical updates",
        flush=True,
    )
    final_state = run_benchmark_stage(
        assembly,
        state=warm_state,
        target_successful_updates=args.warmup_updates + args.measure_updates,
        diagnostic_root=diagnostic_root,
        collect=measured,
        reset_peak_memory_per_update=True,
    )
    assert final_state.successful_updates == (
        args.warmup_updates + args.measure_updates
    )
    if len(measured) != args.measure_updates:
        raise RuntimeError(
            f"expected {args.measure_updates} measured samples, got {len(measured)}"
        )

    write_rank_samples(measured, output_root / f"samples-{label}-rank{rank}.json")

    snapshot_markdown: str | None = None
    if world > 1:
        import torch.distributed as dist

        dist.barrier()
    if rank == 0:
        all_samples: dict[int, list] = {}
        for r in range(world):
            all_samples[r] = load_rank_samples(
                output_root / f"samples-{label}-rank{r}.json"
            )
        summary = summarize(
            all_samples, warmup_iterations=args.warmup_updates, world_size=world
        )
        summary_payload = summary.to_dict()
        summary_payload["rank_step_skew_pct"] = rank_step_skew_pct(
            {
                r: summary.per_rank_step_seconds[r]["mean"]
                for r in summary.per_rank_step_seconds
            }
        )
        config_identity = {
            "model": "canonical G1 (train_g1_cmuon_production.toml)",
            "resolution": assembly.config.train.resolution,
            "local_batch": assembly.config.train.local_batch,
            "accumulation": assembly.config.train.accumulation,
            "global_batch": (
                assembly.config.train.local_batch
                * assembly.config.train.accumulation
                * world
            ),
            "world_size": world,
            "optimizer": "hybrid_cmuon (production assembly)",
            "growth_alpha": assembly.runtime.growth_alpha,
            "overlap_claim": "NOT_EVALUATED_IN_P0",
            "data": "synthetic in-memory (data phase ~0 by design)",
        }
        (output_root / f"{label}.json").write_text(
            json.dumps(summary_payload, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
    else:
        summary_payload = None

    # Stage 3 (opt-in): the SEPARATE post-baseline profiler scratch stage.
    # It runs AFTER the measured baseline, captures exactly
    # profiler_updates logical updates, appends nothing to the measured
    # samples, and advances scratch model/optimizer state only.
    if args.profiler:
        available = profiler_device_trace_available()
        print(
            f"[bench rank{rank}] profiler tiny device-event probe: "
            f"available={available}",
            flush=True,
        )
        all_ranks_available = available
        if world > 1:
            import torch.distributed as dist

            flag = torch.tensor([1 if available else 0], device=device)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            all_ranks_available = bool(flag.item() == 1)
            if all_ranks_available != available:
                print(
                    f"[bench rank{rank}] profiler all-rank consensus overrides "
                    f"local probe: available={all_ranks_available}",
                    flush=True,
                )
        if all_ranks_available:
            print(
                f"[bench rank{rank}] profiler scratch stage: capturing "
                f"{profiler_updates} logical update(s) (separate from baseline)",
                flush=True,
            )
            snapshot_box: list = []

            def wrap_scratch(workload) -> None:
                snapshot_box.append(
                    capture_profiler_window(
                        workload,
                        output_root=output_root,
                        rank=rank,
                        export_trace=(rank == 0),
                        profiler_updates=profiler_updates,
                    )
                )

            run_benchmark_stage(
                assembly,
                state=final_state,
                target_successful_updates=final_state.successful_updates
                + profiler_updates,
                diagnostic_root=diagnostic_root,
                wrap_workload=wrap_scratch,
            )
            if world > 1:
                import torch.distributed as dist

                dist.barrier()
            if rank == 0 and snapshot_box:
                snapshot = snapshot_box[0]
                write_snapshot_json(snapshot, output_root / "profiler-summary.json")
                snapshot_markdown = render_snapshot_markdown(snapshot)
        else:
            # No extra workload anywhere: an ungated capture in this state
            # hung the DTK runtime; emit the honest UNAVAILABLE record.
            if rank == 0 and summary_payload is not None:
                write_snapshot_json(
                    unavailable_snapshot(profiler_updates),
                    output_root / "profiler-summary.json",
                )
                snapshot_markdown = render_snapshot_markdown(
                    unavailable_snapshot(profiler_updates)
                )

    if rank == 0 and summary_payload is not None:
        limitations = [
            (
                "synthetic in-memory data: the `data` phase is an in-memory "
                "handoff (~0 s); production shard download/decode/prefetch is "
                "not measured"
            ),
            (
                "1-GPU mode = rank-local compute baseline (global batch "
                "local*accumulation); not the same effective batch as the 2-GPU run"
            ),
            ("growth_alpha held at steady-state 1.0 (mature-run benchmark assumption)"),
            ("DDP overlap not characterized: overlap_claim = NOT_EVALUATED_IN_P0"),
            "no committed artifacts; scratch output root only",
        ]
        if args.profiler:
            limitations.append(
                "profiler = separate post-baseline scratch stage "
                f"({profiler_updates} logical update(s)); it never wraps the "
                "measured baseline; on stacks where the tiny device-event "
                "probe yields no real device event, no extra workload is "
                "executed and the record is device_trace_available=false"
            )
        else:
            limitations.append(
                "profiler off for this run (opt-in via --profiler; separate "
                "post-baseline stage, default 1 update)"
            )
        md = _render_markdown(
            label,
            summary_payload,
            fingerprint.to_dict(),
            config_identity,
            snapshot_markdown,
            limitations,
        )
        (output_root / f"{label}.md").write_text(md, encoding="utf-8")
        print(
            f"[bench rank0] {label} done: step mean "
            f"{summary_payload['global_step_seconds']['mean']:.3f} s | "
            f"global {summary_payload['global_samples_per_second']:.2f} samples/s | "
            f"artifacts in {output_root}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
