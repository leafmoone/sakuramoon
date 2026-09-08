#!/usr/bin/env python3
"""SakuraMoon P0 training-performance baseline runner.

Runs ONE canonical logical-update benchmark (synthetic in-memory data,
current production components) and writes JSON/Markdown artifacts under a
scratch output root.  Never touches production data services, checkpoints,
W&B, publishers, or evaluation.

Canonical timing baseline:

    warmup 5 updates (excluded) + measure 10 updates (reported)

The measured stage ALWAYS runs without a torch.profiler context.  The
profiler (``--profiler``) is a SEPARATE scratch stage that starts AFTER
the measured baseline and captures exactly ``--profiler-updates`` logical
updates (default 1; 0 is rejected); it never appends to the measured
baseline and may advance scratch model/optimizer state only.

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
        (
            f"Warmup updates: {summary['warmup_iterations']} (excluded) | "
            f"Measured updates: {summary['measured_iterations']} | "
            f"world_size: {summary['world_size']}"
        ),
        "",
        "## GLOBAL STEP — per-update slowest-rank wall distribution",
        "",
        "All ranks aligned by logical-update id (missing/duplicate/mismatched",
        "identities fail closed); per-update global wall = the SLOWEST rank,",
        "the real distributed logical-update critical path.  This is NOT a",
        "pooled all-rank distribution and NOT max-of-means.",
        "",
        "| stat | mean | p50 | p90 | p95 | min | max | stddev |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    global_block = summary["global_step_seconds"]
    lines.append(
        f"| global | {global_block['mean']:.3f} | {global_block['p50']:.3f} "
        f"| {global_block['p90']:.3f} | {global_block['p95']:.3f} "
        f"| {global_block['min']:.3f} | {global_block['max']:.3f} "
        f"| {global_block['stddev']:.3f} |"
    )
    lines += [
        "",
        "## PER-RANK STEP — rank-local distributions (seconds)",
        "",
        "| rank | mean | p50 | p90 | p95 | min | max | stddev |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for rank in sorted(ranks):
        block = ranks[rank]
        lines.append(
            f"| {rank} | {block['mean']:.3f} | {block['p50']:.3f} | {block['p90']:.3f} "
            f"| {block['p95']:.3f} | {block['min']:.3f} | {block['max']:.3f} "
            f"| {block['stddev']:.3f} |"
        )
    lines += [
        "",
        "## THROUGHPUT — summed cross-rank volumes / summed aligned global walls",
        "",
        f"- global samples/s: {summary['global_samples_per_second']:.2f}",
        f"- image tokens/s: {summary['image_tokens_per_second']:.0f}",
        f"- text tokens/s: {summary['text_tokens_per_second']:.0f}",
        (
            "Volumes are the ACTUAL per-update sums across ranks (rank token "
            "counts are not assumed equal); the denominator is the sum of "
            "the aligned per-update global walls."
        ),
        (
            f"- DiT forward matmul: "
            f"{summary['dit']['max_across_ranks_tflops_per_second']:.2f} TFLOP/s "
            "(DiT FORWARD matmul FLOP counter / dit_forward phase seconds — NOT MFU)"
        ),
        "",
        "## Phase seconds (pooled rank-local component statistics)",
        "",
        "Phase means are pooled rank-local component costs; share_of_step",
        "uses the MEAN aligned global logical-update wall as its",
        "denominator.  Phase means do NOT decompose one exact global",
        "critical path when overlap exists.",
        "",
        "| phase | mean s | p50 s | p95 s | share |",
        "| --- | --- | --- | --- | --- |",
    ]
    for name in sorted(summary["phase_seconds"]):
        block = summary["phase_seconds"][name]
        share = block["share_of_step"]
        lines.append(
            f"| {name} | {block['mean']:.3f} | {block['p50']:.3f} | {block['p95']:.3f} "
            f"| {share:.1%} |"
            if share is not None
            else f"| {name} | {block['mean']:.3f} | {block['p50']:.3f} | {block['p95']:.3f} | n/a |"
        )
    lines += [
        "",
        "## MEMORY — true per-update peak counters (GiB)",
        "",
        "peak = the max over measured updates of the per-update",
        "``torch.cuda.max_memory_*`` counters; each measured update's peak",
        "window is reset immediately before that update (no extra",
        "synchronization), so each sample's peak is the peak of exactly",
        "that logical update and warmup memory never enters measured",
        "peaks.  final = the CURRENT post-update counters of the last",
        "measured update — they are NOT peaks.",
        "",
    ]
    for rank, block in sorted(summary["memory"]["per_rank"].items()):
        lines.append(
            f"- rank {rank}: peak allocated {block['peak_allocated_bytes'] / 2**30:.2f} GiB "
            f"| peak reserved {block['peak_reserved_bytes'] / 2**30:.2f} GiB "
            f"| final allocated {block['final_allocated_bytes'] / 2**30:.2f} GiB "
            f"| final reserved {block['final_reserved_bytes'] / 2**30:.2f} GiB"
        )
    top = summary["memory"]["max_across_ranks"]
    lines.append(
        f"- max across ranks: peak allocated {top['peak_allocated_bytes'] / 2**30:.2f} GiB "
        f"| peak reserved {top['peak_reserved_bytes'] / 2**30:.2f} GiB"
    )
    if snapshot_markdown is not None:
        lines += ["", snapshot_markdown.rstrip(), ""]
    lines += ["## Limitations", ""]
    lines += [f"- {item}" for item in limitations]
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    # Single-GPU mode: expose exactly GPU 0 before any CUDA use (the DTK
    # runtime caches the visible-device list at first CUDA call).
    if args.gpus == 1:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sys.path.insert(0, str(args.repository_root / "src"))

    import torch

    from sakuramoon.perf.fingerprint import (
        capture_runtime_fingerprint,
    )
    from sakuramoon.perf.harness import (
        assemble_benchmark,
        git_head_sha,
        load_rank_samples,
        run_benchmark_stage,
        write_rank_samples,
    )
    from sakuramoon.perf.profiler import (
        capture_profiler_window,
        probe_consensus_decision,
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

    if args.gpus == 2 and os.environ.get("LOCAL_RANK") is None:
        print(
            "2-GPU mode requires an accelerate/torchrun launcher, e.g.\n"
            "  accelerate launch --multi_gpu --num_processes 2 --num_machines 1 "
            "--mixed_precision no --dynamo_backend no --main_process_port 29511 "
            f"{Path(sys.argv[0]).name} ...",
            file=sys.stderr,
        )
        return 2
    if args.gpus == 1 and torch.cuda.device_count() != 1:
        print(
            f"1-GPU mode expects exactly one visible device, "
            f"found {torch.cuda.device_count()}",
            file=sys.stderr,
        )
        return 2

    label = args.label or f"baseline-{args.gpus}gpu"
    output_root: Path = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    assembly = assemble_benchmark(
        config_path=Path(args.config),
        config_root=Path(args.config_root)
        if Path(args.config_root).is_absolute()
        else args.repository_root / args.config_root,
        repository_root=args.repository_root,
        environment=PLACEHOLDER_ENVIRONMENT,
        seed=args.seed,
        single_rank=(args.gpus == 1),
        warmup_updates=args.warmup_updates,
        measure_updates=args.measure_updates,
    )
    rank = assembly.rank
    world = assembly.world_size
    device = assembly.device
    config = assembly.config

    fingerprint = capture_runtime_fingerprint(
        git_sha=git_head_sha(args.repository_root),
        distributed_backend=config.distributed.backend,
        world_size=world,
        device_index=int(device.index or 0),
    )

    diagnostic_root = output_root / f"diagnostics-rank{rank}"
    diagnostic_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[bench rank{rank}] warmup {args.warmup_updates} updates "
        f"(model load/compile/lazy state excluded)",
        flush=True,
    )
    warm_state = run_benchmark_stage(
        assembly,
        state=SingleGpuUpdateState.initial(),
        target_successful_updates=args.warmup_updates,
        diagnostic_root=diagnostic_root,
    )

    # The measured stage NEVER runs under a profiler context.  Per-update
    # peak-memory reset makes each sample's peak the TRUE peak of exactly
    # that logical update (no one-shot window reset, no extra sync).
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

    if world > 1:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

    # SEPARATE post-baseline profiler scratch stage (opt-in).  It runs
    # AFTER the measured baseline, captures exactly profiler_updates
    # logical updates, appends nothing to the measured samples, and may
    # advance scratch model/optimizer state.
    #
    # Distributed liveness contract: each rank performs EXACTLY ONE tiny
    # local probe; the probes are reduced to ONE canonical all-rank
    # decision (all_reduce(MIN) == AND of the local flags).  That
    # decision is FINAL: if true, EVERY rank enters the profiler
    # workload; if false, EVERY rank skips it.  Capture is invoked with
    # availability_verified=True and must NOT probe again — a second
    # per-rank probe could transiently disagree and let one rank skip
    # the DDP workload while another enters it (collective hang).  An
    # ungated capture in the unavailable state was observed to hang the
    # DTK runtime (CPU spin, device idle).
    all_ranks_available = False
    scratch_snapshot_box: list = []
    if args.profiler:
        available = profiler_device_trace_available()
        print(
            f"[bench rank{rank}] profiler tiny device-event probe: "
            f"available={available}",
            flush=True,
        )
        if world > 1:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                flag = torch.tensor([1 if available else 0], device=device)
                dist.all_reduce(flag, op=dist.ReduceOp.MIN)
                all_ranks_available = bool(flag.item() == 1)
            else:
                all_ranks_available = probe_consensus_decision([available])
            if all_ranks_available != available:
                print(
                    f"[bench rank{rank}] profiler all-rank consensus "
                    f"overrides local probe: available={all_ranks_available}",
                    flush=True,
                )
        else:
            all_ranks_available = probe_consensus_decision([available])
        if all_ranks_available:
            print(
                f"[bench rank{rank}] profiler scratch stage: capturing "
                f"{profiler_updates} logical update(s) (separate from the "
                f"measured baseline)",
                flush=True,
            )

            def wrap_scratch(workload) -> None:
                # The all-rank decision above is FINAL; capture must not
                # probe again (availability_verified=True).
                scratch_snapshot_box.append(
                    capture_profiler_window(
                        workload,
                        output_root=output_root,
                        rank=rank,
                        export_trace=(rank == 0),
                        profiler_updates=profiler_updates,
                        availability_verified=True,
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

                if dist.is_available() and dist.is_initialized():
                    dist.barrier()

    # Rank 0 aggregates all ranks.
    if rank == 0:
        rank_samples = {
            r: load_rank_samples(output_root / f"samples-{label}-rank{r}.json")
            for r in range(world)
        }
        summary = summarize(
            rank_samples,
            warmup_iterations=args.warmup_updates,
            world_size=world,
        )
        summary_payload = summary.to_dict()
        rank_means = {
            int(r): sum(s.wall_seconds for s in samples) / len(samples)
            for r, samples in rank_samples.items()
        }
        summary_payload["rank_step_skew_pct"] = rank_step_skew_pct(rank_means)
        summary_path = output_root / f"{label}.json"
        summary_path.write_text(
            json.dumps(summary_payload, ensure_ascii=True, indent=2, sort_keys=True)
            + "\n",
            encoding="utf-8",
        )
        fingerprint_path = output_root / "runtime-fingerprint.json"
        fingerprint.write_json(fingerprint_path)

        config_identity = {
            "config": args.config,
            "run_id": config.run.run_id,
            "seed": args.seed,
            "resolution": config.train.resolution,
            "local_batch": config.train.local_batch,
            "accumulation": config.train.accumulation,
            "global_batch": config.train.global_batch,
            "world_size": world,
            "optimizer": config.optimizer.name,
            "attention_backend": getattr(config.kernels, "attention_backend", None),
            "torch_compile": config.kernels.torch_compile_enabled,
            "torch_compile_mode": config.kernels.torch_compile_mode,
        }
        limitations = [
            (
                "Synthetic in-memory data: the data phase measures in-memory batch "
                "handoff (~0 s), not production shard download/decode/prefetch."
            ),
            "growth_alpha held at 1.0 (ramp-complete steady state).",
            (
                "1-GPU mode is a RANK-LOCAL COMPUTE BASELINE: same local batch and "
                "accumulation, but the effective global training batch differs from "
                "the 2-GPU run."
            ),
            (
                "DiT TFLOP/s uses the exact DiT forward matmul FLOP counter over "
                "the dit_forward phase only; it is not MFU and not whole-step FLOPS."
            ),
            (
                "Text-length mix is deterministic bucket coverage, not the exact "
                "production caption-length distribution."
            ),
        ]
        snapshot_md = None
        if args.profiler:
            if all_ranks_available and scratch_snapshot_box:
                snapshot = scratch_snapshot_box[0]
                write_snapshot_json(snapshot, output_root / "profiler-summary.json")
                limitations.append(
                    f"Profiler = SEPARATE post-baseline scratch stage: "
                    f"{profiler_updates} logical update(s) captured after the "
                    f"measured baseline; it never wrapped the measured window. "
                    f"Device trace available: {snapshot.device_trace_available}. "
                    f"Trace: {snapshot.trace_path or 'not exported'}."
                )
                snapshot_md = render_snapshot_markdown(snapshot)
            else:
                unavailable = unavailable_snapshot(profiler_updates)
                write_snapshot_json(unavailable, output_root / "profiler-summary.json")
                limitations.append(
                    "Profiler device trace UNAVAILABLE: the tiny device-event "
                    "probe yielded no real device event, so NO extra workload "
                    "was executed (an ungated capture in that state hangs the "
                    "DTK runtime); no kernel rows are invented."
                )
                snapshot_md = render_snapshot_markdown(unavailable)
        md = _render_markdown(
            label,
            summary_payload,
            fingerprint.to_dict(),
            config_identity,
            snapshot_md,
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
