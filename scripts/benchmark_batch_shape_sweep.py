#!/usr/bin/env python3
"""P1-R1A microbatch / accumulation throughput sweep orchestrator.

Drives the canonical 2-GPU batch-shape sweep:

    A0 = 20 x 20   baseline control
    S  = 16 x 25   smaller-microbatch control
    C1 = 25 x 16   candidate
    C2 = 40 x 10   HIGH (safety-gated from C1's measured peaks)
    A1 = 20 x 20   baseline repeat (A0/A1 bracket -> drift gate)

Isolation rules (P1-R1A boundary):

* Every candidate runs in its OWN fresh process group
  (``python -m accelerate.commands.launch ... benchmark_training_perf.py``
  with ``start_new_session``) — never several shapes inside one Python
  process — so allocator, compile-graph, lazy-optimizer and DTK runtime
  state can never contaminate between shapes.
* Each candidate gets its own scratch root
  ``<output-root>/<shape>/`` (plus ``<shape>.log`` next to it); no
  production artifact directories are ever written.
* Each candidate has a bounded wall-clock timeout; on timeout the whole
  process group is killed and the candidate is recorded as TIMEOUT.
* ``--profiler`` is NEVER passed: R1A measures whole-step batch-shape
  throughput only (the P0 profiler evidence already exists).
* OOM is a valid benchmark result: it is classified from the candidate
  log, recorded with its evidence, and the candidate is NOT retried or
  silently shrunk.

The sweep compares shapes at the SAME effective global batch (the
canonical 800; the per-candidate runner fails closed otherwise) and
writes a machine-readable ``sweep-summary.json`` plus a human
``sweep-summary.md`` under the sweep root.  No production configuration,
model, optimizer, kernel or DDP setting is touched.

Launch (DTK machine, from the repository worktree):

    source /opt/dtk/env.sh
    OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \\
    FA_SO_PATH=<dtk-venv>/.../flash_attn_2_cuda*.so \\
    python scripts/benchmark_batch_shape_sweep.py \\
        --config train_g1_cmuon_production.toml \\
        --config-root config \\
        --repository-root <repo> \\
        --output-root /sakuramoon-runtime/infra-bench/p1-r1a/<sha>/<ts>
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="train_g1_cmuon_production.toml")
    parser.add_argument("--config-root", default="config")
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        required=True,
        type=Path,
        help="sweep root; per-shape scratch roots are created inside it",
    )
    parser.add_argument("--gpus", type=int, choices=(2,), default=2)
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measure-updates", type=int, default=10)
    parser.add_argument("--expected-global-batch", type=int, default=800)
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="per-candidate wall-clock bound in seconds (default 1800)",
    )
    parser.add_argument(
        "--base-port",
        type=int,
        default=29600,
        help="accelerate main_process_port base; candidate i uses base+i",
    )
    parser.add_argument(
        "--skip-c2",
        action="store_true",
        help="force C2 to SKIPPED_SAFETY_GATE without evaluating the gate",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip candidates whose scratch summary JSON already exists "
        "(reused as PASS; the subprocess is not relaunched)",
    )
    return parser.parse_args(argv)


def _git_head_sha(repository_root: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    value = result.stdout.strip()
    return value if len(value) == 40 else "unavailable"


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill the candidate's whole process group (accelerate + rank children)."""

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.SubprocessError:
        proc.kill()


def _tail_lines(log_path: Path, count: int = 15) -> str:
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-count:])


def run_candidate(
    candidate: Any,
    *,
    python: str,
    repository_root: Path,
    config: str,
    config_root: str,
    sweep_root: Path,
    gpus: int,
    seed: int,
    warmup_updates: int,
    measure_updates: int,
    expected_global_batch: int,
    timeout: int,
    port: int,
    resume: bool,
) -> dict[str, Any]:
    """Run one candidate shape in its own fresh process group."""

    from sakuramoon.perf.batch_shape import (
        STATUS_PASS,
        STATUS_TIMEOUT,
        classify_candidate_status,
        oom_marker_found,
        summarize_candidate,
    )

    label = f"baseline-{gpus}gpu"
    shape_root = sweep_root / candidate.dir_name
    shape_root.mkdir(parents=True, exist_ok=True)
    log_path = sweep_root / f"{candidate.dir_name}.log"
    summary_path = shape_root / f"{label}.json"

    entry: dict[str, Any] = {
        "label": candidate.label,
        "dir_name": candidate.dir_name,
        "local_batch": candidate.local_batch,
        "accumulation": candidate.accumulation,
        "output_root": str(shape_root),
        "log": str(log_path),
        "exit_code": None,
        "timed_out": False,
        "resumed": False,
    }

    if resume and summary_path.exists():
        entry["status"] = STATUS_PASS
        entry["resumed"] = True
        entry["exit_code"] = 0
        entry["summary"] = summarize_candidate(
            json.loads(summary_path.read_text(encoding="utf-8"))
        )
        print(f"[sweep] {candidate.dir_name}: resumed (existing summary)", flush=True)
        return entry

    cmd = [
        python,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        str(gpus),
        "--num_machines",
        "1",
        "--mixed_precision",
        "no",
        "--dynamo_backend",
        "no",
        "--main_process_port",
        str(port),
        str(repository_root / "scripts" / "benchmark_training_perf.py"),
        "--config",
        config,
        "--config-root",
        config_root,
        "--repository-root",
        str(repository_root),
        "--output-root",
        str(shape_root),
        "--gpus",
        str(gpus),
        "--label",
        label,
        "--seed",
        str(seed),
        "--warmup-updates",
        str(warmup_updates),
        "--measure-updates",
        str(measure_updates),
        "--local-batch",
        str(candidate.local_batch),
        "--accumulation",
        str(candidate.accumulation),
        "--expected-global-batch",
        str(expected_global_batch),
    ]
    print(
        f"[sweep] starting {candidate.dir_name} (lb={candidate.local_batch} "
        f"acc={candidate.accumulation}) port={port} timeout={timeout}s",
        flush=True,
    )
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            cwd=repository_root,
        )
    timed_out = False
    try:
        exit_code = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_process_group(proc)
        exit_code = -signal.SIGKILL
    entry["exit_code"] = exit_code
    entry["timed_out"] = timed_out

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    if timed_out:
        entry["status"] = STATUS_TIMEOUT
        entry["note"] = (
            f"bounded timeout ({timeout}s) exceeded; the whole candidate "
            "process group was killed"
        )
    else:
        status = classify_candidate_status(exit_code, log_text, summary_path.exists())
        entry["status"] = status
        if status == "OOM":
            marker = oom_marker_found(log_text)
            entry["oom_evidence"] = {
                "marker": marker,
                "exit_code": exit_code,
                "last_log_lines": _tail_lines(log_path),
            }
        elif status == "ERROR":
            entry["note"] = f"exit code {exit_code}; last log lines below"
            entry["last_log_lines"] = _tail_lines(log_path)
        elif status == STATUS_PASS:
            entry["summary"] = summarize_candidate(
                json.loads(summary_path.read_text(encoding="utf-8"))
            )
    extra = ""
    if entry.get("summary"):
        extra = (
            f" (p50 {entry['summary']['p50_s']:.3f} s, "
            f"{entry['summary']['samples_per_s']:.2f} samples/s)"
        )
    print(f"[sweep] {candidate.dir_name} -> {entry['status']}{extra}", flush=True)
    return entry


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root: Path = args.repository_root
    sweep_root: Path = args.output_root
    sweep_root.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(repository_root / "src"))
    from sakuramoon.perf.batch_shape import (
        CANONICAL_SWEEP,
        STATUS_PASS,
        STATUS_SKIPPED_SAFETY_GATE,
        baseline_drift_pct,
        classify_memory_safety,
        high_candidate_gate_passed,
        is_environment_stable,
        median_pair,
        rank_best_safe,
        speed_class,
        speedup_pct,
    )

    config_root = (
        args.config_root
        if Path(args.config_root).is_absolute()
        else str(repository_root / args.config_root)
    )
    generated = datetime.now(UTC).isoformat()

    candidates: dict[str, Any] = {}
    c2_gate: dict[str, Any] = {"evaluated": False, "passed": False, "reason": ""}

    for index, candidate in enumerate(CANONICAL_SWEEP):
        if candidate.label == "C2" and (args.skip_c2 or not c2_gate["passed"]):
            reason = (
                "--skip-c2"
                if args.skip_c2
                else (
                    "safety gate not passed: "
                    + (c2_gate["reason"] or "C1 did not produce a PASS summary")
                )
            )
            candidates[candidate.dir_name] = {
                "label": candidate.label,
                "dir_name": candidate.dir_name,
                "local_batch": candidate.local_batch,
                "accumulation": candidate.accumulation,
                "output_root": str(sweep_root / candidate.dir_name),
                "log": str(sweep_root / f"{candidate.dir_name}.log"),
                "exit_code": None,
                "timed_out": False,
                "resumed": False,
                "status": STATUS_SKIPPED_SAFETY_GATE,
                "gate_reason": reason,
            }
            print(
                f"[sweep] {candidate.dir_name} -> SKIPPED_SAFETY_GATE ({reason})",
                flush=True,
            )
            continue
        candidates[candidate.dir_name] = run_candidate(
            candidate,
            python=sys.executable,
            repository_root=repository_root,
            config=args.config,
            config_root=config_root,
            sweep_root=sweep_root,
            gpus=args.gpus,
            seed=args.seed,
            warmup_updates=args.warmup_updates,
            measure_updates=args.measure_updates,
            expected_global_batch=args.expected_global_batch,
            timeout=args.timeout,
            port=args.base_port + index,
            resume=args.resume,
        )
        if candidate.label == "C1" and (
            candidates[candidate.dir_name]["status"] == STATUS_PASS
        ):
            c1_summary = candidates[candidate.dir_name]["summary"]
            passed = high_candidate_gate_passed(
                c1_summary["max_peak_allocated_gib"],
                c1_summary["max_peak_reserved_gib"],
            )
            c2_gate = {
                "evaluated": True,
                "passed": passed,
                "reason": (
                    "C1 peak allocated "
                    f"{c1_summary['max_peak_allocated_gib']:.2f} GiB <= 48 GiB "
                    "and peak reserved "
                    f"{c1_summary['max_peak_reserved_gib']:.2f} GiB <= 56 GiB"
                    if passed
                    else (
                        "C1 exceeded the C2 gate (peak allocated "
                        f"{c1_summary['max_peak_allocated_gib']:.2f} GiB "
                        "vs <= 48 GiB; peak reserved "
                        f"{c1_summary['max_peak_reserved_gib']:.2f} GiB "
                        "vs <= 56 GiB)"
                    )
                ),
            }

    # A0/A1 bracket: the baseline reference for every comparison.
    a0 = candidates.get("lb20-acc20-a")
    a1 = candidates.get("lb20-acc20-b")
    both_pass = (
        a0 is not None
        and a1 is not None
        and a0.get("status") == STATUS_PASS
        and a1.get("status") == STATUS_PASS
    )
    baseline_reference: dict[str, Any] = {
        "a0_p50_s": a0["summary"]["p50_s"] if both_pass else None,
        "a1_p50_s": a1["summary"]["p50_s"] if both_pass else None,
        "p50_s": None,
        "samples_per_s": None,
        "drift_pct": None,
        "stable": False,
    }
    if both_pass:
        ref_p50 = median_pair(a0["summary"]["p50_s"], a1["summary"]["p50_s"])
        ref_sps = median_pair(
            a0["summary"]["samples_per_s"], a1["summary"]["samples_per_s"]
        )
        baseline_reference.update(
            {
                "p50_s": ref_p50,
                "samples_per_s": ref_sps,
                "drift_pct": baseline_drift_pct(
                    a0["summary"]["p50_s"], a1["summary"]["p50_s"]
                ),
                "stable": is_environment_stable(
                    a0["summary"]["p50_s"], a1["summary"]["p50_s"]
                ),
            }
        )

    # Per-candidate classification (speed vs the bracket reference, memory).
    safe_entries: dict[str, Any] = {}
    for name in sorted(candidates):
        entry = candidates[name]
        if entry["status"] != STATUS_PASS or not entry.get("summary"):
            entry["speedup_vs_baseline_pct"] = None
            entry["speed_class"] = None
            entry["memory_class"] = (
                "UNSAFE" if entry["status"] in ("OOM", "TIMEOUT", "ERROR") else None
            )
            continue
        summary = entry["summary"]
        entry["memory_class"] = classify_memory_safety(
            summary["max_peak_allocated_gib"],
            summary["max_peak_reserved_gib"],
            oom=False,
            measured_updates=summary["measured_updates"],
        )
        if baseline_reference["p50_s"] is not None:
            improvement = speedup_pct(baseline_reference["p50_s"], summary["p50_s"])
            entry["speedup_vs_baseline_pct"] = improvement
            entry["speed_class"] = speed_class(improvement)
        else:
            entry["speedup_vs_baseline_pct"] = None
            entry["speed_class"] = None
        # The A0 control represents the BASE shape (20x20) in the
        # best-safe pool; the A1 repeat is excluded (same shape, drift probe).
        if entry["memory_class"] == "SAFE" and name != "lb20-acc20-b":
            safe_entries[name] = summary

    best_safe: dict[str, Any] | None = None
    if baseline_reference["p50_s"] is not None and safe_entries:
        pick = rank_best_safe(safe_entries)
        if pick is not None:
            name, reason = pick
            summary = safe_entries[name]
            best_safe = {
                "dir_name": name,
                "label": candidates[name]["label"],
                "shape": (
                    f"{candidates[name]['local_batch']}x"
                    f"{candidates[name]['accumulation']}"
                ),
                "p50_speedup_pct": speedup_pct(
                    baseline_reference["p50_s"], summary["p50_s"]
                ),
                "throughput_gain_pct": (
                    (summary["samples_per_s"] / baseline_reference["samples_per_s"])
                    - 1.0
                )
                * 100.0,
                "memory_class": "SAFE",
                "peak_allocated_gib": summary["max_peak_allocated_gib"],
                "peak_reserved_gib": summary["max_peak_reserved_gib"],
                "reason": reason,
            }

    sweep_summary = {
        "schema_version": 1,
        "generated": generated,
        "hostname": os.uname().nodename,
        "repository_root": str(repository_root),
        "git_sha": _git_head_sha(repository_root),
        "workload": {
            "world_size": args.gpus,
            "global_batch": args.expected_global_batch,
            "samples_per_rank_update": args.expected_global_batch // args.gpus,
            "resolution": 256,
            "seed": args.seed,
            "warmup_updates": args.warmup_updates,
            "measured_updates": args.measure_updates,
            "profiler": "off",
        },
        "baseline_reference": baseline_reference,
        "c2_gate": c2_gate,
        "candidates": candidates,
        "best_safe": best_safe,
    }
    summary_path = sweep_root / "sweep-summary.json"
    summary_path.write_text(
        json.dumps(sweep_summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path = sweep_root / "sweep-summary.md"
    markdown_path.write_text(_render_markdown(sweep_summary), encoding="utf-8")
    print(
        f"[sweep] done: summary in {summary_path} / {markdown_path}",
        flush=True,
    )
    return 0


def _render_markdown(summary: dict[str, Any]) -> str:
    workload = summary["workload"]
    reference = summary["baseline_reference"]
    lines = [
        "# P1-R1A batch-shape sweep",
        "",
        (
            f"Generated: {summary['generated']} | host {summary['hostname']} "
            f"| git {summary['git_sha']}"
        ),
        "",
        (
            f"Workload: world {workload['world_size']} | global batch "
            f"{workload['global_batch']} | "
            f"{workload['samples_per_rank_update']} samples/rank/update | "
            f"resolution {workload['resolution']} | seed {workload['seed']} | "
            f"warmup {workload['warmup_updates']} / measured "
            f"{workload['measured_updates']} | profiler OFF"
        ),
        "",
        (
            "Baseline reference = median of the A0/A1 20x20 bracket "
            "(never the faster of the two)."
        ),
        "",
    ]
    if reference["p50_s"] is not None:
        lines += [
            (
                f"- A0 p50 {reference['a0_p50_s']:.3f} s | A1 p50 "
                f"{reference['a1_p50_s']:.3f} s | drift "
                f"{reference['drift_pct']:.3f}% (limit 2%) | reference p50 "
                f"{reference['p50_s']:.3f} s | reference "
                f"{reference['samples_per_s']:.2f} samples/s"
            ),
            "",
        ]
    lines += [
        (
            "| candidate | shape | status | p50 s | p95 s | samples/s | qwen s "
            "| dit_fwd s | backward s | optimizer s | peak alloc GiB (r0/r1) | "
            "max peak reserved GiB | skew % | speedup % | speed | memory |"
        ),
        (
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | "
            "--- | --- | --- | --- | --- | --- |"
        ),
    ]
    for name in sorted(summary["candidates"]):
        entry = summary["candidates"][name]
        s = entry.get("summary")
        shape = f"{entry['local_batch']}x{entry['accumulation']}"
        if s is None:
            extra = entry.get("gate_reason") or entry.get("note") or ""
            memory_cell = "UNSAFE" if entry["memory_class"] == "UNSAFE" else "-"
            row = (
                f"| {name} ({entry['label']}) | {shape} | {entry['status']} "
                "| - | - | - | - | - | - | - | - | - | - | - "
                f"| {memory_cell} |"
            )
            if extra:
                row += f" -- {extra[:80]}"
            lines.append(row)
            continue
        phases = s["phases_s"]
        alloc = s["peak_allocated_gib"]
        speedup = entry.get("speedup_vs_baseline_pct")
        speedup_cell = f"{speedup:.2f}" if speedup is not None else "-"
        lines.append(
            f"| {name} ({entry['label']}) | {shape} | {entry['status']} "
            f"| {s['p50_s']:.3f} | {s['p95_s']:.3f} | {s['samples_per_s']:.2f} "
            f"| {phases.get('qwen', float('nan')):.3f} "
            f"| {phases.get('dit_forward', float('nan')):.3f} "
            f"| {phases.get('backward', float('nan')):.3f} "
            f"| {phases.get('optimizer', float('nan')):.3f} "
            f"| {alloc.get(0, float('nan')):.2f}/"
            f"{alloc.get(1, float('nan')):.2f} "
            f"| {s['max_peak_reserved_gib']:.2f} | {s['rank_skew_pct']:.3f} "
            f"| {speedup_cell} | {entry['speed_class'] or '-'} "
            f"| {entry['memory_class']} |"
        )
    lines += [
        "",
        (
            f"C2 gate: evaluated={summary['c2_gate']['evaluated']} "
            f"passed={summary['c2_gate']['passed']} "
            f"({summary['c2_gate']['reason'] or 'not reached'})"
        ),
        "",
    ]
    best = summary["best_safe"]
    if best is not None:
        lines += [
            "Best safe candidate:",
            "",
            (
                f"- shape {best['shape']} ({best['label']}, {best['dir_name']}) "
                f"| p50 speedup {best['p50_speedup_pct']:.2f}% | throughput gain "
                f"{best['throughput_gain_pct']:.2f}% | memory "
                f"{best['memory_class']} | peak allocated "
                f"{best['peak_allocated_gib']:.2f} GiB | peak reserved "
                f"{best['peak_reserved_gib']:.2f} GiB"
            ),
            f"- {best['reason']}",
            "",
        ]
    else:
        lines += ["Best safe candidate: NONE (no SAFE PASS candidate)", ""]
    lines += [
        (
            "Per-shape artifacts: `<dir_name>/` (rank samples, schema-2 summary "
            "JSON/MD, runtime fingerprint, diagnostics) and `<dir_name>.log` "
            "under the sweep root.  OOM/timeouts are reported, never hidden."
        ),
        "",
        (
            "This sweep is a BENCHMARK recommendation only: no production "
            "configuration, model, optimizer, kernel or DDP setting was "
            "modified."
        ),
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
