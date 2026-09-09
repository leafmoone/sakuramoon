"""SakuraMoon P1-R1B compile + persistent-cache validation orchestrator.

Runs the P1-R1B matrix of fresh-process benchmarks on a DTK dual-HCU
machine and writes ``compile-cache-summary.json`` / ``.md`` under a
scratch root.  It never hides ERROR/TIMEOUT and never mutates production
state: every candidate is a fresh process group with its own cache root,
a hard timeout, and a retained log.

Candidates (canonical order, GO P1-R1B §16):

  A0 — COLD CONTROL       unique empty cache root (``cold-a0/``)
  P0 — PERSISTENT POPULATE persistent cache root starts EMPTY
  P1 — PERSISTENT REUSE   NEW fresh process, SAME persistent root
  A1 — COLD CONTROL REPEAT another unique empty root (``cold-a1/``)

Optional diagnostic (GO §12):

  D  — 1-HCU rank-local compiler diagnostic (TORCH_LOGS on, no profiler,
       no throughput claims).

Performance candidates: 2 HCU, 256px, 20x20, GBS 800, seed 44, warmup 5,
measured 10, profiler OFF, compiler verbose logs OFF.  AOTAutograd-cache
fallback (GO §17): if the FX+AOT candidate pair fails with an AOT-cache
error, ONE FX-graph-cache-only populate/reuse pair is rerun; no further
retries.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sakuramoon.perf.compile import (
    CompileCacheSnapshot,
    CompileEnvironment,
    build_cache_env,
    classify_cache_benefit,
    classify_graph_breaks,
    environment_bracket_drift_pct,
    parse_compiler_log,
    pct_improvement,
    sanitize_cache_identity,
    split_lines_at_measured_banner,
    split_recompiles_by_window,
    steady_state_regression_pass,
    top_recompile_reason,
)

CANDIDATE_ORDER = ("A0", "P0", "P1", "A1")
STATUS_PASS = "PASS"
STATUS_ERROR = "ERROR"
STATUS_TIMEOUT = "TIMEOUT"
AOT_ERROR_MARKER_RE = re.compile(r"autograd.{0,24}cache", re.IGNORECASE)
RECOMPILE_COUNTER_RE = re.compile(r"recompil", re.IGNORECASE)
GIb = 1024**3


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


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


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _as_dict(value: Any) -> dict[str, Any]:
    """Coerce an untyped JSON payload to a clean dict[str, Any] boundary."""

    return cast("dict[str, Any]", value) if isinstance(value, dict) else {}


def _dtk_identity() -> str:
    env_value = os.environ.get("DTK_VERSION", "").strip()
    if env_value:
        return env_value
    for candidate in (Path("/opt/dtk/VERSION"), Path("/opt/dtk/version.txt")):
        try:
            value = candidate.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value.splitlines()[0].strip()
    opt = Path("/opt")
    if opt.is_dir():
        for entry in sorted(opt.iterdir()):
            if entry.is_dir() and entry.name.startswith("dtk-") and entry.name != "dtk":
                return entry.name
    return "unavailable"


def _kill_process_group(proc: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (OSError, ProcessLookupError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.SubprocessError:
        proc.kill()


def _probe_runtime(python: str) -> dict[str, Any]:
    """Probe torch/HIP/Triton/HCU facts from a short child python (no CUDA use)."""

    code = (
        "import json\n"
        "import torch\n"
        "import triton\n"
        "probe = {\n"
        "  'torch': torch.__version__,\n"
        "  'hip': getattr(torch.version, 'hip', None),\n"
        "  'triton': triton.__version__,\n"
        "  'hcu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,\n"
        "}\n"
        "print(json.dumps(probe))\n"
    )
    result = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    for line in reversed(result.stdout.splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"runtime probe failed: {result.stderr[-800:]}")


def _resolve_fa_so_path(env: dict[str, str], python: str) -> str | None:
    existing = env.get("FA_SO_PATH", "").strip()
    if existing:
        return existing
    code = (
        "import os, sys\n"
        "sp = os.path.dirname(os.sys.modules['torch'].__file__) if 'torch' in sys.modules else None\n"
        "import importlib.util\n"
        "spec = importlib.util.find_spec('flash_attn')\n"
        "print(os.path.join(os.path.dirname(spec.origin), 'flash_attn_2_cuda.cpython-311-x86_64-linux-gnu.so') if spec else '')\n"
    )
    result = subprocess.run(
        [python, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    value = (
        result.stdout.strip().splitlines()[-1].strip() if result.stdout.strip() else ""
    )
    return value or None


def _move_dir_away(directory: Path, suffix: str) -> Path | None:
    """Non-destructively rename a cache directory aside (returns new path)."""

    if not directory.exists():
        return None
    moved = directory.with_name(f"{directory.name}.{suffix}")
    counter = 1
    while moved.exists():
        moved = directory.with_name(f"{directory.name}.{suffix}-{counter}")
        counter += 1
    directory.rename(moved)
    return moved


# ---------------------------------------------------------------------------
# Candidate launch
# ---------------------------------------------------------------------------


def _candidate_command(
    candidate: str,
    *,
    python: str,
    repository_root: Path,
    config: str,
    config_root: str,
    output_root: Path,
    cache_root: Path,
    seed: int,
    warmup_updates: int,
    measure_updates: int,
    port: int,
    fx_graph_cache: bool,
    aot_autograd_cache: bool,
) -> tuple[list[str], dict[str, str], str]:
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "32"
    env["MKL_NUM_THREADS"] = "32"
    # Cache env for THIS candidate (benchmark subprocess only).
    env.update(
        build_cache_env(
            cache_root,
            fx_graph_cache=fx_graph_cache,
            aot_autograd_cache=aot_autograd_cache,
        )
    )
    env.pop("TORCH_LOGS", None)

    if candidate == "D":
        # 1-HCU rank-local diagnostic; verbose compiler logs ON (diagnostic
        # run only — never during performance comparison runs).
        env["TORCH_LOGS"] = "recompiles,graph_breaks,dynamic"
        env["CUDA_VISIBLE_DEVICES"] = "0"
        cmd = [
            python,
            str(repository_root / "scripts" / "benchmark_training_perf.py"),
            "--config",
            config,
            "--config-root",
            config_root,
            "--repository-root",
            str(repository_root),
            "--output-root",
            str(output_root),
            "--gpus",
            "1",
            "--label",
            "diag-1gpu",
            "--seed",
            str(seed),
            "--warmup-updates",
            str(warmup_updates),
            "--measure-updates",
            str(measure_updates),
            "--expected-global-batch",
            "400",
        ]
        label = "diag-1gpu"
    else:
        cmd = [
            python,
            "-m",
            "accelerate.commands.launch",
            "--multi_gpu",
            "--num_processes",
            "2",
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
            str(output_root),
            "--gpus",
            "2",
            "--label",
            "baseline-2gpu",
            "--seed",
            str(seed),
            "--warmup-updates",
            str(warmup_updates),
            "--measure-updates",
            str(measure_updates),
            "--expected-global-batch",
            "800",
        ]
        label = "baseline-2gpu"
    return cmd, env, label


def run_candidate(
    candidate: str,
    *,
    python: str,
    repository_root: Path,
    config: str,
    config_root: str,
    ts_root: Path,
    persistent_root: Path | None,
    seed: int,
    warmup_updates: int,
    measure_updates: int,
    diagnostic_measure_updates: int,
    timeout: int,
    port: int,
    fx_graph_cache: bool,
    aot_autograd_cache: bool,
    resume: bool,
) -> dict[str, Any]:
    """Run ONE candidate in a fresh process group with a hard timeout."""

    if candidate == "D":
        cand_root = ts_root / "diagnostic"
        cache_root = ts_root / "diagnostic-cache"
        label = "diag-1gpu"
        measure = diagnostic_measure_updates
    else:
        cand_root = ts_root / candidate.lower()
        if candidate in ("A0", "A1"):
            cache_root = ts_root / f"cold-{candidate.lower()}"
        else:  # P0 / P1 share the persistent root
            assert persistent_root is not None
            cache_root = persistent_root
        label = "baseline-2gpu"
        measure = measure_updates

    summary_path = cand_root / f"{label}.json"
    log_path = ts_root / f"{candidate}.log"

    entry: dict[str, Any] = {
        "label": candidate,
        "output_root": str(cand_root),
        "cache_root": str(cache_root),
        "log": str(log_path),
        "exit_code": None,
        "timed_out": False,
        "resumed": False,
        "status": STATUS_ERROR,
    }

    if resume and summary_path.exists():
        entry["status"] = STATUS_PASS
        entry["resumed"] = True
        entry["exit_code"] = 0
        print(f"[r1b] {candidate}: resumed (existing summary)", flush=True)
        return entry

    cand_root.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    entry["cache_before"] = CompileCacheSnapshot.capture(
        cache_root, captured_at=_utc_now()
    ).to_dict()

    # A0/P1 must see the SAME persistent root P0 populated; for A0/A1/D the
    # cache root is a unique EMPTY directory (true cold control, GO §15).
    cmd, env, _ = _candidate_command(
        candidate,
        python=python,
        repository_root=repository_root,
        config=config,
        config_root=config_root,
        output_root=cand_root,
        cache_root=cache_root,
        seed=seed,
        warmup_updates=warmup_updates,
        measure_updates=measure,
        port=port,
        fx_graph_cache=fx_graph_cache,
        aot_autograd_cache=aot_autograd_cache,
    )
    print(
        f"[r1b] {candidate}: launch (cache={cache_root} "
        f"fx={fx_graph_cache} aot={aot_autograd_cache} port={port} timeout={timeout}s)",
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
            env=env,
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
    entry["cache_after"] = CompileCacheSnapshot.capture(
        cache_root, captured_at=_utc_now()
    ).to_dict()

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    entry["log_tail"] = "\n".join(log_text.splitlines()[-15:])

    if timed_out:
        entry["status"] = STATUS_TIMEOUT
    elif exit_code != 0 or not summary_path.exists():
        entry["status"] = STATUS_ERROR
    else:
        entry["status"] = STATUS_PASS
        entry["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
        entry["startup_rank0"] = json.loads(
            (cand_root / "startup-rank0.json").read_text(encoding="utf-8")
        )
        entry["aot_error_marker"] = bool(AOT_ERROR_MARKER_RE.search(log_text))
    return entry


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def _startup_headline(entry: dict[str, Any]) -> dict[str, Any]:
    """Headline restart metrics: the SLOWEST rank gates readiness (max)."""

    startup: dict[str, Any] = _as_dict(entry.get("startup_rank0"))
    per_rank: dict[str, Any] = {}
    for rank in ("0", "1"):
        try:
            # co-located on the shared scratch filesystem
            path = Path(entry["output_root"]) / f"startup-rank{rank}.json"
            if path.exists():
                per_rank[f"rank{rank}"] = json.loads(path.read_text(encoding="utf-8"))
        except OSError:
            continue

    def _max_across_ranks(key: str) -> float | None:
        values: list[float] = []
        for payload in list(per_rank.values()):
            value = payload.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        value0 = startup.get(key)
        if isinstance(value0, (int, float)):
            values.append(float(value0))
        return max(values) if values else None

    return {
        "time_to_first_successful_update_s": _max_across_ranks(
            "time_to_first_successful_update_s"
        ),
        "warmup_total_wall_s": _max_across_ranks("warmup_total_wall_s"),
        "time_to_measured_window_s": _max_across_ranks("time_to_measured_window_s"),
        "assembly_wall_s": _max_across_ranks("assembly_wall_s"),
        "compile_install_wall_s": _max_across_ranks("compile_install_wall_s"),
        "warmup_update_walls_s_rank0": startup.get("warmup_update_walls_s"),
        "per_rank": per_rank,
    }


def _steady_metrics(entry: dict[str, Any]) -> dict[str, Any] | None:
    summary: dict[str, Any] = _as_dict(entry.get("summary"))
    if not summary:
        return None
    step: dict[str, Any] = _as_dict(summary.get("global_step_seconds"))
    memory: dict[str, Any] = _as_dict(
        _as_dict(summary.get("memory")).get("max_across_ranks")
    )
    peak_allocated: Any = memory.get("peak_allocated_bytes")
    peak_reserved: Any = memory.get("peak_reserved_bytes")
    return {
        "p50_s": step.get("p50"),
        "p95_s": step.get("p95"),
        "samples_per_s": summary.get("global_samples_per_second"),
        "peak_allocated_gib": (
            peak_allocated / GIb if isinstance(peak_allocated, (int, float)) else None
        ),
        "peak_reserved_gib": (
            peak_reserved / GIb if isinstance(peak_reserved, (int, float)) else None
        ),
    }


def _recompile_delta(entry: dict[str, Any]) -> dict[str, Any]:
    """Measured-window recompile count from the in-process counter DELTA."""

    startup: dict[str, Any] = _as_dict(entry.get("startup_rank0"))
    warmup_end: dict[str, Any] = _as_dict(
        _as_dict(startup.get("dynamo_counters_warmup_end")).get("counters")
    )
    measured_end: dict[str, Any] = _as_dict(
        _as_dict(startup.get("dynamo_counters_measured_end")).get("counters")
    )
    key: str | None = next(
        (
            k
            for k in sorted(set(warmup_end) & set(measured_end))
            if RECOMPILE_COUNTER_RE.search(k)
        ),
        None,
    )
    if key is None:
        return {
            "method": "counters-unavailable",
            "counter_key": None,
            "total": None,
            "warmup": None,
            "measured": None,
        }
    total = int(measured_end[key])
    warmup = int(warmup_end[key])
    measured = total - warmup
    try:
        w, m = split_recompiles_by_window(total, measured_window_recompiles=measured)
    except ValueError:
        w, m = max(0, total - measured), measured
    return {
        "method": "in-process-counter-delta",
        "counter_key": key,
        "total": total,
        "warmup": w,
        "measured": m,
    }


def _diagnostic_summary(entry: dict[str, Any]) -> dict[str, Any]:
    """Compiler evidence for the 1-HCU diagnostic run (GO P1-R1B \u00a724).

    Recompile evidence prefers the in-process dynamo COUNTER delta
    (authoritative).  When this torch build exposes no recompile-style
    counter in ``torch._dynamo.utils.counters`` (observed on 2.9/DTK), it
    falls back to the retained TORCH_LOGS recompile lines split by the
    benchmark measured-stage banner.  Graph breaks come from the retained
    log; the measured-window split uses the same banner.  Counts are
    reported exactly, never fabricated.
    """

    log_text = ""
    try:
        log_text = Path(entry["log"]).read_text(encoding="utf-8", errors="replace")
    except OSError:
        pass
    parsed = parse_compiler_log(log_text)
    split = split_lines_at_measured_banner(log_text.splitlines())
    measured_nos: set[int] = set(split["measured"])

    expected, unexpected = classify_graph_breaks(parsed)
    _, unexpected_meas = classify_graph_breaks(parsed, line_nos=measured_nos)
    breaks_in_measured = sum(
        1 for e in parsed["graph_breaks"] if int(e.get("line_no", -1)) in measured_nos
    )

    recompile_counts = _recompile_delta(entry)
    if recompile_counts["method"] == "in-process-counter-delta":
        recomp_total: int = int(recompile_counts["total"])
        recomp_warmup: int = int(recompile_counts["warmup"])
        recomp_measured: int = int(recompile_counts["measured"])
        method = "in-process-counter-delta"
    else:
        recomp_total = len(parsed["recompiles"])
        recomp_measured = sum(
            1 for e in parsed["recompiles"] if int(e.get("line_no", -1)) in measured_nos
        )
        recomp_warmup = recomp_total - recomp_measured
        method = "log-banner-window" if split["found"] else "log-banner-not-found"
    return {
        "graph_breaks_total": len(parsed["graph_breaks"]),
        "graph_breaks_expected": expected,
        "graph_breaks_unexpected": unexpected,
        "graph_breaks_measured_window": breaks_in_measured,
        "graph_breaks_unexpected_measured_window": unexpected_meas,
        "graph_break_lines": [e["line"] for e in parsed["graph_breaks"]][:40],
        "recompiles_total": recomp_total,
        "recompiles_warmup": recomp_warmup,
        "recompiles_measured": recomp_measured,
        "recompile_counter_method": method,
        "recompile_counter_key": recompile_counts["counter_key"],
        "recompile_log_lines": [e["line"] for e in parsed["recompiles"]][:40],
        "top_recompile_reason": top_recompile_reason(parsed),
        "dynamic_shape_notes": parsed["dynamic_notes"][:20],
        "measured_banner_found": split["found"],
        "raw_log_preserved": True,
    }


# ---------------------------------------------------------------------------
# Matrix + computation
# ---------------------------------------------------------------------------


def _compute(
    candidates: dict[str, Any],
    *,
    environment: CompileEnvironment,
    fx_graph_cache: bool,
    aot_autograd_cache: bool,
    aot_fallback: str | None,
) -> dict[str, Any]:
    a0: dict[str, Any] = candidates.get("A0") or {}
    p0: dict[str, Any] = candidates.get("P0") or {}
    p1: dict[str, Any] = candidates.get("P1") or {}
    a1: dict[str, Any] = candidates.get("A1") or {}

    a0_steady = _steady_metrics(a0)
    a1_steady = _steady_metrics(a1)
    p1_steady = _steady_metrics(p1)
    a0_up = _startup_headline(a0)
    a1_up = _startup_headline(a1)
    p1_up = _startup_headline(p1)
    p0_up = _startup_headline(p0)

    def _steady_ref(key: str) -> float | None:
        values = [
            m[key]
            for m in (a0_steady, a1_steady)
            if m is not None and isinstance(m.get(key), (int, float))
        ]
        return sum(values) / len(values) if values else None

    def _startup_ref(key: str) -> float | None:
        values = [
            u[key] for u in (a0_up, a1_up) if isinstance(u.get(key), (int, float))
        ]
        return (sum(values) / len(values)) if values else None

    steady_ref_p50 = _steady_ref("p50_s")
    steady_ref_p95 = _steady_ref("p95_s")
    cold_first = _startup_ref("time_to_first_successful_update_s")
    reuse_first = p1_up.get("time_to_first_successful_update_s")

    first_saved = None
    first_improvement = None
    cache_class = None
    if isinstance(cold_first, float) and isinstance(reuse_first, float):
        first_saved = cold_first - reuse_first
        first_improvement = pct_improvement(cold_first, reuse_first)
        cache_class = classify_cache_benefit(first_improvement, first_saved)

    def _saved(key: str) -> float | None:
        ref = _startup_ref(key)
        val = p1_up.get(key)
        if isinstance(ref, float) and isinstance(val, float):
            return ref - val
        return None

    # Steady-state regression gate: reuse vs the COLD BRACKET REFERENCE
    # (mean of A0/A1). Positive delta = reuse is SLOWER (regression).
    p1_p50 = (p1_steady or {}).get("p50_s")
    p1_p95 = (p1_steady or {}).get("p95_s")
    p1_mem = (p1_steady or {}).get("peak_allocated_gib")
    p50_delta = (
        (p1_p50 - steady_ref_p50) / steady_ref_p50 * 100.0
        if isinstance(steady_ref_p50, float) and isinstance(p1_p50, (int, float))
        else None
    )
    p95_delta = (
        (p1_p95 - steady_ref_p95) / steady_ref_p95 * 100.0
        if isinstance(steady_ref_p95, float) and isinstance(p1_p95, (int, float))
        else None
    )
    ref_mem = _steady_ref("peak_allocated_gib")
    mem_delta = (
        (p1_mem - ref_mem) / ref_mem * 100.0
        if isinstance(ref_mem, float) and isinstance(p1_mem, (int, float))
        else None
    )
    gate = (
        steady_state_regression_pass(p50_delta, mem_delta)
        if p50_delta is not None and mem_delta is not None
        else None
    )

    # Environment bracket (GO §18).
    steady_drift = (
        environment_bracket_drift_pct(a0_steady["p50_s"], a1_steady["p50_s"])
        if a0_steady
        and a1_steady
        and isinstance(a0_steady.get("p50_s"), (int, float))
        and isinstance(a1_steady.get("p50_s"), (int, float))
        else None
    )
    cold_drift = (
        environment_bracket_drift_pct(
            a0_up["time_to_first_successful_update_s"],
            a1_up["time_to_first_successful_update_s"],
        )
        if isinstance(a0_up.get("time_to_first_successful_update_s"), (int, float))
        and isinstance(a1_up.get("time_to_first_successful_update_s"), (int, float))
        else None
    )

    # Decision tree (GO §32): YES only with clean compiler evidence AND a
    # worthwhile, steady-neutral persistent cache.  Missing compiler
    # evidence is not "clean": it DEFERS the recommendation.
    d: dict[str, Any] = candidates.get("D") or {}
    diag: dict[str, Any] = d.get("diagnostic") or {}
    # GO P1-R1B \u00a732 Case A: clean = NO recompiles and NO unexpected
    # graph breaks in the steady (MEASURED) window.
    recompiles_measured: Any = diag.get("recompiles_measured")
    unexpected_breaks: Any = diag.get("graph_breaks_unexpected_measured_window")
    compiler_clean = (
        recompiles_measured == 0 and unexpected_breaks == 0
        if isinstance(recompiles_measured, int) and isinstance(unexpected_breaks, int)
        else None
    )
    if compiler_clean is not True:
        recommend = "DEFER"
    elif cache_class in ("USEFUL", "STRONG") and gate is True:
        recommend = "YES"
    else:
        recommend = "NO"

    return {
        "environment": {
            "steady_A0_A1_drift_pct": steady_drift,
            "cold_start_A0_A1_drift_pct": cold_drift,
            "steady_stable_le_2pct": (steady_drift is not None and steady_drift <= 2.0),
            "cold_start_observed_le_15pct": (
                cold_drift is not None and cold_drift <= 15.0
            ),
        },
        "cache_effect": {
            "cold_reference_first_update_s": cold_first,
            "reuse_first_update_s": reuse_first,
            "first_update_saved_s": first_saved,
            "first_update_improvement_pct": first_improvement,
            "warmup_saved_s": _saved("warmup_total_wall_s"),
            "time_to_measured_saved_s": _saved("time_to_measured_window_s"),
            "populate_first_update_s": p0_up.get("time_to_first_successful_update_s"),
            "class": cache_class,
        },
        "steady_state": {
            "cold_ref_p50_s": steady_ref_p50,
            "reuse_p50_s": (p1_steady or {}).get("p50_s"),
            "reuse_vs_cold_p50_delta_pct": p50_delta,
            "cold_ref_p95_s": steady_ref_p95,
            "reuse_p95_s": (p1_steady or {}).get("p95_s"),
            "reuse_vs_cold_p95_delta_pct": p95_delta,
            "cold_ref_peak_allocated_gib": ref_mem,
            "reuse_peak_allocated_gib": (p1_steady or {}).get("peak_allocated_gib"),
            "memory_delta_pct": mem_delta,
            "regression_gate": "PASS"
            if gate is True
            else "FAIL"
            if gate is False
            else None,
        },
        "decision": {
            "compiler_steady_state_clean": compiler_clean,
            "persistent_cache_worthwhile": cache_class in ("USEFUL", "STRONG"),
            "recommend_persistent_cache_integration": recommend,
        },
        "cache_candidate": {
            "fx_graph_cache": fx_graph_cache,
            "aot_autograd_cache": aot_autograd_cache,
            "aot_fallback": aot_fallback,
        },
    }


def _render_markdown(summary: dict[str, Any]) -> str:
    env = summary["environment_bracket"]
    eff = summary["cache_effect"]
    ss = summary["steady_state"]
    dec = summary["decision"]
    lines = [
        "# SakuraMoon Infra P1-R1B — Compile + Persistent Cache",
        "",
        f"- git SHA: `{summary['git_sha']}`",
        f"- generated: {summary['generated']}",
        f"- machine: {summary['hostname']}",
        f"- cache identity: `{summary['cache']['identity']}`",
        f"- persistent root: `{summary['cache']['persistent_root']}`",
        "",
        "## Candidates",
        "",
        "| cand | status | exit | ttf (rank-max, s) | steady p50 (s) | samples/s | cache bytes after |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in ("A0", "P0", "P1", "A1", "D"):
        cand = summary["candidates"].get(key)
        if not cand:
            continue
        steady: dict[str, Any] = cand.get("steady") or {}
        up: dict[str, Any] = cand.get("startup_headline") or {}
        after: Any = _as_dict(cand.get("cache_after")).get("total_bytes")
        lines.append(
            f"| {key} | {cand.get('status')} | {cand.get('exit_code')} "
            f"| {up.get('time_to_first_successful_update_s')} "
            f"| {steady.get('p50_s')} "
            f"| {steady.get('samples_per_s')} "
            f"| {after} |"
        )
    lines += [
        "",
        "## Environment bracket",
        f"- steady A0/A1 drift: {env['steady_A0_A1_drift_pct']}% (gate <= 2%)",
        f"- cold-start A0/A1 drift: {env['cold_start_A0_A1_drift_pct']}% (observe <= 15%)",
        "",
        "## Cache effect (restart latency)",
        f"- cold reference ttf: {eff['cold_reference_first_update_s']} s",
        f"- reuse ttf: {eff['reuse_first_update_s']} s",
        f"- saved: {eff['first_update_saved_s']} s ({eff['first_update_improvement_pct']}%)",
        f"- class: {eff['class']}",
        "",
        "## Steady state (reuse vs cold bracket)",
        f"- p50 delta: {ss['reuse_vs_cold_p50_delta_pct']}% (gate <= 2%)",
        f"- p95 delta: {ss['reuse_vs_cold_p95_delta_pct']}%",
        f"- memory delta: {ss['memory_delta_pct']}% (gate <= 5%)",
        f"- regression gate: {ss['regression_gate']}",
        "",
        "## Decision",
        f"- compiler steady-state clean: {dec['compiler_steady_state_clean']}",
        f"- persistent cache worthwhile: {dec['persistent_cache_worthwhile']}",
        f"- recommend integration: {dec['recommend_persistent_cache_integration']}",
        "",
    ]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/sakuramoon-runtime/infra-bench/p1-r1b"),
    )
    parser.add_argument(
        "--persistent-root",
        type=Path,
        default=Path("/sakuramoon-runtime/compile-cache/p1-r1b"),
    )
    parser.add_argument("--config", default="train_g1.toml")
    parser.add_argument("--config-root", default="config")
    parser.add_argument("--seed", type=int, default=44)
    parser.add_argument("--warmup-updates", type=int, default=5)
    parser.add_argument("--measure-updates", type=int, default=10)
    parser.add_argument("--diagnostic-measure-updates", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=2400)
    parser.add_argument("--base-port", type=int, default=29611)
    parser.add_argument(
        "--candidates",
        default=",".join(CANDIDATE_ORDER),
        help="comma-separated subset of A0,P0,P1,A1 (run in canonical order)",
    )
    parser.add_argument(
        "--diagnostic", action="store_true", help="also run diagnostic D (1 HCU)"
    )
    parser.add_argument("--no-fx-graph-cache", action="store_true")
    parser.add_argument(
        "--no-aot-autograd-cache",
        action="store_true",
        help="disable the AOTAutograd cache candidate (FX only)",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--ts-root",
        type=Path,
        default=None,
        help="reuse an existing ts_root in place (requires --resume)",
    )
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repository_root = args.repository_root.resolve()

    git_sha = _git_head_sha(repository_root)
    ts = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    if args.ts_root is not None:
        if not args.resume:
            print("[r1b] --ts-root requires --resume", file=sys.stderr)
            return 2
        ts_root = args.ts_root
        if not ts_root.is_dir():
            print(f"[r1b] --ts-root does not exist: {ts_root}", file=sys.stderr)
            return 2
        ts = ts_root.name
    else:
        ts_root = args.scratch_root / git_sha / ts
        ts_root.mkdir(parents=True, exist_ok=True)

    probe = _probe_runtime(args.python)
    fx = not args.no_fx_graph_cache
    aot = fx and not args.no_aot_autograd_cache
    aot_fallback: str | None = None

    identity = sanitize_cache_identity(
        f"torch{probe.get('torch')}",
        f"hip{probe.get('hip')}",
        _dtk_identity(),
        probe.get("hcu") or "hcu",
        "inductor",
        "max-autotune-no-cudagraphs",
        "dynamic1",
    )
    persistent_root = args.persistent_root / identity
    # P0 requires the persistent root to START EMPTY (GO §16).  A stale
    # non-empty root is moved aside (non-destructively), never deleted.
    stale = None
    if not args.resume and persistent_root.exists() and any(persistent_root.iterdir()):
        stale = _move_dir_away(persistent_root, f"stale-{ts}")
    persistent_root.mkdir(parents=True, exist_ok=True)

    fa_so = _resolve_fa_so_path(dict(os.environ), args.python)
    if fa_so:
        os.environ["FA_SO_PATH"] = fa_so

    env_facts = CompileEnvironment(
        torch_version=str(probe.get("torch")),
        hip_version=probe.get("hip"),
        dtk_identity=_dtk_identity(),
        hcu_model=str(probe.get("hcu") or "unavailable"),
        triton_version=str(probe.get("triton")),
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
        compile_dynamic=True,
        world_size=2,
        resolution=256,
        local_batch=20,
        accumulation=20,
        cache_root=str(persistent_root),
        git_sha=git_sha,
        cache_env=tuple(
            f"{k}={v}"
            for k, v in sorted(
                build_cache_env(
                    persistent_root, fx_graph_cache=fx, aot_autograd_cache=aot
                ).items()
            )
        ),
    )

    requested = [c for c in CANDIDATE_ORDER if c in args.candidates.split(",")]
    candidates: dict[str, Any] = {}
    port_index = 0
    for candidate in requested:
        port = args.base_port + port_index
        port_index += 1
        entry = run_candidate(
            candidate,
            python=args.python,
            repository_root=repository_root,
            config=args.config,
            config_root=args.config_root,
            ts_root=ts_root,
            persistent_root=persistent_root,
            seed=args.seed,
            warmup_updates=args.warmup_updates,
            measure_updates=args.measure_updates,
            diagnostic_measure_updates=args.diagnostic_measure_updates,
            timeout=args.timeout,
            port=port,
            fx_graph_cache=fx,
            aot_autograd_cache=aot,
            resume=args.resume,
        )
        entry["steady"] = _steady_metrics(entry)
        entry["startup_headline"] = _startup_headline(entry)
        candidates[candidate] = entry

        # AOT fallback (GO §17): ONE rerun of the populate/reuse pair with
        # FX graph cache only, if the FX+AOT pair failed on an AOT-cache
        # error.  No further retries.
        if (
            candidate in ("P0", "P1")
            and aot
            and aot_fallback is None
            and entry["status"] != STATUS_PASS
            and entry.get("aot_error_marker") is True
        ):
            print(
                "[r1b] AOT-cache error: rerunning P0/P1 with FX graph cache only",
                flush=True,
            )
            aot = False
            aot_fallback = "fx_graph_cache_only"
            stale2 = (
                _move_dir_away(persistent_root, f"aot-failed-{ts}")
                if any(persistent_root.iterdir())
                else None
            )
            persistent_root.mkdir(parents=True, exist_ok=True)
            stale = stale or stale2
            for retry in ("P0", "P1"):
                retry_port = args.base_port + port_index
                port_index += 1
                retry_entry = run_candidate(
                    retry,
                    python=args.python,
                    repository_root=repository_root,
                    config=args.config,
                    config_root=args.config_root,
                    ts_root=ts_root,
                    persistent_root=persistent_root,
                    seed=args.seed,
                    warmup_updates=args.warmup_updates,
                    measure_updates=args.measure_updates,
                    diagnostic_measure_updates=args.diagnostic_measure_updates,
                    timeout=args.timeout,
                    port=retry_port,
                    fx_graph_cache=fx,
                    aot_autograd_cache=False,
                    resume=False,
                )
                retry_entry["steady"] = _steady_metrics(retry_entry)
                retry_entry["startup_headline"] = _startup_headline(retry_entry)
                candidates[retry] = retry_entry
            # The failure that triggered the fallback is retained for audit.
            candidates[f"{candidate}-aot-failed"] = entry

    if args.diagnostic:
        d_entry = run_candidate(
            "D",
            python=args.python,
            repository_root=repository_root,
            config=args.config,
            config_root=args.config_root,
            ts_root=ts_root,
            persistent_root=persistent_root,
            seed=args.seed,
            warmup_updates=args.warmup_updates,
            measure_updates=args.measure_updates,
            diagnostic_measure_updates=args.diagnostic_measure_updates,
            timeout=args.timeout,
            port=args.base_port + port_index,
            fx_graph_cache=fx,
            aot_autograd_cache=aot,
            resume=args.resume,
        )
        if d_entry["status"] == STATUS_PASS:
            d_entry["diagnostic"] = _diagnostic_summary(d_entry)
        d_entry["steady"] = _steady_metrics(d_entry)
        d_entry["startup_headline"] = _startup_headline(d_entry)
        candidates["D"] = d_entry

    computed = _compute(
        candidates,
        environment=env_facts,
        fx_graph_cache=fx,
        aot_autograd_cache=aot,
        aot_fallback=aot_fallback,
    )

    diag_candidate = candidates.get("D")
    diag_detail: Any = diag_candidate.get("diagnostic") if diag_candidate else None
    summary: dict[str, Any] = {
        "schema": "compile-cache-v1",
        "generated": _utc_now(),
        "git_sha": git_sha,
        "hostname": subprocess.run(
            ["hostname"], capture_output=True, text=True, check=False
        ).stdout.strip(),
        "repository_root": str(repository_root),
        "ts_root": str(ts_root),
        "persistent_root_moved_away": str(stale) if stale else None,
        "workload": {
            "resolution": 256,
            "world_size": 2,
            "global_batch": 800,
            "samples_per_rank_update": 400,
            "seed": args.seed,
            "warmup_updates": args.warmup_updates,
            "measured_updates": args.measure_updates,
            "profiler": "off",
            "compiler_verbose_logs": "off (diagnostic D only)",
        },
        "cache": {
            "identity": identity,
            "persistent_root": str(persistent_root),
            "persistent_starts_empty": "enforced (stale root moved aside)",
            **computed.pop("cache_candidate"),
        },
        "environment": env_facts.to_dict(),
        "candidates": {
            key: {
                k: v for k, v in cand.items() if k not in ("summary", "startup_rank0")
            }
            for key, cand in candidates.items()
        },
        "diagnostic_D_detail": diag_detail,
        "environment_bracket": computed["environment"],
        "cache_effect": computed["cache_effect"],
        "steady_state": computed["steady_state"],
        "decision": computed["decision"],
    }

    json_path = ts_root / "compile-cache-summary.json"
    json_path.write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    md_path = ts_root / "compile-cache-summary.md"
    md_path.write_text(_render_markdown(summary), encoding="utf-8")

    print(
        f"[r1b] summary written: {json_path}\n[r1b] decision: "
        f"{json.dumps(summary['decision'], sort_keys=True)}",
        flush=True,
    )

    # Never hide failure: exit non-zero if any requested performance
    # candidate (or the diagnostic, when requested) did not PASS.
    failed = [
        key for key in requested if candidates.get(key, {}).get("status") != STATUS_PASS
    ]
    if args.diagnostic and candidates.get("D", {}).get("status") != STATUS_PASS:
        failed.append("D")
    if failed:
        print(f"[r1b] FAILED candidates: {', '.join(failed)}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
