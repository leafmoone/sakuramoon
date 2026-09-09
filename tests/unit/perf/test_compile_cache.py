"""Focused unit tests for P1-R1B compile / persistent-cache observability.

Covers (GO P1-R1B §29): cache identity / path sanitization, cache snapshot
file counting / bytes, startup-metrics validation, cold/warm improvement
math, benefit-classification boundaries, steady-state regression math,
environment-bracket drift, cache-env construction, the AOT-requires-FX
fail-closed rule, the unsupported-capability path, the diagnostic parser,
graph-break classification, and recompile classification.  No GPU, no
1.57B model: the pure core of ``sakuramoon.perf.compile`` is torch-free by
design.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from sakuramoon.perf.compile import (
    EXPECTED_BREAK,
    UNEXPECTED_BREAK,
    CompileCacheSnapshot,
    CompileDiagnosticSummary,
    CompileEnvironment,
    CompileStartupSample,
    build_cache_env,
    classify_cache_benefit,
    classify_graph_break,
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

# ---------------------------------------------------------------------------
# Cache identity / path sanitization
# ---------------------------------------------------------------------------


def test_sanitize_identity_collapses_and_lowercases() -> None:
    value = sanitize_cache_identity(
        "torch2.9.0",
        "hip6.3.26093",
        "dtk-26.04",
        "Hygon BW",
        "inductor",
        "max-autotune-no-cudagraphs",
        "dynamic1",
    )
    assert (
        value
        == "torch2-9-0-hip6-3-26093-dtk-26-04-hygon-bw-inductor-max-autotune-no-cudagraphs-dynamic1"
    )
    # Path-safe: no separators, no leading/trailing hyphen.
    assert "/" not in value and "\\" not in value
    assert not value.startswith("-") and not value.endswith("-")


def test_sanitize_identity_skips_none_and_empty() -> None:
    assert sanitize_cache_identity("torch2.9.0", None, "", "  ", "x") == "torch2-9-0-x"


def test_sanitize_identity_all_empty_is_unknown() -> None:
    assert sanitize_cache_identity(None, "", "   ") == "unknown"


def test_sanitize_identity_is_stable() -> None:
    a = sanitize_cache_identity("torch2.9.0", "dtk-26.04", "BW")
    b = sanitize_cache_identity("torch2.9.0", "dtk-26.04", "BW")
    assert a == b


# ---------------------------------------------------------------------------
# Cache env construction (+ AOT requires FX)
# ---------------------------------------------------------------------------


def test_build_cache_env_fx_only() -> None:
    env = build_cache_env("/cache/root", fx_graph_cache=True)
    assert env == {
        "TORCHINDUCTOR_CACHE_DIR": "/cache/root",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
    }
    # Never set TRITON_CACHE_DIR (Inductor owns the triton subcache).
    assert "TRITON_CACHE_DIR" not in env


def test_build_cache_env_aot_with_fx() -> None:
    env = build_cache_env("/cache/root", fx_graph_cache=True, aot_autograd_cache=True)
    assert env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "1"
    assert env["TORCHINDUCTOR_AUTOGRAD_CACHE"] == "1"


def test_build_cache_env_aot_without_fx_fails_closed() -> None:
    with pytest.raises(ValueError, match="requires fx_graph_cache"):
        build_cache_env("/cache/root", fx_graph_cache=False, aot_autograd_cache=True)


def test_build_cache_env_no_fx_no_aot_sets_only_root() -> None:
    env = build_cache_env(Path("/cache/root"), fx_graph_cache=False)
    assert env == {"TORCHINDUCTOR_CACHE_DIR": "/cache/root"}


def test_build_cache_env_never_sets_unsafe_flags() -> None:
    env = build_cache_env("/r", fx_graph_cache=True, aot_autograd_cache=True)
    forbidden = (
        "TORCHINDUCTOR_REMOTE_CACHE",
        "TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE",
        "TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE",
        "TORCH_COMPILE_ALLOW_TF32_CUBLAS_OVERRIDE",
    )
    for key in forbidden:
        assert key not in env


# ---------------------------------------------------------------------------
# CompileEnvironment
# ---------------------------------------------------------------------------


def test_compile_environment_to_dict_parses_cache_env() -> None:
    env = CompileEnvironment(
        torch_version="2.9.0",
        hip_version="6.3.26093",
        dtk_identity="dtk-26.04",
        hcu_model="Hygon BW",
        triton_version="3.3.0",
        compile_backend="inductor",
        compile_mode="max-autotune-no-cudagraphs",
        compile_dynamic=True,
        world_size=2,
        resolution=256,
        local_batch=20,
        accumulation=20,
        cache_root="/cache/root",
        git_sha="e" * 40,
        cache_env=(
            "TORCHINDUCTOR_FX_GRAPH_CACHE=1",
            "TORCHINDUCTOR_CACHE_DIR=/cache/root",
        ),
    )
    payload = env.to_dict()
    assert payload["cache_env"] == {
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_CACHE_DIR": "/cache/root",
    }
    assert payload["git_sha"] == "e" * 40
    assert payload["compile_dynamic"] is True


# ---------------------------------------------------------------------------
# Cache snapshot
# ---------------------------------------------------------------------------


def test_cache_snapshot_counts_files_and_bytes(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    (root / "fxgraph").mkdir(parents=True)
    (root / "triton" / "0").mkdir(parents=True)
    (root / "top.txt").write_bytes(b"abcd")
    (root / "fxgraph" / "a.bin").write_bytes(b"0123456789")
    (root / "triton" / "0" / "kern.cubin").write_bytes(b"xy")
    snapshot = CompileCacheSnapshot.capture(root, captured_at="t0")
    assert snapshot.exists is True
    assert snapshot.file_count == 3
    assert snapshot.total_bytes == 4 + 10 + 2
    assert snapshot.subdir_bytes == {"(root)": 4, "fxgraph": 10, "triton": 2}
    payload = snapshot.to_dict()
    assert payload["root"] == str(root)
    assert payload["file_count"] == 3


def test_cache_snapshot_absent_root_is_honest_zero(tmp_path: Path) -> None:
    snapshot = CompileCacheSnapshot.capture(tmp_path / "missing", captured_at="t0")
    assert snapshot.exists is False
    assert snapshot.file_count == 0
    assert snapshot.total_bytes == 0
    assert snapshot.subdir_bytes == {}


# ---------------------------------------------------------------------------
# Startup metrics validation
# ---------------------------------------------------------------------------


def _marks(base: float, **offsets: float) -> dict[str, float]:
    return {"process_start": base, **{k: base + v for k, v in offsets.items()}}


def test_startup_sample_derives_headline_metrics() -> None:
    marks = _marks(
        100.0,
        assembly_start=0.5,
        config_loaded=1.0,
        accelerator_ready=2.0,
        model_built=12.0,
        optimizer_built=14.0,
        encoders_loaded=24.0,
        ddp_prepared=27.0,
        compile_installed=60.0,
        runtime_built=61.0,
        batches_pregenerated=70.0,
        assembly_end=71.0,
        warmup_stage_start=72.0,
        first_update_complete=90.0,
        warmup_stage_end=120.0,
        measured_stage_start=121.0,
    )
    sample = CompileStartupSample.from_marks(
        marks, warmup_update_walls=[16.0, 15.5, 15.6, 15.7, 15.9]
    )
    assert sample.process_start_to_assembly_start_seconds == pytest.approx(0.5)
    assert sample.assembly_wall_seconds == pytest.approx(70.5)
    assert sample.model_build_wall_seconds == pytest.approx(10.0)
    assert sample.optimizer_build_wall_seconds == pytest.approx(2.0)
    assert sample.encoder_load_wall_seconds == pytest.approx(10.0)
    assert sample.ddp_prepare_wall_seconds == pytest.approx(3.0)
    # compile_install is bounded by ddp_prepared when present:
    assert sample.compile_install_wall_seconds == pytest.approx(33.0)
    assert sample.warmup_total_wall_seconds == pytest.approx(48.0)
    assert sample.warmup_update_walls_seconds == (16.0, 15.5, 15.6, 15.7, 15.9)
    assert sample.time_to_first_successful_update_seconds == pytest.approx(90.0)
    assert sample.time_to_measured_window_seconds == pytest.approx(121.0)


def test_startup_sample_single_rank_omits_ddp_honestly() -> None:
    marks = _marks(
        0.0,
        assembly_start=0.1,
        config_loaded=0.2,
        accelerator_ready=0.3,
        model_built=3.0,
        optimizer_built=4.0,
        encoders_loaded=9.0,
        compile_installed=30.0,
        runtime_built=31.0,
        batches_pregenerated=40.0,
        assembly_end=41.0,
        warmup_stage_start=42.0,
        first_update_complete=58.0,
        warmup_stage_end=90.0,
        measured_stage_start=91.0,
    )
    sample = CompileStartupSample.from_marks(marks)
    # No ddp_prepared mark -> ddp_prepare is ABSENT (not a fake 0.0), and
    # compile_install is bounded by encoders_loaded instead.
    assert sample.ddp_prepare_wall_seconds is None
    assert sample.compile_install_wall_seconds == pytest.approx(21.0)
    payload = sample.to_dict()
    assert "ddp_prepare_wall_s" not in payload
    assert payload["compile_install_wall_s"] == pytest.approx(21.0)
    assert payload["schema"] == "startup-v1"


def test_startup_sample_requires_process_start() -> None:
    with pytest.raises(ValueError, match="process_start"):
        CompileStartupSample.from_marks({"assembly_start": 1.0})


def test_startup_sample_rejects_mark_before_process_start() -> None:
    with pytest.raises(ValueError, match="precedes process_start"):
        CompileStartupSample.from_marks({"process_start": 10.0, "x": 9.0})


def test_startup_sample_rejects_non_float_mark() -> None:
    with pytest.raises(ValueError, match="float"):
        CompileStartupSample.from_marks({"process_start": 0.0, "x": "1.0"})  # type: ignore[dict-item]


def test_startup_sample_rejects_bad_warmup_wall() -> None:
    marks = _marks(0.0, assembly_start=0.1, assembly_end=1.0)
    for bad in (0.0, -1.0, float("nan"), "1.0"):  # type: ignore[list-item]
        with pytest.raises(ValueError):
            CompileStartupSample.from_marks(marks, warmup_update_walls=[bad])


def test_startup_sample_roundtrip_dict_stable() -> None:
    marks = _marks(0.0, assembly_start=0.1, assembly_end=1.0)
    sample = CompileStartupSample.from_marks(marks)
    payload = sample.to_dict()
    assert payload["assembly_wall_s"] == pytest.approx(0.9)
    assert payload["warmup_update_walls_s"] == []
    assert "marks" in payload and payload["marks"]["process_start"] == 0.0


# ---------------------------------------------------------------------------
# Cold/warm improvement math
# ---------------------------------------------------------------------------


def test_pct_improvement_basic() -> None:
    assert pct_improvement(100.0, 80.0) == pytest.approx(20.0)
    assert pct_improvement(100.0, 100.0) == pytest.approx(0.0)
    # A SLOWER candidate is a negative improvement.
    assert pct_improvement(100.0, 120.0) == pytest.approx(-20.0)


def test_pct_improvement_zero_baseline_is_zero() -> None:
    assert pct_improvement(0.0, 5.0) == 0.0
    assert pct_improvement(-1.0, 5.0) == 0.0


# ---------------------------------------------------------------------------
# Benefit classification boundaries
# ---------------------------------------------------------------------------


def test_benefit_classification_boundaries() -> None:
    # NEUTRAL: below both thresholds.
    assert classify_cache_benefit(9.99, 14.99) == "NEUTRAL"
    # USEFUL: either threshold crossed (>= 10% OR >= 15 s).
    assert classify_cache_benefit(10.0, 0.0) == "USEFUL"
    assert classify_cache_benefit(0.0, 15.0) == "USEFUL"
    # Still USEFUL just below STRONG.
    assert classify_cache_benefit(24.99, 59.99) == "USEFUL"
    # STRONG: >= 25% OR >= 60 s.
    assert classify_cache_benefit(25.0, 0.0) == "STRONG"
    assert classify_cache_benefit(0.0, 60.0) == "STRONG"
    # Non-finite input is NEUTRAL (never a false positive).
    assert classify_cache_benefit(float("nan"), 100.0) == "NEUTRAL"


def test_benefit_classification_negative_is_neutral() -> None:
    # Reuse SLOWER than cold: no benefit.
    assert classify_cache_benefit(-5.0, -10.0) == "NEUTRAL"


# ---------------------------------------------------------------------------
# Environment bracket drift
# ---------------------------------------------------------------------------


def test_environment_bracket_drift_formula() -> None:
    # |10 - 10.2| / 10.1 * 100 = 1.980...
    assert environment_bracket_drift_pct(10.0, 10.2) == pytest.approx(
        0.2 / 10.1 * 100.0
    )
    assert environment_bracket_drift_pct(10.0, 10.0) == 0.0


def test_environment_bracket_drift_nonpositive_is_nan() -> None:
    assert math.isnan(environment_bracket_drift_pct(0.0, 5.0))
    assert math.isnan(environment_bracket_drift_pct(-1.0, 5.0))


# ---------------------------------------------------------------------------
# Steady-state regression math
# ---------------------------------------------------------------------------


def test_steady_state_regression_gate() -> None:
    assert steady_state_regression_pass(1.5, 3.0) is True
    assert steady_state_regression_pass(0.0, 0.0) is True
    # Negative deltas = reuse is faster / smaller: pass.
    assert steady_state_regression_pass(-1.0, -2.0) is True
    # p50 regression above 2%: fail.
    assert steady_state_regression_pass(2.01, 0.0) is False
    # Memory increase above 5%: fail.
    assert steady_state_regression_pass(0.0, 5.01) is False
    # Non-finite: fail closed.
    assert steady_state_regression_pass(float("nan"), 0.0) is False


# ---------------------------------------------------------------------------
# Diagnostic parser + graph-break classification
# ---------------------------------------------------------------------------

_SAMPLE_LOG = """\
V0909 02:58:42.751000 418943 site-packages/torch/_dynamo/symbolic_convert.py:611] [0/0] [__graph_breaks] Graph break in user code at /sakuramoon/model/attention.py:532
V0909 02:58:42.751000 418943 site-packages/torch/_dynamo/symbolic_convert.py:611] [0/0] [__graph_breaks] Graph Break Reason: Skip calling `torch.compiler.disable()`d function
V0909 02:58:42.751000 418943 site-packages/torch/_dynamo/symbolic_convert.py:611] [0/0] [__graph_breaks]  For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0098.html
SingleProcess AUTOTUNE benchmarking takes 3.2658 seconds and 6.2050 seconds precompiling for 37 choices
I0909 02:59:01.000000 418943 site-packages/torch/_dynamo/output_graph.py:3135] torch._dynamo: recompiling function packed_block_forward because guard L['x'].size()[1] failed
[bench rank0] measuring 5 logical updates
dynamic: specializing on tensor size 400
"""


def test_parse_compiler_log_event_level_breaks() -> None:
    parsed = parse_compiler_log(_SAMPLE_LOG)
    # ONE break event: the location line starts it, the reason line
    # attaches, and the "For more details ..." URL line is noise.
    assert len(parsed["graph_breaks"]) == 1
    entry = parsed["graph_breaks"][0]
    assert entry["kind"] == "location"
    assert entry["file"] == "/sakuramoon/model/attention.py"
    assert "torch.compiler.disable" in entry["reason"]
    assert entry["line_no"] == 1


def test_parse_compiler_log_excludes_autotune_precompiling() -> None:
    parsed = parse_compiler_log(_SAMPLE_LOG)
    # The Inductor autotune "precompiling for 37 choices" line contains the
    # substring "recompil" (inside "precompiling") but is NOT a recompile.
    assert len(parsed["recompiles"]) == 1
    assert (
        "recompiling function packed_block_forward" in parsed["recompiles"][0]["line"]
    )
    assert any("specializing" in line for line in parsed["dynamic_notes"])


def test_standalone_reason_line_is_own_event() -> None:
    parsed = parse_compiler_log(
        "WARNING graph break reason: unsupported tensor mutation\n"
    )
    assert len(parsed["graph_breaks"]) == 1
    assert parsed["graph_breaks"][0]["kind"] == "reason"
    assert "unsupported tensor mutation" in parsed["graph_breaks"][0]["reason"]


def test_classify_graph_breaks_expected_vs_unexpected() -> None:
    assert (
        classify_graph_break("Skip calling `torch.compiler.disable()`d function")
        == EXPECTED_BREAK
    )
    assert (
        classify_graph_break("fa4_varlen_attention is a torchdynamo_disable boundary")
        == EXPECTED_BREAK
    )
    assert (
        classify_graph_break("torch._dynamo.disable boundary in external kernel")
        == EXPECTED_BREAK
    )
    assert (
        classify_graph_break(
            "", file="/x/model/flash_attn2.py", function="das_fa2_forward"
        )
        == EXPECTED_BREAK
    )
    assert classify_graph_break("dynamic shape scalar extraction") == UNEXPECTED_BREAK
    assert classify_graph_break("unsupported tensor mutation") == UNEXPECTED_BREAK
    assert classify_graph_break("print() call in compiled region") == UNEXPECTED_BREAK


def test_classify_graph_breaks_counts_split() -> None:
    parsed = parse_compiler_log(_SAMPLE_LOG)
    expected, unexpected = classify_graph_breaks(parsed)
    assert expected == 1
    assert unexpected == 0


def test_classify_graph_breaks_line_nos_filter() -> None:
    parsed = parse_compiler_log(_SAMPLE_LOG)
    # Only the measured window (after the banner): the single break event is
    # in the warmup window, so the measured count is zero/zero.
    split = split_lines_at_measured_banner(_SAMPLE_LOG.splitlines())
    expected_meas, unexpected_meas = classify_graph_breaks(
        parsed, line_nos=set(split["measured"])
    )
    assert expected_meas == 0
    assert unexpected_meas == 0


def test_top_recompile_reason_most_common() -> None:
    parsed = parse_compiler_log(_SAMPLE_LOG)
    reason = top_recompile_reason(parsed)
    assert reason is not None
    assert "guard" in reason


def test_split_lines_at_measured_banner_found() -> None:
    split = split_lines_at_measured_banner(_SAMPLE_LOG.splitlines())
    assert split["found"] is True
    assert split["before"] == [1, 2, 3, 4, 5]
    assert split["measured"] == [6, 7]


def test_split_lines_at_measured_banner_absent_is_undefined() -> None:
    lines = ["a", "b", "c"]
    split = split_lines_at_measured_banner(lines)
    assert split["found"] is False
    assert split["measured"] == []
    assert split["before"] == [1, 2, 3]


def test_parse_compiler_log_empty_input() -> None:
    parsed = parse_compiler_log("")
    assert parsed == {"graph_breaks": [], "recompiles": [], "dynamic_notes": []}
    assert top_recompile_reason(parsed) is None
    assert classify_graph_breaks(parsed) == (0, 0)
    split = split_lines_at_measured_banner([])
    assert split["found"] is False


# ---------------------------------------------------------------------------
# Recompile classification (window split)
# ---------------------------------------------------------------------------


def test_split_recompiles_by_window_basic() -> None:
    warmup, measured = split_recompiles_by_window(5, measured_window_recompiles=0)
    assert (warmup, measured) == (5, 0)
    warmup, measured = split_recompiles_by_window(5, measured_window_recompiles=2)
    assert (warmup, measured) == (3, 2)


def test_split_recompiles_by_window_rejects_inconsistent() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        split_recompiles_by_window(3, measured_window_recompiles=4)
    with pytest.raises(ValueError, match="nonnegative"):
        split_recompiles_by_window(-1, measured_window_recompiles=0)


def test_diagnostic_summary_to_dict() -> None:
    summary = CompileDiagnosticSummary(
        graph_breaks_total=2,
        graph_breaks_expected=1,
        graph_breaks_unexpected=1,
        recompiles_total=3,
        recompiles_warmup=3,
        recompiles_measured=0,
        top_recompile_reason="guard size failed",
        dynamic_shape_notes=None,
        raw_log_preserved=True,
    )
    payload = summary.to_dict()
    assert payload["recompiles_measured"] == 0
    assert payload["raw_log_preserved"] is True
    assert payload["top_recompile_reason"] == "guard size failed"


def test_run_candidate_resume_loads_payloads(tmp_path: Path) -> None:
    """A resume must re-derive the SAME payloads a fresh PASS run loaded."""

    scripts_dir = Path(__file__).resolve().parents[3] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        import benchmark_compile_cache as bcc
    finally:
        sys.path.pop(0)

    ts_root = tmp_path / "ts"
    cand_root = ts_root / "a0"
    cand_root.mkdir(parents=True)
    (ts_root / "cold-a0").mkdir(parents=True)
    summary = {
        "global_step_seconds": {"p50": 15.0, "p95": 15.5, "mean": 15.1},
        "global_samples_per_second": 52.0,
        "memory": {
            "max_across_ranks": {
                "peak_allocated_bytes": 1 << 30,
                "peak_reserved_bytes": 2 << 30,
            }
        },
    }
    (cand_root / "baseline-2gpu.json").write_text(json.dumps(summary), encoding="utf-8")
    (cand_root / "startup-rank0.json").write_text(
        json.dumps({"time_to_first_successful_update_s": 650.0}),
        encoding="utf-8",
    )
    entry = bcc.run_candidate(
        "A0",
        python=sys.executable,
        repository_root=tmp_path,
        config="train_g1.toml",
        config_root="config",
        ts_root=ts_root,
        persistent_root=None,
        seed=44,
        warmup_updates=5,
        measure_updates=10,
        diagnostic_measure_updates=5,
        timeout=60,
        port=29611,
        fx_graph_cache=True,
        aot_autograd_cache=True,
        resume=True,
    )
    assert entry["status"] == "PASS"
    assert entry["resumed"] is True
    assert entry["summary"] == summary
    assert entry["startup_rank0"]["time_to_first_successful_update_s"] == 650.0
    steady = bcc._steady_metrics(entry)
    assert steady is not None
    assert steady["p50_s"] == 15.0
    up = bcc._startup_headline(entry)
    assert up["time_to_first_successful_update_s"] == 650.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
