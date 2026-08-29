"""P1-4 clean score: blockiness index fix, horizontal (gy) coverage, and
the frozen read-only sidecar + report-only gate.

Contract under test:
  * blockiness: 8px block boundaries are at 7|8, 15|16, ... so the boundary
    step is the gradient at index 7::8 (the legacy 8::8 read the first
    INTERIOR pixel after the boundary and diluted the ratio); a step placed
    exactly at the 7|8 boundary must collapse the blockiness component
    (asserted on the component, not the 6-way aggregate, which the
    upscale-suspicion term can offset on synthetic 8px-periodic content);
  * horizontal block edges (gy) are counted — the legacy gx-only version
    was blind to them (its block score would stay ~1 on row steps);
  * the sidecar is READ-ONLY at training time: lookups never append/write;
  * the gate is fail-closed (no sidecar row = excluded) and disabled by a
    negative threshold;
  * the report exposes percentiles + per-group stats + candidate
    keep/exclude counts.
"""

from __future__ import annotations

import json

import pytest
import torch
from anime_sr.data.clean_score import (
    CLEAN_SCORE_CACHE_NAME,
    CleanScoreCache,
    _component_scores,
    build_clean_score_report,
    clean_score_gate_retained,
    score_percentiles,
)


def test_block_boundary_index_78() -> None:
    """The 7::8-vs-8::8 fix: a step placed EXACTLY at the 7|8 block
    boundary (the gx column-7 step) must collapse the blockiness component.
    The legacy 8::8 read measured the first INTERIOR pixel after the
    boundary (step 8|9 = 0 here), so it would have kept block ~= 1."""
    h, w = 96, 128
    smooth = torch.linspace(0.0, 0.6, w).unsqueeze(0).expand(h, w).contiguous()
    blocky = smooth.clone()
    blocky[:, 8::8] += 0.3  # discontinuity between columns 7|8, 15|16, ...
    c_smooth = _component_scores(smooth)
    c_blocky = _component_scores(blocky)
    assert c_smooth["block"] > 0.9, f"smooth ramp should score ~clean: {c_smooth['block']}"
    assert c_blocky["block"] < 0.5 * c_smooth["block"], (
        f"boundary steps not penalized: {c_blocky['block']} vs {c_smooth['block']}"
    )


def test_horizontal_block_edges_counted() -> None:
    """Steps at each 7|8 ROW boundary (the gy row-7 step) must also collapse
    the blockiness component — the legacy gx-only version was blind to
    horizontal block edges (its block score would stay ~1 here)."""
    h, w = 128, 96
    smooth = torch.linspace(0.0, 0.6, h).unsqueeze(1).expand(h, w).contiguous()
    blocky = smooth.clone()
    blocky[8::8, :] += 0.3  # horizontal block edges at rows 7|8, 15|16, ...
    c_smooth = _component_scores(smooth)
    c_blocky = _component_scores(blocky)
    assert c_smooth["block"] > 0.9, f"smooth ramp should score ~clean: {c_smooth['block']}"
    assert c_blocky["block"] < 0.5 * c_smooth["block"], (
        f"horizontal block edges not counted: {c_blocky['block']} vs {c_smooth['block']}"
    )


def test_sidecar_last_line_wins(tmp_path) -> None:
    sidecar = tmp_path / CLEAN_SCORE_CACHE_NAME
    sidecar.write_text(
        json.dumps({"sample_id": "a", "score": 0.1}) + "\n"
        + json.dumps({"sample_id": "a", "score": 0.8}) + "\n",
        encoding="utf-8",
    )
    assert CleanScoreCache(tmp_path).get("a") == 0.8


def test_gate_disabled_and_fail_closed(tmp_path) -> None:
    sidecar = tmp_path / CLEAN_SCORE_CACHE_NAME
    sidecar.write_text(
        json.dumps({"sample_id": "a", "score": 0.9}) + "\n"
        + json.dumps({"sample_id": "c", "score": 0.4}) + "\n",
        encoding="utf-8",
    )
    # disabled (negative threshold) -> None (no filtering)
    assert clean_score_gate_retained(["a", "b", "c"], tmp_path, -1.0) is None
    # enabled at 0.5: keep only score >= 0.5; "b" (no row) is EXCLUDED
    # (fail-closed: unverified is not clean)
    assert clean_score_gate_retained(["a", "b", "c"], tmp_path, 0.5) == {"a"}
    # threshold 0.0 keeps everything that has a row
    assert clean_score_gate_retained(["a", "b", "c"], tmp_path, 0.0) == {"a", "c"}


def test_percentiles() -> None:
    p = score_percentiles([float(i) for i in range(1, 101)])
    assert p["p50"] == 50.5
    assert p["p10"] < p["p25"] < p["p50"] < p["p75"] < p["p90"]
    empty = score_percentiles([])
    assert all(v == 0.0 for v in empty.values())


def _write_eligibility(index_dir, rows: list[dict]) -> None:
    (index_dir / "sr-eligibility-v1.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
    )


def test_report_percentiles_groups_candidates(tmp_path) -> None:
    rows = [
        {"sample_id": "a", "sampling_pool": "priority", "quality": "great",
         "anime_completeness": "polished", "anime_classification": "illustration"},
        {"sample_id": "b", "sampling_pool": "regular", "quality": "normal",
         "anime_completeness": "normal", "anime_classification": "illustration"},
        {"sample_id": "c", "sampling_pool": "aux", "quality": "low",
         "anime_completeness": "rough", "anime_classification": "3d"},
        {"sample_id": "d", "sampling_pool": "priority", "quality": "best",
         "anime_completeness": "polished", "anime_classification": "bangumi"},
    ]
    _write_eligibility(tmp_path, rows)
    (tmp_path / CLEAN_SCORE_CACHE_NAME).write_text(
        json.dumps({"sample_id": "a", "score": 0.9}) + "\n"
        + json.dumps({"sample_id": "b", "score": 0.6}) + "\n"
        + json.dumps({"sample_id": "c", "score": 0.3}) + "\n"
        + json.dumps({"sample_id": "d", "score": 0.7}) + "\n",
        encoding="utf-8",
    )
    rep = build_clean_score_report(tmp_path, ["a", "b", "c", "d"], [0.5, 0.7])
    assert rep["n_covered"] == 4 and rep["n_requested"] == 4
    assert rep["coverage"] == 1.0
    p = rep["percentiles"]
    assert p["p50"] == pytest.approx(0.65, abs=0.02)
    # group stats: priority pool has 2 samples
    assert rep["groups"]["sampling_pool"]["priority"]["n"] == 2
    assert rep["groups"]["sampling_pool"]["aux"]["n"] == 1
    assert rep["groups"]["anime_classification"]["3d"]["n"] == 1
    # candidate thresholds: min=0.5 keeps a,b,d (>=0.5), excludes c
    assert rep["candidate_thresholds"]["0.5"] == {"kept": 3, "excluded": 1}
    assert rep["candidate_thresholds"]["0.7"] == {"kept": 2, "excluded": 2}
