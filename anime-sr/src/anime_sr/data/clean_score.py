"""§10.5 图像级 clean score: frozen offline sidecar + read-only training.

Two-stage filter (plan §10.5): stage 1 is the metadata/size filter at index
time; stage 2 (this module) scores each HR image with six deterministic,
pure-tensor heuristics (no RNG, no ML):

- ``blockiness``      8px block-boundary vs interior gradient ratio,
                      BOTH axes (gx for vertical block edges, gy for
                      horizontal — the legacy code only looked at gx)
- ``ringing``         2nd-derivative oscillation near strong edges
- ``blur``            2nd/1st-derivative energy ratio (sharpness proxy)
- ``flat noise``      Laplacian std in flat regions (low-quantile gradient)
- ``upscale suspicion`` 2x down-up residual energy (soft detail)
- ``edge overshoot``  Laplacian magnitude on the strong-edge band

Each component is normalized to [0, 1] where higher = cleaner; the final
score is their weighted mean. The normalization constants are calibration
freezes for Phase II (plan §10.5 leaves the thresholds to the fidelity
stage).

P1-4 (M4-prep work order, 2026-08-29) — two fixes:

1. block boundaries: 8px blocks put their boundary between pixels 7|8,
   15|16, ... so the boundary step is the gradient at index ``7::8`` — the
   legacy code measured ``8::8`` (the first INTERIOR pixel after the
   boundary) and therefore saw most of the block discontinuity as "interior
   gradient", diluting the ratio. Both axes are now measured (the gx-only
   version was blind to horizontal block edges).

2. the sidecar is FROZEN and READ-ONLY at training time. The legacy lazy
   path had every DDP rank / producer worker racing O_APPEND on the same
   JSONL behind a process-local ``threading.Lock`` (not cross-process
   synchronization at all). Scores are now precomputed OFFLINE by
   ``cli/clean_score_precompute`` (single writer, incremental); training
   loads the frozen sidecar once at start-up, prints the distribution
   report (percentiles, per-pool/quality/completeness/classification
   stats, candidate-threshold keep/exclude counts), and applies the
   ``[filter] clean_score_min`` gate only when the user has enabled it
   (default: disabled / report-only).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

__all__ = [
    "CLEAN_SCORE_CACHE_NAME",
    "CleanScoreCache",
    "build_clean_score_report",
    "clean_score_gate_retained",
    "compute_clean_score",
    "score_percentiles",
]

#: frozen sidecar file name under the index dir (data/index/clean-score-v1.jsonl)
CLEAN_SCORE_CACHE_NAME = "clean-score-v1.jsonl"


def _luminance(hr: torch.Tensor) -> torch.Tensor:
    """[3,H,W] in [-1,1] -> [H,W] in [0,1] (BT.601 luminance)."""
    g = 0.299 * hr[0] + 0.587 * hr[1] + 0.114 * hr[2]
    return (g * 0.5 + 0.5).clamp(0.0, 1.0)


def _laplacian(g: torch.Tensor) -> torch.Tensor:
    k = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
    return F.conv2d(
        g.unsqueeze(0).unsqueeze(0), k.unsqueeze(0).unsqueeze(0), padding=1
    ).squeeze()


def _component_scores(g: torch.Tensor) -> dict[str, float]:
    """Six bounded [0,1] component scores (higher = cleaner)."""
    eps = 1e-6
    h, w = g.shape[-2:]
    gx = g[:, 1:] - g[:, :-1]
    gy = g[1:] - g[:-1]
    gmag = torch.cat([gx.reshape(-1), gy.reshape(-1)], 0).abs()
    l = _laplacian(g)

    # --- blockiness: 8px block-boundary discontinuities vs interior --------
    # 8px blocks put their boundary between pixels 7|8, 15|16, ...: the
    # boundary step is the gradient AT index 7, 15, ... (7::8) — NOT 8::8,
    # which is the first interior pixel AFTER the boundary (the legacy bug).
    # gx (horizontal gradient) carries VERTICAL block edges; gy (vertical
    # gradient) carries HORIZONTAL ones — both must be counted (the legacy
    # gx-only version was blind to half the block structure).
    def _boundary_interior(diff: torch.Tensor, axis: int) -> tuple[float, float]:
        # the boundary step is at index 7, 15, ... ALONG the block
        # direction: axis 1 for gx (column steps), axis 0 for gy (row
        # steps)
        n = diff.shape[axis]
        if n < 8:
            return 0.0, 0.0
        bnd_idx = torch.arange(7, n, 8)  # 7, 15, 23, ...
        int_idx = torch.tensor(
            [i for i in range(n) if i % 8 != 7], dtype=torch.long
        )  # interior = every index except 7,15,...
        if axis == 1:  # gx: [H, W-1], steps indexed by column
            bnd = diff[:, bnd_idx].abs().mean().item()
            interior = diff[:, int_idx].abs().mean().item() if len(int_idx) else 0.0
        else:  # gy: [H-1, W], steps indexed by row
            bnd = diff[bnd_idx, :].abs().mean().item()
            interior = diff[int_idx, :].abs().mean().item() if len(int_idx) else 0.0
        return bnd, interior

    b_x, i_x = _boundary_interior(gx, axis=1)  # vertical block edges
    b_y, i_y = _boundary_interior(gy, axis=0)  # horizontal block edges
    b = 0.5 * (b_x + b_y)
    i = 0.5 * (i_x + i_y)
    ratio = (b + eps) / (i + eps)
    s_block = 1.0 / (1.0 + max(ratio - 1.0, 0.0))

    # --- edge / flat masks (per-pixel |gx| proxy; gx is [H, W-1]) --------
    ap = gx.abs()
    strong_q = torch.quantile(ap.flatten().float(), 0.90).item()
    strong = ap >= strong_q
    edge_mask = strong | torch.roll(strong, 1, 1) | torch.roll(strong, -1, 1)
    e_mag = ap[edge_mask].mean().item()  # edge strength, before the pad below
    flat_q = torch.quantile(ap.flatten().float(), 0.10).item()
    flat_mask = ap <= flat_q
    # ap is [H, W-1] but the Laplacian is [H, W]: pad the masks to full width
    edge_band = F.pad(edge_mask, (0, 1))
    flat = F.pad(flat_mask, (0, 1))

    # --- ringing / overshoot: |L| on the edge band, scaled by edge strength
    e_osc = l[edge_band].abs().mean().item()
    osc = e_osc / (e_mag + eps)
    s_ring = 1.0 / (1.0 + 4.0 * osc)
    s_over = max(0.0, 1.0 - 1.5 * osc)

    # --- blur: 2nd/1st derivative energy ratio (sharpness proxy) ----------
    r21 = l.abs().mean().item() / (gmag.mean().item() + eps)
    s_blur = min(max(r21, 0.0), 1.0)

    # --- flat-region noise: Laplacian std where |gx| ~ 0 ------------------
    flat_std = l[flat].std().item() if int(flat.sum()) > 0 else 0.0
    s_flat = max(0.0, 1.0 - flat_std / 0.02)

    # --- upscale suspicion: 2x down-up residual (soft detail) --------------
    g2 = F.interpolate(
        g.unsqueeze(0).unsqueeze(0), size=(h // 2, w // 2), mode="bilinear"
    ).squeeze()
    g2 = F.interpolate(g2.unsqueeze(0).unsqueeze(0), size=(h, w), mode="bilinear").squeeze()
    res = (g - g2).abs().mean().item() / (g.abs().mean().item() + eps)
    s_up = min(res * 8.0, 1.0)

    return {
        "block": s_block,
        "ring": s_ring,
        "blur": s_blur,
        "flat": s_flat,
        "up": s_up,
        "over": s_over,
    }


def compute_clean_score(hr: torch.Tensor) -> float:
    """Clean score in [0,1] (higher = cleaner) for a full HR image.

    ``hr``: ``[3, H, W]`` fp32 in ``[-1, 1]``. Pure and deterministic:
    same pixels -> same score (no RNG anywhere in the path).
    """
    if hr.ndim != 3 or hr.shape[0] != 3:
        raise ValueError(f"expected [3,H,W], got {tuple(hr.shape)}")
    g = _luminance(hr.float().clamp(-1.0, 1.0))
    c = _component_scores(g)
    # unweighted mean; weights are a Phase-II calibration knob (plan §10.5)
    return sum(c.values()) / len(c)


class CleanScoreCache:
    """READ-ONLY frozen sidecar (P1-4, 2026-08-29).

    Training NEVER computes or appends: the legacy lazy path had every DDP
    rank / producer worker racing O_APPEND on the same JSONL behind a
    process-local ``threading.Lock`` — not cross-process synchronization.
    Scores are precomputed OFFLINE by ``cli/clean_score_precompute``
    (single writer, incremental, resumable); this class only loads the
    frozen sidecar at start-up and serves lookups. A missing file is not an
    error (scores simply unavailable -> report-only mode).

    Lines: one per sample ``{"sample_id": ..., "score": ...}``; when the
    same id appears twice (incremental re-runs), the LAST line wins.
    """

    def __init__(self, index_dir: str | Path) -> None:
        self.path = Path(index_dir) / CLEAN_SCORE_CACHE_NAME
        self.scores: dict[str, float] = {}
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self.scores[str(rec["sample_id"])] = float(rec["score"])

    def __len__(self) -> int:
        return len(self.scores)

    def get(self, sample_id: str) -> float | None:
        """Score for ``sample_id``, or None when the sidecar has no row."""
        return self.scores.get(sample_id)


def clean_score_gate_retained(
    sample_ids: list[str], index_dir: str | Path, min_score: float
) -> set[str] | None:
    """Fail-closed retained set for the training gate.

    ``min_score < 0`` -> None (gate disabled, report-only mode). Otherwise
    the retained set is ``{sid: score >= min_score}`` — a sample WITHOUT a
    sidecar row is EXCLUDED (fail-closed: unverified is not clean). The set
    depends only on the frozen sidecar + config, so every DDP rank computes
    the same one."""
    if min_score is None or min_score < 0:
        return None
    scores = CleanScoreCache(index_dir).scores
    return {
        sid for sid in sample_ids
        if scores.get(sid) is not None and scores[sid] >= min_score
    }


def score_percentiles(scores: list[float] | torch.Tensor) -> dict[str, float]:
    """p10/p25/p50/p75/p90 (+mean) of a score vector; empty -> all 0.0."""
    if len(scores) == 0:
        return {"p10": 0.0, "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "mean": 0.0}
    v = torch.as_tensor(scores, dtype=torch.float32)
    q = torch.quantile(v, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], dtype=torch.float32))
    return {
        "p10": round(float(q[0]), 6),
        "p25": round(float(q[1]), 6),
        "p50": round(float(q[2]), 6),
        "p75": round(float(q[3]), 6),
        "p90": round(float(q[4]), 6),
        "mean": round(float(v.mean()), 6),
    }


def build_clean_score_report(
    index_dir: str | Path,
    sample_ids: list[str],
    candidate_thresholds: list[float] | None = None,
) -> dict:
    """Distribution report for the frozen sidecar (P1-4, report-only).

    Reads the sidecar + the eligibility table and assembles:
      * overall percentiles (p10/p25/p50/p75/p90) of the covered samples;
      * per sampling_pool / quality / anime_completeness /
        anime_classification stats (n, percentiles, mean);
      * for each candidate threshold: how many samples a ``clean_score_min``
        gate would KEEP vs EXCLUDE (the user decides the threshold from
        these numbers — the report never drops data itself).
    """
    from anime_sr.data import index as index_mod
    from anime_sr.data.pipeline import find_eligibility

    index_dir = Path(index_dir)
    cache = CleanScoreCache(index_dir)
    candidates = list(candidate_thresholds or [])
    ids = list(dict.fromkeys(sample_ids))  # dedupe, keep order
    idset = set(ids)
    covered = {sid: s for sid, s in cache.scores.items() if sid in idset}
    vals = list(covered.values())

    # index rows provide the grouping fields (pool/quality/completeness/
    # classification); the report degrades gracefully without the table
    groups: dict[str, dict[str, list[float]]] = {
        "sampling_pool": {},
        "quality": {},
        "anime_completeness": {},
        "anime_classification": {},
    }
    try:
        for row in index_mod.iter_index(find_eligibility(index_dir)):
            sid = str(row["sample_id"])
            if sid not in covered:
                continue
            s = covered[sid]
            for field in ("sampling_pool", "quality", "anime_completeness", "anime_classification"):
                key = str(row.get(field) or "unknown")
                groups[field].setdefault(key, []).append(s)
    except FileNotFoundError:
        pass

    group_stats = {
        field: {
            key: {"n": len(v), **score_percentiles(v)}
            for key, v in sorted(sub.items())
        }
        for field, sub in groups.items()
    }
    thresholds = {
        str(round(t, 4)): {
            "kept": sum(1 for s in vals if s >= t),
            "excluded": sum(1 for s in vals if s < t),
        }
        for t in candidates
    }
    return {
        "sidecar": str(cache.path),
        "n_requested": len(ids),
        "n_covered": len(vals),
        "coverage": round(len(vals) / len(ids), 4) if ids else 0.0,
        "percentiles": score_percentiles(vals),
        "groups": group_stats,
        "candidate_thresholds": thresholds,
    }
