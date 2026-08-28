"""§10.5 图像级 clean score: lazy computation on first read + cache.

Two-stage filter (plan §10.5): stage 1 is the metadata/size filter at index
time; stage 2 (this module) computes the clean score on FIRST read of each
HR image and caches it, so training never waits on a whole-dataset decode
pass. Six deterministic, pure-tensor heuristics (no RNG, no ML):

- ``blockiness``      8px block-boundary vs interior gradient ratio
- ``ringing``         2nd-derivative oscillation near strong edges
- ``blur``            2nd/1st-derivative energy ratio (sharpness proxy)
- ``flat noise``      Laplacian std in flat regions (low-quantile gradient)
- ``upscale suspicion`` 2x down-up residual energy (soft detail)
- ``edge overshoot``  Laplacian magnitude on the strong-edge band

Each component is normalized to [0, 1] where higher = cleaner; the final
score is their weighted mean. The normalization constants are calibration
freezes for Phase II (plan §10.5 leaves the thresholds to the fidelity
stage) — what Phase I needs is the plumbing: lazy, cached, deterministic.

Cache: one JSONL line per sample (``{"sample_id", "score"}``) under the
index dir. DDP ranks share the file; lines are short (O_APPEND is atomic
enough for <4 KiB lines), each rank reads the existing lines at start-up
and keeps an in-memory read-through dict so a sample is computed exactly
once per process lifetime.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import torch
import torch.nn.functional as F

__all__ = ["CLEAN_SCORE_CACHE_NAME", "CleanScoreCache", "compute_clean_score"]

#: cache file name under the index dir (data/index/clean-score-v1.jsonl)
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
    b = gx[:, 8::8].abs().mean().item()
    mask = torch.ones(gx.shape[1], dtype=torch.bool)
    mask[8::8] = False
    i = gx[:, mask].abs().mean().item()
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
    """Read-through JSONL cache: compute on first read, then serve from memory.

    Concurrent DDP ranks: each rank loads the file at start-up; appends are
    short O_APPEND lines (atomic enough in practice for <4 KiB writes).
    A sample is computed exactly once per process lifetime.
    """

    def __init__(self, index_dir: str | Path) -> None:
        self.path = Path(index_dir) / CLEAN_SCORE_CACHE_NAME
        self._lock = threading.Lock()
        self._scores: dict[str, float] = {}
        if self.path.is_file():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    self._scores[str(rec["sample_id"])] = float(rec["score"])

    def __len__(self) -> int:
        return len(self._scores)

    def get(self, sample_id: str, hr: torch.Tensor) -> float:
        """Return the score for ``hr`` (sample_id), computing+appending once."""
        with self._lock:
            if sample_id in self._scores:
                return self._scores[sample_id]
            score = round(compute_clean_score(hr), 6)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"sample_id": sample_id, "score": score}) + "\n")
            self._scores[sample_id] = score
            return score
