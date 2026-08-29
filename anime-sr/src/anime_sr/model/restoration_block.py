"""Restoration block (plan §7.3): GQA attention + SwiGLU-FFN with dw3x3.

Two sub-blocks per RestorationBlock (plan §7.3, frozen):

    x -> RMSNorm -> stage FiLM -> GQA attention -> output projection
      -> LayerScale -> residual

    x -> RMSNorm -> stage FiLM -> SwiGLU -> depthwise 3x3 on the expanded
      branch -> output projection -> LayerScale -> residual

Decisions (documented in design.md):
    * "SwiGLU" is the standard SiLU-gated linear pair: h = W2 (SiLU(W1 h)
      * (W3 h)) with W1, W3: dim -> 3*dim and W2: 3*dim -> dim (FFN widths
      1152/1536/2304 are exactly 3*dim, §7.2).
    * The depthwise 3x3 acts on the 3*dim expanded intermediate (spatial
      local mixing inside the FFN, "on the expanded branch"), applied with
      the block's (H, W) layout.
    * LayerScale: per-channel parameter, init 1e-3 (plan §7.3).
    * High-resolution stages alternate normal / shifted 8x8 windows
      (plan §7.3); the bottleneck uses global attention (§7.2).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from anime_sr.model.conditioning import GlobalConditioner, StageFilms
from anime_sr.model.window_attention import RMSNorm2d, WindowAttention
from anime_sr.model.window_attention_sdpa import WindowAttentionSDPA

__all__ = [
    "RestorationBlock",
    "build_attention",
]

#: ``hardware.attention_backend`` values (U233 P2-2): the default is the
#: frozen, verified CORRECTNESS (manual) core; the SDPA variants are
#: benchmark-ready and only become the default after a parity/benchmark
#: decision — never by construction.
_CORRECTNESS_BACKENDS = ("correctness", "manual", "sdpa-correctness")
_SDPA_BACKENDS = ("sdpa-repeat", "sdpa-native-gqa")


def build_attention(
    dim: int,
    num_heads: int,
    num_kv_heads: int,
    *,
    backend: str,
    head_dim: int = 64,
    window_size: int = 8,
    global_attention: bool = False,
    qk_norm: bool = True,
) -> WindowAttention:
    """Select the attention implementation for ``hardware.attention_backend``.

    * ``correctness`` / ``manual`` / ``sdpa-correctness`` (default): the
      frozen explicit core — the verified production default;
    * ``sdpa-repeat``: SDPA core with repeat_interleave GQA (bit-safest
      SDPA variant);
    * ``sdpa-native-gqa``: SDPA core with native GQA (``enable_gqa=True``,
      Hq != Hkv — backend support required).

    All variants share identical weights (state_dict-compatible)."""
    b = str(backend).strip().lower()
    common = {
        "dim": dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "window_size": window_size,
        "global_attention": global_attention,
        "qk_norm": qk_norm,
    }
    if b in _CORRECTNESS_BACKENDS:
        return WindowAttention(**common)
    if b == "sdpa-repeat":
        return WindowAttentionSDPA(gqa_native=False, **common)
    if b == "sdpa-native-gqa":
        return WindowAttentionSDPA(gqa_native=True, **common)
    raise ValueError(
        f"unknown hardware.attention_backend {backend!r}; expected one of "
        f"{sorted(_CORRECTNESS_BACKENDS + _SDPA_BACKENDS)}"
    )


class RestorationBlock(nn.Module):
    """One §7.3 restoration block: windowed GQA attention + SwiGLU-dw FFN."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        ffn: int | None = None,
        *,
        global_attention: bool = False,
        head_dim: int = 64,
        window_size: int = 8,
        layerscale_init: float = 1e-3,
        qk_norm: bool = True,
        attention_backend: str = "correctness",
    ) -> None:
        super().__init__()
        if num_kv_heads < 1 or num_heads % num_kv_heads:
            raise ValueError(
                f"num_heads {num_heads} must be a positive multiple of num_kv_heads {num_kv_heads}"
            )
        expanded = ffn if ffn is not None else 3 * dim
        if expanded <= 0 or dim > expanded:
            raise ValueError(f"ffn {expanded} must be a positive int >= dim")
        self.dim = dim
        self.attention_backend = str(attention_backend).strip().lower()
        self.attn = build_attention(
            dim,
            num_heads,
            num_kv_heads,
            backend=self.attention_backend,
            head_dim=head_dim,
            window_size=window_size,
            global_attention=global_attention,
            qk_norm=qk_norm,
        )
        self.norm1 = RMSNorm2d(dim)
        self.attn_scale = nn.Parameter(torch.full((dim,), layerscale_init))
        self.norm2 = RMSNorm2d(dim)
        self.w1 = nn.Linear(dim, expanded, bias=False)
        self.w3 = nn.Linear(dim, expanded, bias=False)
        self.w2 = nn.Linear(expanded, dim, bias=False)
        self.dw3x3 = nn.Conv2d(expanded, expanded, 3, padding=1, groups=expanded)
        self.ffn_scale = nn.Parameter(torch.full((dim,), layerscale_init))

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        stage: str,
        films: StageFilms,
        conditioner: GlobalConditioner,
        shift: bool = False,
        offset: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """x: (B, N, dim) row-major over the (H, W) grid; N == H*W."""
        B, N, _ = x.shape
        if N != H * W:
            raise ValueError(f"x has {N} tokens but grid {H}x{W} expects {H * W}")
        # ---- attention sub-block ----
        h = self.norm1(x)
        h = conditioner.apply(stage, h, films)
        h = self.attn(h, H, W, shift=shift, offset=offset)
        x = x + self.attn_scale * h
        # ---- SwiGLU-dw FFN sub-block ----
        h = self.norm2(x)
        h = conditioner.apply(stage, h, films)
        h_exp = F.silu(self.w1(h)) * self.w3(h)  # (B, N, 3dim) expanded branch
        C = h_exp.shape[-1]
        h_exp = h_exp.view(B, H, W, C).permute(0, 3, 1, 2)
        h_exp = self.dw3x3(h_exp).permute(0, 2, 3, 1).reshape(B, N, C)
        h = self.w2(h_exp)
        return x + self.ffn_scale * h
