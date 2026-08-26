"""2D continuous RoPE + grouped-query window attention (plan §7.1, §7.2, §7.3).

Layout decisions (all frozen by plan §7.2):
    * head_dim = 64, qk_norm = RMSNorm on q and k, SwiGLU elsewhere, dropout = 0
    * position = continuous 2D RoPE: x- and y-integer coordinates, absolute at
      inference (tiles pass their offset so large images see global coordinates)
    * local stages: 8x8 windows, alternating normal / shifted (shift 4,4)
    * bottleneck: global attention

2D RoPE head layout: head_dim 64 -> dims [0:16]+[16:32] rotate-pair on the x
coordinate, dims [32:48]+[48:64] on y (16 geometric frequencies each,
base 10000). Coordinates may be non-integer (continuous); the cos/sin tables
are cached per (H, W, offset, device).
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

__all__ = [
    "RMSNorm2d",
    "RoPE2D",
    "WindowAttention",
]


class RMSNorm2d(nn.Module):
    """RMSNorm over the feature dim (plan §7.2: norm = RMSNorm)."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


class RoPE2D(nn.Module):
    """Continuous 2D rotary embeddings for head_dim 64 (16 x-pairs, 16 y-pairs).

    ``apply`` rotates q/k of shape (..., head_dim) using per-token (x, y)
    integer/float coordinates of shape (N, 2).
    """

    inv_freq: torch.Tensor

    def __init__(self, head_dim: int = 64, base: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 32:
            raise ValueError(f"head_dim must be divisible by 32 for 2D RoPE, got {head_dim}")
        self.head_dim = head_dim
        self.half_dim = head_dim // 4  # 16
        half = self.half_dim
        inv_freq = base ** (-2.0 * torch.arange(half, dtype=torch.float32) / (2.0 * half))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def _tables(self, coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """coords: (N, 2) float (x=col, y=row). Returns cos/sin (N, head_dim//2)."""
        device = coords.device
        inv = self.inv_freq.to(device=device, dtype=torch.float32)
        c = coords.to(dtype=torch.float32)
        angle_x = torch.outer(c[:, 0], inv)  # (N, half)
        angle_y = torch.outer(c[:, 1], inv)
        angles = torch.cat([angle_x, angle_y], dim=1)  # (N, head_dim//2)
        return angles.cos(), angles.sin()

    # Overrides nn.Module.apply(fn) on purpose: RoPE2D is called as
    # rope.apply(q, k, coords) and is never fed through module.apply.
    def apply(  # type: ignore[reportIncompatibleMethodOverride]
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """q/k: (B, H, N, head_dim); coords: (N, 2). Returns rotated q/k.

        Head layout: dims [0:16]+[16:32] rotate-pair on x, [32:48]+[48:64] on y."""
        cos, sin = self._tables(coords)
        cos = cos.view(1, 1, q.shape[2], 2 * self.half_dim)
        sin = sin.view(1, 1, q.shape[2], 2 * self.half_dim)
        cos_x, cos_y = cos[..., : self.half_dim], cos[..., self.half_dim :]
        sin_x, sin_y = sin[..., : self.half_dim], sin[..., self.half_dim :]

        def rotate(x: torch.Tensor) -> torch.Tensor:
            # x: (B, H, N, 64); pairs: [0:16]+[16:32] on x, [32:48]+[48:64] on y
            a0 = x[..., 0 : self.half_dim]
            a1 = x[..., self.half_dim : self.half_dim * 2]
            b0 = x[..., self.half_dim * 2 : self.half_dim * 3]
            b1 = x[..., self.half_dim * 3 : self.half_dim * 4]
            out_x = torch.cat([a0 * cos_x - a1 * sin_x, a1 * cos_x + a0 * sin_x], dim=-1)
            out_y = torch.cat([b0 * cos_y - b1 * sin_y, b1 * cos_y + b0 * sin_y], dim=-1)
            return torch.cat([out_x, out_y], dim=-1)

        return rotate(q), rotate(k)


class WindowAttention(nn.Module):
    """GQA attention with 8x8 window partitioning and shifted windows.

    Input x: (B, N, dim), tokens row-major over an (H, W) grid.
    Output projection (num_heads*head_dim -> dim) is included (plan §7.3
    "output projection").
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int = 64,
        window_size: int = 8,
        global_attention: bool = False,
        qk_norm: bool = True,
    ) -> None:
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        if num_heads % num_kv_heads:
            raise ValueError(f"num_heads {num_heads} must be a multiple of num_kv_heads {num_kv_heads}")
        kv_dim = num_kv_heads * head_dim
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.global_attention = global_attention
        self.gqa_rep = num_heads // num_kv_heads
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)
        self.q_norm = RMSNorm2d(head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm2d(head_dim) if qk_norm else nn.Identity()
        self.rope = RoPE2D(head_dim)
        self._coord_cache: OrderedDict[tuple[int, int, int, int], torch.Tensor] = OrderedDict()

    # ------------------------------------------------------------------
    def _coords(self, H: int, W: int, offset: tuple[int, int], device: torch.device) -> torch.Tensor:
        """(N, 2) absolute (x=col, y=row) coordinates, cached."""
        key = (H, W, offset[0], offset[1])
        cached = self._coord_cache.get(key)
        if cached is not None:
            self._coord_cache.move_to_end(key)
            return cached.to(device)
        ys, xs = torch.meshgrid(
            torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij"
        )
        coords = torch.stack([xs + offset[0], ys + offset[1]], dim=-1).reshape(-1, 2).to(torch.float32)
        self._coord_cache[key] = coords
        while len(self._coord_cache) > 32:
            self._coord_cache.popitem(last=False)
        return coords

    def _windowed_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        H: int,
        W: int,
        shift: bool,
    ) -> torch.Tensor:
        """q/k/v: (B, Hn | kv_h, N, head_dim) -> (B, N, kv_dim); 8x8 windows."""
        B, Hn, N, D = q.shape
        w = self.window_size
        s = w // 2
        gh, gw = H // w, W // w
        k = k.repeat_interleave(self.gqa_rep, dim=1)  # (B, Hn, N, D)
        v = v.repeat_interleave(self.gqa_rep, dim=1)

        if shift:
            q = torch.roll(q.view(B, Hn, H, W, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, N, D)
            k = torch.roll(k.view(B, Hn, H, W, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, N, D)
            v = torch.roll(v.view(B, Hn, H, W, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, N, D)

        def partition(t: torch.Tensor) -> torch.Tensor:
            # (B, Hn, N, D) -> (B*gh*gw, Hn, w*w, D)
            x = t.view(B, Hn, gh, w, gw, w, D)
            x = x.permute(0, 2, 4, 1, 3, 5, 6)
            return x.reshape(B * gh * gw, Hn, w * w, D)

        qw, kw, vw = partition(q), partition(k), partition(v)
        scale = D**-0.5
        logits = torch.matmul(qw, kw.transpose(-1, -2)) * scale
        attn = logits.float().softmax(dim=-1)
        out = torch.matmul(attn.to(v.dtype), vw)  # (B*gh*gw, Hn, w*w, D)
        out = out.view(B, gh, gw, Hn, w, w, D)
        out = out.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, Hn, N, D)
        if shift:
            out = torch.roll(out.view(B, Hn, H, W, D), shifts=(s, s), dims=(2, 3)).view(B, Hn, N, D)
        return out.transpose(1, 2).reshape(B, N, Hn * D)

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        shift: bool = False,
        offset: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        """x: (B, N, dim) row-major; H*W == N. offset = absolute tile origin (x, y)."""
        B, N, _ = x.shape
        N = H * W
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        coords = self._coords(H, W, offset, x.device)
        q, k = self.rope.apply(q, k, coords)
        if self.global_attention:
            scale = self.head_dim**-0.5
            k_r = k.repeat_interleave(self.gqa_rep, dim=1)
            v_r = v.repeat_interleave(self.gqa_rep, dim=1)
            logits = torch.matmul(q, k_r.transpose(-1, -2)) * scale
            attn = logits.float().softmax(dim=-1)
            out = torch.matmul(attn.to(v.dtype), v_r)  # (B, Hn, N, D)
            out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        else:
            out = self._windowed_attention(q, k, v, H, W, shift)  # (B, N, Hn*D)
        return self.o_proj(out)
