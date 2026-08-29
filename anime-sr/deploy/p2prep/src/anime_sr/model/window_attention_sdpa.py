"""SDPA-backed window attention (P2-prep item 3).

``WindowAttentionSDPA`` is a drop-in subclass of :class:`WindowAttention`
(plan §7.1-§7.3): identical weights, identical 2D-RoPE / qk-RMSNorm /
GQA / 8x8 shifted-window / clipped-boundary semantics, but the attention
core (QK^T scale -> softmax -> V) runs on
:func:`torch.nn.functional.scaled_dot_product_attention` so it can use
the platform's fused/efficient kernels (HCU) instead of the explicit
``matmul -> fp32 masked_fill -> softmax -> matmul`` path.

Everything outside the core is shared with the parent (projections,
norms, RoPE tables, shift/boundary masks, roll and partition helpers),
so parity with the frozen parent is the acceptance gate (see
``tests/test_p2_sdpa_parity.py``); the throughput comparison is
``tools/bench_attention_backends.py``.

GQA handling: ``gqa_native=False`` (default) repeats k/v heads with
``repeat_interleave`` exactly like the parent -- bit-safest across every
SDPA backend. ``gqa_native=True`` passes q (Hq heads) and k/v (Hkv
heads) directly and relies on the backend's native GQA (Hq % Hkv == 0);
math fallbacks that cannot broadcast will reject it, so the benchmark
tries the native variant explicitly and reports.

Mask convention: the parent's masks are bool with True = "attend"
(``_shift_mask``, padded-path ``attend`` incl. the diagonal exemption);
SDPA's ``attn_mask`` uses the same True-attends convention, so the
masks are passed through after a ``repeat(B, ...)`` along the window dim.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from anime_sr.model.window_attention import WindowAttention

__all__ = ["WindowAttentionSDPA"]


class WindowAttentionSDPA(WindowAttention):
    """Same frozen semantics as ``WindowAttention``; SDPA attention core."""

    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int = 64,
                 window_size: int = 8, global_attention: bool = False, qk_norm: bool = True,
                 gqa_native: bool = False) -> None:
        super().__init__(dim=dim, num_heads=num_heads, num_kv_heads=num_kv_heads, head_dim=head_dim,
                          window_size=window_size, global_attention=global_attention, qk_norm=qk_norm)
        self.gqa_native = bool(gqa_native)
        self._qk_norm_flag = bool(qk_norm)  # the parent does not store the flag itself

    # ------------------------------------------------------------------
    def _sdpa_core(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    attn_mask: torch.Tensor | None) -> torch.Tensor:
        """Fused core with the parent's dtype flow.

        The parent's RoPE (fp32 cos/sin tables) upcasts q/k to fp32 even for
        bf16 modules, while v keeps the module dtype; the parent computes
        logits/softmax in fp32 then casts back before the final V matmul
        (``attn.to(v.dtype) @ v``).  We mirror that: SDPA runs in q's dtype
        (v cast up), and the output is cast back to v's dtype so ``o_proj``
        sees the same dtype the parent produces.
        """
        out_dtype = v.dtype
        if v.dtype != q.dtype:
            v = v.to(q.dtype)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        return out.to(out_dtype) if out.dtype != out_dtype else out

    def _qkv_rope(self, x: torch.Tensor, H: int, W: int, offset: tuple[int, int]):
        """Front half of the parent forward: projections + qk RMSNorm + 2D RoPE."""
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q, k = self.q_norm(q), self.k_norm(k)
        coords = self._coords(H, W, offset, x.device)
        q, k = self.rope.apply(q, k, coords)
        return q, k, v

    def forward(
        self,
        x: torch.Tensor,
        H: int,
        W: int,
        shift: bool = False,
        offset: tuple[int, int] = (0, 0),
    ) -> torch.Tensor:
        B, N, _ = x.shape
        N = H * W
        q, k, v = self._qkv_rope(x, H, W, offset)
        if self.global_attention:
            if not self.gqa_native:
                k = k.repeat_interleave(self.gqa_rep, dim=1)
                v = v.repeat_interleave(self.gqa_rep, dim=1)
            out = self._sdpa_core(q, k, v, None)  # (B, Hn, N, D)
            out = out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
            return self.o_proj(out)
        out = self._windowed_sdpa(q, k, v, H, W, shift)
        return self.o_proj(out)

    # ------------------------------------------------------------------
    def _windowed_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                       H: int, W: int, shift: bool) -> torch.Tensor:
        w = self.window_size
        Hp = ((H + w - 1) // w) * w
        Wp = ((W + w - 1) // w) * w
        if not self.gqa_native:
            k = k.repeat_interleave(self.gqa_rep, dim=1)
            v = v.repeat_interleave(self.gqa_rep, dim=1)
        if Hp != H or Wp != W:
            return self._padded_sdpa(q, k, v, H, W, Hp, Wp, shift)
        return self._exact_sdpa(q, k, v, H, W, shift)

    def _exact_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                    H: int, W: int, shift: bool) -> torch.Tensor:
        """Un-padded path: same roll/partition as the parent, SDPA core."""
        B, Hn, N, D = q.shape
        w = self.window_size
        s = w // 2
        gh, gw = H // w, W // w

        if shift:
            q = torch.roll(q.view(B, Hn, H, W, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, N, D)
            k = torch.roll(k.view(B, Hn, H, W, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, N, D)
            v = torch.roll(v.view(B, Hn, H, W, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, N, D)

        def partition(t: torch.Tensor) -> torch.Tensor:
            x = t.view(B, Hn, gh, w, gw, w, D)
            x = x.permute(0, 2, 4, 1, 3, 5, 6)
            return x.reshape(B * gh * gw, Hn, w * w, D)

        qw, kw, vw = partition(q), partition(k), partition(v)
        attn_mask = None
        if shift:
            # parent mask: (G, w*w, w*w) bool, True = attend (clipped neighbourhood)
            mask = self._shift_mask(H, W, q.device)
            attn_mask = mask.repeat(B, 1, 1).unsqueeze(1)  # (B*G, 1, w*w, w*w)
        out = self._sdpa_core(qw, kw, vw, attn_mask)  # (B*G, Hn, w*w, D)
        out = out.view(B, gh, gw, Hn, w, w, D)
        out = out.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, Hn, N, D)
        if shift:
            out = torch.roll(out.view(B, Hn, H, W, D), shifts=(s, s), dims=(2, 3)).view(B, Hn, N, D)
        return out.transpose(1, 2).reshape(B, N, Hn * D)

    def _padded_sdpa(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     H: int, W: int, Hp: int, Wp: int, shift: bool) -> torch.Tensor:
        """Padded path: same zero-pad + roll + key-validity + boundary masks
        as the parent (incl. the diagonal self-token exemption), SDPA core."""
        B, Hn, N, D = q.shape  # N == H*W
        w = self.window_size
        s = w // 2
        gh, gw = Hp // w, Wp // w

        def pad(t: torch.Tensor) -> torch.Tensor:
            g = t.view(B, Hn, H, W, D)
            g = F.pad(g, (0, 0, 0, Wp - W, 0, Hp - H))
            return g.view(B, Hn, Hp * Wp, D)

        q, k, v = pad(q), pad(k), pad(v)
        dev = q.device

        if shift:
            y_pre = (torch.arange(Hp, device=dev) + s) % Hp
            x_pre = (torch.arange(Wp, device=dev) + s) % Wp
            key_ok = (y_pre.view(gh, 1, w, 1) < H) & (x_pre.view(1, gw, 1, w) < W)
        else:
            rows = torch.arange(Hp, device=dev)
            cols = torch.arange(Wp, device=dev)
            key_ok = (rows.view(gh, 1, w, 1) < H) & (cols.view(1, gw, 1, w) < W)
        key_ok_g = key_ok.reshape(gh * gw, w * w)  # (G, w*w) True = real key

        if shift:
            q = torch.roll(q.view(B, Hn, Hp, Wp, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, Hp * Wp, D)
            k = torch.roll(k.view(B, Hn, Hp, Wp, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, Hp * Wp, D)
            v = torch.roll(v.view(B, Hn, Hp, Wp, D), shifts=(-s, -s), dims=(2, 3)).view(B, Hn, Hp * Wp, D)
            boundary = self._shift_mask(Hp, Wp, dev)  # (G, w*w, w*w)
            attend = key_ok_g.view(gh * gw, 1, w * w) & boundary  # (G, w*w, w*w)
        else:
            attend = key_ok_g.view(gh * gw, 1, w * w)
        attend = attend | torch.eye(w * w, dtype=torch.bool, device=dev).view(1, w * w, w * w)
        attend = attend.view(gh * gw, w * w, w * w)

        def partition(t: torch.Tensor) -> torch.Tensor:
            x = t.view(B, Hn, gh, w, gw, w, D)
            x = x.permute(0, 2, 4, 1, 3, 5, 6)
            return x.reshape(B * gh * gw, Hn, w * w, D)

        qw, kw, vw = partition(q), partition(k), partition(v)
        attn_mask = attend.repeat(B, 1, 1).unsqueeze(1)  # (B*G, 1, w*w, w*w)
        out = self._sdpa_core(qw, kw, vw, attn_mask)
        out = out.view(B, gh, gw, Hn, w, w, D)
        out = out.permute(0, 3, 1, 4, 2, 5, 6).reshape(B, Hn, Hp * Wp, D)
        if shift:
            out = torch.roll(out.view(B, Hn, Hp, Wp, D), shifts=(s, s), dims=(2, 3)).view(B, Hn, Hp * Wp, D)
        out = out.view(B, Hn, Hp, Wp, D)[:, :, :H, :W, :].reshape(B, Hn, N, D)
        return out.transpose(1, 2).reshape(B, N, Hn * D)


def sdpa_variant(module: WindowAttentionSDPA) -> WindowAttentionSDPA:
    """Return the same module with the opposite GQA mode (for A/B benches)."""
    other = WindowAttentionSDPA(
        dim=module.dim,
        num_heads=module.num_heads,
        num_kv_heads=module.num_kv_heads,
        head_dim=module.head_dim,
        window_size=module.window_size,
        global_attention=module.global_attention,
        qk_norm=module._qk_norm_flag,
        gqa_native=not module.gqa_native,
    )
    other.load_state_dict(module.state_dict())
    # match the source module's device AND dtype (load_state_dict copies
    # into freshly fp32-initialized weights; the upcast-then-downcast
    # round-trip of load + to(dtype) is lossless for the source values)
    param = next(iter(module.parameters()))
    other.to(device=param.device, dtype=param.dtype)
    return other
