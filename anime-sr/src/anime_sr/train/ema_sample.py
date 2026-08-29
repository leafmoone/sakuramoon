"""Sample-based weight EMA (P2-prep, for M4 production checkpoints).

The M4 loop runs dynamic U-Flow: the per-step batch composition (and thus
the number of *samples* consumed per optimizer step) varies across steps
and resolution buckets. A fixed per-step decay therefore drifts with the
batch mix. This module ties the decay to the number of samples actually
consumed: the configured ``decay`` is the EMA retention after exactly
``ref_samples`` samples; an update that consumes ``n`` samples applies

    beta_n = decay ** (n / ref_samples)
    ema <- beta_n * ema + (1 - beta_n) * param      (fp32)

so an update that processes twice the reference batch ages the EMA twice
as far. The EMA copy is always kept in fp32 regardless of the live
parameter dtype (bf16 training); ``apply``/``restore`` bridge to the live
dtype.

DDP note: each rank keeps its own EMA copy and updates it with its own
per-rank sample count. DDP keeps the live parameters bit-identical
across ranks and the per-rank update count is identical, so the recurrences
start from the same seed and stay bit-identical; the rank-0 copy is what
gets written to the checkpoint.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from torch import nn

__all__ = ["SampleEMA"]


class SampleEMA:
    """Sample-rate-based EMA over a module's parameters (fp32 shadow state)."""

    def __init__(self, module: nn.Module, decay: float, ref_samples: int) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"decay must be in (0, 1), got {decay}")
        if ref_samples < 1:
            raise ValueError(f"ref_samples must be >= 1, got {ref_samples}")
        # DDP: unwrap so fqn refers to the bare module (no "module." prefix).
        m = self._unwrap(module)
        self.decay = float(decay)
        self.ref_samples = int(ref_samples)
        self.n_samples_total = 0
        # fqn -> fp32 shadow parameter (insertion-ordered, stable across updates)
        self._shadow: OrderedDict[str, torch.Tensor] = OrderedDict()
        for fqn, p in m.named_parameters():
            self._shadow[fqn] = p.detach().to(torch.float32).clone()

    # ------------------------------------------------------------------
    def _unwrap(self, module: nn.Module) -> nn.Module:
        if hasattr(module, "module") and isinstance(module.module, nn.Module):
            return module.module
        return module

    def _match(self, module: nn.Module) -> Mapping[str, torch.Tensor]:
        live = dict(self._unwrap(module).named_parameters())
        if set(live) != set(self._shadow):
            missing = set(self._shadow) - set(live)
            extra = set(live) - set(self._shadow)
            raise ValueError(f"EMA/param mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        return live

    def update(self, module: nn.Module, n_samples: int) -> float:
        """One EMA update after an optimizer step that consumed ``n_samples``.

        Returns the effective retention ``beta_n`` applied.
        """
        if n_samples < 1:
            raise ValueError(f"n_samples must be >= 1, got {n_samples}")
        beta = self.decay ** (float(n_samples) / float(self.ref_samples))
        live = self._match(module)
        one_minus = 1.0 - beta
        for fqn, p in live.items():
            # in-place: the fp32 shadow casts the live (e.g. bf16) value up
            self._shadow[fqn].mul_(beta).add_(p.detach(), alpha=one_minus)
        self.n_samples_total += int(n_samples)
        return beta

    # ------------------------------------------------------------------
    def avg_state_dict(self) -> dict[str, torch.Tensor]:
        """EMA weights only (for saving / diffing against live weights)."""
        return {fqn: e.detach().clone() for fqn, e in self._shadow.items()}

    def state_dict(self) -> dict:
        return {
            "decay": self.decay,
            "ref_samples": self.ref_samples,
            "n_samples_total": self.n_samples_total,
            "params": self.avg_state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        if float(sd["decay"]) != self.decay or int(sd["ref_samples"]) != self.ref_samples:
            raise ValueError(
                f"EMA config mismatch: file ({sd['decay']}, {sd['ref_samples']}) "
                f"vs live ({self.decay}, {self.ref_samples})"
            )
        params = sd["params"]
        if set(params) != set(self._shadow):
            missing = set(self._shadow) - set(params)
            extra = set(params) - set(self._shadow)
            raise ValueError(f"EMA param keys mismatch: missing={sorted(missing)} extra={sorted(extra)}")
        for fqn, e in self._shadow.items():
            e.copy_(params[fqn].to(torch.float32).to(e.device))
        self.n_samples_total = int(sd.get("n_samples_total", 0))

    # ------------------------------------------------------------------
    def apply(self, module: nn.Module) -> dict[str, torch.Tensor]:
        """Swap the EMA weights into the live module (live dtype).

        Returns the previous live weights so ``restore`` can put them back
        (used for val probes under the EMA model, e.g. before a M4 launch).
        """
        live = self._match(module)
        prev: dict[str, torch.Tensor] = {}
        for fqn, p in live.items():
            prev[fqn] = p.detach().clone()
            p.data.copy_(self._shadow[fqn].to(dtype=p.dtype, device=p.device))
        return prev

    def restore(self, module: nn.Module, prev: dict[str, torch.Tensor]) -> None:
        live = self._match(module)
        for fqn, p in live.items():
            p.data.copy_(prev[fqn].to(dtype=p.dtype, device=p.device))
