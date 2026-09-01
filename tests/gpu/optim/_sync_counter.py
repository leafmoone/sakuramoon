"""Host-sync counter for optimizer-step instrumentation (cleanup spec §14).

Wraps the host readback entry points that force a device->host
synchronization on CUDA/HCU tensors:

  * Tensor.item()      (scalar readback)
  * Tensor.tolist()    (vector readback)
  * Tensor.cpu()       (explicit host copy)
  * torch.cuda.synchronize / stream.synchronize (explicit fences)

Collective operations (all_reduce / broadcast) do not block the host by
themselves; only their readbacks do, and those are counted through the same
hooks. The counter is a context manager so a step (or a set of steps) can be
measured without leaking the monkeypatches.
"""

from __future__ import annotations

from typing import Self

import torch

__all__ = ["SyncCounter"]


class SyncCounter:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {
            "item": 0,
            "tolist": 0,
            "cpu_copy": 0,
            "synchronize": 0,
        }
        self._active = False
        self._orig_item = torch.Tensor.item
        self._orig_tolist = torch.Tensor.tolist
        self._orig_cpu = torch.Tensor.cpu
        self._orig_sync = torch.cuda.synchronize

    def __enter__(self) -> Self:
        self._active = True
        oc = self

        def item(self_t):
            if oc._active and self_t.is_cuda:
                oc.counts["item"] += 1
            return oc._orig_item(self_t)

        def tolist(self_t):
            if oc._active and self_t.is_cuda:
                oc.counts["tolist"] += 1
            return oc._orig_tolist(self_t)

        def cpu(self_t):
            if oc._active and self_t.is_cuda:
                oc.counts["cpu_copy"] += 1
            return oc._orig_cpu(self_t)

        def sync(*a, **k):
            if oc._active:
                oc.counts["synchronize"] += 1
            return oc._orig_sync(*a, **k)

        torch.Tensor.item = item
        torch.Tensor.tolist = tolist
        torch.Tensor.cpu = cpu
        torch.cuda.synchronize = sync
        return self

    def __exit__(self, *exc) -> bool:
        torch.Tensor.item = self._orig_item
        torch.Tensor.tolist = self._orig_tolist
        torch.Tensor.cpu = self._orig_cpu
        torch.cuda.synchronize = self._orig_sync
        self._active = False
        return False

    def total(self) -> int:
        return sum(self.counts.values())
