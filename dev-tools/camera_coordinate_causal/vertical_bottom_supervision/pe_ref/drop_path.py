# ruff: noqa  (vendored verbatim - see LICENSE.PE / NOTICE)
# Vendored from huggingface timm (MIT license), file timm/layers/drop_path.py,
# class DropPath - the exact implementation imported by the official PE
# vision encoder (pe_vision.py).  Vendored because the VBS audit venv does not
# install timm; in PE-Spatial-B16-512 drop_path=0.0, so this module is loaded
# but never active.  Copied verbatim (no functional changes).
import torch
from torch import nn


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with dim=(b,1,...,1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
        random_tensor.trunc_()  # binarize
        return x * random_tensor

    def extra_repr(self):
        return f"p={self.drop_prob}"


__all__ = ["DropPath"]
