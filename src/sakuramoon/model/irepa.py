"""iREPA training-only alignment projector (SakuraMoon iREPA v1).

Phase 2 scope: the projector module and its artifact metadata only.  The
frozen PE-Spatial teacher, the DiT tap, the spatial normalization, and the
cosine alignment loss are deliberately NOT part of this module; they belong
to later integration phases.
"""

from __future__ import annotations

import torch
from torch import nn

from sakuramoon.model.mixed_precision_conv import MixedPrecisionConv2d

# The locked iREPA v1 projector contract (PE-Spatial-B16-512 teacher width).
IREPA_TEACHER_FEATURE_WIDTH = 768
IREPA_PROJECTOR_KERNEL_SIZE = 3
IREPA_ARTIFACT_CLASS = "IRepaAlignment"
IREPA_ARTIFACT_SCHEMA_VERSION = 1

_DTYPE_NAMES = {
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
}


def irepa_alignment_metadata(in_channels: int) -> dict[str, object]:
    """The canonical v4 architecture document for one iREPA auxiliary.

    This is the single source of truth shared by the module export and the
    config assembly so that ``export(build(document)) == document`` holds
    without constructing a module.
    """

    if type(in_channels) is not int or in_channels <= 0:
        raise ValueError("irepa projector input width must be a positive int")
    return {
        "class": IREPA_ARTIFACT_CLASS,
        "schema_version": IREPA_ARTIFACT_SCHEMA_VERSION,
        "in_channels": in_channels,
        "out_channels": IREPA_TEACHER_FEATURE_WIDTH,
        "kernel_size": IREPA_PROJECTOR_KERNEL_SIZE,
        "stride": 1,
        "padding": 1,
        "dilation": 1,
        "groups": 1,
        "bias": True,
        "weight_dtype": "bfloat16",
        "bias_dtype": "float32",
    }


class IRepaAlignment(nn.Module):
    """Pure projector from DiT image hidden states to teacher feature width.

    The projector is a :class:`MixedPrecisionConv2d` (BF16 weight, FP32
    bias) so the Phase 1 generic convolution parameter policy audits it:
    ``weight`` -> matrix_decay (BF16), ``bias`` -> sensitive_no_decay
    (FP32).  No optimizer special-casing is added.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        out_channels: int = IREPA_TEACHER_FEATURE_WIDTH,
        kernel_size: int = IREPA_PROJECTOR_KERNEL_SIZE,
        stride: int = 1,
        padding: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        # The DTK venv torch stubs leave nn.Module.__init__ partially
        # unknown; same inline treatment as mixed_precision_conv.py.
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if type(in_channels) is not int or in_channels <= 0:
            raise ValueError("irepa projector input width must be a positive int")
        locked = (
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        if locked != (768, 3, 1, 1, 1, 1, True):
            raise ValueError(
                "irepa v1 projector contract is locked to "
                "out=768, kernel=3, stride=1, padding=1, dilation=1, "
                "groups=1, bias=true"
            )
        self.projector = MixedPrecisionConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(
        self,
        image_hidden: torch.Tensor,
        image_shape: tuple[int, int],
    ) -> torch.Tensor:
        """Project packed image tokens to the teacher feature width.

        ``image_hidden`` is ``[B, T, D]`` with ``T = H * W`` row-major;
        ``image_shape`` is ``(H, W)`` (non-square grids are supported).
        Returns ``[B, T, 768]``.
        """

        if image_hidden.dtype is not torch.bfloat16:
            raise TypeError(
                "IRepaAlignment requires a bfloat16 hidden state, "
                f"got {image_hidden.dtype}"
            )
        if image_hidden.ndim != 3:
            raise ValueError(
                "image_hidden must be packed tokens [B, T, D], "
                f"got {image_hidden.ndim} dimensions"
            )
        batch, tokens, width = image_hidden.shape
        if type(image_shape) is not tuple or len(image_shape) != 2:
            raise ValueError("image_shape must be an (H, W) pair of ints")
        height, grid_width = image_shape
        if type(height) is not int or type(grid_width) is not int:
            raise ValueError("image_shape must be an (H, W) pair of ints")
        if batch <= 0 or tokens <= 0 or width <= 0:
            raise ValueError("image_hidden dimensions must be positive")
        if height <= 0 or grid_width <= 0:
            raise ValueError("image_shape dimensions must be positive")
        if height * grid_width != tokens:
            raise ValueError(
                "image tokens do not cover the image grid "
                f"(H*W={height * grid_width} != T={tokens})"
            )
        if width != self.projector.in_channels:
            raise ValueError(
                "hidden width does not match the projector input width "
                f"(D={width} != in_channels={self.projector.in_channels})"
            )
        features = self.projector(
            image_hidden.reshape(batch, width, height, grid_width)
        )
        # flatten(2) keeps the spatial axis row-major (token t = h*W + w);
        # the transpose produces the documented [B, T, C] layout.
        return features.flatten(2).transpose(1, 2)

    def artifact_config(self) -> dict[str, object]:
        """Self-describing v4 artifact metadata for this projector."""

        conv = self.projector
        weight_name = _DTYPE_NAMES.get(conv.weight.dtype)
        bias_name = _DTYPE_NAMES.get(conv.bias.dtype) if conv.bias is not None else None
        if weight_name is None or bias_name is None:
            raise ValueError("irepa projector parameter dtypes are not locked")
        return {
            "class": IREPA_ARTIFACT_CLASS,
            "schema_version": IREPA_ARTIFACT_SCHEMA_VERSION,
            "in_channels": conv.in_channels,
            "out_channels": conv.out_channels,
            "kernel_size": conv.kernel_size[0],
            "stride": conv.stride[0],
            "padding": conv.padding[0],
            "dilation": conv.dilation[0],
            "groups": conv.groups,
            "bias": conv.bias is not None,
            "weight_dtype": weight_name,
            "bias_dtype": bias_name,
        }


__all__ = [
    "IREPA_ARTIFACT_CLASS",
    "IREPA_ARTIFACT_SCHEMA_VERSION",
    "IREPA_PROJECTOR_KERNEL_SIZE",
    "IREPA_TEACHER_FEATURE_WIDTH",
    "IRepaAlignment",
    "irepa_alignment_metadata",
]
