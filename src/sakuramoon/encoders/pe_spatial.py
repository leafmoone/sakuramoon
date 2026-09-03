"""Frozen PE-Spatial-B16-512 teacher encoder for iREPA (training-only).

Phase 3 integrates the official PE-Spatial-B16-512 vision encoder as a
frozen teacher: it is loaded from the fingerprint-bound local asset, cast
to BF16, and run under ``torch.no_grad()`` on the exact final GPU training
image tensor (``[B,3,H,W]`` bfloat16 in ``[-1,1]``, as produced by
``SingleGpuBatchRuntime.prepare``).  The teacher is deliberately kept out
of ``TrainableComposite``: it has no optimizer entry, no DDP wrapping, and
no checkpoint state.  The production training gate stays fail-closed until
Phase 4 installs the student tap and loss integration.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

from sakuramoon.assets.pe_spatial import (
    APPROVED_PE_SPATIAL_B16_512,
    PeSpatialTeacherFingerprint,
    require_local_pe_spatial_teacher,
)
from sakuramoon.pe_spatial.config import PE_VISION_CONFIG
from sakuramoon.pe_spatial.pe_vision import VisionTransformer

TEACHER_NAME = "PE-Spatial-B16-512"
TEACHER_PATCH_SIZE = 16
TEACHER_FEATURE_WIDTH = 768


@dataclass(frozen=True)
class PESpatialTeacherOutput:
    """Raw per-patch teacher features for one training image batch.

    ``patch_features`` is ``[B, T, 768]`` with ``T = (H // 16) * (W // 16)``
    in row-major raster order (token index ``t = y * grid_w + x``), matching
    the SakuraMoon latent raster order.  The CLS prefix token is stripped
    per the official model semantics (``use_cls_token = True``) and no
    pooled feature is produced.
    """

    patch_features: torch.Tensor
    image_shape: tuple[int, int]


def _normalize_checkpoint_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Apply the official checkpoint key normalization (module./visual. strips)."""

    normalized = {key.replace("module.", ""): value for key, value in state_dict.items()}
    if any(key.startswith("visual.") for key in normalized):
        normalized = {
            key.replace("visual.", ""): value
            for key, value in normalized.items()
            if "visual" in key
        }
    return normalized


class FrozenPESpatialEncoder(nn.Module):
    """Frozen, BF16, no-grad PE-Spatial-B16-512 vision teacher.

    Construction is private: use :meth:`load_asset`, which verifies the
    fingerprint-bound asset, performs a strict (fail-closed) state-dict
    load, and freezes the module.  Only the single approved model is
    representable: the architecture is fixed to the official
    ``PE-Spatial-B16-512`` configuration.
    """

    def __init__(self) -> None:
        # The DTK venv torch stubs leave nn.Module.__init__ partially
        # unknown; same inline treatment as mixed_precision_conv.py.
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        vision_config = PE_VISION_CONFIG[TEACHER_NAME]
        self.visual: VisionTransformer = VisionTransformer(**asdict(vision_config))
        if self.visual.use_cls_token is not True:
            raise RuntimeError(
                "PE-Spatial-B16-512 is defined with a CLS prefix token; "
                "the teacher CLS-stripping contract would be invalid"
            )
        if (
            self.visual.patch_size != TEACHER_PATCH_SIZE
            or self.visual.width != TEACHER_FEATURE_WIDTH
            or self.visual.layers != APPROVED_PE_SPATIAL_B16_512.depth
        ):
            raise RuntimeError("vendored PE-Spatial-B16-512 config drifted")
        self.eval()

    @classmethod
    def load_asset(
        cls,
        repository_root: Path,
        teacher_local_path: str,
        *,
        device: torch.device,
        expected: PeSpatialTeacherFingerprint = APPROVED_PE_SPATIAL_B16_512,
    ) -> FrozenPESpatialEncoder:
        """Verify the local asset and load the frozen teacher on ``device``."""

        weight_path = require_local_pe_spatial_teacher(
            repository_root, teacher_local_path, expected=expected
        )
        encoder = cls()
        raw = torch.load(weight_path, weights_only=True)
        if not isinstance(raw, dict):
            raise TypeError("teacher checkpoint must be a state-dict mapping")
        raw_map = cast("dict[object, object]", raw)
        inner: object = raw_map.get("state_dict", raw_map.get("weights", raw_map))
        if not isinstance(inner, dict):
            raise TypeError("teacher checkpoint state dict is invalid")
        state_dict: dict[str, torch.Tensor] = {}
        for key, value in cast("dict[object, object]", inner).items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                raise TypeError(
                    "teacher checkpoint state dict must map str to torch.Tensor"
                )
            state_dict[key] = value
        encoder.visual.load_state_dict(
            _normalize_checkpoint_state_dict(state_dict), strict=True
        )
        encoder.visual = encoder.visual.to(
            device=device, dtype=torch.bfloat16
        )  # pyright: ignore[reportUnknownMemberType]
        encoder.eval()
        for parameter in encoder.parameters():
            parameter.requires_grad_(False)
        return encoder

    def train(self, mode: bool = True) -> FrozenPESpatialEncoder:
        if mode:
            raise RuntimeError("the frozen PE-Spatial teacher must never enter train()")
        return super().train(mode)

    @property
    def device(self) -> torch.device:
        return self.visual.conv1.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.visual.conv1.weight.dtype

    def forward(self, images: torch.Tensor) -> PESpatialTeacherOutput:
        """Encode ``[B,3,H,W]`` bfloat16 ``[-1,1]`` images to patch features.

        The input must already be the final GPU training tensor on this
        module's device: no resize, crop, pad, re-normalization, CPU
        roundtrip, or second host-to-device copy is performed.
        """

        if not isinstance(images, torch.Tensor):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("teacher input must be a torch.Tensor")
        if images.ndim != 4:
            raise ValueError(f"teacher input must be [B,3,H,W], got {tuple(images.shape)}")
        if images.shape[1] != 3:
            raise ValueError(
                "teacher input must have 3 channels, "
                f"got {images.shape[1]}"
            )
        if images.dtype is not torch.bfloat16:
            raise TypeError(
                "teacher input must be bfloat16, "
                f"got {images.dtype}"
            )
        if images.device != self.device:
            raise RuntimeError(
                "teacher input must already reside on the teacher device; "
                "a second host-to-device copy is forbidden"
            )
        height, width = images.shape[2], images.shape[3]
        if height % TEACHER_PATCH_SIZE or width % TEACHER_PATCH_SIZE:
            raise ValueError(
                "teacher input height/width must be divisible by 16, "
                f"got {height}x{width}"
            )
        grid_h, grid_w = height // TEACHER_PATCH_SIZE, width // TEACHER_PATCH_SIZE
        with torch.no_grad():
            features = self.visual(images)
        expected_tokens = 1 + grid_h * grid_w
        if features.shape[0] != images.shape[0] or features.shape[1] != expected_tokens:
            raise RuntimeError(
                "PE-Spatial teacher returned an unexpected token layout: "
                f"{tuple(features.shape)} for {images.shape[0]} images at "
                f"{grid_h}x{grid_w} grid"
            )
        # Strip the CLS prefix token per the official model semantics
        # (use_cls_token=True); never guess by token count.
        patch_features = features[:, 1:, :]
        return PESpatialTeacherOutput(
            patch_features=patch_features,
            image_shape=(grid_h, grid_w),
        )


def prepare_teacher_targets(
    encoder: FrozenPESpatialEncoder,
    images: torch.Tensor,
) -> PESpatialTeacherOutput:
    """Standalone teacher-target preparation for iREPA (Phase 3).

    Consumes the exact final GPU training image tensor
    (``[B,3,H,W]`` bfloat16 in ``[-1,1]``, as produced by
    ``SingleGpuBatchRuntime.prepare``).  In Phase 3 this method is only
    invoked from the HCU audit and tests; it is not executed from the
    production measure/backward path.  Phase 4 wires it into the runtime
    once the student tap and loss integration are installed.
    """

    return encoder(images)


__all__ = [
    "TEACHER_FEATURE_WIDTH",
    "TEACHER_NAME",
    "TEACHER_PATCH_SIZE",
    "FrozenPESpatialEncoder",
    "PESpatialTeacherOutput",
    "prepare_teacher_targets",
]
