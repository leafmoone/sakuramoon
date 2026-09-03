# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Vendored from facebookresearch/perception_models, commit
# 3e352cca660658d4b5c90f42a7808b11469e4c66, file
# core/vision_encoder/config.py, reduced to the vision PEConfig dataclass and
# the exact PE-Spatial-B16-512 definition required by the SakuraMoon iREPA
# frozen teacher.  The text-tower configs, other model entries, and the
# huggingface_hub checkpoint fetcher are intentionally not vendored.
# Apache-2.0; see LICENSE.PE.

"""PE-Spatial-B16-512 vision tower configuration (official definition)."""

from dataclasses import dataclass, replace

__all__ = ["PE_VISION_CONFIG", "PEConfig"]


@dataclass
class PEConfig:
    """ Vision Tower Config. """

    patch_size: int
    width: int
    layers: int
    heads: int
    mlp_ratio: float
    output_dim: int | None

    ls_init_value: float = None
    drop_path: float = 0.0

    image_size: int = 224
    use_abs_posemb: bool = True
    use_cls_token: bool = False
    use_rope2d: bool = True

    pool_type: str = "attn"
    attn_pooler_heads: int = 8

    use_ln_pre: bool = True
    use_ln_post: bool = True


PE_VISION_CONFIG: dict[str, PEConfig] = {}


PE_VISION_CONFIG["PE-Core-B16-224"] = PEConfig(
    image_size=224,
    patch_size=16,
    width=768,
    layers=12,
    heads=12,
    mlp_ratio=4.0,
    pool_type="attn",
    output_dim=1024,
    use_cls_token=True,
)


PE_VISION_CONFIG["PE-Spatial-B16-512"] = replace(
    PE_VISION_CONFIG["PE-Core-B16-224"],
    image_size=512,
    pool_type="none",
    use_ln_post=False,
    output_dim=None,
)
