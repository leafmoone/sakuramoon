"""Vendored official PE vision encoder (Apache-2.0, Meta Platforms).

Contains the frozen-teacher-only subset of
``facebookresearch/perception_models`` (commit
``3e352cca660658d4b5c90f42a7808b11469e4c66``) needed to run
PE-Spatial-B16-512 as a frozen iREPA teacher: the vision transformer
classes, the 2D rotary positional embedding, and the official
PE-Spatial-B16-512 configuration.  See ``NOTICE`` and ``LICENSE.PE``.
"""

from sakuramoon.pe_spatial.config import PE_VISION_CONFIG, PEConfig
from sakuramoon.pe_spatial.pe_vision import VisionTransformer
from sakuramoon.pe_spatial.rope import Rope2D

__all__ = ["PE_VISION_CONFIG", "PEConfig", "Rope2D", "VisionTransformer"]
