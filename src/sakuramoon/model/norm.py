"""Normalization used by the DiT reference path."""

from sakuramoon.conditioning.norm import FP32RMSNorm

RMSNorm = FP32RMSNorm

__all__ = ["RMSNorm"]
