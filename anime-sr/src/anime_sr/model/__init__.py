"""U-Flow Transformer core + Pixel Condition Encoder (plan §6-§7, milestone M3).

Modules:
    pixel_encoder     §6.2 LQ RGB condition encoder (5 scales + GAP)
    window_attention  §7.1-§7.2 continuous 2D RoPE + GQA window attention
    restoration_block §7.3 attention + SwiGLU-dw restoration block
    conditioning     §7.6 timestep/sigma/GAP(p16) -> five stage FiLMs
    output_head       §7.7 zero-init 128-channel velocity head
    uflow             §7.4/§7.5 trunk assembly + full AnimeSRModel (§8)
"""

from anime_sr.model.conditioning import GlobalConditioner, StageFilms, TimestepEmbedding
from anime_sr.model.output_head import OutputHead
from anime_sr.model.pixel_encoder import (
    PixelConditionBlock,
    PixelConditionEncoder,
    PixelConditionOutputs,
)
from anime_sr.model.restoration_block import RestorationBlock
from anime_sr.model.uflow import AnimeSRModel, UFlowSR, count_parameters
from anime_sr.model.window_attention import RMSNorm2d, RoPE2D, WindowAttention

__all__ = [
    "AnimeSRModel",
    "GlobalConditioner",
    "OutputHead",
    "PixelConditionBlock",
    "PixelConditionEncoder",
    "PixelConditionOutputs",
    "RMSNorm2d",
    "RestorationBlock",
    "RoPE2D",
    "StageFilms",
    "TimestepEmbedding",
    "UFlowSR",
    "WindowAttention",
    "count_parameters",
]
