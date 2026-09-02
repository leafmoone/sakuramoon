"""Dense DiT reference primitives."""

from sakuramoon.model.attention import (
    DenseGQAAttention,
    FA4VarlenGQAAttention,
    dense_attention_mask,
    fa4_varlen_attention,
)
from sakuramoon.model.block import DiTBlock, PackedDiTBlock
from sakuramoon.model.dit import (
    DenseDiT,
    DenseDiTFeatures,
    PackedDiT,
    PackedDiTFeatures,
)
from sakuramoon.model.growth import active_slot_ids, new_slot_ids, slot_name
from sakuramoon.model.mixed_precision_conv import MixedPrecisionConv2d
from sakuramoon.model.mlp import SwiGLU
from sakuramoon.model.norm import RMSNorm
from sakuramoon.model.output_head import FinalOutputHead

__all__ = [
    "DenseDiT",
    "DenseDiTFeatures",
    "DenseGQAAttention",
    "DiTBlock",
    "FA4VarlenGQAAttention",
    "FinalOutputHead",
    "MixedPrecisionConv2d",
    "PackedDiT",
    "PackedDiTBlock",
    "PackedDiTFeatures",
    "RMSNorm",
    "SwiGLU",
    "active_slot_ids",
    "dense_attention_mask",
    "fa4_varlen_attention",
    "new_slot_ids",
    "slot_name",
]
