"""Dense DiT reference primitives."""

from sakuramoon.model.attention import DenseGQAAttention, dense_attention_mask
from sakuramoon.model.block import DiTBlock
from sakuramoon.model.dit import DenseDiT, DenseDiTFeatures
from sakuramoon.model.growth import active_slot_ids, new_slot_ids, slot_name
from sakuramoon.model.mlp import SwiGLU
from sakuramoon.model.norm import RMSNorm
from sakuramoon.model.output_head import FinalOutputHead

__all__ = [
    "DenseDiT",
    "DenseDiTFeatures",
    "DenseGQAAttention",
    "DiTBlock",
    "FinalOutputHead",
    "RMSNorm",
    "SwiGLU",
    "active_slot_ids",
    "dense_attention_mask",
    "new_slot_ids",
    "slot_name",
]
