"""Global canvas/time conditioning and zero-initialized modulation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

FREQUENCY_BASE = 10000.0


@dataclass(frozen=True)
class BlockModulation:
    attention_scale: torch.Tensor
    attention_shift: torch.Tensor
    attention_gate: torch.Tensor
    mlp_scale: torch.Tensor
    mlp_shift: torch.Tensor
    mlp_gate: torch.Tensor

    def for_active_index(self, active_index: int) -> BlockModulation:
        if active_index < 0 or active_index >= self.attention_scale.shape[1]:
            raise IndexError("active modulation index is outside the selected slots")
        return BlockModulation(
            attention_scale=self.attention_scale[:, active_index],
            attention_shift=self.attention_shift[:, active_index],
            attention_gate=self.attention_gate[:, active_index],
            mlp_scale=self.mlp_scale[:, active_index],
            mlp_shift=self.mlp_shift[:, active_index],
            mlp_gate=self.mlp_gate[:, active_index],
        )


@dataclass(frozen=True)
class GlobalConditionOutput:
    condition_hidden: torch.Tensor
    block: BlockModulation
    final_scale: torch.Tensor
    final_shift: torch.Tensor


def fixed_sinusoidal_embedding(values: torch.Tensor, dimension: int) -> torch.Tensor:
    if values.ndim != 1 or values.dtype != torch.float32:
        raise ValueError("fixed embedding values must be a 1D FP32 tensor")
    if dimension <= 0 or dimension % 2:
        raise ValueError("fixed embedding dimension must be a positive even integer")
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(FREQUENCY_BASE)
        * torch.arange(half, device=values.device, dtype=torch.float32)
        / half
    )
    arguments = values[:, None] * frequencies[None]
    return torch.cat((arguments.cos(), arguments.sin()), dim=-1)


class GlobalConditioner(nn.Module):
    def __init__(
        self,
        *,
        timestep_dim: int,
        size_dim: int,
        aspect_dim: int,
        hidden_dim: int,
        model_dim: int,
        slot_count: int,
        active_slot_ids: tuple[int, ...],
        modulation_chunks: int,
        final_modulation_size: int,
    ) -> None:
        super().__init__()
        if (
            timestep_dim != 256
            or size_dim != 64
            or aspect_dim != 64
            or hidden_dim != 1024
            or model_dim <= 0
            or slot_count <= 0
            or not active_slot_ids
            or len(set(active_slot_ids)) != len(active_slot_ids)
            or any(slot_id < 0 or slot_id >= slot_count for slot_id in active_slot_ids)
            or modulation_chunks != 6
            or final_modulation_size != 2 * model_dim
        ):
            raise ValueError("global condition dimensions violate the locked contract")
        self.timestep_dim = timestep_dim
        self.size_dim = size_dim
        self.aspect_dim = aspect_dim
        self.hidden_dim = hidden_dim
        self.model_dim = model_dim
        self.slot_count = slot_count
        self.active_slot_ids = active_slot_ids
        self.modulation_chunks = modulation_chunks
        input_dim = timestep_dim + size_dim + aspect_dim
        self.condition_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        modulation_size = modulation_chunks * model_dim
        self.shared_block_projection = nn.Linear(
            hidden_dim, modulation_size, bias=False
        )
        self.block_biases = nn.ParameterDict(
            {
                f"slot_{slot_id:02d}": nn.Parameter(torch.zeros(modulation_size))
                for slot_id in active_slot_ids
            }
        )
        self.final_activation = nn.SiLU()
        self.final_projection = nn.Linear(hidden_dim, final_modulation_size, bias=True)
        nn.init.zeros_(self.shared_block_projection.weight)
        for bias in self.block_biases.values():
            nn.init.zeros_(bias)
        nn.init.zeros_(self.final_projection.weight)
        nn.init.zeros_(self.final_projection.bias)

    def forward(
        self,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        slot_ids: tuple[int, ...],
    ) -> GlobalConditionOutput:
        if (
            timestep.ndim != 1
            or size_scale.shape != timestep.shape
            or aspect.shape != timestep.shape
            or timestep.dtype != torch.float32
            or size_scale.dtype != torch.float32
            or aspect.dtype != torch.float32
        ):
            raise ValueError(
                "timestep, size_scale, and aspect must be matching FP32 vectors"
            )
        if not slot_ids or any(
            slot_id not in self.active_slot_ids for slot_id in slot_ids
        ):
            raise ValueError("slot_ids are empty or not present in this topology")

        device_type = timestep.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            embedded = torch.cat(
                (
                    fixed_sinusoidal_embedding(timestep, self.timestep_dim),
                    fixed_sinusoidal_embedding(size_scale, self.size_dim),
                    fixed_sinusoidal_embedding(aspect, self.aspect_dim),
                ),
                dim=-1,
            )
            condition_hidden = self.condition_mlp(embedded)
            shared = self.shared_block_projection(condition_hidden)
            selected_biases = torch.stack(
                tuple(self.block_biases[f"slot_{slot_id:02d}"] for slot_id in slot_ids)
            )
            per_block = shared[:, None] + selected_biases[None]
            chunks = per_block.view(
                timestep.shape[0],
                len(slot_ids),
                self.modulation_chunks,
                self.model_dim,
            ).unbind(dim=2)
            final_modulation = self.final_projection(
                self.final_activation(condition_hidden)
            )
            final_scale, final_shift = final_modulation.chunk(2, dim=-1)
        return GlobalConditionOutput(
            condition_hidden=condition_hidden,
            block=BlockModulation(*chunks),
            final_scale=final_scale,
            final_shift=final_shift,
        )


__all__ = [
    "FREQUENCY_BASE",
    "BlockModulation",
    "GlobalConditionOutput",
    "GlobalConditioner",
    "fixed_sinusoidal_embedding",
]
