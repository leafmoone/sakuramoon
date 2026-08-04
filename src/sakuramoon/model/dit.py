"""Stable-slot dense reference and flat-varlen production DiT models."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, cast

import torch
from torch import nn
from torch.utils import checkpoint as torch_checkpoint

from sakuramoon.conditioning.global_condition import (
    BlockModulation,
    GlobalConditioner,
    GlobalConditionOutput,
)
from sakuramoon.conditioning.modality import ModalityEmbedding
from sakuramoon.conditioning.packing import PackedSequences, pack_sequences
from sakuramoon.conditioning.rope import image_coordinates, packed_coordinates
from sakuramoon.model.attention import (
    AcceptedCuSeqlens,
    accept_fa4_boundaries,
    accepted_sample_indices,
    dense_attention_mask,
)
from sakuramoon.model.block import DiTBlock, PackedDiTBlock
from sakuramoon.model.growth import active_slot_ids, slot_growth, slot_name
from sakuramoon.model.output_head import FinalOutputHead

ActivationCheckpointMode = Literal["none", "alternating", "all"]
_ACTIVATION_CHECKPOINT_MODES = frozenset({"none", "alternating", "all"})


class _ActivationCheckpoint(Protocol):
    def __call__(
        self,
        function: Callable[..., torch.Tensor],
        *args: torch.Tensor,
        use_reentrant: bool,
        preserve_rng_state: bool,
    ) -> object: ...


_activation_checkpoint = cast(
    _ActivationCheckpoint,
    torch_checkpoint.checkpoint,  # pyright: ignore[reportUnknownMemberType]
)


def _require_activation_checkpoint_mode(value: str) -> ActivationCheckpointMode:
    if value not in _ACTIVATION_CHECKPOINT_MODES:
        raise ValueError("activation checkpoint mode is invalid")
    return cast(ActivationCheckpointMode, value)


def _checkpoint_block_at(
    mode: ActivationCheckpointMode,
    active_index: int,
    *,
    training: bool,
) -> bool:
    if not training or not torch.is_grad_enabled() or mode == "none":
        return False
    return mode == "all" or active_index % 2 == 0


def _checkpoint_tensor(
    operation: Callable[..., torch.Tensor],
    *inputs: torch.Tensor,
) -> torch.Tensor:
    result: object = _activation_checkpoint(
        operation,
        *inputs,
        use_reentrant=False,
        preserve_rng_state=False,
    )
    if not isinstance(result, torch.Tensor):
        raise TypeError("activation-checkpointed DiT block returned a non-tensor")
    return result


def _dense_checkpoint_operation(
    block: DiTBlock,
    token_mask: torch.Tensor,
    attention_mask: torch.Tensor,
    coordinates: torch.Tensor,
    *,
    attention_growth: float,
    mlp_growth: float,
) -> Callable[..., torch.Tensor]:
    def operation(
        tokens: torch.Tensor,
        attention_scale: torch.Tensor,
        attention_shift: torch.Tensor,
        attention_gate: torch.Tensor,
        mlp_scale: torch.Tensor,
        mlp_shift: torch.Tensor,
        mlp_gate: torch.Tensor,
    ) -> torch.Tensor:
        return block(
            tokens,
            token_mask,
            attention_mask,
            coordinates,
            BlockModulation(
                attention_scale,
                attention_shift,
                attention_gate,
                mlp_scale,
                mlp_shift,
                mlp_gate,
            ),
            attention_growth=attention_growth,
            mlp_growth=mlp_growth,
        )

    return operation


def _packed_checkpoint_operation(
    block: PackedDiTBlock,
    boundaries: AcceptedCuSeqlens,
    coordinates: torch.Tensor,
    sample_indices: torch.Tensor,
    *,
    attention_growth: float,
    mlp_growth: float,
) -> Callable[..., torch.Tensor]:
    def operation(
        tokens: torch.Tensor,
        attention_scale: torch.Tensor,
        attention_shift: torch.Tensor,
        attention_gate: torch.Tensor,
        mlp_scale: torch.Tensor,
        mlp_shift: torch.Tensor,
        mlp_gate: torch.Tensor,
    ) -> torch.Tensor:
        return block(
            tokens,
            boundaries,
            coordinates,
            sample_indices,
            BlockModulation(
                attention_scale,
                attention_shift,
                attention_gate,
                mlp_scale,
                mlp_shift,
                mlp_gate,
            ),
            attention_growth=attention_growth,
            mlp_growth=mlp_growth,
        )

    return operation


@dataclass(frozen=True)
class DenseDiTFeatures:
    joint_hidden: torch.Tensor
    token_mask: torch.Tensor
    image_start: int
    image_shape: tuple[int, int]
    condition: GlobalConditionOutput


@dataclass(frozen=True)
class PackedDiTFeatures:
    joint_hidden: torch.Tensor
    packed: PackedSequences
    condition: GlobalConditionOutput
    sample_indices: torch.Tensor


class DenseDiT(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        input_channels: int,
        hidden_size: int,
        intermediate_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        rope_nope_dim: int,
        rope_y_dim: int,
        rope_x_dim: int,
        rope_position_scale: float,
        rope_theta: float,
        norm_eps: float,
        timestep_dim: int,
        size_dim: int,
        aspect_dim: int,
        condition_hidden_size: int,
        stable_slot_count: int,
        modulation_chunks: int,
        final_modulation_size: int,
        out_channels: int,
        modality_init_std: float,
        linear_dtype: torch.dtype,
        sensitive_dtype: torch.dtype,
        projection_bias: bool,
        attention_dropout: float,
        mlp_dropout: float,
        output_weight_zero_init: bool,
        output_bias_zero_init: bool,
    ) -> None:
        super().__init__()
        slots = active_slot_ids(depth)
        if input_channels != out_channels or stable_slot_count != 24:
            raise ValueError("DiT requires 128-channel latent I/O and 24 stable slots")
        if linear_dtype not in (torch.float32, torch.bfloat16):
            raise ValueError("linear_dtype must be float32 or bfloat16")
        if sensitive_dtype != torch.float32:
            raise ValueError("sensitive DiT parameters must use float32")
        self.depth = depth
        self.hidden_size = hidden_size
        self.active_slot_ids = slots
        self._activation_checkpoint_mode: ActivationCheckpointMode = "none"
        self._artifact_config: dict[str, object] = {
            "active_slot_ids": list(slots),
            "aspect_dim": aspect_dim,
            "attention_backend": "dense_sdpa",
            "attention_dropout": attention_dropout,
            "condition_hidden_size": condition_hidden_size,
            "depth": depth,
            "final_modulation_size": final_modulation_size,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "input_channels": input_channels,
            "intermediate_size": intermediate_size,
            "kv_heads": kv_heads,
            "linear_dtype": str(linear_dtype).removeprefix("torch."),
            "mlp_dropout": mlp_dropout,
            "modality_init_std": modality_init_std,
            "modulation_chunks": modulation_chunks,
            "norm_eps": norm_eps,
            "out_channels": out_channels,
            "output_bias_zero_init": output_bias_zero_init,
            "output_weight_zero_init": output_weight_zero_init,
            "projection_bias": projection_bias,
            "q_heads": q_heads,
            "rope_nope_dim": rope_nope_dim,
            "rope_position_scale": rope_position_scale,
            "rope_theta": rope_theta,
            "rope_x_dim": rope_x_dim,
            "rope_y_dim": rope_y_dim,
            "sensitive_dtype": str(sensitive_dtype).removeprefix("torch."),
            "size_dim": size_dim,
            "stable_slot_count": stable_slot_count,
            "timestep_dim": timestep_dim,
        }
        self.input_projection = nn.Linear(
            input_channels,
            hidden_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )
        self.modality = ModalityEmbedding(hidden_size, modality_init_std)
        self.conditioner = GlobalConditioner(
            timestep_dim=timestep_dim,
            size_dim=size_dim,
            aspect_dim=aspect_dim,
            hidden_dim=condition_hidden_size,
            model_dim=hidden_size,
            slot_count=stable_slot_count,
            active_slot_ids=slots,
            modulation_chunks=modulation_chunks,
            final_modulation_size=final_modulation_size,
        )
        self.blocks = nn.ModuleDict(
            {
                slot_name(slot_id): DiTBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    rope_nope_dim=rope_nope_dim,
                    rope_y_dim=rope_y_dim,
                    rope_x_dim=rope_x_dim,
                    rope_position_scale=rope_position_scale,
                    rope_theta=rope_theta,
                    norm_eps=norm_eps,
                    linear_dtype=linear_dtype,
                    projection_bias=projection_bias,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                )
                for slot_id in slots
            }
        )
        self.output_head = FinalOutputHead(
            hidden_size=hidden_size,
            out_channels=out_channels,
            norm_eps=norm_eps,
            projection_dtype=sensitive_dtype,
            weight_zero_init=output_weight_zero_init,
            bias_zero_init=output_bias_zero_init,
        )

    @property
    def activation_checkpoint_mode(self) -> ActivationCheckpointMode:
        return self._activation_checkpoint_mode

    def set_activation_checkpoint_mode(self, mode: str) -> None:
        """Apply the explicit runtime-only block recomputation policy."""

        self._activation_checkpoint_mode = _require_activation_checkpoint_mode(mode)

    def _prepare_tokens(
        self,
        latent: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        style_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, tuple[int, int]]:
        if latent.ndim != 4 or latent.shape[1] != self.input_projection.in_features:
            raise ValueError("latent must have shape [B,input_channels,H,W]")
        batch, _, height, width = latent.shape
        if text_tokens.ndim != 3 or text_tokens.shape[0] != batch:
            raise ValueError("text_tokens must have shape [B,L,hidden_size]")
        if text_tokens.shape[-1] != self.hidden_size:
            raise ValueError("text token width differs from the DiT hidden size")
        if text_mask.shape != text_tokens.shape[:2] or text_mask.dtype != torch.bool:
            raise ValueError("text_mask must be boolean with shape [B,L]")
        if style_tokens.shape != (batch, 4, self.hidden_size):
            raise ValueError("style_tokens must have shape [B,4,hidden_size]")
        if (
            latent.dtype != self.input_projection.weight.dtype
            or text_tokens.dtype != latent.dtype
            or style_tokens.dtype != latent.dtype
        ):
            raise ValueError(
                "latent, text, style, and DiT linear weights must share dtype"
            )
        image_tokens = latent.permute(0, 2, 3, 1).reshape(
            batch,
            height * width,
            latent.shape[1],
        )
        image_tokens = self.input_projection(image_tokens)
        text_tokens = self.modality(text_tokens, "text")
        style_tokens = self.modality(style_tokens, "style")
        image_tokens = self.modality(image_tokens, "image")
        joint = torch.cat((text_tokens, style_tokens, image_tokens), dim=1)
        style_mask = torch.ones(batch, 4, device=latent.device, dtype=torch.bool)
        image_mask = torch.ones(
            batch,
            height * width,
            device=latent.device,
            dtype=torch.bool,
        )
        token_mask = torch.cat((text_mask, style_mask, image_mask), dim=1)
        coordinates = torch.zeros(
            batch,
            joint.shape[1],
            2,
            device=latent.device,
            dtype=torch.float32,
        )
        image_start = text_tokens.shape[1] + 4
        coordinates[:, image_start:] = image_coordinates(
            height,
            width,
            device=latent.device,
        )
        return joint, token_mask, coordinates, image_start, (height, width)

    def forward_features(
        self,
        latent: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> DenseDiTFeatures:
        joint, token_mask, coordinates, image_start, image_shape = self._prepare_tokens(
            latent,
            text_tokens,
            text_mask,
            style_tokens,
        )
        condition = self.conditioner(
            timestep,
            size_scale,
            aspect,
            self.active_slot_ids,
        )
        attention_mask = dense_attention_mask(token_mask)
        for active_index, slot_id in enumerate(self.active_slot_ids):
            growth = slot_growth(self.depth, slot_id, growth_alpha)
            block = self.blocks[slot_name(slot_id)]
            modulation = condition.block.for_active_index(active_index)
            if _checkpoint_block_at(
                self._activation_checkpoint_mode,
                active_index,
                training=self.training,
            ):
                if not isinstance(block, DiTBlock):
                    raise TypeError("dense DiT block registry is invalid")
                joint = _checkpoint_tensor(
                    _dense_checkpoint_operation(
                        block,
                        token_mask,
                        attention_mask,
                        coordinates,
                        attention_growth=growth,
                        mlp_growth=growth,
                    ),
                    joint,
                    modulation.attention_scale,
                    modulation.attention_shift,
                    modulation.attention_gate,
                    modulation.mlp_scale,
                    modulation.mlp_shift,
                    modulation.mlp_gate,
                )
            else:
                joint = block(
                    joint,
                    token_mask,
                    attention_mask,
                    coordinates,
                    modulation,
                    attention_growth=growth,
                    mlp_growth=growth,
                )
        return DenseDiTFeatures(
            joint_hidden=joint,
            token_mask=token_mask,
            image_start=image_start,
            image_shape=image_shape,
            condition=condition,
        )

    def forward(
        self,
        latent: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> torch.Tensor:
        features = self.forward_features(
            latent,
            text_tokens,
            text_mask,
            style_tokens,
            timestep,
            size_scale,
            aspect,
            growth_alpha=growth_alpha,
        )
        height, width = features.image_shape
        image_tokens = height * width
        image_hidden = features.joint_hidden[
            :,
            features.image_start : features.image_start + image_tokens,
        ]
        return self.output_head(
            image_hidden,
            features.condition.final_scale,
            features.condition.final_shift,
            features.image_shape,
        )

    def model_metadata(self) -> dict[str, int | str]:
        return {
            "prediction_type": "x",
            "out_channels": self.output_head.out_channels,
            "depth": self.depth,
            "stable_slot_count": 24,
        }

    def artifact_config(self) -> dict[str, object]:
        return dict(self._artifact_config)


class PackedDiT(nn.Module):
    """Production DiT that keeps T024 sequences flat through every block."""

    def __init__(
        self,
        *,
        depth: int,
        input_channels: int,
        hidden_size: int,
        intermediate_size: int,
        q_heads: int,
        kv_heads: int,
        head_dim: int,
        rope_nope_dim: int,
        rope_y_dim: int,
        rope_x_dim: int,
        rope_position_scale: float,
        rope_theta: float,
        norm_eps: float,
        timestep_dim: int,
        size_dim: int,
        aspect_dim: int,
        condition_hidden_size: int,
        stable_slot_count: int,
        modulation_chunks: int,
        final_modulation_size: int,
        out_channels: int,
        modality_init_std: float,
        linear_dtype: torch.dtype,
        sensitive_dtype: torch.dtype,
        projection_bias: bool,
        attention_dropout: float,
        mlp_dropout: float,
        output_weight_zero_init: bool,
        output_bias_zero_init: bool,
    ) -> None:
        super().__init__()
        slots = active_slot_ids(depth)
        if input_channels != out_channels or stable_slot_count != 24:
            raise ValueError("DiT requires 128-channel latent I/O and 24 stable slots")
        if linear_dtype != torch.bfloat16:
            raise ValueError("packed FA4 DiT requires BF16 linear parameters")
        if sensitive_dtype != torch.float32:
            raise ValueError("sensitive DiT parameters must use float32")
        self.depth = depth
        self.hidden_size = hidden_size
        self.active_slot_ids = slots
        self._activation_checkpoint_mode: ActivationCheckpointMode = "none"
        self._artifact_config: dict[str, object] = {
            "active_slot_ids": list(slots),
            "aspect_dim": aspect_dim,
            "attention_backend": "fa4_varlen",
            "attention_dropout": attention_dropout,
            "condition_hidden_size": condition_hidden_size,
            "depth": depth,
            "final_modulation_size": final_modulation_size,
            "head_dim": head_dim,
            "hidden_size": hidden_size,
            "input_channels": input_channels,
            "intermediate_size": intermediate_size,
            "kv_heads": kv_heads,
            "linear_dtype": str(linear_dtype).removeprefix("torch."),
            "mlp_dropout": mlp_dropout,
            "modality_init_std": modality_init_std,
            "modulation_chunks": modulation_chunks,
            "norm_eps": norm_eps,
            "out_channels": out_channels,
            "output_bias_zero_init": output_bias_zero_init,
            "output_weight_zero_init": output_weight_zero_init,
            "projection_bias": projection_bias,
            "q_heads": q_heads,
            "rope_nope_dim": rope_nope_dim,
            "rope_position_scale": rope_position_scale,
            "rope_theta": rope_theta,
            "rope_x_dim": rope_x_dim,
            "rope_y_dim": rope_y_dim,
            "sensitive_dtype": str(sensitive_dtype).removeprefix("torch."),
            "size_dim": size_dim,
            "stable_slot_count": stable_slot_count,
            "timestep_dim": timestep_dim,
        }
        self.input_projection = nn.Linear(
            input_channels,
            hidden_size,
            bias=projection_bias,
            dtype=linear_dtype,
        )
        self.modality = ModalityEmbedding(hidden_size, modality_init_std)
        self.conditioner = GlobalConditioner(
            timestep_dim=timestep_dim,
            size_dim=size_dim,
            aspect_dim=aspect_dim,
            hidden_dim=condition_hidden_size,
            model_dim=hidden_size,
            slot_count=stable_slot_count,
            active_slot_ids=slots,
            modulation_chunks=modulation_chunks,
            final_modulation_size=final_modulation_size,
        )
        self.blocks = nn.ModuleDict(
            {
                slot_name(slot_id): PackedDiTBlock(
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    rope_nope_dim=rope_nope_dim,
                    rope_y_dim=rope_y_dim,
                    rope_x_dim=rope_x_dim,
                    rope_position_scale=rope_position_scale,
                    rope_theta=rope_theta,
                    norm_eps=norm_eps,
                    linear_dtype=linear_dtype,
                    projection_bias=projection_bias,
                    attention_dropout=attention_dropout,
                    mlp_dropout=mlp_dropout,
                )
                for slot_id in slots
            }
        )
        self.output_head = FinalOutputHead(
            hidden_size=hidden_size,
            out_channels=out_channels,
            norm_eps=norm_eps,
            projection_dtype=sensitive_dtype,
            weight_zero_init=output_weight_zero_init,
            bias_zero_init=output_bias_zero_init,
        )

    @property
    def activation_checkpoint_mode(self) -> ActivationCheckpointMode:
        return self._activation_checkpoint_mode

    def set_activation_checkpoint_mode(self, mode: str) -> None:
        """Apply the explicit runtime-only block recomputation policy."""

        self._activation_checkpoint_mode = _require_activation_checkpoint_mode(mode)

    def prepare_packed_sequences(
        self,
        latents: tuple[torch.Tensor, ...],
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        text_lengths: tuple[int, ...],
        style_tokens: torch.Tensor,
    ) -> PackedSequences:
        batch = text_tokens.shape[0] if text_tokens.ndim == 3 else -1
        if batch <= 0 or len(latents) != batch:
            raise ValueError(
                "latents and text tokens must describe the same nonempty batch"
            )
        if text_tokens.shape[-1] != self.hidden_size:
            raise ValueError("text token width differs from the DiT hidden size")
        if text_mask.shape != text_tokens.shape[:2] or text_mask.dtype != torch.bool:
            raise ValueError("text_mask must be boolean with shape [B,L]")
        if style_tokens.shape != (batch, 4, self.hidden_size):
            raise ValueError("style_tokens must have shape [B,4,hidden_size]")
        expected_dtype = self.input_projection.weight.dtype
        if text_tokens.dtype != expected_dtype or style_tokens.dtype != expected_dtype:
            raise ValueError("text, style, and DiT linear weights must share dtype")

        image_tokens: list[torch.Tensor] = []
        image_shapes: list[tuple[int, int]] = []
        for latent in latents:
            if (
                latent.ndim != 3
                or latent.shape[0] != self.input_projection.in_features
                or latent.dtype != expected_dtype
                or latent.device != text_tokens.device
            ):
                raise ValueError(
                    "each latent must be [input_channels,H,W] on the token device and dtype"
                )
            _, height, width = latent.shape
            projected = self.input_projection(
                latent.permute(1, 2, 0).reshape(height * width, latent.shape[0])
            )
            image_tokens.append(self.modality(projected, "image"))
            image_shapes.append((height, width))
        return pack_sequences(
            self.modality(text_tokens, "text"),
            text_mask,
            text_lengths,
            self.modality(style_tokens, "style"),
            tuple(image_tokens),
            tuple(image_shapes),
        )

    @staticmethod
    def _sample_indices(boundaries: AcceptedCuSeqlens) -> torch.Tensor:
        return accepted_sample_indices(boundaries)

    def forward_packed_features(
        self,
        packed: PackedSequences,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> PackedDiTFeatures:
        if packed.tokens.ndim != 2 or packed.tokens.shape[-1] != self.hidden_size:
            raise ValueError("packed tokens must have shape [T,hidden_size]")
        if len(packed.spans) != timestep.shape[0]:
            raise ValueError("packed sample count must equal the condition batch")
        boundaries = accept_fa4_boundaries(
            packed.boundaries,
            total_tokens=packed.tokens.shape[0],
            batch_size=len(packed.spans),
            device=packed.tokens.device,
        )
        coordinates = packed_coordinates(packed)
        sample_indices = self._sample_indices(boundaries)
        condition = self.conditioner(
            timestep,
            size_scale,
            aspect,
            self.active_slot_ids,
        )
        joint = packed.tokens
        for active_index, slot_id in enumerate(self.active_slot_ids):
            growth = slot_growth(self.depth, slot_id, growth_alpha)
            block = self.blocks[slot_name(slot_id)]
            modulation = condition.block.for_active_index(active_index)
            if _checkpoint_block_at(
                self._activation_checkpoint_mode,
                active_index,
                training=self.training,
            ):
                if not isinstance(block, PackedDiTBlock):
                    raise TypeError("packed DiT block registry is invalid")
                joint = _checkpoint_tensor(
                    _packed_checkpoint_operation(
                        block,
                        boundaries,
                        coordinates,
                        sample_indices,
                        attention_growth=growth,
                        mlp_growth=growth,
                    ),
                    joint,
                    modulation.attention_scale,
                    modulation.attention_shift,
                    modulation.attention_gate,
                    modulation.mlp_scale,
                    modulation.mlp_shift,
                    modulation.mlp_gate,
                )
            else:
                joint = block(
                    joint,
                    boundaries,
                    coordinates,
                    sample_indices,
                    modulation,
                    attention_growth=growth,
                    mlp_growth=growth,
                )
        return PackedDiTFeatures(
            joint_hidden=joint,
            packed=packed,
            condition=condition,
            sample_indices=sample_indices,
        )

    def predict_from_features(
        self,
        features: PackedDiTFeatures,
    ) -> tuple[torch.Tensor, ...]:
        image_hidden = torch.cat(
            tuple(
                features.joint_hidden[spans.image.start : spans.image.end]
                for spans in features.packed.spans
            )
        )
        image_lengths = torch.tensor(
            tuple(height * width for height, width in features.packed.image_shapes),
            device=image_hidden.device,
            dtype=torch.int64,
        )
        image_sample_indices = torch.repeat_interleave(
            torch.arange(
                len(features.packed.spans),
                device=image_hidden.device,
                dtype=torch.int64,
            ),
            image_lengths,
            output_size=image_hidden.shape[0],
        )
        return self.output_head.forward_packed(
            image_hidden,
            image_sample_indices,
            features.condition.final_scale,
            features.condition.final_shift,
            features.packed.image_shapes,
        )

    def forward_packed(
        self,
        packed: PackedSequences,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> tuple[torch.Tensor, ...]:
        return self.predict_from_features(
            self.forward_packed_features(
                packed,
                timestep,
                size_scale,
                aspect,
                growth_alpha=growth_alpha,
            )
        )

    def forward(
        self,
        latents: tuple[torch.Tensor, ...],
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        text_lengths: tuple[int, ...],
        style_tokens: torch.Tensor,
        timestep: torch.Tensor,
        size_scale: torch.Tensor,
        aspect: torch.Tensor,
        *,
        growth_alpha: float,
    ) -> tuple[torch.Tensor, ...]:
        packed = self.prepare_packed_sequences(
            latents,
            text_tokens,
            text_mask,
            text_lengths,
            style_tokens,
        )
        return self.forward_packed(
            packed,
            timestep,
            size_scale,
            aspect,
            growth_alpha=growth_alpha,
        )

    def model_metadata(self) -> dict[str, int | str]:
        return {
            "prediction_type": "x",
            "out_channels": self.output_head.out_channels,
            "depth": self.depth,
            "stable_slot_count": 24,
            "attention_backend": "fa4_varlen",
        }

    def artifact_config(self) -> dict[str, object]:
        return dict(self._artifact_config)


__all__ = ["DenseDiT", "DenseDiTFeatures", "PackedDiT", "PackedDiTFeatures"]
