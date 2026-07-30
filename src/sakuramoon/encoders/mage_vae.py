"""Local Mage-VAE architecture and frozen inference wrapper.

The module implements the checkpoint structure released by Microsoft Mage at
commit 8c94a0ac905167f40b05b09332b78752b7f9fbef. It intentionally supports only
the locally prepared safetensors checkpoint used by SakuraMoon.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol, cast

import torch
import torch.nn.functional as F
from safetensors.torch import load_file  # pyright: ignore[reportUnknownVariableType]
from torch import nn

from sakuramoon.assets import require_local_vae


def _silu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * torch.sigmoid(tensor)


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(32, channels, eps=1e-6, affine=True)


def _modulate(
    tensor: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    batch, channels = tensor.shape[:2]
    return tensor * (1 + scale.view(batch, channels, 1, 1)) + shift.view(
        batch, channels, 1, 1
    )


class _LayerNorm2d(nn.LayerNorm):
    def __init__(self, channels: int, *, affine: bool = True) -> None:
        super().__init__(  # pyright: ignore[reportUnknownMemberType]
            channels, eps=1e-6, elementwise_affine=affine
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        channels_last = input.permute(0, 2, 3, 1).contiguous()
        normalized = F.layer_norm(
            channels_last,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return normalized.permute(0, 3, 1, 2).contiguous()


class _EncoderLayerNorm2d(_LayerNorm2d):
    pass


class _RMSNorm(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        source_dtype = tensor.dtype
        normalized = tensor.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + 1e-6
        )
        return self.weight.to(source_dtype) * normalized.to(source_dtype)


class _TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(256, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        half = 128
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        ).to(timestep.device)
        arguments = timestep[:, None].float() * frequencies[None]
        embedding = torch.cat((arguments.cos(), arguments.sin()), dim=-1)
        first_layer = cast(nn.Linear, self.mlp[0])
        return self.mlp(embedding.to(first_layer.weight.dtype))


class _BottleneckPatchEmbed(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj1 = nn.Conv2d(3, 128, kernel_size=16, stride=16, bias=False)
        self.proj2 = nn.Conv2d(512, 384, kernel_size=1, bias=True)

    def forward(self, image: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.proj2(torch.cat((self.proj1(image), condition), dim=1))


class _DiCoBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.conv2 = nn.Conv2d(
            hidden_size, hidden_size, 3, padding=1, groups=hidden_size
        )
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.Sigmoid(),
        )
        self.conv4 = nn.Conv2d(hidden_size, 4 * hidden_size, 1)
        self.conv5 = nn.Conv2d(4 * hidden_size, hidden_size, 1)
        self.norm1 = _LayerNorm2d(hidden_size, affine=False)
        self.norm2 = _LayerNorm2d(hidden_size, affine=False)
        self.adaLN_modulation: nn.Module = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, source: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(condition).chunk(6, dim=1)
        )
        hidden = _modulate(self.norm1(source), shift_attn, scale_attn)
        hidden = F.gelu(self.conv2(self.conv1(hidden)))
        hidden = self.conv3(hidden * self.ca(hidden))
        hidden = source + gate_attn[..., None, None] * hidden
        mlp = self.conv5(
            F.gelu(self.conv4(_modulate(self.norm2(hidden), shift_mlp, scale_mlp)))
        )
        return hidden + gate_mlp[..., None, None] * mlp


class _EncoderDiCoBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.conv2 = nn.Conv2d(
            hidden_size, hidden_size, 3, padding=1, groups=hidden_size
        )
        self.conv3 = nn.Conv2d(hidden_size, hidden_size, 1)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_size, hidden_size, 1),
            nn.Sigmoid(),
        )
        self.conv4 = nn.Conv2d(hidden_size, 4 * hidden_size, 1)
        self.conv5 = nn.Conv2d(4 * hidden_size, hidden_size, 1)
        self.norm1 = _EncoderLayerNorm2d(hidden_size)
        self.norm2 = _EncoderLayerNorm2d(hidden_size)

    def forward(self, source: torch.Tensor) -> torch.Tensor:
        hidden = self.norm1(source)
        hidden = F.gelu(self.conv2(self.conv1(hidden)))
        hidden = self.conv3(hidden * self.ca(hidden))
        hidden = source + hidden
        return hidden + self.conv5(F.gelu(self.conv4(self.norm2(hidden))))


class _NerfEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.max_freqs = 8
        self.embedder = nn.Sequential(nn.Linear(35 + 64, 32))

    def _positions(
        self,
        patch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        position = torch.linspace(0, 1, patch_size, device=device, dtype=dtype)
        position_y, position_x = torch.meshgrid(position, position, indexing="ij")
        position_x = position_x.reshape(-1, 1, 1)
        position_y = position_y.reshape(-1, 1, 1)
        frequencies = torch.linspace(
            0, self.max_freqs, self.max_freqs, device=device, dtype=dtype
        )
        frequency_x = frequencies[None, :, None]
        frequency_y = frequencies[None, None, :]
        coefficients = (1 + frequency_x * frequency_y) ** -1
        basis_x = torch.cos(position_x * frequency_x * torch.pi)
        basis_y = torch.cos(position_y * frequency_y * torch.pi)
        return (basis_x * basis_y * coefficients).view(1, -1, self.max_freqs**2)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        batch, positions, _ = tensor.shape
        patch_size = math.isqrt(positions)
        if patch_size * patch_size != positions:
            raise ValueError("decoder patch position count must be square")
        basis = self._positions(patch_size, tensor.device, tensor.dtype).expand(
            batch, -1, -1
        )
        return self.embedder(torch.cat((tensor, basis), dim=-1))


class _NerfFinalLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = _RMSNorm(32)
        self.linear = nn.Linear(32, 3)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm(tensor))


class _MLPResBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_ln = nn.LayerNorm(32, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(32, 32), nn.SiLU(), nn.Linear(32, 32))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(32, 96))

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        shift, scale, gate = self.adaLN_modulation(condition).chunk(3, dim=-1)
        hidden = self.in_ln(tensor) * (1 + scale) + shift
        return tensor + gate * self.mlp(hidden)


class _SimpleMLPAdaLN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 32
        self.model_channels = 32
        self.out_channels = 3
        self.num_res_blocks = 3
        self.patch_size = 16
        self.cond_embed = nn.Linear(384, 16**2 * 32)
        self.input_proj = nn.Linear(32, 32)
        self.res_blocks = nn.ModuleList(_MLPResBlock() for _ in range(3))

    def forward(self, tensor: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        tensor = self.input_proj(tensor)
        condition = self.cond_embed(condition).reshape(condition.shape[0], 16**2, -1)
        for block in self.res_blocks:
            tensor = block(tensor, condition)
        return tensor


class _ResnetBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 384
        self.out_channels = 384
        self.norm1 = _group_norm(384)
        self.conv1 = nn.Conv2d(384, 384, 3, padding=1)
        self.norm2 = _group_norm(384)
        self.dropout = nn.Dropout(0.0)
        self.conv2 = nn.Conv2d(384, 384, 3, padding=1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(_silu(self.norm1(tensor)))
        hidden = self.conv2(self.dropout(_silu(self.norm2(hidden))))
        return tensor + hidden


class _PatchAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 384
        self.patch_size = 32
        self.norm = _group_norm(384)
        self.q = nn.Conv2d(384, 384, 1)
        self.k = nn.Conv2d(384, 384, 1)
        self.v = nn.Conv2d(384, 384, 1)
        self.proj_out = nn.Conv2d(384, 384, 1)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        normalized = self.norm(tensor)
        query = self.q(normalized)
        key = self.k(normalized)
        value = self.v(normalized)
        batch, channels, height, width = query.shape
        patch = self.patch_size
        pad_h = (patch - height % patch) % patch
        pad_w = (patch - width % patch) % patch
        if pad_h or pad_w:
            padding = (0, pad_w, 0, pad_h)
            query = F.pad(query, padding, mode="replicate")
            key = F.pad(key, padding, mode="replicate")
            value = F.pad(value, padding, mode="replicate")
        padded_h, padded_w = query.shape[-2:]
        patches_h, patches_w = padded_h // patch, padded_w // patch
        patch_count = patches_h * patches_w

        def split_patches(source: torch.Tensor) -> torch.Tensor:
            return (
                source.reshape(batch, channels, patches_h, patch, patches_w, patch)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(batch * patch_count, channels, patch * patch)
            )

        query = split_patches(query)
        key = split_patches(key)
        value = split_patches(value)
        weights = torch.bmm(query.transpose(1, 2), key) * channels**-0.5
        weights = weights.softmax(dim=2).transpose(1, 2)
        hidden = (
            torch.bmm(value, weights)
            .reshape(batch, patches_h, patches_w, channels, patch, patch)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(batch, channels, padded_h, padded_w)
        )
        hidden = hidden[:, :, :height, :width]
        return tensor + self.proj_out(hidden)


class _ConstAdaLN(nn.Module):
    modulation: torch.Tensor

    def __init__(self, modulation: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("modulation", modulation.detach().clone())

    def forward(self, condition: torch.Tensor) -> torch.Tensor:
        if self.modulation.shape[0] != condition.shape[0]:
            return self.modulation.expand(condition.shape[0], -1)
        return self.modulation


def _fold_adaln(module: nn.Module, condition: torch.Tensor) -> None:
    for child in module.modules():
        if isinstance(child, _DiCoBlock) and not isinstance(
            child.adaLN_modulation, _ConstAdaLN
        ):
            child.adaLN_modulation = _ConstAdaLN(child.adaLN_modulation(condition))


class _Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(128, 384, 3, padding=1)
        self.block = nn.Sequential(
            _ResnetBlock(),
            _PatchAttention(),
            _ResnetBlock(),
            _PatchAttention(),
            _ResnetBlock(),
        )
        self.norm_out = _group_norm(384)
        self.conv_out = nn.Conv2d(384, 384, 3, padding=1)
        self.ada = nn.Identity()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.block(self.conv_in(latent))
        return self.ada(self.conv_out(_silu(self.norm_out(hidden))))


class _DConvEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.z_ch = 128
        self.patch_size = 16
        self.patch_cond_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.head_blocks = nn.ModuleList(_EncoderDiCoBlock(768) for _ in range(2))
        self.proj_down = nn.Conv2d(768, 384, 1)
        self.z_proj = nn.Conv2d(128, 384, 1)
        self.fuse_proj = nn.Conv2d(768, 384, 1)
        self.t_embedder = _TimestepEmbedder(384)
        self.blocks = nn.ModuleList(_DiCoBlock(384) for _ in range(21))
        self.norm_out = _LayerNorm2d(384)
        self.proj_out = nn.Conv2d(384, 256, 1)

    def forward_pred(
        self,
        noisy_latent: torch.Tensor,
        timestep: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        condition = self.patch_cond_embed(image)
        for block in self.head_blocks:
            condition = block(condition)
        condition = self.proj_down(condition)
        hidden = self.fuse_proj(torch.cat((condition, self.z_proj(noisy_latent)), dim=1))
        embedded_timestep = self.t_embedder(timestep.view(-1))
        for block in self.blocks:
            hidden = block(hidden, embedded_timestep)
        return self.proj_out(self.norm_out(hidden))


class _YEmbedder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = _Decoder()


class _DConvDenoiser(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.in_channels = 3
        self.patch_size = 16
        self.hidden_size = 384
        self.num_cond_blocks = 21
        self.t_embedder = _TimestepEmbedder(384)
        self.y_embedder_x = nn.Conv2d(384, 32 * 16**2, 1)
        self.x_embedder = _NerfEmbedder()
        self.s_embedder = _BottleneckPatchEmbed()
        self.blocks = nn.ModuleList(_DiCoBlock(384) for _ in range(21))
        self.dec_net = _SimpleMLPAdaLN()
        self.final_layer = _NerfFinalLayer()
        self.y_embedder = _YEmbedder()

    def forward(
        self,
        image: torch.Tensor,
        timestep: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        batch, _, height, width = image.shape
        embedded_timestep = self.t_embedder(timestep.view(-1))
        hidden = self.s_embedder(image, condition)
        for block in self.blocks:
            hidden = block(hidden, embedded_timestep)
        length = hidden.shape[-2] * hidden.shape[-1]
        hidden = hidden.permute(0, 2, 3, 1).reshape(-1, self.hidden_size)
        patches = F.unfold(image, kernel_size=16, stride=16)
        patches = torch.cat((patches, self.y_embedder_x(condition).flatten(2)), dim=1)
        patches = (
            patches.reshape(batch, -1, 16**2, length)
            .permute(0, 3, 2, 1)
            .flatten(0, 1)
        )
        patches = self.final_layer(self.dec_net(self.x_embedder(patches), hidden))
        patches = patches.transpose(1, 2).reshape(batch, length, -1)
        return F.fold(
            patches.transpose(1, 2).contiguous(),
            (height, width),
            kernel_size=16,
            stride=16,
        )


class MageVAE(nn.Module):
    """One-step Mage encoder/decoder using posterior mean latents."""

    latent_channels = 128
    downsample_factor = 16

    def __init__(self, checkpoint: Path) -> None:
        super().__init__()
        if not checkpoint.is_file() or checkpoint.suffix != ".safetensors":
            raise FileNotFoundError(f"local Mage-VAE checkpoint is missing: {checkpoint}")
        self.dconv_encoder = _DConvEncoder()
        self.decoder_model = _DConvDenoiser()
        state = load_file(checkpoint, device="cpu")
        encoder_prefix = "student.dconv_encoder."
        encoder_state = {
            key.removeprefix(encoder_prefix): value
            for key, value in state.items()
            if key.startswith(encoder_prefix)
        }
        if not encoder_state:
            raise RuntimeError("Mage-VAE checkpoint has no encoder weights")
        self.dconv_encoder.load_state_dict(encoder_state, strict=True)

        decoder_prefix = "pipeline."
        expected_decoder = self.decoder_model.state_dict()
        decoder_state = {
            key.removeprefix(decoder_prefix): value
            for key, value in state.items()
            if key.startswith(decoder_prefix)
            and key.removeprefix(decoder_prefix) in expected_decoder
            and value.shape == expected_decoder[key.removeprefix(decoder_prefix)].shape
        }
        missing_decoder = sorted(set(expected_decoder) - set(decoder_state))
        if missing_decoder:
            raise RuntimeError(
                "Mage-VAE checkpoint is missing decoder weights: "
                + ", ".join(missing_decoder[:5])
            )
        self.decoder_model.load_state_dict(decoder_state, strict=True)
        del state

        with torch.no_grad():
            timestep = torch.zeros(1)
            _fold_adaln(self.dconv_encoder, self.dconv_encoder.t_embedder(timestep))
            _fold_adaln(self.decoder_model, self.decoder_model.t_embedder(timestep))

    @torch.no_grad()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError("Mage-VAE input must have shape [B,3,H,W]")
        height, width = image.shape[-2:]
        if height % 16 or width % 16:
            raise ValueError("Mage-VAE input height and width must be multiples of 16")
        noisy_latent = torch.zeros(
            image.shape[0],
            128,
            height // 16,
            width // 16,
            device=image.device,
            dtype=image.dtype,
        )
        timestep = torch.zeros(image.shape[0], device=image.device, dtype=image.dtype)
        moments = self.dconv_encoder.forward_pred(noisy_latent, timestep, image)
        return moments[:, :128]

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 4 or latent.shape[1] != 128:
            raise ValueError("Mage-VAE latent must have shape [B,128,H,W]")
        condition = self.decoder_model.y_embedder.decoder(latent)
        image = torch.zeros(
            latent.shape[0],
            3,
            latent.shape[2] * 16,
            latent.shape[3] * 16,
            device=latent.device,
            dtype=latent.dtype,
        )
        timestep = torch.zeros(latent.shape[0], device=latent.device, dtype=latent.dtype)
        return self.decoder_model(image, timestep, condition)


class _MageBackend(Protocol):
    def encode(self, image: torch.Tensor) -> torch.Tensor: ...

    def decode(self, latent: torch.Tensor) -> torch.Tensor: ...


class FrozenMageVAE(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        super().train(False)

    def train(self, mode: bool = True) -> FrozenMageVAE:
        del mode
        super().train(False)
        self.model.eval()
        return self

    @torch.inference_mode()
    def encode(self, image: torch.Tensor) -> torch.Tensor:
        latent = cast(_MageBackend, self.model).encode(image)
        expected = (image.shape[0], 128, image.shape[2] // 16, image.shape[3] // 16)
        if latent.shape != expected:
            raise RuntimeError(f"Mage-VAE returned latent shape {tuple(latent.shape)}, expected {expected}")
        return latent.detach()

    @torch.inference_mode()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        image = cast(_MageBackend, self.model).decode(latent)
        expected = (latent.shape[0], 3, latent.shape[2] * 16, latent.shape[3] * 16)
        if image.shape != expected:
            raise RuntimeError(f"Mage-VAE returned image shape {tuple(image.shape)}, expected {expected}")
        return image.detach()


def load_local_mage_vae(repository_root: Path, device: torch.device) -> FrozenMageVAE:
    if device.type != "cuda":
        raise ValueError("the production Mage-VAE requires a CUDA device")
    model_path = require_local_vae(repository_root)
    checkpoint = model_path / "diffusion_pytorch_model.safetensors"
    model = MageVAE(checkpoint).to(device=device, dtype=torch.bfloat16)
    return FrozenMageVAE(model)


__all__ = ["FrozenMageVAE", "MageVAE", "load_local_mage_vae"]
