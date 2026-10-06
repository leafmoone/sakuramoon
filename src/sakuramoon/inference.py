"""Checkpoint-only generation using the training caption and flow contracts."""

from __future__ import annotations

import dataclasses
import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch

from sakuramoon.conditioning.rope import image_coordinates
from sakuramoon.data.caption import CaptionPlan, empty_caption_dropout_hits
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
    TokenEncoder,
    serialize_caption,
)
from sakuramoon.encoders.mage_vae import FrozenMageVAE
from sakuramoon.encoders.qwen import QwenRuntime
from sakuramoon.objective.flow import guided_velocity
from sakuramoon.sampling.profiles import SamplingProfile
from sakuramoon.sampling.sampler import sample_profile
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs


@dataclass(frozen=True)
class InferenceSettings:
    profile: SamplingProfile
    cfg_scale: float
    noise_scale: float
    t_eps: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.cfg_scale) or self.cfg_scale < 0:
            raise ValueError("guidance scale must be finite and nonnegative")
        if not math.isfinite(self.noise_scale) or self.noise_scale <= 0:
            raise ValueError("noise scale must be finite and positive")
        if not math.isfinite(self.t_eps) or not 0 < self.t_eps < 1:
            raise ValueError("t_eps must be in (0, 1)")


def load_inference_settings(
    path: Path,
    *,
    profile: str | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
) -> InferenceSettings:
    """Read sampling fields without loading datasets, secrets or optimizers."""
    with path.open("rb") as handle:
        config: dict[str, Any] = tomllib.load(handle)
    if "extends" in config:
        raise ValueError("use resolved_config.toml or a standalone inference config")
    sampling = config["sampling"]
    name = profile or sampling["profile"]
    preset = sampling["profiles"][name]

    def number(value: object, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a TOML number")
        return float(value)

    return InferenceSettings(
        SamplingProfile(
            name=name,
            solver=preset["solver"],
            steps=steps if steps is not None else preset["steps"],
            time_schedule=preset["time_schedule"],
        ),
        cfg_scale=number(
            guidance_scale if guidance_scale is not None else config["cfg"]["scale"],
            "cfg.scale",
        ),
        noise_scale=number(config["timestep"]["noise_scale"], "timestep.noise_scale"),
        t_eps=number(config["timestep"]["t_eps"], "timestep.t_eps"),
    )


def _indices(
    rows: tuple[tuple[int, ...], ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max(map(len, rows), default=0)
    indices = torch.full((len(rows), width), -1, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.bool, device=device)
    for i, row in enumerate(rows):
        if row:
            indices[i, : len(row)] = torch.tensor(row, device=device)
            mask[i, : len(row)] = True
    return indices, mask


@torch.inference_mode()
def generate_image(
    plan: CaptionPlan,
    *,
    composite: TrainableComposite,
    qwen: QwenRuntime,
    vae: FrozenMageVAE,
    settings: InferenceSettings,
    width: int,
    height: int,
    seed: int,
    growth_alpha: float,
    device: torch.device,
) -> torch.Tensor:
    """Return one uint8 [3,H,W] image; CFG is applied in velocity space."""
    if any(type(size) is not int or size <= 0 or size % 16 for size in (width, height)):
        raise ValueError("width and height must be positive multiples of 16")
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    if not math.isfinite(growth_alpha) or not 0 <= growth_alpha <= 1:
        raise ValueError("growth alpha must be in [0, 1]")
    if composite.training:
        raise ValueError("inference requires composite.eval()")
    tokenizer = cast(TokenEncoder, qwen.tokenizer)
    pad = qwen.tokenizer.pad_token_id
    if type(pad) is not int:
        raise ValueError("Qwen tokenizer has no integer padding token")
    framing = FramingContract(EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, pad)
    unconditional = CaptionPlan(
        tags=(),
        condition=None,
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=True,
        dropout_hits=empty_caption_dropout_hits(all_condition=True),
    )
    captions = tuple(
        serialize_caption(p, tokenizer, framing) for p in (plan, unconditional)
    )
    if captions[0].truncated:
        raise ValueError("prompt exceeds the model's caption budget; shorten it")
    length = max(c.dense_length for c in captions)
    input_ids = torch.full((2, length), pad, dtype=torch.long, device=device)
    attention_mask = torch.zeros((2, length), dtype=torch.bool, device=device)
    for i, caption in enumerate(captions):
        count = len(caption.input_ids)
        input_ids[i, :count] = torch.tensor(caption.input_ids, device=device)
        attention_mask[i, :count] = True
    states = qwen.encoder(input_ids, attention_mask).hidden_states
    main_indices, main_mask = _indices(
        tuple(c.main_token_indices for c in captions), device
    )
    condition_indices, condition_mask = _indices(
        tuple(c.condition_token_indices for c in captions), device
    )
    coordinates = image_coordinates(height // 16, width // 16, device=device)
    channels = composite.dit.input_projection.in_features
    shape = (channels, height // 16, width // 16)
    inputs = TrainableCompositeInputs(
        qwen_states=states,
        main_token_indices=main_indices,
        main_mask=main_mask,
        main_token_lengths=tuple(len(c.main_token_indices) for c in captions),
        condition_token_indices=condition_indices,
        condition_mask=condition_mask,
        use_null_condition=torch.tensor(
            [c.use_null_condition for c in captions], dtype=torch.bool, device=device
        ),
        active_condition_sample_indices=torch.tensor(
            [i for i, c in enumerate(captions) if not c.use_null_condition],
            dtype=torch.long,
            device=device,
        ),
        latents=tuple(
            torch.empty(shape, dtype=torch.bfloat16, device=device) for _ in captions
        ),
        image_coordinates=(coordinates, coordinates),
        timestep=torch.zeros(2, dtype=torch.float32, device=device),
        # The 512^2 reference area is part of the trained condition convention.
        size_scale=torch.full(
            (2,), 0.5 * math.log2(height * width / 512**2), device=device
        ),
        aspect=torch.full((2,), math.log2(width / height), device=device),
        growth_alpha=growth_alpha,
    )
    conditioning = composite.forward_conditioning(inputs)
    generator = torch.Generator(device=device).manual_seed(seed)
    noise = torch.randn(
        (1, *shape), generator=generator, device=device, dtype=torch.float32
    )
    noise.mul_(settings.noise_scale)

    def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        branches = torch.cat((state, state)).to(torch.bfloat16)
        step_inputs = dataclasses.replace(
            inputs,
            latents=tuple(branches.unbind()),
            timestep=torch.cat((timestep, timestep)),
        )
        predicted = composite.forward_dit(step_inputs, conditioning)
        return guided_velocity(
            predicted[0].unsqueeze(0),
            predicted[1].unsqueeze(0),
            state,
            timestep,
            t_eps=settings.t_eps,
            guidance_scale=settings.cfg_scale,
        )

    sampled = sample_profile(velocity, noise, profile=settings.profile)
    if not bool(torch.isfinite(sampled.state).all()):
        raise FloatingPointError("sampler produced nonfinite latents")
    decoded = vae.decode(sampled.state.to(torch.bfloat16))
    if not bool(torch.isfinite(decoded).all()):
        raise FloatingPointError("VAE produced nonfinite pixels")
    return decoded[0].float().add(1).mul(127.5).round().clamp(0, 255).to(torch.uint8)
