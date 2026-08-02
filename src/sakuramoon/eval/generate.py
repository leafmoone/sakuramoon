"""Read-only checkpoint generation for formal evaluator jobs."""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

from sakuramoon.checkpoint.artifact import export_trainable_composite
from sakuramoon.checkpoint.load import (
    load_inference_artifact,
    read_checkpoint_manifest,
    read_raw_checkpoint_state,
)
from sakuramoon.checkpoint.schema import CheckpointKind as ArtifactCheckpointKind
from sakuramoon.config.assembly import trainable_composite_spec
from sakuramoon.config.schema import RuntimeConfig
from sakuramoon.data.caption import CaptionDropoutHits, CaptionPlan
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
    SerializedCaption,
    TokenEncoder,
    serialize_caption,
)
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.eval.spec import CheckpointRef, PromptCase
from sakuramoon.objective.flow import guided_velocity
from sakuramoon.sampling.sampler import (
    GenerationMetadata,
    build_generation_metadata,
    sample_profile,
)
from sakuramoon.train.step import TrainableComposite, TrainableCompositeInputs


class GenerationContractError(RuntimeError):
    """Checkpoint generation cannot honor the formal evaluator contract."""


@dataclass(frozen=True, slots=True)
class GeneratedBatch:
    cases: tuple[PromptCase, ...]
    images: torch.Tensor
    metadata: GenerationMetadata

    def __post_init__(self) -> None:
        has_mixed_shapes = any(
            case.height != self.cases[0].height
            or case.width != self.cases[0].width
            for case in self.cases[1:]
        ) if self.cases else False
        if (
            not self.cases
            or has_mixed_shapes
            or self.images.device.type != "cpu"
            or self.images.dtype != torch.uint8
            or self.images.shape
            != (
                len(self.cases),
                3,
                self.cases[0].height,
                self.cases[0].width,
            )
        ):
            raise ValueError("generated batch image contract is invalid")


_NO_DROPOUT = CaptionDropoutHits(
    all_condition=False,
    nsfw=False,
    character=False,
    copyright=False,
    general=False,
    artist=False,
    candidate_source=False,
    long_names=False,
    long_no_names=False,
    short_vibes=False,
    nl2=False,
    nl3=False,
)
_ALL_CONDITION_DROPOUT = dataclasses.replace(_NO_DROPOUT, all_condition=True)


def _conditional_plan(case: PromptCase) -> CaptionPlan:
    if case.conditions:
        raise GenerationContractError(
            "prompt condition semantics are not governed for production evaluation"
        )
    return CaptionPlan(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        nl_text=case.prompt,
        selected_nl="long_names",
        all_condition_dropped=False,
        dropout_hits=_NO_DROPOUT,
    )


def _unconditional_plan() -> CaptionPlan:
    return CaptionPlan(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        nl_text=None,
        selected_nl=None,
        all_condition_dropped=True,
        dropout_hits=_ALL_CONDITION_DROPOUT,
    )


def _initial_gaussian_noise(
    cases: tuple[PromptCase, ...], *, device: torch.device
) -> torch.Tensor:
    if not cases:
        raise GenerationContractError("initial noise requires prompt cases")
    height = cases[0].height
    width = cases[0].width
    if any(case.height != height or case.width != width for case in cases[1:]):
        raise GenerationContractError("initial noise requires one prompt shape")
    noise_items: list[torch.Tensor] = []
    for case in cases:
        generator = torch.Generator(device=device)
        generator.manual_seed(case.seed)
        noise_items.append(
            torch.randn(
                (128, height // 16, width // 16),
                device=device,
                dtype=torch.float32,
                generator=generator,
            )
        )
    return torch.stack(noise_items)


def _index_tensor(
    values: tuple[tuple[int, ...], ...], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    width = max((len(item) for item in values), default=0)
    indices = torch.full((len(values), width), -1, dtype=torch.long, device=device)
    mask = torch.zeros((len(values), width), dtype=torch.bool, device=device)
    for row, item in enumerate(values):
        if item:
            indices[row, : len(item)] = torch.tensor(
                item, dtype=torch.long, device=device
            )
            mask[row, : len(item)] = True
    return indices, mask


def _serialized_conditioning(
    cases: tuple[PromptCase, ...], tokenizer: object, device: torch.device
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    tuple[int, ...],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    encoder = cast(TokenEncoder, tokenizer)
    padding_token_id = getattr(encoder, "pad_token_id", None)
    encode = getattr(encoder, "encode", None)
    if type(padding_token_id) is not int or not callable(encode):
        raise GenerationContractError("Qwen tokenizer contract is invalid")
    framing = FramingContract(
        EXPECTED_PREFIX_TOKENS,
        EXPECTED_SUFFIX_TOKENS,
        padding_token_id,
    )
    serialized: list[SerializedCaption] = []
    for case in cases:
        conditional = serialize_caption(_conditional_plan(case), encoder, framing)
        if conditional.truncated:
            raise GenerationContractError(
                f"prompt {case.prompt_id} exceeds the fixed condition budget"
            )
        serialized.append(conditional)
    unconditional = serialize_caption(_unconditional_plan(), encoder, framing)
    serialized.extend(unconditional for _ in cases)
    captions = tuple(serialized)
    dense_length = max(item.dense_length for item in captions)
    input_ids = torch.full(
        (len(captions), dense_length),
        padding_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(captions), dense_length), dtype=torch.bool, device=device
    )
    for row, item in enumerate(captions):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(
            item.input_ids, dtype=torch.long, device=device
        )
        attention_mask[row, :length] = True
    main_indices, main_mask = _index_tensor(
        tuple(item.main_token_indices for item in captions), device
    )
    artist_indices, artist_mask = _index_tensor(
        tuple(item.artist_token_indices for item in captions), device
    )
    use_null = torch.tensor(
        tuple(item.use_null_style for item in captions),
        dtype=torch.bool,
        device=device,
    )
    active_style = torch.tensor(
        tuple(index for index, item in enumerate(captions) if not item.use_null_style),
        dtype=torch.long,
        device=device,
    )
    return (
        input_ids,
        attention_mask,
        main_indices,
        main_mask,
        tuple(len(item.main_token_indices) for item in captions),
        artist_indices,
        artist_mask,
        use_null,
        active_style,
    )


class CheckpointGenerator:
    """Load one explicit artifact and generate immutable ordered prompt batches."""

    def __init__(
        self,
        *,
        config: RuntimeConfig,
        checkpoint_path: Path,
        checkpoint: CheckpointRef,
        repository_root: Path,
        device: torch.device,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise GenerationContractError("formal checkpoint generation requires CUDA")
        manifest = read_checkpoint_manifest(checkpoint_path)
        if (
            manifest.identity.checkpoint_id != checkpoint.checkpoint_id
            or manifest.identity.update != checkpoint.successful_update
            or manifest.identity.config_sha256 != checkpoint.resolved_config_sha256
            or manifest.kind.value != checkpoint.artifact_kind
        ):
            raise GenerationContractError("checkpoint identity changed after preflight")
        module = load_inference_artifact(
            checkpoint_path,
            manifest.identity,
            device=device,
        )
        if type(module) is not TrainableComposite:
            raise GenerationContractError("checkpoint is not a trainable composite")
        if export_trainable_composite(module) != trainable_composite_spec(config):
            raise GenerationContractError(
                "checkpoint architecture differs from evaluator configuration"
            )
        growth_alpha = 1.0
        if manifest.kind is ArtifactCheckpointKind.RAW:
            _raw_manifest, raw_state = read_raw_checkpoint_state(checkpoint_path)
            growth_alpha = raw_state.growth.alpha
        elif config.growth.enabled:
            raise GenerationContractError(
                "non-raw checkpoint lacks required growth-alpha provenance"
            )
        module.requires_grad_(False)
        module.eval()
        qwen = load_local_qwen(repository_root, device)
        vae = load_local_mage_vae(repository_root, device)
        self.config = config
        self.checkpoint = checkpoint
        self.composite = module
        self.qwen = qwen
        self.vae = vae
        self.device = device
        self.growth_alpha = growth_alpha

    @torch.inference_mode()
    def generate(self, cases: tuple[PromptCase, ...]) -> GeneratedBatch:
        if not cases:
            raise GenerationContractError("generation batch must not be empty")
        height = cases[0].height
        width = cases[0].width
        if any(case.height != height or case.width != width for case in cases):
            raise GenerationContractError(
                "each explicit evaluator batch must use one image shape"
            )
        (
            input_ids,
            attention_mask,
            main_indices,
            main_mask,
            main_lengths,
            artist_indices,
            artist_mask,
            use_null,
            active_style,
        ) = _serialized_conditioning(cases, self.qwen.tokenizer, self.device)
        qwen_output = self.qwen.encoder(input_ids, attention_mask)
        qwen_states = qwen_output.hidden_states
        branch_count = len(cases) * 2
        size_scale_value = 0.5 * math.log2(
            (float(height) * float(width)) / float(512 * 512)
        )
        aspect_value = math.log2(float(width) / float(height))
        size_scale = torch.full(
            (branch_count,), size_scale_value, device=self.device, dtype=torch.float32
        )
        aspect = torch.full(
            (branch_count,), aspect_value, device=self.device, dtype=torch.float32
        )
        placeholder_timestep = torch.zeros(
            branch_count, device=self.device, dtype=torch.float32
        )
        placeholder_latents = tuple(
            torch.empty(
                128,
                height // 16,
                width // 16,
                device=self.device,
                dtype=torch.bfloat16,
            )
            for _ in range(branch_count)
        )
        static_inputs = TrainableCompositeInputs(
            qwen_states=qwen_states,
            main_token_indices=main_indices,
            main_mask=main_mask,
            main_token_lengths=main_lengths,
            artist_token_indices=artist_indices,
            artist_mask=artist_mask,
            use_null_style=use_null,
            active_style_sample_indices=active_style,
            latents=placeholder_latents,
            timestep=placeholder_timestep,
            size_scale=size_scale,
            aspect=aspect,
            growth_alpha=self.growth_alpha,
        )
        conditioning = self.composite.forward_conditioning(static_inputs)
        initial_noise = _initial_gaussian_noise(cases, device=self.device)

        def velocity(state: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
            branches = torch.cat((state, state), dim=0).to(torch.bfloat16)
            branch_timestep = torch.cat((timestep, timestep), dim=0)
            inputs = dataclasses.replace(
                static_inputs,
                latents=tuple(branches.unbind(0)),
                timestep=branch_timestep,
            )
            predictions = self.composite.forward_dit(inputs, conditioning)
            if len(predictions) != branch_count:
                raise GenerationContractError("DiT prediction count changed")
            conditional = torch.stack(predictions[: len(cases)])
            unconditional = torch.stack(predictions[len(cases) :])
            return guided_velocity(
                conditional,
                unconditional,
                state,
                timestep,
                t_eps=self.config.timestep.t_eps,
                guidance_scale=self.config.cfg.scale,
            )

        sampled = sample_profile(velocity, initial_noise, profile="reference")
        decoded = self.vae.decode(sampled.state.to(torch.bfloat16))
        if not bool(torch.isfinite(decoded).all().item()):
            raise GenerationContractError("Mage-VAE produced nonfinite pixels")
        images = (
            decoded.float()
            .add(1.0)
            .mul(127.5)
            .round()
            .clamp(0.0, 255.0)
            .to(device="cpu", dtype=torch.uint8)
        )
        metadata = build_generation_metadata(
            sampled,
            checkpoint_id=self.checkpoint.checkpoint_id,
            checkpoint_kind=self.checkpoint.artifact_kind,
            objective_provenance=self.checkpoint.objective_provenance,
            cfg_scale=self.config.cfg.scale,
        )
        return GeneratedBatch(cases=cases, images=images, metadata=metadata)


__all__ = [
    "CheckpointGenerator",
    "GeneratedBatch",
    "GenerationContractError",
]
