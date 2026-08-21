"""Text-only access to the frozen local Qwen3.5 language model."""

from __future__ import annotations

import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
)
from transformers.models.qwen3_5.modeling_qwen3_5 import is_fast_path_available

from sakuramoon.assets import require_local_qwen
from sakuramoon.data.serialize import CONDITION_BUCKETS, EXPECTED_PREFIX_TOKENS

HIDDEN_STATE_BLOCKS = (2, 4, 8, 12, 16, 20, 24)
QWEN_DENSE_LENGTHS = tuple(
    EXPECTED_PREFIX_TOKENS + condition_bucket
    for condition_bucket in CONDITION_BUCKETS
)
# Default dense launch group for the length-aware Qwen fast path.  The
# production local batch is 19, so any group size >= 19 is a single padded
# launch (a no-op relative to per-bucket tiling).  A batch-19 benchmark
# (scripts/benchmark_qwen_dense_groups.py, --batch 19) shows group size 10
# splits the sorted rows into two launches (padded to 226 and 482) that avoid
# padding the short rows out to 482, running ~15% faster than the no-op
# 32; smaller groups add launch overhead (8 ~ break-even, 4 is slower).
# 10 is optimal for the fixed local batch 19; re-measure if the batch moves.
QWEN_DENSE_GROUP_SIZE = 10
QWEN_FAST_PATH_PROBE_LENGTH = 290


@dataclass(frozen=True)
class QwenEncoderOutput:
    """Seven selected states from one Qwen forward."""

    hidden_states: torch.Tensor
    attention_mask: torch.Tensor


@dataclass(frozen=True)
class QwenRuntime:
    encoder: FrozenQwenEncoder
    tokenizer: PreTrainedTokenizerBase


class FrozenQwenEncoder(nn.Module):
    """Keep Qwen frozen and expose only the approved seven text states."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.model.requires_grad_(False)
        self.model.eval()
        super().train(False)

    def train(self, mode: bool = True) -> FrozenQwenEncoder:
        """A frozen encoder remains in evaluation mode."""

        del mode
        super().train(False)
        self.model.eval()
        return self

    def _forward_selected(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        layers = getattr(self.model, "layers", None)
        if not isinstance(layers, nn.ModuleList) or len(layers) != 24:
            raise RuntimeError("Qwen must expose exactly 24 decoder layers")
        captured: dict[int, list[torch.Tensor]] = {
            block: [] for block in HIDDEN_STATE_BLOCKS
        }

        def capture_block(
            block: int,
        ) -> Callable[
            [nn.Module, tuple[object, ...], object],
            None,
        ]:
            def hook(
                _module: nn.Module,
                _inputs: tuple[object, ...],
                block_output: object,
            ) -> None:
                if not isinstance(block_output, torch.Tensor):
                    raise TypeError(f"Qwen block {block} must return one raw tensor")
                captured[block].append(block_output)

            return hook

        handles = tuple(
            layers[block - 1].register_forward_hook(capture_block(block))
            for block in HIDDEN_STATE_BLOCKS
        )
        try:
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        finally:
            for handle in handles:
                handle.remove()
        for block, values in captured.items():
            if len(values) != 1:
                raise RuntimeError(
                    f"Qwen block {block} must run exactly once, got {len(values)}"
                )

        selected = torch.stack(
            tuple(captured[block][0] for block in HIDDEN_STATE_BLOCKS),
            dim=2,
        )
        if selected.shape[-1] != 2048:
            raise RuntimeError(f"Qwen hidden size must be 2048, got {selected.shape[-1]}")
        return selected.detach()

    @staticmethod
    def _group_rows(
        dense_lengths: tuple[int, ...],
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        grouped: dict[int, list[int]] = {}
        for row, length in enumerate(dense_lengths):
            grouped.setdefault(length, []).append(row)
        return tuple(
            (length, tuple(rows)) for length, rows in sorted(grouped.items())
        )

    @staticmethod
    def _chunk_rows(
        dense_lengths: tuple[int, ...],
        group_size: int,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        ordered = sorted(
            range(len(dense_lengths)),
            key=lambda row: (dense_lengths[row], row),
        )
        groups: list[tuple[int, tuple[int, ...]]] = []
        for start in range(0, len(ordered), group_size):
            rows = tuple(ordered[start : start + group_size])
            groups.append((max(dense_lengths[row] for row in rows), rows))
        return tuple(groups)

    @staticmethod
    def _validate_declared_lengths(
        attention_mask: torch.Tensor,
        dense_lengths: tuple[int, ...],
    ) -> None:
        for length, rows in FrozenQwenEncoder._group_rows(dense_lengths):
            row_indices = torch.tensor(
                rows, device=attention_mask.device, dtype=torch.long
            )
            trailing_mask_is_empty = (
                ~attention_mask.index_select(0, row_indices)[:, length:]
            ).all()
            if attention_mask.is_cuda:
                torch._assert_async(  # pyright: ignore[reportPrivateUsage,reportPrivateImportUsage]
                    trailing_mask_is_empty
                )
            elif not bool(trailing_mask_is_empty):
                raise ValueError("dense length truncates a valid Qwen token")

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        dense_lengths: tuple[int, ...] | None = None,
        dense_group_size: int = QWEN_DENSE_GROUP_SIZE,
    ) -> QwenEncoderOutput:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("input_ids and attention_mask must have shape [batch, length]")
        if input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if attention_mask.dtype != torch.bool:
            raise TypeError(
                "attention_mask must use torch.bool with True meaning valid token"
            )

        if dense_lengths is None:
            selected = self._forward_selected(input_ids, attention_mask)
        else:
            if (
                type(dense_lengths) is not tuple
                or len(dense_lengths) != input_ids.shape[0]
                or any(
                    type(length) is not int
                    or length not in QWEN_DENSE_LENGTHS
                    or length > input_ids.shape[1]
                    for length in dense_lengths
                )
            ):
                raise ValueError(
                    "dense_lengths must contain one supported Qwen bucket per row"
                )
            if type(dense_group_size) is not int or dense_group_size <= 0:
                raise ValueError("dense_group_size must be a positive integer")
            self._validate_declared_lengths(attention_mask, dense_lengths)
            homogeneous_length = dense_lengths[0]
            if all(length == homogeneous_length for length in dense_lengths):
                # A length-aware data batch can use one large Qwen launch and
                # avoid index_select/index_copy plus four BS32 forwards.
                selected = self._forward_selected(
                    input_ids[:, :homogeneous_length],
                    attention_mask[:, :homogeneous_length],
                )
            else:
                groups = self._chunk_rows(dense_lengths, dense_group_size)
                selected: torch.Tensor | None = None
                for length, rows in groups:
                    row_indices = torch.tensor(
                        rows, device=input_ids.device, dtype=torch.long
                    )
                    group_selected = self._forward_selected(
                        input_ids.index_select(0, row_indices)[:, :length].contiguous(),
                        attention_mask.index_select(0, row_indices)[
                            :, :length
                        ].contiguous(),
                    )
                    if selected is None:
                        selected = torch.zeros(
                            (input_ids.shape[0], input_ids.shape[1], 7, 2048),
                            device=input_ids.device,
                            dtype=group_selected.dtype,
                        )
                    selected[:, :length].index_copy_(
                        0, row_indices, group_selected
                    )
                if selected is None:
                    raise RuntimeError("Qwen dense grouping produced no output")
        return QwenEncoderOutput(
            hidden_states=selected,
            attention_mask=attention_mask,
        )


@dataclass(frozen=True)
class QwenFastPathFacts:
    """Observed evidence for the Qwen fast-path preflight gate.

    ``conv_layer_count`` counts only the linear-attention layers, which are
    the ones that carry a causal-conv1d kernel (``Qwen3_5GatedDeltaNet``).
    The full-attention layers do not carry conv kernels and are excluded.
    """

    flag_available: bool
    conv_module: str | None
    wired_conv_layers: int
    conv_layer_count: int
    layer_count: int


def _iter_linear_conv_kernels(layers: nn.ModuleList) -> list:
    """Return the per-linear-layer conv kernel (or None) in layer order."""
    kernels = []
    for layer in layers:
        if getattr(layer, "block_type", None) == "linear_attention":
            kernels.append(
                getattr(getattr(layer, "linear_attn", None), "causal_conv1d_fn", None)
            )
    return kernels


def require_qwen_fast_path(encoder: FrozenQwenEncoder) -> QwenFastPathFacts:
    """Production training must not run on the Transformers PyTorch fallback.

    The DTK Transformers fork selects the FLA/causal-conv1d kernels at import
    time.  When the FLA wheel or the causal-conv1d shim is missing the model
    silently trains on the slower PyTorch implementation, so the production
    preflight fails closed on either missing piece: the import-time flag or
    the per-linear-layer conv wiring.
    """

    layers = getattr(encoder.model, "layers", None)
    if not isinstance(layers, nn.ModuleList) or len(layers) != 24:
        raise RuntimeError("Qwen fast path cannot be verified: model exposes no decoder layers")
    conv_functions = _iter_linear_conv_kernels(layers)
    conv_layer_count = len(conv_functions)
    wired_conv_layers = sum(conv is not None for conv in conv_functions)
    conv_module = next(
        (getattr(conv, "__module__", None) for conv in conv_functions if conv is not None),
        None,
    )
    facts = QwenFastPathFacts(
        flag_available=bool(is_fast_path_available),
        conv_module=conv_module,
        wired_conv_layers=wired_conv_layers,
        conv_layer_count=conv_layer_count,
        layer_count=len(layers),
    )
    if (
        not facts.flag_available
        or conv_layer_count == 0
        or facts.wired_conv_layers != facts.conv_layer_count
    ):
        raise RuntimeError(
            "Qwen3.5 fast path is not fully active "
            f"(is_fast_path_available={facts.flag_available}, conv kernels wired on "
            f"{facts.wired_conv_layers}/{facts.conv_layer_count} linear-attention layers, "
            f"conv module={facts.conv_module!r}). "
            "Production training must not run on the Transformers PyTorch "
            "fallback. Install the FLA wheel and keep the causal-conv1d "
            "shim on PYTHONPATH."
        )
    return facts


def probe_qwen_fast_path(
    encoder: FrozenQwenEncoder,
    probe_rows: int,
    *,
    probe_length: int = QWEN_FAST_PATH_PROBE_LENGTH,
    timed_launches: int = 2,
) -> float:
    """Return the mean wall-clock seconds per homogeneous Qwen launch.

    One warm-up launch absorbs allocation and first-kernel costs.  The probe
    follows the length-aware single-launch path and is only used as preflight
    evidence, never as a training input.

    The full-attention layers use SDPA; the probe pins the always-available
    "math" SDPA backend so the probe stays robust even when the flash /
    efficient kernels are absent from the environment.  The FLA fast path
    under test lives in the linear-attention layers and is independent of the
    SDPA backend choice, so this does not weaken the fast-path evidence.
    """

    if type(probe_rows) is not int or probe_rows <= 0:
        raise ValueError("probe_rows must be a positive integer")
    if type(timed_launches) is not int or timed_launches <= 0:
        raise ValueError("timed_launches must be a positive integer")
    if probe_length not in QWEN_DENSE_LENGTHS:
        raise ValueError("probe_length must be a supported Qwen dense bucket")
    device = next(encoder.model.parameters()).device
    input_ids = torch.randint(
        1, 32_768, (probe_rows, probe_length), device=device, dtype=torch.long
    )
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
    dense_lengths = (probe_length,) * probe_rows
    sdpa_backend = _probe_sdpa_backend()
    with torch.inference_mode():
        with sdpa_backend():
            encoder.forward(input_ids, attention_mask, dense_lengths=dense_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        with sdpa_backend():
            for _ in range(timed_launches):
                encoder.forward(input_ids, attention_mask, dense_lengths=dense_lengths)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    return (time.perf_counter() - started) / timed_launches


def _probe_sdpa_backend():
    """A context manager pinning the always-available "math" SDPA backend.

    Falls back to a no-op context when the SDPA backend selector is not
    present in the environment (the probe then uses whatever backend the
    process default picks).
    """

    from contextlib import contextmanager

    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel
    except ImportError:  # very old torch without the SDPA backend selector

        @contextmanager
        def _no_op():
            yield

        return _no_op

    return lambda: sdpa_kernel(SDPBackend.MATH)


def load_local_qwen(
    repository_root: Path,
    device: torch.device,
    *,
    attention_backend: str = "sdpa",
) -> QwenRuntime:
    """Load the fixed local text model without creating the visual tower."""

    if device.type != "cuda":
        raise ValueError("the production Qwen encoder requires a CUDA device")
    if attention_backend not in {"sdpa", "flash_attention_2"}:
        raise ValueError("Qwen attention backend is invalid")
    model_path = require_local_qwen(repository_root)
    if not is_fast_path_available:
        warnings.warn(
            "Qwen3.5 FLA/causal-conv1d kernels are unavailable; using the "
            "Transformers PyTorch fallback. Training will be slower.",
            RuntimeWarning,
            stacklevel=2,
        )

    config = Qwen3_5TextConfig.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_path, local_files_only=True
    )
    if config.num_hidden_layers != 24 or config.hidden_size != 2048:
        raise RuntimeError("local Qwen text config must be 24 layers with hidden size 2048")
    config.use_cache = False

    causal_lm = Qwen3_5ForCausalLM.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
        model_path,
        config=config,
        local_files_only=True,
        trust_remote_code=False,
        attn_implementation=attention_backend,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    text_model = causal_lm.model
    text_model.to(  # pyright: ignore[reportUnknownMemberType, reportArgumentType, reportCallIssue]
        device=device
    )
    encoder = FrozenQwenEncoder(text_model)
    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(  # pyright: ignore[reportUnknownMemberType]
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            use_fast=True,
            padding_side="right",
        ),
    )
    if tokenizer.pad_token_id != 248044:
        raise RuntimeError("Qwen padding token must be <|endoftext|> (248044)")
    return QwenRuntime(encoder=encoder, tokenizer=tokenizer)


__all__ = [
    "HIDDEN_STATE_BLOCKS",
    "QWEN_DENSE_GROUP_SIZE",
    "QWEN_DENSE_LENGTHS",
    "QWEN_FAST_PATH_PROBE_LENGTH",
    "FrozenQwenEncoder",
    "QwenEncoderOutput",
    "QwenFastPathFacts",
    "QwenRuntime",
    "load_local_qwen",
    "probe_qwen_fast_path",
    "require_qwen_fast_path",
]
