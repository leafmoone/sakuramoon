"""Diagnostic Newton-Schulz replay tracer (F1 forensic, telemetry-only).

This module exists to answer, AFTER a hard-failure verdict is already
settled, "what did the NS iteration do to the exact failing input, in
which dtype, at which iteration?" It is a second, analysis-only pass over
a clone of the exact rescue input.

Hard guarantees (see reports/cmuon-fp32-rescue-forensic-audit.md §7):

  * The op sequence (casts, transpose-to-wide, Frobenius clamp
    normalization, the quintic addmm iterations) replicates
    ``cmuon_zeroth_power_bf16`` / ``cmuon_zeroth_power_fp32`` exactly.
    The extra norm/max reductions are side-effect-free reads; they never
    feed back into the working matrix.
  * The replay output is NEVER written back: it cannot change the fail
    flag, the staged delta, momentum, parameters, the owner broadcast, or
    the commit. The production result already happened before this runs.
  * If the replay itself fails, callers record ``forensic_trace_error``
    and the original ``CMuonSafetyError`` is still raised: an analysis
    failure never masks a production failure.

This module is imported only by the forensic dump path of
``fp32_rescue.py`` and by the developer tools
(``dev-tools/cmuon_fp32_rescue_replay.py`` /
``dev-tools/cmuon_fp32_rescue_stress.py``). ``cmuon.py`` is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

_WORKING_DTYPES = ("bfloat16", "float32")


@dataclass(frozen=True)
class NSIterationTrace:
    """One Newton-Schulz iteration of the replay (post-iteration working
    matrix stats + the gram matrix that produced it)."""

    iteration: int  # 1-based
    working_dtype: str
    # working matrix (post-iteration):
    frobenius_norm: float
    rms: float
    abs_max: float
    finite: bool
    # gram matrix of this iteration (pre-update):
    gram_frobenius_norm: float
    gram_rms: float
    gram_abs_max: float
    gram_finite: bool


@dataclass(frozen=True)
class NSReplayResult:
    """Full diagnostic replay of one NS pass on one exact input.

    ``final_delta_rms`` / ``final_delta_finite`` are the Moonlight-scaled
    update statistics the production verdict compared against the
    [rescue_floor, ceiling] band (delta = -alpha * NS output), computed
    the same way the production verdict computes them (fp32
    pow(2).mean().sqrt() on the final working matrix, before any BF16
    staging rounding).
    """

    working_dtype: str
    transposed: bool
    # raw input (as saved; the production BF16 path casts to BF16, the
    # FP32 path casts to FP32 before NS):
    input_frobenius_norm: float
    input_rms: float
    input_abs_max: float
    input_finite: bool
    # input after the Frobenius clamp normalization (what iteration 1 sees):
    normalized_input_rms: float
    normalized_input_finite: bool
    norm_divisor: float  # ortho.norm().clamp(min=eps) value
    iterations: tuple[NSIterationTrace, ...]
    # final working matrix (pre-Moonlight-scale):
    final_frobenius_norm: float
    final_rms: float
    final_abs_max: float
    final_finite: bool
    # Moonlight-scaled delta (delta = -alpha * final), verdict statistics:
    final_delta_rms: float
    final_delta_finite: bool


def _stats(t: torch.Tensor) -> tuple[float, float, float, bool]:
    """(fro, rms, abs_max, finite) of a tensor, reduced in FP32."""
    tf = t.float()
    fro = float(tf.norm().item())
    rms = float(tf.pow(2).mean().sqrt().item())
    abs_max = float(tf.abs().max().item())
    finite = bool(torch.isfinite(tf).all().item())
    return fro, rms, abs_max, finite


def _validate(
    input_tensor: torch.Tensor,
    ns_steps: int,
    coefficients: tuple[float, float, float],
    eps: float,
) -> None:
    """Same input validation as the production NS functions."""
    if ns_steps <= 0 or ns_steps >= 100:
        raise ValueError("ns_steps must be in [1, 99]")
    if input_tensor.ndim != 2:
        raise ValueError("cmuon input must be a 2D matrix")
    if len(coefficients) != 3:
        raise ValueError("ns_coefficients must be a 3-tuple")
    if not (eps > 0.0):
        raise ValueError("eps must be positive")


def trace_ns_replay(
    input_tensor: torch.Tensor,
    *,
    ns_steps: int,
    coefficients: tuple[float, float, float],
    eps: float,
    alpha: float,
    working_dtype: str = "bfloat16",
) -> NSReplayResult:
    """Replay one NS pass on ``input_tensor`` (a clone; the input is never
    modified) and return per-iteration diagnostics.

    ``working_dtype`` selects the exact production path to replicate:
    ``"bfloat16"`` mirrors ``cmuon_zeroth_power_bf16`` (input cast to
    BF16, all arithmetic BF16); ``"float32"`` mirrors
    ``cmuon_zeroth_power_fp32`` (input cast to FP32, all arithmetic FP32).
    ``alpha`` is the Moonlight scale applied in the final delta statistics
    (the production staging delta is ``(-alpha) * ns``).
    """
    if working_dtype not in _WORKING_DTYPES:
        raise ValueError(
            f"working_dtype must be one of {_WORKING_DTYPES}, got {working_dtype!r}"
        )
    _validate(input_tensor, ns_steps, coefficients, eps)
    a, b, c = coefficients

    raw_fro, raw_rms, raw_max, raw_finite = _stats(input_tensor)

    ortho = input_tensor.bfloat16() if working_dtype == "bfloat16" else input_tensor.float()
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T

    # Exact production normalization (Frobenius, clamped by eps).
    divisor = ortho.norm().clamp(min=eps)
    ortho = ortho / divisor
    norm_rms, norm_finite = float(ortho.float().pow(2).mean().sqrt().item()), bool(
        torch.isfinite(ortho.float()).all().item()
    )

    iterations: list[NSIterationTrace] = []
    for k in range(ns_steps):
        gram = ortho @ ortho.T
        g_fro, g_rms, g_max, g_finite = _stats(gram)
        # gram_update = b*gram + c*(gram @ gram); ortho = a*ortho + gram_update@ortho
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
        w_fro, w_rms, w_max, w_finite = _stats(ortho)
        iterations.append(
            NSIterationTrace(
                iteration=k + 1,
                working_dtype=working_dtype,
                frobenius_norm=w_fro,
                rms=w_rms,
                abs_max=w_max,
                finite=w_finite,
                gram_frobenius_norm=g_fro,
                gram_rms=g_rms,
                gram_abs_max=g_max,
                gram_finite=g_finite,
            )
        )
    if transposed:
        ortho = ortho.T

    f_fro, f_rms, f_max, f_finite = _stats(ortho)
    # Verdict statistics: delta = -alpha * ns, fp32 rms exactly like the
    # production rescue readback (sign does not affect rms/finiteness).
    delta = (-float(alpha)) * ortho.float()
    d_rms = float(delta.pow(2).mean().sqrt().item())
    d_finite = bool(torch.isfinite(delta).all().item())

    return NSReplayResult(
        working_dtype=working_dtype,
        transposed=transposed,
        input_frobenius_norm=raw_fro,
        input_rms=raw_rms,
        input_abs_max=raw_max,
        input_finite=raw_finite,
        normalized_input_rms=norm_rms,
        normalized_input_finite=norm_finite,
        norm_divisor=float(divisor.item()),
        iterations=tuple(iterations),
        final_frobenius_norm=f_fro,
        final_rms=f_rms,
        final_abs_max=f_max,
        final_finite=f_finite,
        final_delta_rms=d_rms,
        final_delta_finite=d_finite,
    )


def replay_result_to_json(result: NSReplayResult) -> dict[str, object]:
    """JSON-serializable form of a replay result (artifact metadata)."""
    return {
        "working_dtype": result.working_dtype,
        "transposed": result.transposed,
        "input": {
            "frobenius_norm": result.input_frobenius_norm,
            "rms": result.input_rms,
            "abs_max": result.input_abs_max,
            "finite": result.input_finite,
        },
        "normalized_input": {
            "rms": result.normalized_input_rms,
            "finite": result.normalized_input_finite,
            "norm_divisor": result.norm_divisor,
        },
        "iterations": [
            {
                "iteration": t.iteration,
                "frobenius_norm": t.frobenius_norm,
                "rms": t.rms,
                "abs_max": t.abs_max,
                "finite": t.finite,
                "gram_frobenius_norm": t.gram_frobenius_norm,
                "gram_rms": t.gram_rms,
                "gram_abs_max": t.gram_abs_max,
                "gram_finite": t.gram_finite,
            }
            for t in result.iterations
        ],
        "final": {
            "frobenius_norm": result.final_frobenius_norm,
            "rms": result.final_rms,
            "abs_max": result.final_abs_max,
            "finite": result.final_finite,
            "delta_rms": result.final_delta_rms,
            "delta_finite": result.final_delta_finite,
        },
    }


__all__ = [
    "NSIterationTrace",
    "NSReplayResult",
    "replay_result_to_json",
    "trace_ns_replay",
]
