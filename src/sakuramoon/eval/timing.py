"""Clock-domain uncertainty for single-GPU evaluator cost measurements."""

from __future__ import annotations

import math
import time

_CUDA_EVENT_TIMESTAMP_RESOLUTION_SECONDS = 0.5e-6
_FLOAT32_MIN_SUBNORMAL_MILLISECONDS_EXPONENT = -149


def _float32_millisecond_ulp_seconds(gpu_seconds: float) -> float:
    if gpu_seconds == 0.0:
        return math.ldexp(
            1.0, _FLOAT32_MIN_SUBNORMAL_MILLISECONDS_EXPONENT
        ) / 1000.0
    seconds_mantissa, seconds_exponent = math.frexp(gpu_seconds)
    _milliseconds_mantissa, milliseconds_scale_exponent = math.frexp(
        seconds_mantissa * 1000.0
    )
    milliseconds_exponent = seconds_exponent + milliseconds_scale_exponent
    ulp_exponent = max(
        milliseconds_exponent - 24,
        _FLOAT32_MIN_SUBNORMAL_MILLISECONDS_EXPONENT,
    )
    return math.ldexp(1.0, ulp_exponent) / 1000.0


def allowed_gpu_clock_overshoot_seconds(gpu_seconds: float) -> float:
    """Return measured-duration uncertainty without altering either measurement."""

    if type(gpu_seconds) is not float or not math.isfinite(gpu_seconds) or gpu_seconds < 0:
        raise ValueError("gpu_seconds must be a finite nonnegative float")
    perf_counter_resolution = time.get_clock_info("perf_counter").resolution
    return (
        2.0 * _CUDA_EVENT_TIMESTAMP_RESOLUTION_SECONDS
        + _float32_millisecond_ulp_seconds(gpu_seconds)
        + 2.0 * perf_counter_resolution
        + math.ulp(gpu_seconds)
    )


def require_plausible_single_gpu_timing(
    *, wall_seconds: float, gpu_seconds: float
) -> None:
    """Reject GPU duration beyond explicit cross-clock measurement uncertainty."""

    allowed_overshoot = allowed_gpu_clock_overshoot_seconds(gpu_seconds)
    maximum_gpu_seconds = wall_seconds + allowed_overshoot
    if gpu_seconds > maximum_gpu_seconds:
        overshoot_seconds = gpu_seconds - wall_seconds
        raise ValueError(
            "GPU seconds cannot exceed wall seconds for one GPU beyond clock "
            "quantization: "
            f"gpu_seconds={gpu_seconds!r}, wall_seconds={wall_seconds!r}, "
            f"overshoot_seconds={overshoot_seconds!r}, "
            f"allowed_clock_overshoot_seconds={allowed_overshoot!r}"
        )


__all__ = [
    "allowed_gpu_clock_overshoot_seconds",
    "require_plausible_single_gpu_timing",
]
