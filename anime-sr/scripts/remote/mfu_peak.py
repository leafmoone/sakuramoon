"""MFU rated-peak probe (plan §15.3 gate): measure the HCU's actual bf16 GEMM ceiling.

The earlier MFU number (28.8%) was computed against an *assumed* 50 TFLOPS bf16
peak. This probe measures the device's true raw-GEMM ceiling so we can recompute
MFU against the real rated peak and re-estimate the full-training wall time.

Run:  mfu_peak.py <gpu_index>
"""

from __future__ import annotations

import sys
import time

import torch


def peak_gemm(
    gpu: int,
    n: int = 8192,
    dtype: torch.dtype = torch.bfloat16,
    warmup: int = 5,
    iters: int = 30,
) -> dict:
    torch.cuda.set_device(gpu)
    a = torch.randn(n, n, device=f"cuda:{gpu}", dtype=dtype)
    b = torch.randn(n, n, device=f"cuda:{gpu}", dtype=dtype)
    for _ in range(warmup):
        _ = a @ b
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        _ = a @ b
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    tflops = 2.0 * n**3 / dt / 1e12
    return {
        "gpu": gpu,
        "n": n,
        "dtype": str(dtype),
        "seconds_per_matmul": round(dt, 4),
        "tflops": round(tflops, 1),
    }


def main() -> None:
    gpu = int(sys.argv[1])
    name = torch.cuda.get_device_name(gpu)
    total, free = torch.cuda.mem_get_info(gpu)
    bf16 = peak_gemm(gpu, dtype=torch.bfloat16)
    fp32 = peak_gemm(gpu, dtype=torch.float32)
    fp16 = peak_gemm(gpu, dtype=torch.float16)
    print(
        f"device={name} free={free/1e9:.1f}G total={total/1e9:.1f}G\n"
        f"bf16 peak={bf16['tflops']} TFLOPS\n"
        f"fp16 peak={fp16['tflops']} TFLOPS\n"
        f"fp32 peak={fp32['tflops']} TFLOPS"
    )
    print(
        "MFU_RECOMPUTE "
        f"bf16_28.8pct_of_50T => vs measured bf16 peak {bf16['tflops']}T: "
        f"{0.288 * 50.0 / bf16['tflops'] * 100:.0f}% of measured peak"
    )


if __name__ == "__main__":
    main()
