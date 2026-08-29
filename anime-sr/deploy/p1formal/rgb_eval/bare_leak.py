"""bare_leak.py — code-free control experiment.

Runs ONLY bare torch on the HCU: one 8192x8192 tensor init + repeated matmul.
No anime-sr code, no VAE, no MIOpen-dependent custom kernels, no data pipeline.
If cgroup host memory still balloons to ~100+ GiB on a bad-state host, the
leak is in the DTK/HSA driver layer (project code fully exonerated). If it
stays bounded (a few GiB), the anime-sr workload pattern is a required
trigger (and the fix hunt moves to 'what in that pattern trips the driver').
"""
import time

import torch

print("torch", torch.__version__, "hip", torch.version.hip, flush=True)
x = torch.randn(8192, 8192, device="cuda")
t0 = time.time()
for i in range(20000):
    x = x @ x
    if i % 10 == 0:
        try:
            v = int(open("/sys/fs/cgroup/memory.current").read())
            print(i, f"cgroup={v / 2**30:.2f} GiB t={time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print("cg err", e, flush=True)
    torch.cuda.synchronize()
print("BARE_DONE", flush=True)
