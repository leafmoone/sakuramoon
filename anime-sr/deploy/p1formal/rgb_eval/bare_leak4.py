"""bare_leak4.py — ballast test of the 'HBM free-space path' hypothesis.

All bad-state observations are small-footprint inference-style processes
(~5-8 GiB of 64 GiB HCU); all clean observations are large-footprint
training processes (40-50+ GiB).  This holds a 40 GiB ballast tensor (mimic
the training regime) and then runs the exact bare2 trigger loop (bf16 matmul
+ 1024^2 convs).  If it stays flat on a bad-state host, the workaround for
anime-sr inference is a ballast allocation pushing HCU occupancy into the
training regime; if it still balloons, the footprint theory is dead.
"""
import time

import torch
import torch.nn as nn

print("torch", torch.__version__, "hip", torch.version.hip, flush=True)
# 40 GiB ballast: 20 * 1024^3 elements * 2 bytes (bf16)
ballast = torch.empty(20 * 1024**3, dtype=torch.bfloat16, device="cuda")
torch.cuda.synchronize()
print("ballast 40 GiB allocated", flush=True)
x = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda")
conv_a = nn.Conv2d(32, 64, 3, padding=1, bias=False).to("cuda").bfloat16()
conv_b = nn.Conv2d(64, 32, 3, padding=1, bias=False).to("cuda").bfloat16()
act = torch.randn(1, 32, 1024, 1024, dtype=torch.bfloat16, device="cuda")
t0 = time.time()
for i in range(20000):
    x = x @ x
    act = conv_b(act) if i % 2 else conv_a(act)
    if i % 10 == 0:
        try:
            v = int(open("/sys/fs/cgroup/memory.current").read())
            print(i, f"cgroup={v / 2**30:.2f} GiB t={time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print("cg err", e, flush=True)
    torch.cuda.synchronize()
print("BARE4_DONE", flush=True)
