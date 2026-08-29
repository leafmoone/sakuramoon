"""bare_leak2.py — refined trigger experiment.

vs bare_leak.py (fp32 matmul, clean): adds the pieces anime-sr does beyond a
plain matmul — bfloat16 GEMM + a real conv2d at U-Net scale (32->64ch @ 1024x1024,
exercises the MIOpen conv path). If this balloons on the bad-state host, the
trigger is the bf16/conv(MIOpen) kernel path; if flat, the trigger lies further
up (ckpt load of many tensors / VAE / shape churn).
"""
import time

import torch
import torch.nn as nn

print("torch", torch.__version__, "hip", torch.version.hip, flush=True)
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
print("BARE2_DONE", flush=True)
