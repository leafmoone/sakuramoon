"""bare_leak5.py — backward-pass variant of the bare2 trigger.

Clean observations (training) all run forward+backward; bad observations
(inference, bare2) are forward-only.  This adds matmul-backward and
conv-backward (dgrad+wgrad) every iteration.  If it stays flat on a
bad-state host, the trigger is the forward-only kernel path; if it still
balloons, the forward/backward asymmetry is dead and the next test is the
two-HCU variant.
"""
import time

import torch
import torch.nn as nn

print("torch", torch.__version__, "hip", torch.version.hip, flush=True)
x = torch.randn(8192, 8192, dtype=torch.bfloat16, device="cuda", requires_grad=True)
conv_a = nn.Conv2d(32, 64, 3, padding=1, bias=False).to("cuda").bfloat16()
act = torch.randn(1, 32, 1024, 1024, dtype=torch.bfloat16, device="cuda", requires_grad=True)
t0 = time.time()
for i in range(20000):
    y = x @ x
    z = conv_a(act)
    y.sum().backward()
    z.sum().backward()
    x.grad = None
    act.grad = None
    if i % 10 == 0:
        try:
            v = int(open("/sys/fs/cgroup/memory.current").read())
            print(i, f"cgroup={v / 2**30:.2f} GiB t={time.time() - t0:.0f}s", flush=True)
        except Exception as e:  # noqa: BLE001
            print("cg err", e, flush=True)
    torch.cuda.synchronize()
print("BARE5_DONE", flush=True)
