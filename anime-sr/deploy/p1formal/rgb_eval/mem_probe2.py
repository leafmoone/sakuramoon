"""Probe 2: is the cgroup anon burst caused by the prior HCU OOM event?

mode "fallback"  — reproduce rgb_eval.infer_many: attempt the b5 batched
    path (HCU OOM expected), catch, empty_cache, then run 5x b1
    infer_image, printing host RSS after each.  (Reproduces the smoke
    test's path.)
mode "direct"    — fresh process, NO prior OOM: run 5x b1 directly with
    empty_cache+gc between profiles.  (Candidate fix: skip the batched
    attempt entirely.)

Self-exits at cgroup > 90 GiB to protect the pod.
"""

import gc
import os
import sys

sys.path.insert(0, "/root/anime-sr-p1formal/src")

import torch

import rgb_eval as RE

MODE = sys.argv[1] if len(sys.argv) > 1 else "fallback"


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def cgroup_mb() -> int:
    with open("/sys/fs/cgroup/memory.current") as f:
        return int(f.read()) // (1 << 20)


def mark(tag: str) -> None:
    cur = cgroup_mb()
    print(f"[probe2:{MODE}] {tag}: rss={rss_mb():.0f} MB cgroup={cur} MB", flush=True)
    if cur > 90_000:
        print(f"[probe2:{MODE}] GUARD: cgroup {cur} MB; self-exit", flush=True)
        os._exit(42)


def main() -> None:
    ev = RE.Evaluator(args=None)
    dev, dt = ev.device, torch.bfloat16
    lq_cpu = torch.zeros(1, 3, 256, 256, dtype=torch.float32)  # b1 dummy
    hr = torch.zeros(1, 3, 1024, 1024, dtype=torch.float32)    # b1 infer_image shape
    mark("after-load")

    if MODE == "fallback":
        # --- reproduce infer_many: batched b5 attempt first ---
        lq5 = torch.zeros(5, 3, 256, 256, dtype=torch.float32)
        lq5_dev = lq5.to(dev, dt)
        try:
            bic5 = torch.nn.functional.interpolate(
                lq5_dev.float(), size=(1024, 1024), mode="bicubic", align_corners=False
            ).to(dt)
            z5 = ev.vae.encode(bic5)
            a5 = ev.vae.decode(z5)
            z5_1 = ev.sampler.one_step(z5, (z5, lq5_dev), sigma=0.0)
            z5_4 = ev.sampler.four_step(z5, (z5, lq5_dev), sigma=0.0)
            mark("b5-completed(no-oom?!)")
        except torch.OutOfMemoryError as e:
            print(f"[probe2:fallback] HCU OOM caught ({str(e)[:60]}...); empty_cache", flush=True)
            torch.cuda.empty_cache()
        mark("after-b5-attempt+empty_cache")
        del lq5, lq5_dev
        gc.collect()

    # --- 5x b1, as infer_image does (lq_for skipped: dummy lq) ---
    for i in range(5):
        lq_dev = lq_cpu.to(dev, dt)
        bic = torch.nn.functional.interpolate(
            lq_dev.float(), size=(1024, 1024), mode="bicubic", align_corners=False
        ).to(dt)
        with torch.no_grad():
            z_lr = ev.vae.encode(bic)
            anchor = ev.vae.decode(z_lr)
            z1 = ev.sampler.one_step(z_lr, (z_lr, lq_dev), sigma=0.0)
            z4 = ev.sampler.four_step(z_lr, (z_lr, lq_dev), sigma=0.0)
            o1 = ev.vae.decode(z1)
            o4 = ev.vae.decode(z4)
        _ = o1.float().cpu(), o4.float().cpu()
        del bic, z_lr, anchor, z1, z4, o1, o4
        gc.collect()
        torch.cuda.empty_cache()
        mark(f"b1-{i + 1}/5-done")
    print(f"[probe2:{MODE}] DONE", flush=True)


if __name__ == "__main__":
    main()
