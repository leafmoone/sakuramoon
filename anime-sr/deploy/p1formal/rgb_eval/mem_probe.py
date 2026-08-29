"""Per-op memory probe: attribute the cgroup anon growth (08-30 OOM forensics).

Runs each hot-loop op of rgb_eval in sequence on a single HCU, printing
host RSS (VmRSS) + HCU-side allocator stats after each op, across 3
identical rounds.  If RSS keeps climbing round over round, the leaking
op is identified.  Self-terminates if cgroup memory.current exceeds the
guard threshold (protects the pod from a 4th OOM re-provision).
"""

import os
import sys

sys.path.insert(0, "/root/anime-sr-p1formal/src")

import torch

import rgb_eval as RE


def rss_mb() -> float:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024.0
    return -1.0


def cgroup_mb() -> int:
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read()) // (1 << 20)
    except OSError:
        return -1


def guard(tag: str) -> None:
    cur = cgroup_mb()
    if cur > 100_000:  # ~100 GiB of the 118 GiB cap
        print(f"[probe] GUARD: cgroup {cur} MB > 100 GiB after {tag}; self-exit", flush=True)
        os._exit(42)


def main() -> None:
    ev = RE.Evaluator(args=None)
    dev, dt = ev.device, torch.bfloat16
    print(f"[probe] loaded model keys={ev.n_model_keys} ckpt_step={ev.ckpt_step}", flush=True)
    print(f"[probe] baseline rss={rss_mb():.0f} MB cgroup={cgroup_mb()} MB", flush=True)

    # dummy LQ: 5 profiles of a 256x256 image (batch of 5 like infer_many)
    lq_cpu = torch.zeros(5, 3, 256, 256, dtype=torch.float32)
    lq_dev = lq_cpu.to(dev, dt)

    def step(name: str) -> None:
        nonlocal lq_dev
        torch.cuda.synchronize()
        alloc = torch.cuda.memory_allocated() // (1 << 20)
        resv = torch.cuda.memory_reserved() // (1 << 20)
        print(f"[probe] {name}: rss={rss_mb():.0f} MB cgroup={cgroup_mb()} MB "
              f"hcu_alloc={alloc} MB hcu_resv={resv} MB", flush=True)
        guard(name)

    step("baseline(after load)")
    bic = torch.nn.functional.interpolate(
        lq_dev.float(), size=(1024, 1024), mode="bicubic", align_corners=False
    ).to(dt)
    step("after interpolate1024")
    z_lr = ev.vae.encode(bic)
    step("after vae.encode")
    anchor = ev.vae.decode(z_lr)
    step("after vae.decode(anchor)")
    z1 = ev.sampler.one_step(z_lr, (z_lr, lq_dev), sigma=0.0)
    step("after sampler.one_step")
    z4 = ev.sampler.four_step(z_lr, (z_lr, lq_dev), sigma=0.0)
    step("after sampler.four_step(7 nfe)")
    o1 = ev.vae.decode(z1)
    o4 = ev.vae.decode(z4)
    step("after vae.decode(o1,o4)")

    for rnd in (2, 3):
        z_lr = ev.vae.encode(bic)
        z1 = ev.sampler.one_step(z_lr, (z_lr, lq_dev), sigma=0.0)
        z4 = ev.sampler.four_step(z_lr, (z_lr, lq_dev), sigma=0.0)
        o1 = ev.vae.decode(z1)
        o4 = ev.vae.decode(z4)
        step(f"round{rnd}(encode+1step+4step+2decode)")
        del o1, o4, z1, z4, z_lr
        gc_mark()
    print("[probe] DONE", flush=True)


def gc_mark() -> None:
    import gc

    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
