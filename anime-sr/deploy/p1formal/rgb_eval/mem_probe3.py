"""Probe 3: localize the cgroup-anon burst on the REAL pipeline (08-30 r5.2).

Mirrors the smoke3 group-0 exactly: webp map -> hr_crop_for -> lq_for x5 ->
infer_image x5 (b1), printing host RSS after every op.  Self-exits at
cgroup > 90 GiB (protects the pod).  Run only when no other eval is alive
(cgroup is shared; a co-running eval pollutes the curve).
"""

import gc
import os
import sys

sys.path.insert(0, "/root/anime-sr-p1formal/src")

import rgb_eval as RE


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
    print(f"[probe3] {tag}: rss={rss_mb():.0f} MB cgroup={cur} MB", flush=True)
    if cur > 90_000:
        print(f"[probe3] GUARD: cgroup {cur} MB; self-exit", flush=True)
        os._exit(42)


def main() -> None:
    ev = RE.Evaluator(args=None)
    mark("after-load")

    webp = RE.build_sid_webp_map()
    mark(f"webp-map n={len(webp)}")

    sids = RE.select_set_a(128)[:2]
    hrs = []
    for n, sid in enumerate(sids):
        hr, box = RE.hr_crop_for(sid, webp)
        hrs.append(hr)
        mark(f"hr_crop-{n} ({sid} {box})")

    for p in RE.PROFILES:
        lq = RE.lq_for(hrs[0], ev.profile_cfg[p], sids[0])
        mark(f"lq_for-{p}")
        del lq
        gc.collect()

    for n, (hr, sid) in enumerate(zip(hrs, sids)):
        for p in RE.PROFILES:
            res = ev.infer_image(hr, sid, p)
            del res
            gc.collect()
            torch.cuda.empty_cache() if (n, p) == (0, RE.PROFILES[0]) else None
        mark(f"infer_done-sid{n} ({sid})")
        del hr
        gc.collect()
    print("[probe3] DONE", flush=True)


import torch  # noqa: E402  (needed for empty_cache above)

if __name__ == "__main__":
    main()
