"""Probe 4 (08-30 r5.3): run the REAL smoke (rgb_eval --set A --limit 10,
the same command as smoke3.sh) with a 2s host-RSS + cgroup + GC sampler.

Discriminator: probe3 ran the identical per-op sequence standalone and stayed
bounded (5.8 GB).  If the burst reproduces here, the sampler timeline (rss /
cgroup / gc object counts every 2s, correlated against the eval's own log
lines) localizes WHERE; the gc object count discriminates a python-level
leak (count climbs) from a C/driver-level allocation (rss climbs, count flat).
"""

import gc
import sys
import threading
import time

OUT = "/root/private_data/anime-sr/rgb-eval-logs/probe4-sampler.log"
LOGF = open(OUT, "w")


def _rss_mb() -> int:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except Exception:
        return -1


def _cg_mb() -> int:
    try:
        with open("/sys/fs/cgroup/memory.current") as f:
            return int(f.read()) >> 20
    except Exception:
        return -1


def _loop() -> None:
    n = 0
    while True:
        n += 1
        extra = ""
        if n % 5 == 1:
            try:
                extra = f" gc_objs={len(gc.get_objects())}"
            except Exception:
                pass
        print(f"{time.strftime('%H:%M:%S')} rss={_rss_mb()}MB cg={_cg_mb()}MB{extra}", file=LOGF, flush=True)
        time.sleep(2)


threading.Thread(target=_loop, daemon=True).start()

sys.path.insert(0, "/root/anime-sr-p1formal/src")
sys.argv = ["rgb_eval.py", "--set", "A", "--limit", "10", "--out", "/root/private_data/anime-sr/rgb-eval-out-smoke"]
import rgb_eval  # noqa: E402

rgb_eval.main()
print("[probe4] main() returned", flush=True)
