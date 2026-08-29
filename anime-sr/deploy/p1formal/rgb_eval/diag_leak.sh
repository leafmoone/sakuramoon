#!/bin/bash
# diag_leak.sh (v2) — attribute the ~100 GiB anon balloon to its owner:
# driver-side (huge anonymous mmap / HSA staging) vs code-side (glibc [heap] /
# Python).  Launches a 1-pair Set-A probe (bad-state host: guard-killed ~112 GiB)
# and samples the eval process every 8 s: VmSize/VmRSS, top-10 largest VMAs
# (>256 MB) with perms+pathname, and cgroup anon/file split.
# v2 fixes: wait for the eval pid before first sample (pgrep race); no
# pipefail (head SIGPIPE kills the sampler under set -eo pipefail).
set -eo
LOG=/root/private_data/anime-sr/rgb-eval-logs/diag-$(date +%H%M%S).log
cd /root/anime-sr-p1formal/rgb_eval || exit 1
nohup bash leak_probe.sh 0 1 diag >/dev/null 2>&1 &
P=""
for _ in $(seq 1 30); do
  P=$(pgrep -f rgb_eval.py | head -1)
  [ -n "$P" ] && break
  sleep 5
done
if [ -z "$P" ]; then
  echo "FATAL: eval process never appeared" >> "$LOG"
  exit 1
fi
for i in $(seq 1 60); do
  kill -0 "$P" 2>/dev/null || break
  {
    echo "=== sample $i $(date '+%T') pid=$P"
    grep -E '^(VmSize|VmRSS)' /proc/$P/status 2>/dev/null || true
    # largest VMAs >256MB via pmap (portable): size GB, perms, pathname (blank = anonymous)
    pmap -x "$P" 2>/dev/null | awk 'NR>1 && $3+0>262144 {printf "%8.1f GB  %s  %s\n", $3/1048576, $2, $5}' | sort -rn | head -10 || true
    grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat
  } >> "$LOG" 2>&1
  sleep 8
done
echo "=== diag finished $(date '+%T')" >> "$LOG"
