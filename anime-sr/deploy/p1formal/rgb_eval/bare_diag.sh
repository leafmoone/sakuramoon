#!/bin/bash
# bare_diag.sh — run a code-free torch script (default bare_leak.py) with a
# 112 GiB cgroup guard + 8 s VMA sampler, same instrumentation as
# diag_leak.sh.  Launch:  nohup bash bare_diag.sh [script.py] >/dev/null 2>&1 &
# Result log: rgb-eval-logs/bare<name>-<HHMMSS>.log ; python out: /root/bare-<name>.out
set -eo
SCRIPT=${1:-bare_leak.py}
NAME=${SCRIPT%.py}
LOG=/root/private_data/anime-sr/rgb-eval-logs/bare${NAME}-$(date +%H%M%S).log
cd /root/anime-sr-p1formal/rgb_eval || exit 1
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export OMP_NUM_THREADS=2
nohup /usr/local/bin/python3.11 "$SCRIPT" > "/root/bare-${NAME}.out" 2>&1 &
EP=$!
# 112 GiB guard (same threshold as leak_probe.sh)
(
  for _ in $(seq 1 200); do
    C=$(cat /sys/fs/cgroup/memory.current)
    if [ "$C" -gt 120259084288 ]; then
      echo "$(date '+%F %T') GUARD: cgroup $C B > 112 GiB; SIGKILL $EP" >> "$LOG"
      kill -9 "$EP" 2>/dev/null || true
      break
    fi
    sleep 5
  done
) &
for i in $(seq 1 60); do
  kill -0 "$EP" 2>/dev/null || break
  {
    echo "=== sample $i $(date '+%T') pid=$EP"
    grep -E '^(VmSize|VmRSS)' /proc/$EP/status 2>/dev/null || true
    pmap -x "$EP" 2>/dev/null | awk 'NR>1 && $3+0>262144 {printf "%8.1f GB  %s  %s\n", $3/1048576, $2, $5}' | sort -rn | head -10 || true
    grep -E '^(anon|file) ' /sys/fs/cgroup/memory.stat
  } >> "$LOG" 2>&1
  sleep 8
done
echo "=== barediag finished $(date '+%T')" >> "$LOG"
