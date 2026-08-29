#!/bin/bash
# mem_mon2: attribute cgroup memory growth (file vs anon vs per-process) every 15s.
# Writes to $1 (log path). Runs $2 iterations (default 240 = 1h).
LOG=${1:-/root/private_data/anime-sr/rgb-eval-logs/mon2.log}
N=${2:-240}
i=0
while [ "$i" -lt "$N" ]; do
  i=$((i+1))
  TS=$(date '+%F %T')
  CUR=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)
  STAT=$(grep -E '^(file |anon |shmem |slab )' /sys/fs/cgroup/memory.stat 2>/dev/null | tr '\n' ' ')
  TOP=$(ps aux --sort=-rss 2>/dev/null | awk 'NR>1 {printf "%s(%s)%s ", substr($11,1,26), $6, $2}' | head -c 300)
  # hy-smi cols: HCU(1) Temp(2) AvgPwr(3) Perf(4) PwrCap(5) VRAM%(6) HCU%(7) Dec%(8) Enc%(9)
  HCU=$(/opt/hyhal/bin/hy-smi 2>/dev/null | grep -E '^[0-9]+ ' | awk '{printf "%s:VRAM%s HCU%s Dec%s | ", $1, $6, $7, $8}')
  echo "$TS cur=$CUR | $STAT | top: $TOP | hcu: $HCU" >> "$LOG"
  sleep 15
done
