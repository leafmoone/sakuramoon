#!/bin/bash
# chunk0b measurement launcher (r5.5b): ONE clean process, pair-chunk 0
# (3 pairs = sid 1014612 x P0..P2), backstop guard 110 GiB (below the
# 117.5 GiB cgroup ceiling), 5s cgroup sampler -> chunk0b-sampler.log.
#
# Launch fire-and-forget from ssh (NO &-chain on the ssh line — that makes
# `$!` capture a wrapper subshell, not the python, so the guard watches the
# wrong pid and the real python outlives it as a 100+ GB orphan):
#   nohup bash chunk0b_launch.sh >/dev/null 2>&1 &
cd /root/anime-sr-p1formal/rgb_eval
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/anime-sr-p1formal/src
LOG=/root/private_data/anime-sr/rgb-eval-logs
OUT=/root/private_data/anime-sr/rgb-eval-out
GUARD=118111600640   # 110 GiB backstop (cgroup ceiling = 117.5 GiB)

nohup /usr/local/bin/python3.11 rgb_eval.py --set A --chunk 0 --chunksize 3 --out "$OUT" > "$LOG/chunk0b-smoke.log" 2>&1 &
EPID=$!
nohup bash guard_kill.sh "$EPID" "$GUARD" "$LOG/chunk0b-guard.log" &
GPID=$!
# 5s cgroup sampler for ~6 min (12 iters): record the peak, then stop.
(
  for i in $(seq 1 12); do
    echo "$(date '+%T') cgroup=$(cat /sys/fs/cgroup/memory.current)" >> "$LOG/chunk0b-sampler.log"
    sleep 5
  done
) &
echo "$(date '+%F %T') launched eval=$EPID guard=$GPID guard_th=$GUARD" >> "$LOG/chunk0b-launch.log"
echo "LAUNCHED eval=$EPID guard=$GPID"
