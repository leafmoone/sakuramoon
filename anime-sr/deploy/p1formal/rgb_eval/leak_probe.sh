#!/bin/bash
# leak_probe.sh <chunk> <chunksize> <tag> [ENV1=v1 ENV2=v2 ...]
# One-pair (or N-pair) leak probe on the current HSA state:
# runs rgb_eval --set A --chunk $1 --chunksize $2 with an optional env prefix,
# 112 GiB guard backstop (cgroup ceiling 117.5 GiB), 5s cgroup sampler.
# Launch fire-and-forget: nohup bash leak_probe.sh 0 1 p1pair >/dev/null 2>&1 &
CH=$1; CS=$2; TAG=$3; shift 3
cd /root/anime-sr-p1formal/rgb_eval
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/anime-sr-p1formal/src
LOG=/root/private_data/anime-sr/rgb-eval-logs
OUT=/root/private_data/anime-sr/rgb-eval-out
GUARD=120259084288   # 112 GiB backstop (ceiling 126701535232 = 117.5 GiB)

# shellcheck disable=SC2086
env "$@" nohup /usr/local/bin/python3.11 rgb_eval.py --set A --chunk "$CH" --chunksize "$CS" --out "$OUT" > "$LOG/probe-${TAG}.log" 2>&1 &
EPID=$!
nohup bash guard_kill.sh "$EPID" "$GUARD" "$LOG/probe-${TAG}-guard.log" &
(
  for i in $(seq 1 40); do
    echo "$(date '+%T') cgroup=$(cat /sys/fs/cgroup/memory.current)" >> "$LOG/probe-${TAG}-sampler.log"
    sleep 5
  done
) &
echo "$(date '+%F %T') probe ${TAG}: chunk=${CH} cs=${CS} env=[$*] eval=$EPID" >> "$LOG/probe-launch.log"
echo "PROBE ${TAG} eval=$EPID"
