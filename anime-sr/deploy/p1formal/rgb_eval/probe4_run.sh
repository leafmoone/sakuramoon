#!/bin/bash
# probe4 launcher (08-30 r5.4): real smoke command (rgb_eval --set A --limit 10)
# with the 2s host-RSS/cgroup/GC sampler (mem_probe4.py) + cgroup guard.
# - No `set -u`: /opt/dtk-26.04/env.sh aborts under -u (unbound CMAKE_PREFIX_PATH).
# - Eval stdout/stderr -> probe4-eval.log; sampler -> probe4-sampler.log;
#   guard -> probe4-guard.log (all on the persistent volume).
set -o pipefail
source /opt/dtk-26.04/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export OMP_NUM_THREADS=2 PYTHONPATH=/root/anime-sr-p1formal/src
cd /root/anime-sr-p1formal/rgb_eval
LOG=/root/private_data/anime-sr/rgb-eval-logs
/usr/local/bin/python3.11 mem_probe4.py >> "$LOG/probe4-eval.log" 2>&1 &
EPID=$!
bash guard_kill.sh "$EPID" 98784247808 "$LOG/probe4-guard.log" &
wait "$EPID"
rc=$?
echo "PROBE4 rc=$rc" >> "$LOG/probe4-eval.log"
