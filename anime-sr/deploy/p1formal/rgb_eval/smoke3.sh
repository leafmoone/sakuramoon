#!/bin/bash
# smoke3: Set A 10-pair smoke on the 08-30 r5 code (b1 + chunked + guard).
# Verifies: (a) no HCU OOM event (b5 batched attempt removed), (b) bounded
# host RSS (cgroup stays far below the 118 GiB cap; guard @ 92 GiB as a
# backstop), (c) both groups + report + rgb-eval-out-smoke/A artifacts.
# NB: no `set -u`: /opt/dtk-26.04/env.sh references $CMAKE_PREFIX_PATH without a
# default and aborts under -u (killed the first smoke3 launch, 08-30).
set -o pipefail
cd /root/anime-sr-p1formal/rgb_eval
source /opt/dtk-26.04/env.sh
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/anime-sr-p1formal/src
LOG=/root/private_data/anime-sr/rgb-eval-logs
/usr/local/bin/python3.11 rgb_eval.py --set A --limit 10 --out /root/private_data/anime-sr/rgb-eval-out-smoke >> "$LOG/smoke3.log" 2>&1 &
EPID=$!
bash guard_kill.sh "$EPID" 98784247808 "$LOG/smoke3-guard.log" &
GPID=$!
set +e
wait "$EPID"
rc=$?
set -e
kill "$GPID" 2>/dev/null || true
wait "$GPID" 2>/dev/null || true
echo "SMOKE3 rc=$rc"
exit "$rc"
