#!/bin/bash
# Full RGB-eval launch, PROCESS-ISOLATED (sakrua10, 08-30 r5.5b).
# OOM forensics (this day, ports 10357/10054/10442): the DTK/HSA host-side
# staging leak scales with per-process forward count and is invisible to the
# GC (probe4: gc object count flat at 252k while cgroup climbed 4 -> 99 GiB);
# the pod was saved 3x by guard_kill.sh (92 GiB): 12:55:45, 13:04:56,
# 13:26:24 (chunk0, 5 pairs: 99.1 GiB at the work end, cgroup dropped to
# 1.3 GB after the kill -> the leak is process-owned).
# Mitigation: one fresh process per PAIR-chunk (A: 3 pairs ~60 GiB worst case,
# B: 2 curation entries ~80 GiB worst case, C: 3 items ~70 GiB), each under
# the 92 GiB guard; a guard-kill costs one retried chunk.  Artifacts on the
# persistent volume; logs under rgb-eval-logs/.
# - No `set -u`: /opt/dtk-26.04/env.sh references $CMAKE_PREFIX_PATH without a
#   default and aborts under -u (would kill the whole run at source time).
set -eo pipefail
cd /root/anime-sr-p1formal/rgb_eval
source /opt/dtk-26.04/env.sh 2>/dev/null
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export OMP_NUM_THREADS=2
export PYTHONPATH=/root/anime-sr-p1formal/src

OUT=/root/private_data/anime-sr/rgb-eval-out
LOG=/root/private_data/anime-sr/rgb-eval-logs
mkdir -p "$LOG"
GUARD_TH=98784247808   # bytes = 92 GiB (guard_kill.sh compares memory.current in BYTES)
# Set A at pair granularity: 128 sids x 5 profiles = 640 pairs, 3 pairs per
# fresh process.  Measured leak in bad HSA state: ~18.6 GiB per pair (chunk0,
# 5 pairs, guard-killed at 99.1 GiB 13:26:24) -> 3 pairs lands at ~60 GiB
# worst case, safely under the 92 GiB guard; a guard-kill costs that chunk's
# pairs, the driver retries once with a fresh process.  B (2 entries) and C
# (3 items) are sized the same way against the per-pair leak.
N_PAIRS=640
A_CHUNKSIZE=3
N_CHUNKS=$(( (N_PAIRS + A_CHUNKSIZE - 1) / A_CHUNKSIZE ))

# run_guarded <tag> <retry 0|1> <rgb_eval args...>: eval + cgroup guard;
# retry once when retry=1.  Logs: full-<tag>.log, full-<tag>-guard.log.
run_guarded() {
  local tag=$1 retry=$2
  shift 2
  local attempt rc EPID GPID
  for attempt in 1 2; do
    /usr/local/bin/python3.11 rgb_eval.py "$@" >> "$LOG/full-${tag}.log" 2>&1 &
    EPID=$!
    bash guard_kill.sh "$EPID" "$GUARD_TH" "$LOG/full-${tag}-guard.log" &
    GPID=$!
    set +e
    wait "$EPID"
    rc=$?
    set -e
    kill "$GPID" 2>/dev/null || true
    wait "$GPID" 2>/dev/null || true
    if [ "$rc" -eq 0 ]; then
      return 0
    fi
    echo "$(date '+%F %T') $tag attempt $attempt rc=$rc (guard log: full-${tag}-guard.log)"
    if [ "$retry" -ne 1 ] || [ "$attempt" -ge 2 ]; then
      return "$rc"
    fi
    sleep 20   # let the cgroup page cache / driver state settle before retry
  done
  return "$rc"
}

# whole-run monitor (15s cadence, ~7h -> 1700 iterations)
nohup bash mem_mon2.sh "$LOG/full-mon.log" 1700 > /dev/null 2>&1 &
echo "$(date '+%F %T') MONITOR pid=$!"

# ---------------- Set A: 640 pairs, one fresh process per 3-pair chunk ----------------
echo "=== $(date '+%F %T') set A: ${N_PAIRS} pairs, ${A_CHUNKSIZE} pairs per process (${N_CHUNKS} chunks) ==="
FAILED_A=""
for i in $(seq 0 $((N_CHUNKS - 1))); do
  if ! run_guarded "A-chunk-$i" 1 --set A --chunk "$i" --chunksize "$A_CHUNKSIZE" --out "$OUT"; then
    FAILED_A="$FAILED_A $i"
    echo "$(date '+%F %T') set A chunk $i FAILED after 2 attempts"
  fi
done
[ -z "$FAILED_A" ] || echo "=== set A failed chunks:$FAILED_A (re-run manually: --set A --chunk <i> --chunksize $A_CHUNKSIZE) ==="

# ---------------- Set B: curation entries, 2 per process (stress chunks hit 4 pairs) ----------------
N_B=$( /usr/local/bin/python3.11 -c "import json;print(len(json.load(open('rgb-eval-b-curation.json'))))" )
NB_CHUNKS=$(( (N_B + 1) / 2 ))
echo "=== $(date '+%F %T') set B: ${N_B} curated entries, 2 per process (${NB_CHUNKS} chunks) ==="
FAILED_B=""
for i in $(seq 0 $((NB_CHUNKS - 1))); do
  if ! run_guarded "B-chunk-$i" 1 --set B --chunk "$i" --chunksize 2 --b-curation rgb-eval-b-curation.json --out "$OUT"; then
    FAILED_B="$FAILED_B $i"
    echo "$(date '+%F %T') set B chunk $i FAILED after 2 attempts"
  fi
done
[ -z "$FAILED_B" ] || echo "=== set B failed chunks:$FAILED_B (re-run manually: --set B --chunk <i> --chunksize 2) ==="

# ---------------- Set C: 60 items, 3 per process ----------------
N_C=$( /usr/local/bin/python3.11 -c "import json;print(len(json.load(open('/root/private_data/anime-sr/data/c-selection.json'))))" )
NC_CHUNKS=$(( (N_C + 2) / 3 ))
echo "=== $(date '+%F %T') set C: ${N_C} items, 3 per process (${NC_CHUNKS} chunks) ==="
FAILED_C=""
for i in $(seq 0 $((NC_CHUNKS - 1))); do
  if ! run_guarded "C-chunk-$i" 1 --set C --chunk "$i" --chunksize 3 --c-selection /root/private_data/anime-sr/data/c-selection.json --out "$OUT"; then
    FAILED_C="$FAILED_C $i"
    echo "$(date '+%F %T') set C chunk $i FAILED after 2 attempts"
  fi
done
[ -z "$FAILED_C" ] || echo "=== set C failed chunks:$FAILED_C (re-run manually: --set C --chunk <i> --chunksize 3) ==="

# ---------------- sheets + union metrics.jsonl (no model, seconds; idempotent) ----------------
SHEETS_ARGS=(--set all --sheets --b-curation rgb-eval-b-curation.json
  --c-selection /root/private_data/anime-sr/data/c-selection.json --out "$OUT")
echo "=== $(date '+%F %T') union pass 1 (A complete) ==="
run_guarded A-sheets 0 "${SHEETS_ARGS[@]}" \
  || echo "A-sheets union FAILED (chunk jsonls remain on disk)"
echo "=== $(date '+%F %T') union pass 2 (B/C complete) ==="
run_guarded A-sheets2 0 "${SHEETS_ARGS[@]}" \
  || echo "final union FAILED (per-set jsonls remain on disk)"
echo "=== $(date '+%F %T') ALL SETS DONE (failed A:$FAILED_A B:$FAILED_B C:$FAILED_C) ==="
RC_FAIL=0
if [ -n "$FAILED_A" ] || [ -n "$FAILED_B" ] || [ -n "$FAILED_C" ]; then RC_FAIL=1; fi
exit "$RC_FAIL"
