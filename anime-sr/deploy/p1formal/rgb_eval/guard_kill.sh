#!/bin/bash
# guard_kill.sh <eval_pid> <threshold_bytes> <logfile>
# 08-30 OOM mitigation: watch cgroup memory.current (cgroup v2, BYTES);
# SIGKILL the eval pid if it crosses the threshold (default 92 GiB of the
# 118 GiB cap).  Protects the pod from another cgroup OOM re-provision;
# the eval's own logs stay on the volume for forensics.  Exits when the
# eval pid dies (normally or by guard).
# NB 08-30 r5.1 bug: threshold was compared as MB against a BYTES counter,
# so the guard SIGKILLed every eval at startup (1.2 GB > 94208 "MB").
EPID=$1
TH=${2:-98784247808}   # bytes; 92 GiB (1073741824 * 92)
LOG=$3
while kill -0 "$EPID" 2>/dev/null; do
  CUR=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
  if [ "$CUR" -gt "$TH" ]; then
    echo "$(date '+%F %T') GUARD: cgroup ${CUR} B > ${TH} B (92 GiB); SIGKILL ${EPID}" >> "$LOG"
    kill -9 "$EPID" 2>/dev/null
    break
  fi
  sleep 5
done
echo "$(date '+%F %T') guard: eval pid ${EPID} gone" >> "$LOG"
