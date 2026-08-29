#!/bin/bash
# eval-gate-watch.sh — auto-launch the full RGB eval when the HCU host recovers
# to good state (sakrua10, 08-30).
#
# Context: the DTK/HSA host-staging leak is HOST-level (pod 11112 bad from
# t0 on both HCU0 and HCU1; host UUID f9648edc-c69b-11f0-9249-e8611a67637b).
# Pod re-provisioning does NOT clear it; only a host reboot / driver reset
# (platform ops) does.  This watcher re-probes every INTERVAL seconds:
#   good state  (1-pair probe peak <= PEAK_GATE)  -> launch run_full_eval.sh, exit
#   bad state   (guard-killed at 112 GiB / high peak) -> sleep, retry
#
# Lives on the persistent volume (survives pod re-provision); the PROCESS does
# not — re-arm in every fresh pod (single line, after restoring the repo):
#   nohup bash /root/private_data/anime-sr/eval-gate-watch.sh >/dev/null 2>&1 &
#
# Forensics: every probe is logged to rgb-eval-logs/gate-watch.log (peak +
# guard-kill count), sampler/guard detail under probe-<TAG>-*.log.
set -eo pipefail

V=/root/private_data/anime-sr
LOG="$V/rgb-eval-logs"
mkdir -p "$LOG"
PEAK_GATE=21474836480   # bytes = 20 GiB.  Good state: 1-pair run peaks ~6-15 GiB.
                        # Bad state: balloons >=99 GiB (guard kills at 112 GiB).
INTERVAL=1800           # 30 min between probes while in bad state

# Single-instance guard (stale lock files from dead pods are re-acquirable).
exec 9>"$V/.gate-watch.lock"
flock -n 9 || { echo "$(date '+%F %T') gate-watch already running — skipping" >> "$LOG/gate-watch.log"; exit 0; }

cd /root/anime-sr-p1formal/rgb_eval || {
  echo "$(date '+%F %T') FATAL: /root/anime-sr-p1formal missing — restore from $V/anime-sr-p1formal-backup-20260830.tar.gz first" >> "$LOG/gate-watch.log"
  exit 1
}

echo "$(date '+%F %T') gate-watch armed (PEAK_GATE=${PEAK_GATE} B, INTERVAL=${INTERVAL}s)" >> "$LOG/gate-watch.log"

while :; do
  TAG="gate$(date +%H%M%S)"
  # 1-pair Set-A probe with 112 GiB guard + 40x5s cgroup sampler (leak_probe.sh).
  nohup bash leak_probe.sh 0 1 "$TAG" >/dev/null 2>&1 &
  # Wait for the probe to settle: guard-kill recorded, sampler exhausted (~200s),
  # or the eval process gone.
  for _ in $(seq 1 260); do
    sleep 5
    if grep -q SIGKILL "$LOG/probe-${TAG}-guard.log" 2>/dev/null; then break; fi
    if ! pgrep -f "rgb_eval.py" >/dev/null 2>&1; then
      # brief grace so the sampler can finish its last sample
      sleep 6
      break
    fi
  done
  # sampler lines: "HH:MM:SS cgroup=<bytes>" -> parse the number after '='
  PEAK=$(awk -F'=' '{print $2}' "$LOG/probe-${TAG}-sampler.log" 2>/dev/null | sort -n | tail -1 || true)
  KILLED=$(grep -c SIGKILL "$LOG/probe-${TAG}-guard.log" 2>/dev/null || echo 0)
  echo "$(date '+%F %T') probe=${TAG} peak=${PEAK:-0} guard_kill=${KILLED}" >> "$LOG/gate-watch.log"
  if [ -n "${PEAK:-}" ] && [ "${PEAK}" -le "${PEAK_GATE}" ]; then
    echo "$(date '+%F %T') GOOD STATE (peak ${PEAK} <= ${PEAK_GATE}) -> launching full eval" >> "$LOG/gate-watch.log"
    nohup bash run_full_eval.sh >> "$LOG/full-run.log" 2>&1 &
    exit 0
  fi
  sleep "$INTERVAL"
done
