#!/bin/bash
# bare3env.sh — bare2 (bf16 matmul + bf16 conv) under the G1 stack's accelerator
# environment (DTK tuning vars that anime-sr eval does NOT set).  Attribution
# test: if this stays flat on a bad-state host, the G1 env is the config fix.
# Launch: nohup bash bare3env.sh >/dev/null 2>&1 &
# Result log: rgb-eval-logs/bare3env-<HHMMSS>.log ; python out: /root/bare-bare3env.out
set -eo
LOG=/root/private_data/anime-sr/rgb-eval-logs/bare3env-$(date +%H%M%S).log
cd /root/anime-sr-p1formal/rgb_eval || exit 1
# G1 stack accelerator env (from live data_service /proc/PID/environ, salt2)
export DTKROOT=/opt/dtk
export HIP_PATH=/opt/dtk/hip
export HIP_KERNEL_EVENT_SYSTENFENCE=1
export HIP_USE_GRAPH_QUEUE_POOL=1
export PYTORCH_MIOPEN_SUGGEST_NDHWC=1
export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/dtk/dcc/gcvm/lib:/opt/dtk/hip/lib:/opt/dtk/llvm/lib:/opt/dtk/lib:/opt/dtk/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:/opt/dtk/dushmem/lib:/opt/dtk/opencl/lib:/opt/dtk/.hyhal/rocm_smi/lib:/usr/local/lib/:/usr/local/lib64/:/opt/mpi/lib:/opt/hwloc/lib:/opt/dtk-26.04/dcc/gcvm/lib:/opt/dtk-26.04/hip/lib:/opt/dtk-26.04/llvm/lib:/opt/dtk-26.04/lib:/opt/dtk-26.04/lib64:/opt/dtk-26.04/dushmem/lib:/opt/dtk-26.04/opencl/lib:/opt/dtk-26.04/.hyhal/rocm_smi/lib
export OMP_NUM_THREADS=2
nohup /usr/local/bin/python3.11 bare_leak2.py > /root/bare-bare3env.out 2>&1 &
EP=$!
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
echo "=== bare3env finished $(date '+%T')" >> "$LOG"
