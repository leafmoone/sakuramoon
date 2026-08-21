#!/usr/bin/env bash
# SakuraMoon G1 migration restore + stack launch.
# Run ON THE NEW INSTANCE as root:  bash /root/private_data/sakuramoon-g1/scripts/migrate_restore.sh
# Idempotent: skips anything already present; safe no-op if accidentally run on the old container.
set -euo pipefail
G1=/root/private_data/sakuramoon-g1
R=/sakuramoon-runtime
STAGE=/root/sakuramoon-mig-restore
REPO=leafmoone/docker_tmp
RELAY=DAT/replay_01
log(){ printf '[migrate-restore] %s\n' "$*"; }
die(){ printf '[migrate-restore] ERROR: %s\n' "$*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die 'run as root'

# 1) identity guard: refuse to 'restore' if this looks like the old (source) container
SNAP=$(grep ' / ' /proc/mounts | grep -oE 'snapshots/[0-9]+' | head -1)
log "containerd snapshot: ${SNAP:-unknown}"
if ps -p 340661 >/dev/null 2>&1; then
  log 'WARNING: PID 340661 (old G1 launcher) is visible -> this is the OLD container; nothing to restore. Aborting to protect the live stack.'
  exit 2
fi

# 2) env: proxy + modelscope token from the g1 workload env file
while IFS= read -r -d '' l; do
  case "$l" in http_proxy=*|https_proxy=*|ftp_proxy=*|no_proxy=*|MODELSCOPE_API_TOKEN=*) export "$l" ;; esac
done < "$G1/.env.training-stack.nul"
command -v ms-hub >/dev/null 2>&1 || die 'ms-hub not found'

# 3) fetch + verify payloads that are still missing
mkdir -p "$STAGE"
have_ckpt(){ ls "$R"/output_model/g1/ckpt_*raw*-update-cadence/COMPLETE 2>/dev/null | grep -q . ; }
fetch(){ # <tarball> : download relay file to $STAGE, verify sha256, print local path
  local name=$1 f
  f=$(find "$STAGE" -name "$name" -type f 2>/dev/null | head -1)
  if [ -z "$f" ]; then
    log "downloading $name from relay"
    ms-hub download --repo-type model "$REPO" --local-dir "$STAGE" "$RELAY/$name" --disable-tqdm || die "download $name failed"
    f=$(find "$STAGE" -name "$name" -type f 2>/dev/null | head -1)
    [ -n "$f" ] || die "downloaded $name not found under $STAGE"
  fi
  local sum want
  sum=$(sha256sum "$f" | cut -d' ' -f1)
  want=$(grep " $name\$" "$STAGE/SHA256SUMS" 2>/dev/null | cut -d' ' -f1)
  [ -n "$want" ] && [ "$sum" = "$want" ] || { log "$name sha256 mismatch/missing (got ${sum:-none}); re-downloading"; rm -f "$f"; fetch "$name"; return; }
  log "$name verified"
  printf '%s\n' "$f"
}
# SHA256SUMS itself (tiny; fetch via find-or-download)
if [ ! -s "$STAGE/SHA256SUMS" ]; then
  ms-hub download --repo-type model "$REPO" --local-dir "$STAGE" "$RELAY/SHA256SUMS" --disable-tqdm || true
  cp -f "$(find "$STAGE" -name SHA256SUMS -type f | head -1)" "$STAGE/SHA256SUMS" 2>/dev/null || true
fi
if ! have_ckpt; then
  t0=$(fetch g0-ckpts.tar)
  log 'extracting checkpoints -> /sakuramoon-runtime/output_model/g1'
  mkdir -p "$R/output_model/g1"
  tar -xf "$t0" -C "$R/output_model/g1"
fi
if [ ! -d "$R/model" ] || [ -z "$(ls -A "$R/model" 2>/dev/null)" ]; then
  t1=$(fetch g1-model.tar)
  log 'extracting model -> /sakuramoon-runtime/model'
  tar -xf "$t1" -C "$R"
fi
if [ ! -x "$R/sakuramoon-dtk-venv/bin/python" ] || [ ! -d "$R/torchinductor-cache" ]; then
  t2=$(fetch g2-venv-inductor.tar)
  log 'extracting venv/inductor-cache/bundle/artifacts -> /sakuramoon-runtime'
  tar -xf "$t2" -C "$R"
fi

# 4) venv chain: /root/private_data/sakuramoon/.venv (NFS) -> /opt/sakuramoon-venv -> $R/sakuramoon-dtk-venv
if [ ! -x /root/private_data/sakuramoon/.venv/bin/accelerate ]; then
  log 'recreating /opt/sakuramoon-venv symlink'
  ln -sfn "$R/sakuramoon-dtk-venv" /opt/sakuramoon-venv
fi
/root/private_data/sakuramoon/.venv/bin/python -c 'import torch; print("torch", torch.__version__, "ok")' >/dev/null || die 'venv import torch failed after restore'

# 5) HCU sanity: expect 2 cards for G1 world_size=2
CARDS=$(ls /dev/dri 2>/dev/null | grep -c '^card' || true)
log "visible DRI cards: ${CARDS:-0}"
[ "${CARDS:-0}" -ge 2 ] || log 'WARNING: fewer than 2 visible cards; G1 world_size=2 may not fit'

# 6) launch the full stack via the official script (G1 overrides)
log 'starting training stack (data service will re-fetch data shards; publisher; train resumes from newest COMPLETE ckpt)'
env PROJECT_ROOT="$G1" \
    CONFIG_NAME=train_g1.toml \
    CONFIG_ROOT="$G1/config" \
    VENV_ROOT=/root/private_data/sakuramoon/.venv \
    PYTHON_BIN=/root/private_data/sakuramoon/.venv/bin/python \
    ACCELERATE_BIN=/root/private_data/sakuramoon/.venv/bin/accelerate \
    REQUIRED_HOST_SUBSTRING=sakrua3 \
    MAIN_PROCESS_PORT=29525 \
    bash /root/private_data/sakuramoon/scripts/training_stack.sh start

echo
echo '[migrate-restore] DONE. Verify over the next ~10-20 min:'
echo "  tail -f /root/sakuramoon-logs/train.log        (expect: [train] TorchInductor cache line, first update > restored ckpt)"
echo "  tail -f /sakuramoon-runtime/artifacts/g1/metrics.jsonl | grep -oE '\"update\": *[0-9]+' | tail -1"
echo "  hy-smi                                          (both HCUs busy)"
