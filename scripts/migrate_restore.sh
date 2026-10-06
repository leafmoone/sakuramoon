#!/usr/bin/env bash
# SakuraMoon G1 migration restore + stack launch.
# Set PROJECT_ROOT, RUNTIME_ROOT, REPO and RELAY for the destination instance.
# Idempotent: skips anything already present; safe no-op if accidentally run on the old container.
set -euo pipefail
G1="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
R="${RUNTIME_ROOT:?Set RUNTIME_ROOT to the intended restore directory}"
STAGE="${RESTORE_STAGE:-${TMPDIR:-/tmp}/sakuramoon-restore}"
REPO="${REPO:?Set REPO to your snapshot repository}"
RELAY="${RELAY:?Set RELAY to the snapshot directory within REPO}"
log(){ printf '[migrate-restore] %s\n' "$*" >&2; }
die(){ printf '[migrate-restore] ERROR: %s\n' "$*" >&2; exit 1; }
[ "$(id -u)" = 0 ] || die 'run as root'

# Never restore over a running trainer. The caller supplies proxy/token settings.
if pgrep -f '[s]akuramoon.cli.train' >/dev/null; then
  die 'training is running on this host; restore on an idle destination'
fi
if [[ -n "${WORKLOAD_ENV_FILE:-}" ]]; then
  while IFS= read -r -d '' line; do
    case "$line" in http_proxy=*|https_proxy=*|HTTP_PROXY=*|HTTPS_PROXY=*|no_proxy=*|MODELSCOPE_API_TOKEN=*) export "$line" ;; esac
  done < "${WORKLOAD_ENV_FILE}"
fi
if [[ -n "${DTK_ENV_SH:-}" ]]; then source "${DTK_ENV_SH}"; fi
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
  [ -n "$want" ] && [ "$sum" = "$want" ] || { log "$name sha256 mismatch/missing (got ${sum:-none}); refusing the restore"; return 1; }
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
  log "extracting checkpoints -> $R/output_model/g1"
  mkdir -p "$R/output_model/g1"
  tar -xf "$t0" -C "$R/output_model/g1"
fi
if [ ! -d "$R/model" ] || [ -z "$(ls -A "$R/model" 2>/dev/null)" ]; then
  t1=$(fetch g1-model.tar)
  log "extracting model -> $R/model"
  tar -xf "$t1" -C "$R"
fi
if [ ! -x "$R/sakuramoon-dtk-venv/bin/python" ] || [ ! -d "$R/torchinductor-cache" ]; then
  t2=$(fetch g2-venv-inductor.tar)
  log "extracting runtime environment -> $R"
  tar -xf "$t2" -C "$R"
fi

# The restored environment must be compatible with the target hardware/ABI.
VENV_ROOT="${VENV_ROOT:-${R}/sakuramoon-dtk-venv}"
"${VENV_ROOT}/bin/python" -c 'import torch; print("torch", torch.__version__)' || die 'restored Python environment is incompatible; rebuild it on this host'

# Restoring files does not implicitly restart training or publish checkpoints.
if [[ "${START_TRAINING:-0}" == 1 ]]; then
  env PROJECT_ROOT="$G1" RUNTIME_ROOT="$R" \
    CONFIG_NAME="${CONFIG_NAME:-train_g1.toml}" CONFIG_ROOT="$G1/config" \
    VENV_ROOT="$VENV_ROOT" PYTHON_BIN="$VENV_ROOT/bin/python" \
    ACCELERATE_BIN="$VENV_ROOT/bin/accelerate" \
    MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}" \
    bash "$G1/scripts/training_stack.sh" start
else
  log 'restore complete; inspect the runtime and set START_TRAINING=1 to launch explicitly'
fi
