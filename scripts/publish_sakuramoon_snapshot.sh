#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

# Portable snapshot publisher for the current HCU/DAS environment.
# It does not require Docker. Secrets are intentionally not included.
REPO_ID="${REPO_ID:?Set REPO_ID to the intended ModelScope destination}"
REPO_TYPE="model"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-${PROJECT_ROOT}}"
PYTHON_ENV="${PYTHON_ENV:-${PROJECT_ROOT}/.venv}"
SNAPSHOT_ROOT="${SNAPSHOT_ROOT:-${TMPDIR:-/tmp}/sakuramoon-bundles}"
LATEST_DIR="${SNAPSHOT_ROOT}/latest"
PART_SIZE="${PART_SIZE:-4G}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
ZSTD_THREADS="${ZSTD_THREADS:-8}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-4}"
MS_HUB_BIN="${MS_HUB_BIN:-${PROJECT_ROOT}/.venv/bin/ms-hub}"
LOG_FILE="${LOG_FILE:-${SNAPSHOT_ROOT}/snapshot-publisher.log}"

mkdir -p "${SNAPSHOT_ROOT}"
exec 9>"${SNAPSHOT_ROOT}/snapshot-publisher.lock"
if ! flock -n 9; then
  echo "another snapshot publisher is already running" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T%z')" "$*" | tee -a "${LOG_FILE}"
}

load_environment() {
  : "${MODELSCOPE_API_TOKEN:?MODELSCOPE_API_TOKEN is not set}"
  [[ -x "${MS_HUB_BIN}" ]] || {
    log "missing ModelScope Hub CLI: ${MS_HUB_BIN}"
    return 1
  }
  command -v tar >/dev/null 2>&1 || { log 'tar is unavailable'; return 1; }
  command -v zstd >/dev/null 2>&1 || { log 'zstd is unavailable'; return 1; }
  command -v split >/dev/null 2>&1 || { log 'split is unavailable'; return 1; }
  command -v sha256sum >/dev/null 2>&1 || { log 'sha256sum is unavailable'; return 1; }
}

check_sources() {
  local path
  for path in "${PROJECT_ROOT}" "${RUNTIME_ROOT}" "${PYTHON_ENV}"; do
    [[ -e "${path}" ]] || {
      log "snapshot source is missing: ${path}"
      return 1
    }
  done
}

ensure_private_repo() {
  "${MS_HUB_BIN}" --token "${MODELSCOPE_API_TOKEN}" create "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" --visibility private --exist-ok \
    >>"${LOG_FILE}" 2>&1
}

build_snapshot() {
  check_sources
  local work_dir archive part_dir old_dir parts
  work_dir="$(mktemp -d "${SNAPSHOT_ROOT}/.snapshot.XXXXXX")"
  archive="${work_dir}/sakuramoon-snapshot.tar.zst"
  part_dir="${work_dir}/latest"
  mkdir -p "${part_dir}"

  log "creating snapshot; runtime size may exceed 200 GiB"
  local -a sources=()
  local source included covered
  for source in "${PROJECT_ROOT}" "${RUNTIME_ROOT}" "${PYTHON_ENV}"; do
    source="$(realpath -- "${source}")"
    [[ "${source}" != / ]] || { log 'refusing to archive the filesystem root'; return 1; }
    source="${source#/}"
    covered=0
    local -a retained=()
    for included in "${sources[@]}"; do
      if [[ "${source}" == "${included}" || "${source}" == "${included}/"* ]]; then
        covered=1
      fi
      if [[ "${included}" != "${source}/"* ]]; then
        retained+=("${included}")
      fi
    done
    if [[ "${covered}" == 0 ]]; then
      sources=("${retained[@]}" "${source}")
    fi
  done
  tar --ignore-failed-read --warning=no-file-changed \
    --exclude='.git' --exclude='.ssh' --exclude='.codex' \
    --exclude='.env' --exclude='.env.*' --exclude='auth.json' \
    --exclude='*.pem' --exclude='*.key' --exclude='id_rsa*' --exclude='id_ed25519*' \
    --exclude='core' --exclude='core.*' --exclude='*.log' \
    --exclude='*.partial' --exclude='*.range-*' --exclude='*.tmp' \
    --exclude='docker-data' --exclude='docker-exec' --exclude='docker-package' \
    --exclude="${SNAPSHOT_ROOT#/}" \
    --use-compress-program="zstd -T${ZSTD_THREADS} -3" \
    -cpf "${archive}" -C / "${sources[@]}"

  log "splitting snapshot into ${PART_SIZE} parts"
  split --bytes="${PART_SIZE}" --numeric-suffixes=0 --suffix-length=4 \
    "${archive}" "${part_dir}/sakuramoon-snapshot.tar.zst.part-"
  rm -f "${archive}"

  parts="$(find "${part_dir}" -maxdepth 1 -type f -name '*.part-*' -printf '%f\n' | sort)"
  [[ -n "${parts}" ]] || {
    log 'no snapshot parts were produced'
    rm -rf "${work_dir}"
    return 1
  }
  (
    cd "${part_dir}"
    printf '%s\n' "${parts}" | xargs -r sha256sum > SHA256SUMS
  )
  {
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'project_root=%s\n' "${PROJECT_ROOT}"
    printf 'runtime_root=%s\n' "${RUNTIME_ROOT}"
    printf 'python_env=%s\n' "${PYTHON_ENV}"
    printf 'compression=zstd-3\n'
    printf 'part_size=%s\n' "${PART_SIZE}"
    printf 'parts:\n%s\n' "${parts}"
    printf 'excluded_secrets=.env*,.ssh,.codex,auth.json,private-key-patterns\n'
  } >"${part_dir}/manifest.txt"
  printf 'ready\n' >"${part_dir}/READY"

  old_dir="${SNAPSHOT_ROOT}/.latest.old.$$.${RANDOM}"
  rm -rf "${old_dir}"
  if [[ -e "${LATEST_DIR}" ]]; then
    mv "${LATEST_DIR}" "${old_dir}"
  fi
  mv "${part_dir}" "${LATEST_DIR}"
  rm -rf "${old_dir}" "${work_dir}"
  log "snapshot ready at ${LATEST_DIR}"
}

publish_latest() {
  [[ -f "${LATEST_DIR}/READY" ]] || {
    log "no READY snapshot at ${LATEST_DIR}"
    return 1
  }
  ensure_private_repo
  log "uploading snapshot to ${REPO_ID}"
  "${MS_HUB_BIN}" --token "${MODELSCOPE_API_TOKEN}" upload "${REPO_ID}" \
    "${LATEST_DIR}" --repo-type "${REPO_TYPE}" --no-cache \
    --max-workers "${UPLOAD_WORKERS}" \
    --commit-message "sakuramoon snapshot $(date '+%F %T%z')" \
    >>"${LOG_FILE}" 2>&1
  log 'upload completed'
}

run_once() {
  build_snapshot
  publish_latest
}

load_environment
case "${1:-loop}" in
  snapshot)
    build_snapshot
    ;;
  publish-once)
    publish_latest
    ;;
  once)
    run_once
    ;;
  loop)
    trap 'log "stopping"; exit 0' INT TERM
    while true; do
      if ! run_once; then
        log "cycle failed; retrying after ${INTERVAL_SECONDS}s"
      fi
      sleep "${INTERVAL_SECONDS}"
    done
    ;;
  *)
    printf 'usage: %s {snapshot|publish-once|once|loop}\n' "$0" >&2
    exit 2
    ;;
esac
