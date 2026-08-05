#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

# This script deliberately targets one fixed repository only.
REPO_ID="${REPO_ID:-leafmoone/docker_tmp}"
REPO_TYPE="model"
IMAGE_TAG="${IMAGE_TAG:-sakuramoon:hcu-dtk-26.04}"
BUNDLE_ROOT="${BUNDLE_ROOT:-/sakuramoon-runtime/docker-package}"
LATEST_DIR="${BUNDLE_ROOT}/latest"
PART_SIZE="${PART_SIZE:-4G}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-1800}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-4}"
TRY_DELETE_REPO="${TRY_DELETE_REPO:-0}"
LOG_FILE="${LOG_FILE:-${BUNDLE_ROOT}/publish.log}"

PROJECT_ROOT="${PROJECT_ROOT:-/public/home/acfb8k41va/sakuramoon}"
MS_HUB_BIN="${MS_HUB_BIN:-${PROJECT_ROOT}/.venv/bin/ms-hub}"

mkdir -p "${BUNDLE_ROOT}"
exec 9>"${BUNDLE_ROOT}/publish.lock"
if ! flock -n 9; then
  echo "another publish_docker_tmp.sh instance is already running" >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T%z')" "$*" | tee -a "${LOG_FILE}"
}

load_environment() {
  if [[ -f /root/private_data/.ai_user_info/ai_proxy ]]; then
    # shellcheck disable=SC1091
    source /root/private_data/.ai_user_info/ai_proxy
  fi
  if [[ -f /etc/profile.d/model-tokens.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/model-tokens.sh
  fi
  : "${MODELSCOPE_API_TOKEN:?MODELSCOPE_API_TOKEN is not set}"
  [[ -x "${MS_HUB_BIN}" ]] || {
    log "missing ModelScope Hub CLI: ${MS_HUB_BIN}"
    return 1
  }
}

ensure_private_repo() {
  "${MS_HUB_BIN}" --token "${MODELSCOPE_API_TOKEN}" create "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" --visibility private --exist-ok \
    >>"${LOG_FILE}" 2>&1
}

prepare_bundle() {
  command -v docker >/dev/null 2>&1 || {
    log "docker is unavailable; prepare must run on a Docker-capable host"
    return 1
  }
  command -v split >/dev/null 2>&1 || {
    log "split is unavailable"
    return 1
  }
  docker image inspect "${IMAGE_TAG}" >/dev/null 2>&1 || {
    log "Docker image not found: ${IMAGE_TAG}"
    return 1
  }

  local tmp_dir compressed_format
  tmp_dir="${BUNDLE_ROOT}/.latest.tmp.$$.${RANDOM}"
  rm -rf "${tmp_dir}"
  mkdir -p "${tmp_dir}"

  log "exporting ${IMAGE_TAG} and splitting into ${PART_SIZE} parts"
  if command -v zstd >/dev/null 2>&1; then
    compressed_format="tar.zst"
    docker save "${IMAGE_TAG}" \
      | zstd -T0 -3 -c \
      | split --bytes="${PART_SIZE}" --numeric-suffixes=0 --suffix-length=4 - \
          "${tmp_dir}/sakuramoon-image.tar.zst.part-"
  else
    compressed_format="tar"
    log "zstd unavailable; exporting an uncompressed tar"
    docker save "${IMAGE_TAG}" \
      | split --bytes="${PART_SIZE}" --numeric-suffixes=0 --suffix-length=4 - \
          "${tmp_dir}/sakuramoon-image.tar.part-"
  fi

  local parts
  parts="$(find "${tmp_dir}" -maxdepth 1 -type f -name '*.part-*' -printf '%f\n' | sort)"
  [[ -n "${parts}" ]] || {
    log "no image parts were produced"
    rm -rf "${tmp_dir}"
    return 1
  }
  (
    cd "${tmp_dir}"
    printf '%s\n' "${parts}" | xargs -r sha256sum >"SHA256SUMS"
  )
  {
    printf 'image_tag=%s\n' "${IMAGE_TAG}"
    printf 'created_at=%s\n' "$(date --iso-8601=seconds)"
    printf 'format=%s\n' "${compressed_format}"
    printf 'part_size=%s\n' "${PART_SIZE}"
    printf 'parts:\n%s\n' "${parts}"
  } >"${tmp_dir}/manifest.txt"
  printf 'ready\n' >"${tmp_dir}/READY"

  local old_dir
  old_dir="${BUNDLE_ROOT}/.latest.old.$$.${RANDOM}"
  rm -rf "${old_dir}"
  if [[ -e "${LATEST_DIR}" ]]; then
    mv "${LATEST_DIR}" "${old_dir}"
  fi
  mv "${tmp_dir}" "${LATEST_DIR}"
  rm -rf "${old_dir}"
  log "prepared ${LATEST_DIR}"
}

publish_latest() {
  [[ -f "${LATEST_DIR}/READY" ]] || {
    log "no READY bundle at ${LATEST_DIR}"
    return 1
  }

  # ModelScope currently rejects repository deletion for API-token auth.
  # Keep this opt-in so a future permission change cannot unexpectedly erase
  # the repository during a normal upload loop.
  if [[ "${TRY_DELETE_REPO}" == "1" ]]; then
    if "${MS_HUB_BIN}" --token "${MODELSCOPE_API_TOKEN}" delete "${REPO_ID}" \
      --repo-type "${REPO_TYPE}" --yes >>"${LOG_FILE}" 2>&1; then
      log "deleted ${REPO_ID}; recreating it"
    else
      log "repository deletion is unavailable with the current token; continuing with overwrite upload"
    fi
  fi

  ensure_private_repo
  log "uploading latest bundle to ${REPO_ID}"
  "${MS_HUB_BIN}" --token "${MODELSCOPE_API_TOKEN}" upload "${REPO_ID}" \
    "${LATEST_DIR}" --repo-type "${REPO_TYPE}" --no-cache \
    --max-workers "${UPLOAD_WORKERS}" \
    --commit-message "update docker snapshot $(date '+%F %T%z')" \
    >>"${LOG_FILE}" 2>&1
  log "upload completed"
}

run_once() {
  prepare_bundle
  publish_latest
}

load_environment
case "${1:-loop}" in
  prepare)
    prepare_bundle
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
    printf 'usage: %s {prepare|publish-once|once|loop}\n' "$0" >&2
    exit 2
    ;;
esac
