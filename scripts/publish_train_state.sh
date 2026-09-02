#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

# Mirror the stable contents of output_model/s0 at the repository root.  Raw
# checkpoints keep their on-disk names (for example
# ckpt_2000_raw-2000-update-cadence); incomplete atomic .ckpt_*.tmp directories
# are never staged or published.
REPO_ID="${REPO_ID:-leafmoone/sm_train_state}"
REPO_TYPE="model"
REPO_PATH="${REPO_PATH:-s0}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/private_data/sakuramoon/output_model/s0}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/private_data/sakuramoon}"
STATE_ROOT="${STATE_ROOT:-/root/private_data/.sm-train-state-publisher}"
INTERVAL_SECONDS="${INTERVAL_SECONDS:-600}"
UPLOAD_WORKERS="${UPLOAD_WORKERS:-4}"
MS_HUB_BIN="${MS_HUB_BIN:-${PROJECT_ROOT}/.venv/bin/ms-hub}"
LOG_FILE="${LOG_FILE:-${STATE_ROOT}/publisher.log}"
LAST_PUBLISHED="${LAST_PUBLISHED:-${STATE_ROOT}/last-published-s0.txt}"

if [[ ! "${INTERVAL_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'INTERVAL_SECONDS must be a positive integer\n' >&2
  exit 2
fi

mkdir -p "${STATE_ROOT}"
exec 9>"${STATE_ROOT}/publisher.lock"
# 2026-08-30: bounded retry instead of one-shot: a stop/start race (previous instance's
# SIGTERM trap not yet reaped) can transiently hold the lock right after a stack restart.
_lock_ok=0
for _i in $(seq 1 30); do
  if flock -n 9; then _lock_ok=1; break; fi
  sleep 1
done
if [[ "${_lock_ok}" != "1" ]]; then
  printf 'another train-state publisher is already running\n' >&2
  exit 2
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T%z')" "$*" | tee -a "${LOG_FILE}"
}

load_environment() {
  # Vendor/profile scripts are not guaranteed to be nounset-clean NOR
  # parse-clean: the SCNet platform ai_proxy file starts with human-readable
  # header lines (用户：... / 设备：... / IP端口：...) that fail as commands
  # and, under set -e, abort the whole publisher (observed on salt10,
  # exit 127). A failed source must be non-fatal: the proxy variables also
  # arrive via the training stack's workload env.
  set +u
  # 2026-08-30 fix: only fall back to ai_proxy when no proxy is already configured.
  # ai_proxy went stale (pinned dead pool 10.13.17.166) and silently broke every
  # hub upload while the stack-injected proxy (10.16.1.51) was alive.
  if [[ -f /root/private_data/.ai_user_info/ai_proxy ]] && [[ -z "${http_proxy:-}${HTTP_PROXY:-}" ]]; then
    # shellcheck disable=SC1091
    source /root/private_data/.ai_user_info/ai_proxy 2>/dev/null \
      || log "WARNING: ai_proxy source failed (continuing; proxy comes from the workload env)"
  fi
  if [[ -f /opt/dtk-26.04/env.sh ]]; then
    # shellcheck disable=SC1091
    source /opt/dtk-26.04/env.sh >/dev/null 2>&1 \
      || log "WARNING: DTK env source failed (continuing)"
  fi
  if [[ -f /etc/profile.d/model-tokens.sh ]]; then
    # shellcheck disable=SC1091
    source /etc/profile.d/model-tokens.sh 2>/dev/null \
      || log "WARNING: model-tokens source failed (continuing)"
  fi
  set -u
  : "${MODELSCOPE_API_TOKEN:?MODELSCOPE_API_TOKEN is not set}"
  [[ -d "${SOURCE_ROOT}" ]] || {
    log "source root does not exist: ${SOURCE_ROOT}"
    return 1
  }
  [[ -x "${MS_HUB_BIN}" ]] || {
    log "ModelScope Hub CLI is unavailable: ${MS_HUB_BIN}"
    return 1
  }
  command -v sha256sum >/dev/null 2>&1 || {
    log 'sha256sum is unavailable'
    return 1
  }
}

remove_workdir() {
  local workdir="$1"
  case "${workdir}" in
    "${STATE_ROOT}"/.candidate.*|"${STATE_ROOT}"/.verify.*)
      rm -rf -- "${workdir}"
      ;;
    *)
      log "refusing to remove unexpected work path: ${workdir}"
      return 1
      ;;
  esac
}

stage_source_tree() {
  local candidate entry name complete_count=0
  candidate="$(mktemp -d "${STATE_ROOT}/.candidate.XXXXXX")"
  shopt -s dotglob nullglob
  for entry in "${SOURCE_ROOT}"/*; do
    name="$(basename "${entry}")"
    case "${name}" in
      .ckpt_*.tmp|.ms_upload_cache)
        continue
        ;;
      ckpt_*)
        if [[ ! -d "${entry}" ]] \
          || [[ ! -f "${entry}/COMPLETE" ]] \
          || ! grep -qx 'complete' "${entry}/COMPLETE" \
          || [[ ! -f "${entry}/manifest.json" ]]; then
          continue
        fi
        complete_count=$((complete_count + 1))
        ;;
    esac
    if ! cp -al -- "${entry}" "${candidate}/"; then
      remove_workdir "${candidate}"
      return 1
    fi
  done
  shopt -u dotglob nullglob
  if (( complete_count == 0 )); then
    remove_workdir "${candidate}"
    return 1
  fi
  printf '%s\n' "${candidate}"
}

tree_identity() {
  local candidate="$1"
  (
    cd "${candidate}"
    while IFS= read -r -d '' path; do
      case "${path}" in
        ./ckpt_*/*)
          case "${path}" in
            */COMPLETE|*/manifest.json)
              sha256sum "${path}"
              ;;
          esac
          ;;
        *)
          sha256sum "${path}"
          ;;
      esac
    done < <(find . -type f ! -path './.ms_upload_cache/*' -print0 | sort -z)
  ) | sha256sum | cut -d' ' -f1
}

ensure_private_repo() {
  "${MS_HUB_BIN}" create "${REPO_ID}" \
    --repo-type "${REPO_TYPE}" --visibility private --exist-ok \
    >>"${LOG_FILE}" 2>&1 9>&-
}

verify_remote_checkpoints() {
  local candidate="$1" verify_dir name expected actual
  local -a remote_files=()
  while IFS= read -r name; do
    remote_files+=("${REPO_PATH}/${name}/COMPLETE" "${REPO_PATH}/${name}/manifest.json")
  done < <(find "${candidate}" -mindepth 1 -maxdepth 1 -type d \
    -name 'ckpt_*' -printf '%f\n' | sort -V)
  ((${#remote_files[@]} > 0)) || return 1

  verify_dir="$(mktemp -d "${STATE_ROOT}/.verify.XXXXXX")"
  if ! "${MS_HUB_BIN}" download \
    --repo-type "${REPO_TYPE}" --local-dir "${verify_dir}" --force \
    "${REPO_ID}" "${remote_files[@]}" >>"${LOG_FILE}" 2>&1 9>&-; then
    remove_workdir "${verify_dir}"
    return 1
  fi
  while IFS= read -r name; do
    if [[ ! -f "${verify_dir}/${REPO_PATH}/${name}/COMPLETE" ]] \
      || ! grep -qx 'complete' "${verify_dir}/${REPO_PATH}/${name}/COMPLETE" \
      || [[ ! -f "${verify_dir}/${REPO_PATH}/${name}/manifest.json" ]]; then
      remove_workdir "${verify_dir}"
      return 1
    fi
    expected="$(sha256sum "${candidate}/${name}/manifest.json" | cut -d' ' -f1)"
    actual="$(sha256sum "${verify_dir}/${REPO_PATH}/${name}/manifest.json" | cut -d' ' -f1)"
    if [[ "${actual}" != "${expected}" ]]; then
      remove_workdir "${verify_dir}"
      return 1
    fi
  done < <(find "${candidate}" -mindepth 1 -maxdepth 1 -type d \
    -name 'ckpt_*' -printf '%f\n' | sort -V)
  remove_workdir "${verify_dir}"
}

publish_if_changed() {
  local candidate identity previous='' marker_tmp checkpoint_count upload_log
  candidate="$(stage_source_tree)" || {
    log "no complete checkpoint tree under ${SOURCE_ROOT}"
    return 0
  }
  identity="$(tree_identity "${candidate}")"
  if [[ -f "${LAST_PUBLISHED}" ]]; then
    IFS= read -r previous <"${LAST_PUBLISHED}" || true
  fi
  if [[ "${identity}" == "${previous}" ]]; then
    checkpoint_count="$(find "${candidate}" -mindepth 1 -maxdepth 1 \
      -type d -name 'ckpt_*' | wc -l)"
    remove_workdir "${candidate}"
    log "unchanged: ${checkpoint_count} checkpoint directories"
    return 0
  fi

  if ! ensure_private_repo; then
    remove_workdir "${candidate}"
    log "failed to create or access ${REPO_ID}"
    return 1
  fi
  checkpoint_count="$(find "${candidate}" -mindepth 1 -maxdepth 1 \
    -type d -name 'ckpt_*' | wc -l)"
  log "mirroring ${checkpoint_count} checkpoint directories from ${SOURCE_ROOT} to ${REPO_ID}/${REPO_PATH}"
  upload_log="$(mktemp "${STATE_ROOT}/.upload.XXXXXX")"
  if ! "${MS_HUB_BIN}" upload "${REPO_ID}" "${candidate}" "${REPO_PATH}" \
    --repo-type "${REPO_TYPE}" --use-cache \
    --max-workers "${UPLOAD_WORKERS}" --disable-tqdm \
    --exclude '.ms_upload_cache/**' \
    --commit-message "training state tree ${identity:0:12}" \
    >"${upload_log}" 2>&1 9>&-; then
    cat "${upload_log}" >>"${LOG_FILE}"
    rm -f -- "${upload_log}"
    remove_workdir "${candidate}"
    log "tree upload failed"
    return 1
  fi
  cat "${upload_log}" >>"${LOG_FILE}"
  if grep -Eiq \
    '(^|[[:space:]|])ERROR([[:space:]|:]|$)|commit failed|permanent failure|commit rejected|Failed[[:space:]]*:[[:space:]]*[1-9][0-9]*' \
    "${upload_log}"; then
    rm -f -- "${upload_log}"
    remove_workdir "${candidate}"
    log "tree upload reported a failure despite a zero exit status"
    return 1
  fi
  rm -f -- "${upload_log}"
  if ! verify_remote_checkpoints "${candidate}"; then
    remove_workdir "${candidate}"
    log "remote checkpoint verification failed"
    return 1
  fi

  marker_tmp="${LAST_PUBLISHED}.tmp.$$"
  printf '%s\n' "${identity}" >"${marker_tmp}"
  mv -f "${marker_tmp}" "${LAST_PUBLISHED}"
  remove_workdir "${candidate}"
  log "tree upload complete: ${checkpoint_count} checkpoint directories"
}

run_loop() {
  local started elapsed delay
  trap 'log "stopping"; exit 0' INT TERM
  while true; do
    started="$(date +%s)"
    if ! publish_if_changed; then
      log "cycle failed; the source tree will be retried"
    fi
    elapsed=$(( $(date +%s) - started ))
    delay=$(( INTERVAL_SECONDS - elapsed ))
    (( delay > 0 )) || delay=1
    sleep "${delay}"
  done
}

load_environment
case "${1:-loop}" in
  once)
    publish_if_changed
    ;;
  loop)
    run_loop
    ;;
  *)
    printf 'usage: %s {once|loop}\n' "$0" >&2
    exit 2
    ;;
esac
