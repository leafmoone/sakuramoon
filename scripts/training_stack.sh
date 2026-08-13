#!/usr/bin/env bash

set -Eeuo pipefail
umask 077

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/.." && pwd -P)}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/sakuramoon-runtime}"
LOG_ROOT="${LOG_ROOT:-/root/sakuramoon-logs}"
RUN_ROOT="${RUN_ROOT:-/run/sakuramoon}"
CONFIG_NAME="${CONFIG_NAME:-train_s0.toml}"
CONFIG_ROOT="${CONFIG_ROOT:-${PROJECT_ROOT}/config}"
VENV_ROOT="${VENV_ROOT:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${PYTHON_BIN:-${VENV_ROOT}/bin/python}"
ACCELERATE_BIN="${ACCELERATE_BIN:-${VENV_ROOT}/bin/accelerate}"
MANAGEMENT_PYTHON="${MANAGEMENT_PYTHON:-${PYTHON_BIN}}"
WORKLOAD_ENV_FILE="${WORKLOAD_ENV_FILE:-${PROJECT_ROOT}/.env.training-stack.nul}"
PUBLISH_STATE_ROOT="${PUBLISH_STATE_ROOT:-${RUNTIME_ROOT}/.sm-train-state-publisher}"
PUBLISH_LAST_PUBLISHED="${PUBLISH_LAST_PUBLISHED:-/root/private_data/.sm-train-state-publisher/last-published-s0.txt}"
REQUIRED_HOST_SUBSTRING="${REQUIRED_HOST_SUBSTRING:-leaf6}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29500}"
START_TIMEOUT_SECONDS="${START_TIMEOUT_SECONDS:-180}"
STOP_TIMEOUT_SECONDS="${STOP_TIMEOUT_SECONDS:-30}"

DATA_LOG="${LOG_ROOT}/data-service.log"
TRAIN_LOG="${LOG_ROOT}/train.log"
PUBLISH_LOG="${LOG_ROOT}/checkpoint-publisher.log"
DATA_PID_FILE="${RUN_ROOT}/data-service.pid"
TRAIN_PID_FILE="${RUN_ROOT}/train.pid"
PUBLISH_PID_FILE="${RUN_ROOT}/checkpoint-publisher.pid"
STACK_LOCK_FILE="${RUN_ROOT}/training-stack.lock"

CHECKPOINT_ROOT=''
DATA_SOCKET=''
WORLD_SIZE=''
DEVICE_LIST=''
RESOLVED_PID=''

log() {
  printf '[training-stack] %s\n' "$*"
}

die() {
  printf '[training-stack] FAST-FAIL: %s\n' "$*" >&2
  exit 1
}

usage() {
  cat <<'EOF'
usage: scripts/training_stack.sh ACTION

Actions:
  start           Start any missing data, publisher, and training processes.
  restart         Stop all three processes, then start all three.
  restart-train   Restart only training; reuse the live data service/publisher.
  stop            Stop training, publisher, and data service.
  status          Show managed process, checkpoint, configuration, and log state.
  adopt           Adopt one exact pre-existing process for each component.
  validate        Validate host, environment, config, checkpoint, and executables.
  logs            Follow the three fixed log files.

Training always resumes from the numerically newest complete checkpoint.
Use ALLOW_FRESH_START=1 only when a deliberate fresh run is required.
EOF
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is unavailable: $1"
}

require_leaf_host() {
  local host
  host="$(hostname)"
  [[ "${host}" == *"${REQUIRED_HOST_SUBSTRING}"* ]] \
    || die "wrong host: ${host} does not contain ${REQUIRED_HOST_SUBSTRING}"
}

validate_integer() {
  local name="$1" value="$2"
  [[ "${value}" =~ ^[1-9][0-9]*$ ]] || die "${name} must be a positive integer: ${value}"
}

prepare_paths() {
  [[ -d "${PROJECT_ROOT}" ]] || die "project root is missing: ${PROJECT_ROOT}"
  [[ -d "${RUNTIME_ROOT}" ]] || die "runtime root is missing: ${RUNTIME_ROOT}"
  [[ -d "${CONFIG_ROOT}" ]] || die "config root is missing: ${CONFIG_ROOT}"
  [[ -f "${CONFIG_ROOT}/${CONFIG_NAME}" ]] \
    || die "training config is missing: ${CONFIG_ROOT}/${CONFIG_NAME}"
  [[ -x "${PYTHON_BIN}" ]] || die "Python is not executable: ${PYTHON_BIN}"
  [[ -x "${ACCELERATE_BIN}" ]] || die "Accelerate is not executable: ${ACCELERATE_BIN}"
  [[ -x "${MANAGEMENT_PYTHON}" ]] \
    || die "management Python is not executable: ${MANAGEMENT_PYTHON}"
  [[ -r "${PROJECT_ROOT}/scripts/publish_train_state.sh" ]] \
    || die "checkpoint publisher is not readable"
  mkdir -p -- "${LOG_ROOT}" "${RUN_ROOT}" "${PUBLISH_STATE_ROOT}"
  chmod 700 "${LOG_ROOT}" "${RUN_ROOT}" "${PUBLISH_STATE_ROOT}"
  cd -- "${PROJECT_ROOT}"
}

load_workload_environment() {
  local entry key
  [[ -f "${WORKLOAD_ENV_FILE}" ]] \
    || die "workload environment snapshot is missing: ${WORKLOAD_ENV_FILE}"
  [[ ! -L "${WORKLOAD_ENV_FILE}" ]] \
    || die "workload environment snapshot may not be a symlink"
  [[ "$(stat -c '%a' "${WORKLOAD_ENV_FILE}")" == 600 ]] \
    || die "workload environment snapshot must have mode 600: ${WORKLOAD_ENV_FILE}"
  while IFS= read -r -d '' entry; do
    [[ "${entry}" == *=* ]] || die "malformed workload environment entry"
    key="${entry%%=*}"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] \
      || die "invalid workload environment variable name: ${key}"
    case "${key}" in
      HOSTNAME|HOME|USER|LOGNAME|PWD|OLDPWD|SHLVL|_|SSH_*|CRD_*|KUBERNETES_*|SCNET_*|NOTEBOOK_*|JUPYTER_*|RANK|LOCAL_RANK|WORLD_SIZE|LOCAL_WORLD_SIZE|MASTER_ADDR|MASTER_PORT|RESUME)
        continue
        ;;
    esac
    export "${entry}"
  done <"${WORKLOAD_ENV_FILE}"

  export HOME=/root
  export USER=root
  export LOGNAME=root
  export PATH="${VENV_ROOT}/bin:${PATH}"
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  export PYTHONUNBUFFERED=1
  export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
  export MIOPEN_FIND_MODE="${MIOPEN_FIND_MODE:-NORMAL}"

  unset RANK LOCAL_RANK WORLD_SIZE LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT RESUME

  : "${PATH:?PATH is missing after environment load}"
  : "${LD_LIBRARY_PATH:?LD_LIBRARY_PATH is missing after environment load}"
  : "${PYTHONPATH:?PYTHONPATH is missing after environment load}"
  : "${DTKROOT:?DTKROOT is missing after environment load}"
  : "${MODELSCOPE_API_TOKEN:?MODELSCOPE_API_TOKEN is missing after environment load}"
  : "${WANDB_API_KEY:?WANDB_API_KEY is missing after environment load}"
}

load_config_contract() {
  local config_output
  local -a values
  config_output="$(
    "${MANAGEMENT_PYTHON}" - "${CONFIG_NAME}" "${CONFIG_ROOT}" "${RUNTIME_ROOT}" <<'PY'
from pathlib import Path
from collections.abc import Mapping
import copy
import sys
import tomllib


def merge(base: Mapping[str, object], overlay: Mapping[str, object]) -> dict[str, object]:
    result = copy.deepcopy(dict(base))
    for key, value in overlay.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = merge(existing, value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load(path: Path, root: Path, active: set[Path]) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    resolved.relative_to(root)
    if resolved in active:
        raise RuntimeError(f"config extends cycle: {resolved}")
    active.add(resolved)
    try:
        with resolved.open("rb") as stream:
            payload = tomllib.load(stream)
        includes = payload.pop("extends", [])
        if not isinstance(includes, list) or not all(isinstance(item, str) for item in includes):
            raise TypeError("extends must be a string array")
        merged: dict[str, object] = {}
        for include in includes:
            include_path = Path(include)
            if include_path.is_absolute() or ".." in include_path.parts:
                raise ValueError(f"invalid extends path: {include}")
            merged = merge(merged, load(resolved.parent / include_path, root, active))
        return merge(merged, payload)
    finally:
        active.remove(resolved)

config_name, config_root, runtime_root = sys.argv[1:]
root = Path(config_root).resolve(strict=True)
config = load(root / config_name, root, set())
paths = config["paths"]
data = config["data"]
distributed = config["distributed"]
if not isinstance(paths, Mapping) or not isinstance(data, Mapping) or not isinstance(distributed, Mapping):
    raise TypeError("required config tables are missing")
service = data["service"]
if not isinstance(service, Mapping):
    raise TypeError("data.service table is missing")
checkpoint = Path(str(paths["checkpoint_dir"]))
if not checkpoint.is_absolute():
    checkpoint = Path(runtime_root) / checkpoint
print(checkpoint)
print(service["socket_path"])
print(distributed["world_size"])
PY
  )" || die "failed to load the training config contract"
  mapfile -t values <<<"${config_output}"
  [[ "${#values[@]}" -eq 3 ]] \
    || die "config contract returned ${#values[@]} fields instead of 3"
  CHECKPOINT_ROOT="${values[0]}"
  DATA_SOCKET="${values[1]}"
  WORLD_SIZE="${values[2]}"
  [[ "${CHECKPOINT_ROOT}" == /* ]] || die "checkpoint root is not absolute"
  [[ "${DATA_SOCKET}" == /* ]] || die "data socket path is not absolute"
  validate_integer WORLD_SIZE "${WORLD_SIZE}"
  [[ "${WORLD_SIZE}" -le 8 ]] || die "WORLD_SIZE is unexpectedly large: ${WORLD_SIZE}"
  DEVICE_LIST="$(seq -s, 0 "$((WORLD_SIZE - 1))")"
  export CUDA_VISIBLE_DEVICES="${DEVICE_LIST}"
  export HIP_VISIBLE_DEVICES="${DEVICE_LIST}"
  export ROCR_VISIBLE_DEVICES="${DEVICE_LIST}"
}

pid_file_for() {
  case "$1" in
    data) printf '%s\n' "${DATA_PID_FILE}" ;;
    train) printf '%s\n' "${TRAIN_PID_FILE}" ;;
    publisher) printf '%s\n' "${PUBLISH_PID_FILE}" ;;
    *) die "unknown component: $1" ;;
  esac
}

log_file_for() {
  case "$1" in
    data) printf '%s\n' "${DATA_LOG}" ;;
    train) printf '%s\n' "${TRAIN_LOG}" ;;
    publisher) printf '%s\n' "${PUBLISH_LOG}" ;;
    *) die "unknown component: $1" ;;
  esac
}

pid_matches_component() {
  local component="$1" pid="$2" command cwd
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ -r "/proc/${pid}/cmdline" ]] || return 1
  command="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
  cwd="$(readlink -f "/proc/${pid}/cwd" 2>/dev/null || true)"
  [[ "${cwd}" == "${PROJECT_ROOT}" ]] || return 1
  case "${component}" in
    data)
      [[ "${command}" == *'-m sakuramoon.cli.data_service'* ]]
      ;;
    train)
      [[ "${command}" == *'accelerate launch'* \
        && "${command}" == *'-m sakuramoon.cli.train'* ]]
      ;;
    publisher)
      [[ "${command}" == *"${PROJECT_ROOT}/scripts/publish_train_state.sh loop"* ]]
      ;;
    *)
      return 1
      ;;
  esac
}

discover_component_pids() {
  local component="$1" proc pid
  for proc in /proc/[1-9]*; do
    pid="${proc##*/}"
    if pid_matches_component "${component}" "${pid}"; then
      printf '%s\n' "${pid}"
    fi
  done
}

write_pid_file() {
  local path="$1" pid="$2" temporary
  temporary="${path}.tmp.$$"
  printf '%s\n' "${pid}" >"${temporary}"
  mv -f -- "${temporary}" "${path}"
}

resolve_component_pid() {
  local component="$1" pid_file pid
  local -a discovered
  RESOLVED_PID=''
  pid_file="$(pid_file_for "${component}")"
  if [[ -f "${pid_file}" ]]; then
    IFS= read -r pid <"${pid_file}" || die "cannot read PID file: ${pid_file}"
    [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || die "invalid PID file: ${pid_file}"
    if kill -0 "${pid}" 2>/dev/null; then
      pid_matches_component "${component}" "${pid}" \
        || die "PID ${pid} from ${pid_file} belongs to another process"
      RESOLVED_PID="${pid}"
      return 0
    fi
    log "removing stale ${component} PID file: ${pid_file} (${pid})"
    rm -f -- "${pid_file}"
  fi

  mapfile -t discovered < <(discover_component_pids "${component}")
  case "${#discovered[@]}" in
    0)
      return 1
      ;;
    1)
      RESOLVED_PID="${discovered[0]}"
      write_pid_file "${pid_file}" "${RESOLVED_PID}"
      log "adopted existing ${component} process PID ${RESOLVED_PID}"
      return 0
      ;;
    *)
      die "multiple ${component} processes found: ${discovered[*]}"
      ;;
  esac
}

tail_component_log() {
  local component="$1" count="${2:-120}" log_file
  log_file="$(log_file_for "${component}")"
  if [[ -f "${log_file}" ]]; then
    tail -n "${count}" "${log_file}" >&2
  else
    printf '[training-stack] no %s log exists at %s\n' \
      "${component}" "${log_file}" >&2
  fi
}

start_detached() {
  local component="$1" log_file="$2"
  shift 2
  if resolve_component_pid "${component}"; then
    log "${component} already running: PID ${RESOLVED_PID}"
    return 0
  fi

  : >"${log_file}"
  nohup "$@" >>"${log_file}" 2>&1 </dev/null 9>&- &
  RESOLVED_PID=$!
  write_pid_file "$(pid_file_for "${component}")" "${RESOLVED_PID}"
  sleep 2
  if ! kill -0 "${RESOLVED_PID}" 2>/dev/null \
    || ! pid_matches_component "${component}" "${RESOLVED_PID}"; then
    rm -f -- "$(pid_file_for "${component}")"
    tail_component_log "${component}"
    die "${component} failed during startup"
  fi
  log "started ${component}: PID ${RESOLVED_PID}, log ${log_file}"
}

wait_for_data_service() {
  local elapsed=0
  while (( elapsed < START_TIMEOUT_SECONDS )); do
    if [[ -S "${DATA_SOCKET}" ]]; then
      resolve_component_pid data || die "data socket exists but data process is absent"
      log "data service ready: PID ${RESOLVED_PID}, socket ${DATA_SOCKET}"
      return 0
    fi
    resolve_component_pid data || {
      tail_component_log data
      die "data service exited before creating ${DATA_SOCKET}"
    }
    sleep 1
    elapsed=$((elapsed + 1))
    if (( elapsed % 10 == 0 )); then
      log "waiting for data service: ${elapsed}s"
    fi
  done
  tail_component_log data
  die "data service did not become ready within ${START_TIMEOUT_SECONDS}s"
}

latest_complete_checkpoint() {
  local directory name update raw_update best_update=-1 best=''
  [[ -d "${CHECKPOINT_ROOT}" ]] || die "checkpoint root is missing: ${CHECKPOINT_ROOT}"
  shopt -s nullglob
  for directory in "${CHECKPOINT_ROOT}"/ckpt_*_raw-*-update-cadence; do
    [[ -d "${directory}" ]] || continue
    [[ -f "${directory}/COMPLETE" ]] || continue
    grep -qx 'complete' "${directory}/COMPLETE" || continue
    [[ -f "${directory}/manifest.json" ]] || continue
    name="${directory##*/}"
    if [[ "${name}" =~ ^ckpt_([0-9]+)_raw-([0-9]+)-update-cadence$ ]]; then
      update="${BASH_REMATCH[1]}"
      raw_update="${BASH_REMATCH[2]}"
      [[ "${update}" == "${raw_update}" ]] \
        || die "checkpoint name has mismatched updates: ${name}"
      if (( 10#${update} > best_update )); then
        best_update=$((10#${update}))
        best="${directory}"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -z "${best}" ]]; then
    [[ "${ALLOW_FRESH_START:-0}" == 1 ]] || die "no complete checkpoint under ${CHECKPOINT_ROOT}"
    return 1
  fi
  printf '%s\n' "${best}"
}

start_data() {
  if ! resolve_component_pid data; then
    if [[ -e "${DATA_SOCKET}" ]]; then
      log "removing stale data socket: ${DATA_SOCKET}"
      rm -f -- "${DATA_SOCKET}"
    fi
    start_detached data "${DATA_LOG}" \
      "${PYTHON_BIN}" -u -m sakuramoon.cli.data_service \
      --config "${CONFIG_NAME}" \
      --config-root "${CONFIG_ROOT}" \
      --root "${RUNTIME_ROOT}"
  else
    log "data already running: PID ${RESOLVED_PID}"
  fi
  wait_for_data_service
}

start_publisher() {
  if resolve_component_pid publisher; then
    log "publisher already running: PID ${RESOLVED_PID}"
    return 0
  fi

  : >"${PUBLISH_LOG}"
  nohup env \
    SOURCE_ROOT="${CHECKPOINT_ROOT}" \
    PROJECT_ROOT="${PROJECT_ROOT}" \
    STATE_ROOT="${PUBLISH_STATE_ROOT}" \
    LOG_FILE="${PUBLISH_LOG}" \
      LAST_PUBLISHED="${PUBLISH_LAST_PUBLISHED}" \
    MS_HUB_BIN="${VENV_ROOT}/bin/ms-hub" \
    /usr/bin/bash "${PROJECT_ROOT}/scripts/publish_train_state.sh" loop \
    >/dev/null 2>&1 </dev/null 9>&- &
  RESOLVED_PID=$!
  write_pid_file "${PUBLISH_PID_FILE}" "${RESOLVED_PID}"
  sleep 2
  if ! kill -0 "${RESOLVED_PID}" 2>/dev/null \
    || ! pid_matches_component publisher "${RESOLVED_PID}"; then
    rm -f -- "${PUBLISH_PID_FILE}"
    tail_component_log publisher
    die "publisher failed during startup"
  fi
  log "started publisher: PID ${RESOLVED_PID}, log ${PUBLISH_LOG}"
}

training_rank_count() {
  local launcher_pid="$1"
  ps -eo ppid=,args= | awk -v parent="${launcher_pid}" \
    '$1 == parent && $0 ~ /-m sakuramoon[.]cli[.]train/ { count += 1 } END { print count + 0 }'
}

wait_for_training_ranks() {
  local launcher_pid="$1" elapsed=0 rank_count
  while (( elapsed < START_TIMEOUT_SECONDS )); do
    if ! kill -0 "${launcher_pid}" 2>/dev/null; then
      tail_component_log train 240
      die "training launcher exited during startup"
    fi
    rank_count="$(training_rank_count "${launcher_pid}")"
    if [[ "${rank_count}" -eq "${WORLD_SIZE}" ]]; then
      log "training ranks ready: launcher ${launcher_pid}, ranks ${rank_count}"
      return 0
    fi
    if grep -Eq 'Traceback|OutOfMemory|out of memory|RuntimeError|FAILED|non-finite|NaN' \
      "${TRAIN_LOG}"; then
      tail_component_log train 240
      die "training log reported a startup failure"
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    if (( elapsed % 10 == 0 )); then
      log "waiting for ${WORLD_SIZE} training ranks: ${elapsed}s, current ${rank_count}"
    fi
  done
  tail_component_log train 240
  die "training ranks did not start within ${START_TIMEOUT_SECONDS}s"
}

start_train() {
  local checkpoint
  resolve_component_pid data || die "data service is not running"
  [[ -S "${DATA_SOCKET}" ]] || die "data service socket is missing: ${DATA_SOCKET}"
  resolve_component_pid publisher || die "checkpoint publisher is not running"
  if resolve_component_pid train; then
    log "train already running: PID ${RESOLVED_PID}"
    return 0
  fi

  local -a resume_args=()
  if checkpoint="$(latest_complete_checkpoint)"; then
    resume_args=(--resume "${checkpoint}")
    log "selected resume checkpoint: ${checkpoint}"
  else
    log 'starting a deliberate fresh run because ALLOW_FRESH_START=1'
  fi

  start_detached train "${TRAIN_LOG}" \
    "${PYTHON_BIN}" "${ACCELERATE_BIN}" launch \
    --multi_gpu \
    --num_processes "${WORLD_SIZE}" \
    --num_machines 1 \
    --mixed_precision no \
    --dynamo_backend no \
    --main_process_port "${MAIN_PROCESS_PORT}" \
    -m sakuramoon.cli.train \
    --config "${CONFIG_NAME}" \
    --config-root "${CONFIG_ROOT}" \
    --root "${RUNTIME_ROOT}" \
    "${resume_args[@]}"
  wait_for_training_ranks "${RESOLVED_PID}"
}

collect_descendants() {
  local root_pid="$1" current child
  local -a queue=("${root_pid}") descendants=()
  while (( ${#queue[@]} > 0 )); do
    current="${queue[0]}"
    queue=("${queue[@]:1}")
    while IFS= read -r child; do
      [[ -n "${child}" ]] || continue
      descendants+=("${child}")
      queue+=("${child}")
    done < <(ps -eo pid=,ppid= | awk -v parent="${current}" '$2 == parent { print $1 }')
  done
  if (( ${#descendants[@]} > 0 )); then
    printf '%s\n' "${descendants[@]}"
  fi
}

stop_component() {
  local component="$1" pid elapsed=0 target alive=0
  local -a descendants targets
  if ! resolve_component_pid "${component}"; then
    log "${component} is not running"
    return 0
  fi
  pid="${RESOLVED_PID}"
  mapfile -t descendants < <(collect_descendants "${pid}")
  targets=("${pid}" "${descendants[@]}")
  log "stopping ${component}: PID ${pid}"
  for target in "${targets[@]}"; do
    if kill -0 "${target}" 2>/dev/null; then
      kill -TERM "${target}" 2>/dev/null || {
        kill -0 "${target}" 2>/dev/null \
          && die "failed to signal ${component} PID ${target}"
      }
    fi
  done
  while (( elapsed < STOP_TIMEOUT_SECONDS )); do
    alive=0
    for target in "${targets[@]}"; do
      if kill -0 "${target}" 2>/dev/null; then
        alive=1
        break
      fi
    done
    (( alive == 0 )) && break
    sleep 1
    elapsed=$((elapsed + 1))
  done
  if (( alive != 0 )); then
    log "${component} ignored TERM for ${STOP_TIMEOUT_SECONDS}s; sending KILL"
    for target in "${targets[@]}"; do
      kill -KILL "${target}" 2>/dev/null || true
    done
    sleep 1
  fi
  for target in "${targets[@]}"; do
    kill -0 "${target}" 2>/dev/null \
      && die "${component} PID ${target} survived shutdown"
  done
  rm -f -- "$(pid_file_for "${component}")"
  log "stopped ${component}"
}

start_stack() {
  start_data
  start_publisher
  start_train
}

stop_stack() {
  stop_component train
  stop_component publisher
  stop_component data
}

component_status() {
  local component="$1" log_file pid ranks
  log_file="$(log_file_for "${component}")"
  if resolve_component_pid "${component}"; then
    pid="${RESOLVED_PID}"
    if [[ "${component}" == train ]]; then
      ranks="$(training_rank_count "${pid}")"
      printf '%-10s running pid=%s ranks=%s/%s log=%s\n' \
        "${component}" "${pid}" "${ranks}" "${WORLD_SIZE:-?}" "${log_file}"
    else
      printf '%-10s running pid=%s log=%s\n' "${component}" "${pid}" "${log_file}"
    fi
  else
    printf '%-10s stopped log=%s\n' "${component}" "${log_file}"
  fi
}

show_status() {
  component_status data
  component_status publisher
  component_status train
  if [[ -n "${DATA_SOCKET}" ]]; then
    if [[ -S "${DATA_SOCKET}" ]]; then
      printf 'data socket ready: %s\n' "${DATA_SOCKET}"
    else
      printf 'data socket missing: %s\n' "${DATA_SOCKET}"
    fi
  fi
  if [[ -n "${CHECKPOINT_ROOT}" && -d "${CHECKPOINT_ROOT}" ]]; then
    local checkpoint
    if checkpoint="$(latest_complete_checkpoint 2>/dev/null)"; then
      printf 'latest checkpoint: %s\n' "${checkpoint}"
    else
      printf 'latest checkpoint: none\n'
    fi
  fi
}

adopt_stack() {
  local component
  for component in data publisher train; do
    resolve_component_pid "${component}" \
      || die "cannot adopt ${component}: no exact matching process"
    log "managed ${component}: PID ${RESOLVED_PID}"
  done
}

validate_stack() {
  local checkpoint
  checkpoint="$(latest_complete_checkpoint)" \
    || die "validation requires a complete checkpoint"
  log "host: $(hostname)"
  log "project: ${PROJECT_ROOT}"
  log "config: ${CONFIG_ROOT}/${CONFIG_NAME}"
  log "world size: ${WORLD_SIZE}, devices: ${DEVICE_LIST}"
  log "checkpoint: ${checkpoint}"
  log "logs: ${LOG_ROOT}"
  log "environment snapshot: ${WORKLOAD_ENV_FILE}"
  log 'environment and launch contract are valid'
}

follow_logs() {
  mkdir -p -- "${LOG_ROOT}"
  touch "${DATA_LOG}" "${TRAIN_LOG}" "${PUBLISH_LOG}"
  exec tail -n 100 -F "${DATA_LOG}" "${TRAIN_LOG}" "${PUBLISH_LOG}"
}

main() {
  local action="${1:-status}"
  case "${action}" in
    -h|--help|help)
      usage
      return 0
      ;;
    logs)
      follow_logs
      ;;
  esac

  require_leaf_host
  require_command flock
  require_command seq
  require_command awk
  prepare_paths

  case "${action}" in
    status)
      load_config_contract
      show_status
      return 0
      ;;
    validate)
      load_config_contract
      load_workload_environment
      validate_stack
      return 0
      ;;
    start|restart|restart-train|stop|adopt)
      exec 9>"${STACK_LOCK_FILE}"
      flock -n 9 || die "another training-stack operation holds ${STACK_LOCK_FILE}"
      ;;
    *)
      usage >&2
      die "unknown action: ${action}"
      ;;
  esac

  load_config_contract
  validate_integer START_TIMEOUT_SECONDS "${START_TIMEOUT_SECONDS}"
  validate_integer STOP_TIMEOUT_SECONDS "${STOP_TIMEOUT_SECONDS}"

  case "${action}" in
    start|restart|restart-train)
      load_workload_environment
      ;;
  esac

  case "${action}" in
    start)
      start_stack
      show_status
      ;;
    restart)
      stop_stack
      start_stack
      show_status
      ;;
    restart-train)
      resolve_component_pid data || die "data service must be running"
      [[ -S "${DATA_SOCKET}" ]] || die "data service socket is missing"
      resolve_component_pid publisher || die "checkpoint publisher must be running"
      stop_component train
      start_train
      show_status
      ;;
    stop)
      stop_stack
      show_status
      ;;
    adopt)
      adopt_stack
      show_status
      ;;
  esac
}

main "$@"
