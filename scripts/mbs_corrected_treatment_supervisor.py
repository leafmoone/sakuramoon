#!/usr/bin/env python3
"""MBS treatment machine-side supervisor (tracked, review-branch version).

Python stdlib only. Runs independently of DSH/Codex/session lifetime
(launched via nohup/setsid). Owns the hard safety gates:

  - startup deadline (training process must appear)
  - per-metric-record invariants (schema + mirror conservation)
  - early exposure hard gate (by update 118103)
  - fatal train.log pattern gate (from launch byte offset)
  - stall gate (no new metric for 20 min)
  - wall-clock deadline (4 h from stack launch)
  - terminal verification (exact U118200, COMPLETE, no U118201)

On FAIL: writes result JSON atomically, calls training_stack.sh stop,
never restarts. On PASS: verifies terminal, waits 20 s, confirms no
U118201, writes result JSON, calls stop.

Stop environment (fix review sections 14/15): build_stop_env() passes
the 14 operational variables PLUS the explicit VENV_ROOT.  The previous
untracked supervisor omitted VENV_ROOT, so training_stack.sh stop
re-derived PYTHON_BIN/ACCELERATE_BIN from ${PROJECT_ROOT}/.venv (absent
on DTK hosts) and could not kill the workload.  PYTHON_BIN/
ACCELERATE_BIN/MANAGEMENT_PYTHON are preserved when the launch
environment carries them, otherwise derived from VENV_ROOT with the
exact training_stack.sh convention.

--self-test-stop-env runs a SIDE-EFFECT-FREE preflight (paths absolute,
python executable present + executable, WORKLOAD_ENV_FILE exists +
non-symlink + mode 0600, port/interval ints in range, host constraint)
and exits 0/1 without invoking the stack or touching any log/metric.

No secret VALUES are embedded; only the WORKLOAD_ENV_FILE path is stored
and passed through to the stack's stop action.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

FATAL_PATTERNS = [
    "Traceback",
    "OutOfMemory",
    "out of memory",
    "non-finite",
    "nonfinite",
    "NaN",
    "rank divergence",
    "watchdog timeout",
    "NCCL error",
    "HSA error",
    "VMFault",
    "optimizer failure",
    "CMuonSafetyError",
    "checkpoint verification failed",
]

FINITE_FIELDS = [
    "total_loss",
    "high_noise_loss",
    "low_noise_loss",
    "learning_rate",
    "pre_clip_grad_norm",
    "post_clip_grad_norm",
]

MIRROR_FIELDS = {
    "logical": "camera_mirror_logical_samples",
    "eligible": "camera_mirror_eligible",
    "selected": "camera_mirror_selected",
    "applied": "camera_mirror_applied",
    "extra": "camera_mirror_extra_views",
    "physical": "camera_mirror_physical_views",
    "lt2": "camera_mirror_severity_lt2",
    "s24": "camera_mirror_severity_2to4",
    "ge4": "camera_mirror_severity_ge4",
    "vertical": "camera_mirror_vertical_applied",
    "orig_start": "camera_mirror_original_start",
    "orig_center": "camera_mirror_original_center",
    "orig_end": "camera_mirror_original_end",
    "mir_start": "camera_mirror_mirror_start",
    "mir_center": "camera_mirror_mirror_center",
    "mir_end": "camera_mirror_mirror_end",
}


def is_finite_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


# ---------------------------------------------------------------------------
# Stop environment (fix review sections 14/15)
# ---------------------------------------------------------------------------
# The 14 operational variables the stack's stop action needs, plus the
# explicit VENV_ROOT.  Omitting VENV_ROOT is the bug that made the
# previous (untracked) supervisor's stop action orphan the workload:
# training_stack.sh stop then re-derived PYTHON_BIN/ACCELERATE_BIN from
# ${PROJECT_ROOT}/.venv, which does not exist on the DTK hosts.
STOP_ENV_KEYS = (
    "PROJECT_ROOT",
    "RUNTIME_ROOT",
    "CONFIG_ROOT",
    "CONFIG_NAME",
    "LOG_ROOT",
    "RUN_ROOT",
    "WORKLOAD_ENV_FILE",
    "RESUME_CHECKPOINT",
    "PUBLISH_STATE_ROOT",
    "PUBLISH_LAST_PUBLISHED",
    "REPO_PATH",
    "INTERVAL_SECONDS",
    "REQUIRED_HOST_SUBSTRING",
    "MAIN_PROCESS_PORT",
    "VENV_ROOT",
)

# Derived from VENV_ROOT with the exact training_stack.sh convention
# (lines 13-16 of the stack script) when not carried by the launch env.
DERIVED_BIN_KEYS = ("PYTHON_BIN", "ACCELERATE_BIN", "MANAGEMENT_PYTHON")


def build_stop_env(a) -> dict:
    """Pure builder for the stack-stop environment (unit-testable).

    The 14 operational variables + VENV_ROOT are always set explicitly.
    PYTHON_BIN/ACCELERATE_BIN/MANAGEMENT_PYTHON are preserved when the
    supervisor's own environment carries them (the launch contract
    exports the full stack environment); otherwise they are derived from
    VENV_ROOT exactly like training_stack.sh does.
    """
    env = os.environ.copy()
    env.update({
        "PROJECT_ROOT": a.project_root,
        "RUNTIME_ROOT": a.runtime_root,
        "CONFIG_ROOT": a.config_root,
        "CONFIG_NAME": a.config_name,
        "LOG_ROOT": a.log_root,
        "RUN_ROOT": a.run_root,
        "WORKLOAD_ENV_FILE": a.workload_env_file,
        "RESUME_CHECKPOINT": a.resume_checkpoint,
        "PUBLISH_STATE_ROOT": a.publish_state_root,
        "PUBLISH_LAST_PUBLISHED": a.publish_last_published,
        "REPO_PATH": a.repo_path,
        "INTERVAL_SECONDS": str(a.interval_seconds),
        "REQUIRED_HOST_SUBSTRING": a.required_host_substring,
        "MAIN_PROCESS_PORT": str(a.main_process_port),
        "VENV_ROOT": a.venv_root,
    })
    if not env.get("PYTHON_BIN"):
        env["PYTHON_BIN"] = os.path.join(a.venv_root, "bin", "python")
    if not env.get("ACCELERATE_BIN"):
        env["ACCELERATE_BIN"] = os.path.join(a.venv_root, "bin", "accelerate")
    if not env.get("MANAGEMENT_PYTHON"):
        env["MANAGEMENT_PYTHON"] = env["PYTHON_BIN"]
    return env


def self_test_stop_env(a) -> tuple:
    """SIDE-EFFECT-FREE preflight of the operational/stop environment.

    Read-only checks (never invokes the stack, writes nothing):
      - every STOP_ENV_KEYS field is a non-empty string;
      - every path field is absolute (REPO_PATH is relative by convention);
      - VENV_ROOT exists; the effective PYTHON_BIN and ACCELERATE_BIN are
        existing executable files;
      - WORKLOAD_ENV_FILE exists, is a regular file, is NOT a symlink,
        and has mode exactly 0600;
      - INTERVAL_SECONDS is a positive int; MAIN_PROCESS_PORT is an int
        in 1..65535;
      - REQUIRED_HOST_SUBSTRING is non-empty and CONFIG_NAME ends .toml.

    Returns (ok, problems, stop_env).
    """
    import stat as _stat

    problems = []
    env = build_stop_env(a)

    for key in STOP_ENV_KEYS:
        value = env.get(key)
        if not isinstance(value, str) or not value:
            problems.append(f"{key}: empty or not a string (got {value!r})")

    for key in STOP_ENV_KEYS:
        if key in {"REPO_PATH", "CONFIG_NAME", "INTERVAL_SECONDS", "REQUIRED_HOST_SUBSTRING", "MAIN_PROCESS_PORT"}:
            continue  # REPO_PATH/CONFIG_NAME relative by convention; ints/substr checked below
        value = env.get(key, "")
        if isinstance(value, str) and value and not os.path.isabs(value):
            problems.append(f"{key}: not an absolute path: {value!r}")

    venv_root = env.get("VENV_ROOT", "")
    if (
        isinstance(venv_root, str)
        and venv_root
        and os.path.isabs(venv_root)
        and not os.path.isdir(venv_root)
    ):
        problems.append(f"VENV_ROOT: directory missing: {venv_root!r}")
    for key in ("PYTHON_BIN", "ACCELERATE_BIN"):
        value = env.get(key, "")
        if isinstance(value, str) and value and os.path.isabs(value):
            if not os.path.isfile(value):
                problems.append(f"{key}: file missing: {value!r}")
            elif not os.access(value, os.X_OK):
                problems.append(f"{key}: not executable: {value!r}")

    env_file = env.get("WORKLOAD_ENV_FILE", "")
    if isinstance(env_file, str) and env_file and os.path.isabs(env_file):
        if os.path.islink(env_file):
            problems.append(f"WORKLOAD_ENV_FILE: is a symlink: {env_file!r}")
        elif not os.path.isfile(env_file):
            problems.append(
                f"WORKLOAD_ENV_FILE: missing or not a regular file: {env_file!r}"
            )
        else:
            mode = _stat.S_IMODE(os.stat(env_file).st_mode)
            if mode != 0o600:
                problems.append(f"WORKLOAD_ENV_FILE: mode {oct(mode)} != 0600: {env_file!r}")

    if type(a.interval_seconds) is not int or a.interval_seconds <= 0:
        problems.append(f"INTERVAL_SECONDS: not a positive int: {a.interval_seconds!r}")
    if type(a.main_process_port) is not int or not (1 <= a.main_process_port <= 65535):
        problems.append(f"MAIN_PROCESS_PORT: not an int in 1..65535: {a.main_process_port!r}")
    if not isinstance(a.required_host_substring, str) or not a.required_host_substring:
        problems.append("REQUIRED_HOST_SUBSTRING: empty")
    if not isinstance(a.config_name, str) or not a.config_name.endswith(".toml"):
        problems.append(f"CONFIG_NAME: must be a non-empty .toml name: {a.config_name!r}")

    return (not problems, problems, env)


class Supervisor:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.state = "WAITING_FOR_START"
        self.result_written = False
        self._train_exit_ts: float | None = None
        self._startup_pass_logged = False
        self.log_path = Path(a.supervisor_log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._logf = self.log_path.open("a", buffering=1)

    # ---------- logging ----------
    def log(self, msg: str) -> None:
        line = f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}"
        try:
            self._logf.write(line + "\n")
            self._logf.flush()
        except Exception:  # noqa: BLE001, S110 - log loss must not kill supervision
            pass
        print(line, flush=True)

    # ---------- atomic result ----------
    def _atomic_write(self, path: Path, obj: dict) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".sup-")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(obj, f, indent=2)
                f.write("\n")
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                try:
                    os.unlink(tmp)
                except OSError:
                    pass

    def write_result(self, status: str, reason: str) -> None:
        a = self.a
        obj = {
            "schema": "mbs_corrected_treatment_supervisor_result_v1",
            "status": status,
            "reason": reason,
            "state": self.state,
            "first_update": a.first_update,
            "terminal_update": a.terminal_update,
            "last_successful_update": getattr(self, "last_update", a.first_update - 1),
            "cumulative": getattr(self, "cumulative", {}),
            "early_exposure_gate": getattr(self, "early_gate", "not_reached"),
            "timestamp_unix": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._atomic_write(Path(a.result_path), obj)
        self.result_written = True
        self.log(f"RESULT WRITTEN: status={status} reason={reason}")

    # ---------- process / stack ----------
    def train_pid(self) -> int | None:
        pid_file = Path(self.a.run_root) / "train.pid"
        try:
            raw = pid_file.read_text().strip()
            pid = int(raw)
        except Exception:  # noqa: BLE001 - pid-file probe is best-effort
            return None
        try:
            os.kill(pid, 0)
            return pid
        except (ProcessLookupError, PermissionError, ValueError):
            return None
        except OSError:
            return None

    def train_alive(self) -> bool:
        return self.train_pid() is not None

    def train_rank_count(self) -> int:
        # count python -m sakuramoon.cli.train processes (world_size ranks)
        try:
            out = subprocess.run(
                ["pgrep", "-f", "sakuramoon.cli.train"],
                capture_output=True, text=True, timeout=15, check=False,
            ).stdout.split()
            return len(out)
        except Exception:  # noqa: BLE001 - rank probe is best-effort
            return 0

    def stack_stop(self) -> None:
        a = self.a
        env = build_stop_env(a)
        cmd = ["bash", os.path.join(a.project_root, "scripts/training_stack.sh"), "stop"]
        self.log(f"INVOKING stack stop: {' '.join(cmd)}")
        try:
            r = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, check=False)
            self.log(f"stack stop rc={r.returncode} "
                     f"stdout_tail={r.stdout[-400:]!r} stderr_tail={r.stderr[-400:]!r}")
        except Exception as e:  # noqa: BLE001 - stop must record any failure mode
            self.log(f"stack stop exception: {e!r}")

    # ---------- metrics ----------
    def _file_size(self, p: Path) -> int:
        try:
            return p.stat().st_size
        except OSError:
            return 0

    def read_new_metric_records(self, offset: int) -> tuple[list[dict], int]:
        p = Path(self.a.metrics_path)
        size = self._file_size(p)
        if size <= offset:
            return [], offset
        try:
            with p.open("rb") as f:
                f.seek(offset)
                chunk = f.read()
        except OSError:
            return [], offset
        # keep only complete lines (ending with \n)
        end = chunk.rfind(b"\n")
        if end < 0:
            return [], offset  # no complete line yet
        complete = chunk[: end + 1]
        new_offset = offset + len(complete)
        recs = []
        for line in complete.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                # partial / corrupt line: do not act on it
                continue
        return recs, new_offset

    def validate_record(self, rec: dict) -> tuple[bool, str]:
        f = MIRROR_FIELDS
        if rec.get("schema_version") != 12:
            return False, f"schema_version != 12 (got {rec.get('schema_version')!r})"
        if rec.get("nonfinite_count") != 0:
            return False, f"nonfinite_count != 0 (got {rec.get('nonfinite_count')!r})"
        for fld in FINITE_FIELDS:
            if not is_finite_num(rec.get(fld)):
                return False, f"non-finite/missing {fld}={rec.get(fld)!r}"
        eff = rec.get("effective_batch")
        if rec.get(f["logical"]) != eff:
            return False, f"logical({rec.get(f['logical'])}) != effective_batch({eff})"
        lt2 = rec.get(f["lt2"]); s24 = rec.get(f["s24"]); ge4 = rec.get(f["ge4"])
        vert = rec.get(f["vertical"])
        if not all(is_finite_num(x) for x in (lt2, s24, ge4, vert)):
            return False, "severity/vertical not numeric"
        if lt2 + s24 + ge4 != vert:
            return False, (f"severity partition {lt2}+{s24}+{ge4} != "
                           f"vertical_applied {vert}")
        severe = s24 + ge4
        elig = rec.get(f["eligible"]); sel = rec.get(f["selected"])
        app = rec.get(f["applied"]); extra = rec.get(f["extra"]); phys = rec.get(f["physical"])
        if not all(is_finite_num(x) for x in (elig, sel, app, extra, phys)):
            return False, "mirror counter not numeric"
        if elig != severe:
            return False, f"eligible({elig}) != severe_vertical({severe})"
        if sel != elig:
            return False, f"selected({sel}) != eligible({elig})"
        if app != sel:
            return False, f"applied({app}) != selected({sel})"
        if sel - app != 0:
            return False, f"degraded = selected-applied = {sel - app} != 0"
        if extra != app:
            return False, f"extra_views({extra}) != applied({app})"
        if phys != rec.get(f["logical"]) + extra:
            return False, (f"physical({phys}) != logical({rec.get(f['logical'])})"
                           f"+extra({extra})")
        osum = rec.get(f["orig_start"]) + rec.get(f["orig_center"]) + rec.get(f["orig_end"])
        msum = rec.get(f["mir_start"]) + rec.get(f["mir_center"]) + rec.get(f["mir_end"])
        if osum != app:
            return False, f"original sides sum {osum} != applied {app}"
        if msum != app:
            return False, f"mirror sides sum {msum} != applied {app}"
        return True, "ok"

    # ---------- train.log fatal gate ----------
    def read_new_log(self, offset: int) -> tuple[str, int]:
        p = Path(self.a.train_log_path)
        size = self._file_size(p)
        if size <= offset:
            return "", offset
        try:
            with p.open("rb") as fh:
                fh.seek(offset)
                chunk = fh.read().decode("utf-8", "replace")
        except OSError:
            return "", offset
        return chunk, size

    # ---------- terminal checkpoint ----------
    def terminal_ckpt_state(self) -> dict:
        ck = Path(self.a.terminal_ckpt_path)
        info = {"exists": ck.is_dir()}
        complete_f = ck / "COMPLETE"
        if info["exists"] and complete_f.is_file():
            try:
                info["complete_text"] = complete_f.read_text()
            except OSError:
                info["complete_text"] = None
        info["complete_ok"] = info.get("complete_text") == "complete\n"
        manifest_f = ck / "manifest.json"
        state_f = ck / "train_state" / "trainer_state.json"
        try:
            m = json.loads(manifest_f.read_text())
            ident = m.get("identity", {})
            info["manifest_update"] = ident.get("update")
            info["checkpoint_id"] = ident.get("checkpoint_id")
            info["reason_ok"] = (
                "update-cadence" in str(ident.get("checkpoint_id", ""))
                and "stage-finalize" not in str(ident.get("checkpoint_id", ""))
                and ident.get("update") == self.a.terminal_update
            )
        except Exception:  # noqa: BLE001 - ckpt inspection is best-effort
            info["reason_ok"] = False
            info["manifest_update"] = None
        try:
            ts = json.loads(state_f.read_text())
            info["trainer_successful"] = ts.get("successful_updates")
            sb = ts.get("stage_budget", {})
            info["budget_ok"] = (
                sb.get("start_successful_update") == 58000
                and sb.get("terminal_successful_update") == 168000
            )
        except Exception:  # noqa: BLE001 - trainer-state read is best-effort
            info["trainer_successful"] = None
            info["budget_ok"] = False
        return info

    def any_u118201(self) -> tuple[bool, str]:
        # metric 118201
        try:
            with open(self.a.metrics_path, "rb") as fh:
                data = fh.read()
            for line in data.split(b"\n"):
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                    if rec.get("successful_update") == self.a.terminal_update + 1:
                        return True, "metric 118201 present"
                except json.JSONDecodeError:
                    continue
        except OSError:
            pass
        # checkpoint > 118200
        root = Path(self.a.terminal_ckpt_path).parent
        try:
            for p in root.iterdir():
                name = p.name
                if name.startswith("ckpt_"):
                    try:
                        num = int(name.split("_")[1])
                    except (IndexError, ValueError):
                        continue
                    if num > self.a.terminal_update:
                        return True, f"checkpoint {name} > terminal"
        except OSError:
            pass
        # train log
        try:
            with open(self.a.train_log_path, "rb") as fh:
                data = fh.read().decode("utf-8", "replace")
            if f"update={self.a.terminal_update + 1}" in data or \
               f"successful_update = {self.a.terminal_update + 1}" in data or \
               f"update {self.a.terminal_update + 1}" in data:
                return True, f"train.log references update {self.a.terminal_update + 1}"
        except OSError:
            pass
        return False, "none"

    # ---------- status file (live, for DSH secondary monitoring) ----------
    def write_status(self) -> None:
        obj = {
            "state": self.state,
            "last_successful_update": getattr(self, "last_update", self.a.first_update - 1),
            "cumulative": getattr(self, "cumulative", {}),
            "early_exposure_gate": getattr(self, "early_gate", "not_reached"),
            "train_alive": self.train_alive(),
            "train_ranks": self.train_rank_count(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        self._atomic_write(Path(self.a.status_path), obj)

    # ---------- fail / pass ----------
    def fail(self, reason: str) -> None:
        if self.result_written:
            return
        self.state = "FAIL"
        self.write_result("FAIL", reason)
        self.stack_stop()
        self.write_status()

    def pass_run(self, reason: str) -> None:
        if self.result_written:
            return
        self.state = "PASS"
        self.write_result("PASS", reason)
        self.stack_stop()
        self.write_status()

    # ---------- main loop ----------
    def run(self) -> int:
        a = self.a
        t0 = time.time()
        self.log(f"SUPERVISOR START pid={os.getpid()} state=WAITING_FOR_START")
        self.log(f"  startup_deadline={a.startup_deadline}s wall_deadline={a.wall_deadline}s "
                 f"stall_timeout={a.stall_timeout}s first={a.first_update} terminal={a.terminal_update}")
        self.write_status()

        # ---- WAITING_FOR_START: training process must appear ----
        t_stack: float | None = None
        while t_stack is None:
            if self.train_alive():
                t_stack = time.time()
                self.state = "RUNNING"
                self.log(f"train process appeared (pid={self.train_pid()}, "
                         f"ranks={self.train_rank_count()}); T_stack set; state=RUNNING")
            elif time.time() - t0 > a.startup_deadline:
                self.fail("startup deadline exceeded: no training process within "
                          f"{a.startup_deadline}s")
                return 1
            else:
                time.sleep(5)
                self.write_status()

        # record train.log offset at launch; metrics start fresh
        train_log_offset = self._file_size(Path(a.train_log_path))
        self.log(f"train.log launch offset={train_log_offset}")
        metrics_offset = self._file_size(Path(a.metrics_path))
        self.log(f"metrics launch offset={metrics_offset}")
        self.write_status()

        self.last_update = a.first_update - 1
        self.cumulative = {
            "severe_vertical": 0, "eligible": 0, "selected": 0, "applied": 0,
            "extra_views": 0, "physical_views": 0, "logical_samples": 0,
            "vertical_applied": 0, "severity_lt2": 0, "severity_2to4": 0,
            "severity_ge4": 0,
        }
        self.early_gate = "not_reached"
        first_metric_ts: float | None = None
        terminal_seen = False

        while True:
            # ---- wall-clock deadline ----
            if time.time() - t_stack > a.wall_deadline:
                self.fail("wall-clock deadline exceeded: 4h from stack launch "
                          "before U118200")
                return 1

            # ---- fatal log gate (from launch offset) ----
            chunk, train_log_offset = self.read_new_log(train_log_offset)
            if chunk:
                for pat in FATAL_PATTERNS:
                    if pat in chunk:
                        idx = chunk.find(pat)
                        self.fail(f"fatal pattern in train.log: {pat!r} "
                                  f"ctx={chunk[max(0, idx - 80): idx + 80]!r}")
                        return 1

            # ---- metrics gate ----
            recs, metrics_offset = self.read_new_metric_records(metrics_offset)
            for rec in recs:
                ok, why = self.validate_record(rec)
                if not ok:
                    self.fail(f"record invariant violation @ update "
                              f"{rec.get('successful_update')!r}: {why}")
                    return 1
                up = rec["successful_update"]
                if up > a.terminal_update:
                    self.fail(f"update {up} exceeds terminal {a.terminal_update}")
                    return 1
                if up != self.last_update + 1:
                    self.fail(f"non-contiguous update: expected "
                              f"{self.last_update + 1}, got {up}")
                    return 1
                self.last_update = up
                f = MIRROR_FIELDS
                self.cumulative["logical_samples"] += rec[f["logical"]]
                self.cumulative["vertical_applied"] += rec[f["vertical"]]
                self.cumulative["severity_lt2"] += rec[f["lt2"]]
                self.cumulative["severity_2to4"] += rec[f["s24"]]
                self.cumulative["severity_ge4"] += rec[f["ge4"]]
                self.cumulative["severe_vertical"] += rec[f["s24"]] + rec[f["ge4"]]
                self.cumulative["eligible"] += rec[f["eligible"]]
                self.cumulative["selected"] += rec[f["selected"]]
                self.cumulative["applied"] += rec[f["applied"]]
                self.cumulative["extra_views"] += rec[f["extra"]]
                self.cumulative["physical_views"] += rec[f["physical"]]
                first_metric_ts = time.time()
                self.log(f"METRIC {up}: severe={rec[f['s24']]+rec[f['ge4']]} "
                         f"elig={rec[f['eligible']]} sel={rec[f['selected']]} "
                         f"app={rec[f['applied']]} phys={rec[f['physical']]} "
                         f"loss={rec.get('total_loss')!r} "
                         f"eff_batch={rec.get('effective_batch')}")
                # ---- early exposure hard gate (by 118103) ----
                if up >= 118103 and self.early_gate == "not_reached":
                    c = self.cumulative
                    if (c["severe_vertical"] > 0 and c["eligible"] > 0
                            and c["selected"] > 0 and c["applied"] > 0
                            and c["eligible"] == c["severe_vertical"]
                            and c["selected"] == c["eligible"]
                            and c["applied"] == c["selected"]):
                        self.early_gate = "PASS"
                        self.log(f"EARLY_EXPOSURE_GATE = PASS @ {up} "
                                 f"(cum severe={c['severe_vertical']} "
                                 f"elig={c['eligible']} sel={c['selected']} "
                                 f"app={c['applied']})")
                    else:
                        self.fail("early exposure gate FAILED at/before 118103: "
                                  f"cum severe={c['severe_vertical']} "
                                  f"elig={c['eligible']} sel={c['selected']} "
                                  f"app={c['applied']} (must all be >0 and equal)")
                        return 1
                if up == a.terminal_update:
                    terminal_seen = True
                    self.log(f"TERMINAL METRIC {up} observed; awaiting ckpt + exit")

            # ---- stall gate ----
            if (
                first_metric_ts is not None
                and not terminal_seen
                and time.time() - first_metric_ts > a.stall_timeout
                and self.train_alive()
            ):
                self.fail(f"stall: no new metric for {a.stall_timeout}s while "
                          f"train still running (last update={self.last_update})")
                return 1

            # ---- PASS condition ----
            if terminal_seen:
                ck = self.terminal_ckpt_state()
                alive = self.train_alive()
                if (ck.get("exists") and ck.get("complete_ok")
                        and ck.get("trainer_successful") == a.terminal_update
                        and ck.get("budget_ok") and ck.get("reason_ok")
                        and not alive):
                    if self._train_exit_ts is None:
                        self._train_exit_ts = time.time()
                        self.log(f"trainer exited; terminal ckpt COMPLETE "
                                 f"(update={ck.get('manifest_update')} "
                                 f"reason_ok=True); 20s settle wait begins")
                    elif time.time() - self._train_exit_ts >= 20:
                        found, why = self.any_u118201()
                        if found:
                            self.fail(f"U118201 detected after terminal: {why}")
                            return 1
                        self.pass_run(
                            f"terminal {a.terminal_update} reached; "
                            f"ckpt COMPLETE; no U118201; "
                            f"records={self.last_update - a.first_update + 1}")
                        return 0
                # else: keep polling (checkpoint may still be writing)

            time.sleep(5)
            self.write_status()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--project-root", required=True)
    p.add_argument("--runtime-root", required=True)
    p.add_argument("--config-root", required=True)
    p.add_argument("--config-name", required=True)
    p.add_argument("--log-root", required=True)
    p.add_argument("--run-root", required=True)
    p.add_argument("--workload-env-file", required=True)
    p.add_argument("--resume-checkpoint", required=True)
    p.add_argument("--publish-state-root", required=True)
    p.add_argument("--publish-last-published", required=True)
    p.add_argument("--repo-path", required=True)
    p.add_argument("--interval-seconds", type=int, required=True)
    p.add_argument("--required-host-substring", required=True)
    p.add_argument("--main-process-port", type=int, required=True)
    p.add_argument("--venv-root", required=True)
    p.add_argument(
        "--self-test-stop-env",
        action="store_true",
        help=(
            "validate the operational/stop environment (side-effect-free) "
            "and exit 0/1; never invokes the stack, never writes results"
        ),
    )
    p.add_argument("--metrics-path", required=True)
    p.add_argument("--train-log-path", required=True)
    p.add_argument("--result-path", required=True)
    p.add_argument("--status-path", required=True)
    p.add_argument("--supervisor-log-path", required=True)
    p.add_argument("--terminal-ckpt-path", required=True)
    p.add_argument("--startup-deadline", type=int, default=600)
    p.add_argument("--wall-deadline", type=int, default=14400)
    p.add_argument("--stall-timeout", type=int, default=1200)
    p.add_argument("--first-update", type=int, default=118101)
    p.add_argument("--terminal-update", type=int, default=118200)
    return p.parse_args()


def main() -> int:
    a = parse_args()
    if a.self_test_stop_env:
        ok, problems, env = self_test_stop_env(a)
        print("SELF_TEST_STOP_ENV:", "PASS" if ok else "FAIL", flush=True)
        for key in STOP_ENV_KEYS + DERIVED_BIN_KEYS:
            print(f"  {key}={env.get(key)}", flush=True)
        for problem in problems:
            print(f"  PROBLEM: {problem}", flush=True)
        return 0 if ok else 1
    sup = Supervisor(a)

    def _term(signum, frame):
        # if killed externally, record a FAIL (never silently die)
        try:
            if not sup.result_written:
                sup.fail(f"supervisor received signal {signum}")
        except Exception:  # noqa: BLE001, S110 - signal path must exit regardless
            pass
        sys.exit(130)

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)
    try:
        return sup.run()
    except Exception as e:  # noqa: BLE001 - top-level catch-all by design
        import traceback
        sup.log("SUPERVISOR UNCAUGHT EXCEPTION:\n" + traceback.format_exc())
        if not sup.result_written:
            sup.fail(f"supervisor internal error: {e!r}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
