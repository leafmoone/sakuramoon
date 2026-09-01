#!/usr/bin/env python3
"""SR 6M hub publisher: mirror run progress to the ModelScope hub (sidecar).

Cadence (2026-09-01 user order "每一百步上传到单独的文件夹 sr"):
  * every 100 trainer steps: upload the rolling ``<prefix>/train-state.json``
    (last step line from the train log + run provenance)
  * whenever a new ``step-NNNNNNN.pt`` appears in the out dir (the trainer's
    250k-exposure grid + milestones): upload it to ``<prefix>/`` — the real
    checkpoint mirror that survives pod rebuilds (the local disk is ephemeral)
  * when the trainer process exits: final state + ``latest.pt`` upload, exit

Idempotency: uploaded repo paths + the last state step persist in
``--state-file`` (NFS). The ms- API token is read from the key=value line
``MODELSCOPE_API_TOKEN=`` in ``--token-file`` (never printed, never logged).
The train log is read INCREMENTALLY (offset + truncation reset).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

STEP_LINE = re.compile(
    r"\[latent\] step (\d+)/(\d+) loss=([\d.]+) lr=([\d.e+-]+) "
    r"\( ?([\d.]+) it/s \) data_wait=([\d.]+)%"
)


def _token_from_file(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("MODELSCOPE_API_TOKEN="):
            return line.split("=", 1)[1].strip()
    raise SystemExit(f"--token-file {path}: no MODELSCOPE_API_TOKEN= line")


def _load_state(state_path: Path) -> dict:
    try:
        return json.loads(state_path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {"uploaded": [], "last_state_step": 0, "final_done": False}


def _save_state(state_path: Path, st: dict) -> None:
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(state_path)


class _LogTail:
    """Incremental tail: yields newly parsed step lines since the last call."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0
        self._buf = ""

    def scan(self) -> list[tuple[int, dict]]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.offset:  # log recreated/truncated
            self.offset = 0
            self._buf = ""
        if size == self.offset:
            return []
        found: list[tuple[int, dict]] = []
        try:
            with self.path.open("rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read(size - self.offset)
                self.offset = size
        except FileNotFoundError:
            return []
        self._buf += chunk.decode("utf-8", "replace")
        lines = self._buf.splitlines()
        self._buf = lines.pop() if lines else ""  # keep partial tail line
        for raw in lines:
            m = STEP_LINE.search(raw)
            if m:
                step, total, loss, lr, itps, wait = m.groups()
                found.append(
                    (
                        int(step),
                        {
                            "total_steps": int(total),
                            "loss": float(loss),
                            "lr": float(lr),
                            "it_per_s": float(itps),
                            "data_wait_pct": float(wait),
                        },
                    )
                )
        return found


def _trainer_alive() -> bool:
    r = subprocess.run(
        ["pgrep", "-f", "train_latent_flow"],
        capture_output=True,
        text=True,
    )
    return r.returncode == 0


def _log(msg: str) -> None:
    print(f"[hub-pub] {time.strftime('%m-%d %H:%M:%S')} {msg}", flush=True)


def _upload(api, repo: str, local: Path, in_repo: str, msg: str) -> None:
    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            api.upload_file(
                repo_id=repo,
                path_or_fileobj=str(local),
                path_in_repo=in_repo,
                commit_message=msg,
            )
            return
        except Exception as e:  # noqa: BLE001 - network layer: retry, then surface
            last_err = e
            _log(f"upload {in_repo} attempt {attempt} failed: {type(e).__name__}: {e}")
            time.sleep(15 * attempt)
    raise RuntimeError(f"upload {in_repo} failed after 3 attempts: {last_err}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", required=True, help="hub repo id (model)")
    p.add_argument("--prefix", required=True, help="in-repo dir, e.g. checkpoints/sr")
    p.add_argument("--out-dir", required=True, help="trainer out dir (ckpts)")
    p.add_argument("--train-log", required=True, help="trainer log (step lines)")
    p.add_argument("--token-file", required=True, help="env file with MODELSCOPE_API_TOKEN")
    p.add_argument("--state-file", required=True, help="NFS json idempotency state")
    p.add_argument("--run-name", default="sr-v2-6m")
    p.add_argument("--state-every-steps", type=int, default=100)
    p.add_argument("--interval", type=float, default=30.0)
    args = p.parse_args()

    from modelscope.hub.api import HubApi

    api = HubApi()
    api.login(_token_from_file(Path(args.token_file)))
    _log(f"logged in; repo={args.repo} prefix={args.prefix}")

    out_dir = Path(args.out_dir)
    state_path = Path(args.state_file)
    st = _load_state(state_path)
    uploaded: set[str] = set(st.get("uploaded", []))
    last_state_step: int = int(st.get("last_state_step", 0))
    tail = _LogTail(Path(args.train_log))
    last_line: tuple[int, dict] | None = None
    for _step, _fields in tail.scan():  # prime: parse the existing log once
        last_line = (_step, _fields)

    provenance = {
        "run": args.run_name,
        "config_stack": ["base.toml", "data.toml", "sr_v2_6m.toml"],
        "corpus": "leafmoone/SR_v2 (986 shards, 2188067 train-eligible)",
        "stream": "shard_seq (per-cycle shard permutation + intra-shard streaming)",
        "start": "init-trunk from checkpoints/latent-flow-phase1-small/step-0005000.pt",
        "venue": "salt9 (2x BW DCU)",
        "updated_utc": None,
    }

    def write_state_json(step: int, fields: dict, final: bool) -> Path:
        payload = dict(provenance)
        payload["step"] = step
        payload["metrics"] = fields
        payload["final"] = final
        payload["updated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        local = out_dir / "hub-train-state.json"
        local.write_text(json.dumps(payload, indent=1))
        return local

    while True:
        # 1) rolling train-state.json every --state-every-steps
        for step, fields in tail.scan():
            last_line = (step, fields)
        if (
            last_line is not None
            and not st.get("final_done")
            and last_line[0] >= last_state_step + args.state_every_steps
        ):
            local = write_state_json(last_line[0], last_line[1], final=False)
            try:
                _upload(
                    api,
                    args.repo,
                    local,
                    f"{args.prefix}/train-state.json",
                    f"sr 6m train-state @ step {last_line[0]}",
                )
                last_state_step = last_line[0]
                st["last_state_step"] = last_state_step
                _save_state(state_path, st)
                _log(f"train-state uploaded @ step {last_line[0]}")
            except RuntimeError as e:
                _log(f"state upload failed, will retry: {e}")

        # 2) checkpoint files -> prefix/ (rescanned each cycle)
        for local in sorted(out_dir.glob("step-*.pt")):
            if local.name in uploaded:
                continue
            try:
                _upload(
                    api,
                    args.repo,
                    local,
                    f"{args.prefix}/{local.name}",
                    f"sr 6m checkpoint {local.name}",
                )
                uploaded.add(local.name)
                st["uploaded"] = sorted(uploaded)
                _save_state(state_path, st)
                _log(f"checkpoint uploaded: {local.name} ({local.stat().st_size / 1e9:.2f} GiB)")
            except RuntimeError as e:
                _log(f"ckpt upload deferred, will retry: {e}")

        # 3) trainer exited -> final flush + state, then stop
        if not _trainer_alive() and not st.get("final_done"):
            final_step = last_line[0] if last_line else last_state_step
            final_fields = last_line[1] if last_line else {}
            local = write_state_json(final_step, final_fields, final=True)
            try:
                _upload(
                    api,
                    args.repo,
                    local,
                    f"{args.prefix}/train-state.json",
                    "sr 6m final train-state",
                )
            except RuntimeError as e:
                _log(f"final state upload failed; retrying next cycle: {e}")
                time.sleep(args.interval)
                continue
            st["final_done"] = True
            st["uploaded"] = sorted(uploaded)
            _save_state(state_path, st)
            _log("final state uploaded; publisher exiting")
            return

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
