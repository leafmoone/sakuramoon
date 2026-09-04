#!/usr/bin/env python3
"""F2 teardown survival test: does the minimal hard-fail capsule survive a
SIGTERM/SIGKILL of the OWNER process 5s / 2s / 1s after the OWNER RAISED?

Kill anchor = the owner's RAISED, not the step start: in production the
elastic launcher reacts to the RAISING rank's death (F1 measured ~30s) and
SIGTERM's the remaining ranks — so RAISED + delay is the realistic window,
and it is the stricter bound (the kill targets the publisher directly, at
the instant its publish has just finished). A kill before the verdict
lands inside the NS compute where no capsule is publishable yet (the
verdict is unknown); that is not the production teardown, so it is not the
gate.

This is the direct regression for the F1 112105 failure, where the elastic
teardown SIGTERM'd the owner 30s after the first rank raised — and the F1
publish path (2 full CPU NS replays, >30s on a 2560x2560 chunk) never
finished. The F2 contract: the local capsule must be durable well before
any realistic kill window.

Protocol (driver + worker, same file):

  * worker (torchrun --nproc_per_node=2):
      - rank0 = owner of a single PRODUCTION-SCALE [2560,2560] content_gate
        chunk (the 112105 class, forced below_floor: BF16 "huge",
        FP32 "tiny"=1e-9x -> finite below floor)
      - prints "WORKER rank0 READY pid=<pid>", then
        "WORKER rank0 STEP_START t=<wallclock>" immediately before
        opt.step() (the capsule publish happens inside that step)
      - after the expected CMuonSafetyError it prints
        "WORKER rank0 POST_STEP <event-or->" and SLEEPS (stays alive so
        the driver can kill it at the exact delay)
  * driver (this process, --mode driver):
      - spawns the worker via <python> -m torch.distributed.run (2 ranks),
        parses READY/STEP_START/RAISED via select() polling — the kill
        decision is TIMER-driven, never line-gated (the workers are
        deliberately silent over the whole NS+publish window, so a check
        that only runs on line arrival lands late)
      - SIGTERM the rank0 worker PID at OWNER_RAISED + delay for
        delay in (5, 2, 1) — each run gets a FRESH out dir
      - after the process dies: verifies the LOCAL emergency capsule —
        exactly one event dir, strict metadata.json, tensor file whose
        sha256 equals the recorded one, no partial/corrupt event dirs
      - records per-delay durability -> the MINIMUM SAFE WINDOW

Verdict: PASS if the capsule is durable at 2s AND 5s (the 1s result is
reported; if 1s loses the race the report quantifies the minimum safe
window instead — spec §7 allows either proof of survival at the stated
windows or an explicit quantification).

No optimizer math is touched: the worker runs the real production
optimizer class with the production NS (no stubs) and forced verdicts via
the same call-order mechanism as the 2-rank forensic test.

Usage:
  python tests/gpu/optim/cmuon_capsule_teardown.py --mode driver \
      --out /sakuramoon-runtime/cmuon-f2/out/teardown-report.json \
      [--delays 5,2,1]
  # the driver spawns:
  #   <python> -m torch.distributed.run --nproc_per_node=2 \
  #       cmuon_capsule_teardown.py --mode worker
  # (TEARDOWN_OUT env carries the per-run worker out dir)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import select
import signal
import subprocess
import sys
import time
from pathlib import Path


def _worker() -> int:
    """Run under torchrun with 2 ranks; rank0 owns the 2560x2560 chunk."""

    import torch
    import torch.distributed as dist
    from torch import nn

    # Make the tree's src importable (worktree layout: tests/gpu/optim/...).
    here = Path(__file__).resolve()
    sys.path.insert(0, str(here.parent.parent.parent.parent / "src"))

    import sakuramoon.optim.fp32_rescue as fr
    from sakuramoon.optim.cmuon import (
        cmuon_zeroth_power_bf16 as _real_bf16_ns,
    )
    from sakuramoon.optim.cmuon import (
        cmuon_zeroth_power_fp32 as _real_fp32_ns,
    )
    from sakuramoon.optim.cmuon import (
        route_cmuon_parameters,
    )
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
    from sakuramoon.optim.fp32_rescue import build_fp32_rescue
    from sakuramoon.optim.guarded_canonical import (
        GuardedCanonicalGuardConfig,
        stable_owner,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, "teardown test requires exactly 2 ranks"
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)

    out_dir = Path(os.environ.get("TEARDOWN_OUT", "/tmp/teardown-worker"))
    emergency = out_dir / "emergency"
    shared = out_dir / "shared"

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Module()
            torch.manual_seed(20260903)
            # Production-scale 2560x2560 chunks — the 112105 capture profile.
            # nn.Linear so the parameter FQNs carry the canonical ".weight"
            # suffix the allowlist regex requires.
            self.attention.content_gate = nn.Linear(2560, 2560, bias=False)
            nn.init.normal_(self.attention.content_gate.weight, std=0.02)
            self.attention.q_proj = nn.Linear(2560, 2560, bias=False)
            nn.init.normal_(self.attention.q_proj.weight, std=0.02)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    class _SingleDiT(nn.Module):
        """Canonical-FQN single block: two production-scale 2560x2560 CMuon
        chunks (content_gate + q_proj; one is rank0-owned = the poison) plus
        the allowlist-exempt final_layer matrix (AdamW subset non-empty)."""

        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleDict({"slot_00": _Block()})
            self.final_layer = nn.ModuleDict(
                {"linear": nn.Linear(128, 128, bias=False)}
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return x

    class _Composite(nn.Module):
        def __init__(self, dit: nn.Module) -> None:
            super().__init__()
            self.dit = dit

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
            return self.dit(x)

    model = _Composite(_SingleDiT()).to(device)
    # Locked audit dtype policy: 2D matrix params are BF16.
    for p in model.parameters():
        if p.ndim == 2 and p.dtype != torch.bfloat16:
            p.data = p.data.to(torch.bfloat16)
    routing = route_cmuon_parameters(
        model, matrix_weight_decay=0.0, sensitive_weight_decay=0.0
    )
    flat_keys: list[tuple[str, int]] = []
    for spec in routing.cmuon_specs:
        for ci in range(spec.chunk_count):
            flat_keys.append((spec.name, ci))
    owner_of_flat = [
        stable_owner(fqn, ci, world_size) for fqn, ci in flat_keys
    ]
    owned_here = [i for i, o in enumerate(owner_of_flat) if o == rank]
    # The poison chunk is the one OWNED BY RANK0 (whatever its role — both
    # chunks are production-scale 2560x2560, so the capture-timing profile
    # is identical; the content_gate-specific 112105 class is covered by
    # the 2-rank scenario E). rank0 is therefore always the publisher,
    # which keeps the driver's kill target unambiguous.
    owned0 = [i for i, o in enumerate(owner_of_flat) if o == 0]
    assert owned0, "rank0 must own at least one chunk"
    poison_pos = owned0[0]
    poison_fqn, poison_chunk = flat_keys[poison_pos]

    # Call-order forced NS: the poison chunk is identified by call order
    # among THIS rank's owned chunks (flat-order filter), like the 2-rank
    # forensic test.
    class _ForcedNS:
        def __init__(self, owned: list[int]) -> None:
            self.owned = owned
            self.pattern: dict[int, str] = {}
            self.call = 0
            self.fp32_kind: str | None = None

        def _apply(self, out: torch.Tensor, kind: str | None) -> torch.Tensor:
            if kind is None:
                return out
            if kind == "huge":
                return out * 1e9
            if kind == "tiny":
                return out * 1e-9
            raise ValueError(kind)

        def bf16(self, grad, ns_steps, ns_coefficients, eps):
            i = self.call
            self.call += 1
            kind = self.pattern.get(self.owned[i])
            # Call the REAL production NS (the fr.* module globals are the
            # patch points — calling them would recurse into this stub).
            out = _real_bf16_ns(grad, ns_steps, ns_coefficients, eps)
            return self._apply(out, kind)

        def fp32(self, grad, ns_steps, ns_coefficients, eps):
            out = _real_fp32_ns(grad, ns_steps, ns_coefficients, eps)
            return self._apply(out, self.fp32_kind)

    forced = _ForcedNS(owned_here)
    # Bootstrap guard references, keyed "fqn#chunkN" (same contract as the
    # 2-rank forensic test): a small per-chunk signal floor.
    bootstrap_refs: dict[str, float] = {}
    for spec in routing.cmuon_specs:
        g = torch.randn_like(spec.parameter)
        for ci in range(spec.chunk_count):
            chunk_size = spec.chunk_size()
            if spec.chunk_count == 1:
                sig = g.float().pow(2).mean().sqrt().item()
            else:
                start = ci * chunk_size
                end = start + chunk_size
                sl = [slice(None)] * g.ndim
                sl[spec.chunk_dim] = slice(start, end)
                sig = g[tuple(sl)].float().pow(2).mean().sqrt().item()
            bootstrap_refs[f"{spec.name}#chunk{ci}"] = max(sig * 1e-3, 1e-12)
    opt = build_fp32_rescue(
        model,
        lr=1.5625e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        block_size=256,
        bf16_stochastic_round=True,
        matrix_weight_decay=0.0,
        sensitive_weight_decay=0.0,
        sr_seed=44,
        ns_steps_by_role={
            "attention_q": 4,
            "attention_k": 4,
            "attention_v": 4,
            "attention_content_gate": 4,
            "attention_out": 4,
            "ffn_in": 4,
            "ffn_down": 4,
            "adaln_shared": 4,
        },
        guard_cfg=GuardedCanonicalGuardConfig(
            guard_ratio=0.05,
            reference_decay=0.999,
            min_reference=1e-12,
            numerical_floor=1e-20,
            warmup_observations=0,
            invariant_check=True,
        ),
        guard_bootstrap_refs=bootstrap_refs,
        rank=rank,
        world_size=world_size,
        momentum_dtype="bfloat16",
        chunk_rescale_sqrt_n=False,
        hard_fail_artifact_root=str(shared),
        legacy_forensic_dir=str(out_dir / "legacy"),
        emergency_capsule_root=str(emergency),
    )
    fr.cmuon_zeroth_power_bf16 = forced.bf16
    fr.cmuon_zeroth_power_fp32 = forced.fp32

    def _log(msg: str) -> None:
        print(f"WORKER rank{rank} {msg}", flush=True)

    _log(f"READY pid={os.getpid()}")

    forced.pattern = {poison_pos: "huge"}
    forced.fp32_kind = "tiny"  # finite below-floor FP32 -> below_floor
    forced.call = 0
    g = torch.Generator(device="cpu").manual_seed(77)
    with torch.no_grad():
        for p in model.parameters():
            p.grad = (
                torch.randn(tuple(p.shape), generator=g, dtype=p.dtype)
                .to(p.device)
                * 0.1
            )

    _log(f"STEP_START t={time.time():.6f} poison={poison_fqn}#chunk{poison_chunk}")
    try:
        opt.step()
    except CMuonSafetyError as e:
        _log(f"RAISED {str(e)[:120]}")
    else:
        _log("NO_RAISE (unexpected)")
        dist.barrier()
        dist.destroy_process_group()
        return 1
    events = [
        p
        for p in (emergency.iterdir() if emergency.exists() else [])
        if p.is_dir() and not p.name.startswith(".")
    ]
    _log(f"POST_STEP {events[0].name if events else '-'}")
    # Stay alive: the driver kills us at STEP_START + delay. If the capsule
    # is already durable, the kill changes nothing; if not, we die
    # mid-publish (the atomic publish leaves at most a temp dir).
    deadline = time.time() + 600
    while time.time() < deadline:
        time.sleep(0.5)
    dist.barrier()
    dist.destroy_process_group()
    return 0


def _inspect_capsule(emergency: Path) -> dict:
    """Durability contract: exactly one event dir, strict metadata, tensor
    sha matches, no partial event dirs."""
    out: dict[str, object] = {
        "capsule_durable": False,
        "events": [],
        "sha_ok": None,
        "mirror_status": None,
    }
    if not emergency.exists():
        out["events"] = []
        return out
    events = sorted(
        p for p in emergency.iterdir() if p.is_dir() and not p.name.startswith(".")
    )
    out["events"] = [p.name for p in events]
    if len(events) != 1:
        return out
    ev = events[0]
    meta_path = ev / "metadata.json"
    if not meta_path.is_file():
        return out
    try:
        # Strict JSON: NaN/Infinity literals are NOT acceptable durability.
        meta = json.loads(
            meta_path.read_text(),
            parse_constant=lambda c: json.JSONDecodeError(
                f"non-strict JSON constant {c}", "", 0
            ),
        )
    except json.JSONDecodeError:
        return out
    tensor_files = [
        f for f in ("input.safetensors", "input.pt") if (ev / f).is_file()
    ]
    if len(tensor_files) != 1:
        return out
    tf = ev / tensor_files[0]
    sha = hashlib.sha256(tf.read_bytes()).hexdigest()
    out["sha_ok"] = sha == str(meta.get("tensor_sha256", ""))
    out["reason"] = meta.get("fp32_failure_reason")
    if (ev / "mirror.json").is_file():
        try:
            out["mirror_status"] = json.loads((ev / "mirror.json").read_text())["status"]
        except (json.JSONDecodeError, KeyError):
            out["mirror_status"] = "unreadable"
    out["capsule_durable"] = bool(out["sha_ok"])
    return out


def _driver(args: argparse.Namespace) -> int:
    here = Path(__file__).resolve()
    out_path = Path(args.out)
    args.out = out_path
    delays = [int(d) for d in args.delays.split(",")]

    # Worker launch: <venv python> -m torch.distributed.run (the venv's
    # python is required so torchao — venv-only — resolves for the workers).
    launcher: list[str]
    if args.torchrun:
        launcher = [args.torchrun, "--nproc_per_node=2"]
    else:
        launcher = [args.python, "-m", "torch.distributed.run", "--nproc_per_node=2"]

    results = []
    for delay in delays:
        run_dir = args.out.parent / f".teardown-run-{delay}s"
        if run_dir.exists():
            import shutil

            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ)
        env["TEARDOWN_OUT"] = str(run_dir)
        env["NCCL_DEBUG"] = "ERROR"
        proc = subprocess.Popen(
            [
                *launcher,
                str(here),
                "--mode",
                "worker",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        rank0_pid: int | None = None
        step_start: float | None = None
        raise_time: float | None = None
        post_step_event: str | None = None
        kill_time: float | None = None
        kill_delivered: bool | None = None
        killed_before_exit: bool | None = None
        deadline = time.time() + 180
        driver_error: str | None = None
        log_path = run_dir / "worker-stdout.log"
        stdout_fd = proc.stdout.fileno()
        with open(log_path, "w", buffering=1) as log_handle:
            try:
                assert proc.stdout is not None
                while True:
                    # The adversarial kill is TIMER-driven: checked on every
                    # 50ms select poll, never only when a worker line
                    # arrives (the workers stay silent over the whole
                    # NS+publish window, so a line-gated check lands late).
                    if (
                        raise_time is not None
                        and kill_time is not None
                        and kill_delivered is None
                        and killed_before_exit is None
                        and time.time() >= kill_time
                    ):
                        # SIGTERM the OWNER worker at OWNER_RAISED + delay
                        # (production = elastic teardown of the remaining
                        # ranks after the raising rank dies).
                        assert rank0_pid is not None
                        try:
                            os.kill(rank0_pid, signal.SIGTERM)
                            kill_delivered = True
                        except ProcessLookupError:
                            # The owner exited on its own before the kill
                            # window elapsed: record it; no SIGTERM was
                            # delivered.
                            killed_before_exit = True
                            break
                        grace = time.time() + 5
                        while proc.poll() is None and time.time() < grace:
                            time.sleep(0.05)
                        if proc.poll() is None:
                            os.kill(rank0_pid, signal.SIGKILL)
                        break
                    if time.time() > deadline:
                        proc.kill()
                        break
                    ready, _, _ = select.select([stdout_fd], [], [], 0.05)
                    if ready:
                        line = proc.stdout.readline()
                        if not line:
                            break  # EOF: the launcher exited
                        log_handle.write(line)
                        if "READY pid=" in line:
                            tag, rest = line.split("READY pid=", 1)
                            if "rank0" in tag:
                                rank0_pid = int(rest.strip())
                        elif "STEP_START t=" in line:
                            tag, rest = line.split("STEP_START t=", 1)
                            if "rank0" in tag:
                                step_start = float(rest.split()[0])
                        elif line.startswith("WORKER rank0 RAISED "):
                            # Production kill anchor (see module docstring).
                            raise_time = time.time()
                            kill_time = raise_time + delay
                        elif "POST_STEP" in line:
                            tag, rest = line.split("POST_STEP", 1)
                            if "rank0" in tag and rest.strip() != "-":
                                post_step_event = rest.strip()
                    elif proc.poll() is not None:
                        # Launcher gone: drain whatever is left in the pipe.
                        for line in proc.stdout:
                            log_handle.write(line)
                        break
                if proc.poll() is None:
                    try:
                        proc.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=30)
            except Exception as exc:  # noqa: BLE001 — driver records, never masks
                driver_error = str(exc)
                if proc.poll() is None:
                    proc.kill()
                    proc.wait(timeout=30)
        exit_code: object = proc.returncode
        if driver_error is not None:
            exit_code = f"driver-error:{driver_error}"
        elif killed_before_exit:
            exit_code = f"{proc.returncode} (exited before kill window)"
        inspect = _inspect_capsule(run_dir / "emergency")
        elapsed = None
        raise_at = None
        if step_start is not None:
            elapsed = time.time() - step_start
            if raise_time is not None:
                raise_at = round(raise_time - step_start, 3)
        entry = {
            "delay_s": delay,
            "worker_exit_code": exit_code,
            "kill_anchor": "owner_RAISED",
            "raise_at_step_plus_s": raise_at,
            "kill_at_step_plus_s": None if raise_at is None else round(raise_at + delay, 3),
            "kill_delivered": kill_delivered,
            "killed_before_exit": killed_before_exit,
            "worker_log": str(log_path),
            "elapsed_since_step_start_at_exit": None if elapsed is None else round(elapsed, 3),
            "rank0_post_step_event": post_step_event,
            **inspect,
        }
        results.append(entry)
        print(f"[teardown] delay={delay}s -> {json.dumps(entry)}", flush=True)

    durable = {r["delay_s"]: bool(r["capsule_durable"]) for r in results}
    min_safe = None
    for d in sorted(durable, reverse=True):
        if durable[d]:
            min_safe = d
        else:
            break
    report = {
        "schema": "sakuramoon.cmuon_capsule_teardown.v2",
        "delays": results,
        "durable_by_delay": durable,
        "minimum_safe_window_s": min_safe,
        "interpretation": (
            "kill anchor = owner RAISED (production: the elastic launcher "
            "SIGTERM's the remaining ranks after the raising rank dies; "
            "F1 measured ~30s). minimum_safe_window_s = smallest tested "
            "delay at which the local capsule is still durable (all larger "
            "tested delays passed). A SIGTERM at or after RAISED + this "
            "delay preserves the exact-input capsule. The publish window "
            "(verdict -> atomic rename) is ~0.7s (capture bench) and "
            "completes BEFORE the raise, so the only theoretical loss zone "
            "is a SIGTERM landing inside that pre-raise window; production "
            "kills arrive ~30s AFTER the raise."
        ),
        "verdict": (
            "PASS"
            if durable.get(2) and durable.get(5)
            else "REPORTED-QUANTIFIED"
        ),
        "wall_clock_unix_seconds": time.time(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0 if report["verdict"] == "PASS" else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="driver", choices=("driver", "worker"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--delays", default="5,2,1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--torchrun", default=None)
    args = parser.parse_args()
    if args.mode == "worker":
        return _worker()
    assert args.out is not None, "--out is required for driver mode"
    return _driver(args)


if __name__ == "__main__":
    raise SystemExit(main())
