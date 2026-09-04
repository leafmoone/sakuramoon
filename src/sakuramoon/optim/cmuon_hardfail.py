"""Hard-failure forensic artifacts for the FP32-rescue optimizer (F1).

When an FP32-rescue step hard-fails (BF16 and FP32 verdicts both failed,
the ``CMuonSafetyError`` that is about to be raised on every rank), the
OWNER rank of each failing NS input publishes a self-contained replay
artifact:

  <root>/obs-<obs>-rank<R>-<fqn_safe>-chunk<C>/
    input.safetensors | input.pt   (the EXACT BF16 NS input, original dtype)
    metadata.json              (verdict values + diagnostic replay traces)

Guarantees (see reports/cmuon-fp32-rescue-forensic-audit.md §6/§8/§9):

  * Telemetry-only: nothing here feeds back into the verdict, the staged
    delta, momentum, parameters, the owner broadcast, or the commit. It
    runs strictly AFTER the rank-consistent failure verdict is settled.
  * Exact input: the saved tensor is a contiguous clone of the owner's
    BF16 Nesterov chunk (the input the BF16 NS consumed, and — via the
    exact BF16->FP32 cast — the input the FP32 rescue recomputed from).
    The CPU copy happens only after the hard-fail decision.
  * Owner-only: a non-owner rank never writes an input tensor for a
    chunk it did not compute (no fabricated inputs). Its per-rank
    forensic JSON record keeps null FP32 fields.
  * Atomic + unique: files are written to a temporary sibling, fsynced,
    and the event directory is published with an atomic rename. Event
    directories are unique per (observation, rank, fqn, chunk); a
    repeated failure (crash loop) gets a ``-r2``, ``-r3`` suffix instead
    of overwriting an older event. Old failures are never clobbered.
  * Fail-safe: any I/O error inside the publish is recorded (raised as
    ``HardFailArtifactError``) and the caller logs it and continues to
    raise the original ``CMuonSafetyError``. Telemetry I/O can never
    replace the root-cause exception.

The metadata separates the ORIGINAL production verdict values (computed
by the rescue before the raise) from the DIAGNOSTIC REPLAY values
(a second NS pass over the saved clone, via ``cmuon_ns_trace``), so a
nondeterministic HCU GEMM can never be confused with the value the
verdict actually compared.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import torch

#: Production default artifact root (the live G1 artifacts tree).
DEFAULT_HARD_FAIL_ROOT = "/sakuramoon-runtime/artifacts/g1/cmuon-hard-fail"

#: Artifact metadata schema tag.
HARD_FAIL_ARTIFACT_SCHEMA = "sakuramoon.cmuon_hard_fail_artifact.v1"

# Strict, single-valued FP32 rescue failure reasons. The priority mirrors
# the production verdict's condition order EXACTLY (nonfinite first, then
# below floor, then above ceiling); the string only ever affects report
# fields — never the verdict.
FP32_REASON_NONFINITE = "nonfinite"
FP32_REASON_BELOW_FLOOR = "below_floor"
FP32_REASON_ABOVE_CEILING = "above_ceiling"


def classify_fp32_verdict(
    finite32: bool,
    rms32: float,
    rescue_floor: float,
    ceiling: float,
) -> str | None:
    """The FP32 rescue failure reason for a failed verdict (None = passed).

    Deterministic priority identical to the production condition order::

        not bool(finite32)  ->  "nonfinite"
        or rms32 < floor    ->  "below_floor"
        or rms32 > ceiling  ->  "above_ceiling"
    """
    if not bool(finite32):
        return FP32_REASON_NONFINITE
    if float(rms32) < float(rescue_floor):
        return FP32_REASON_BELOW_FLOOR
    if float(rms32) > float(ceiling):
        return FP32_REASON_ABOVE_CEILING
    return None


def _json_safe(value: object) -> object:
    """Strict-JSON normalization (recursively): non-finite floats become
    None. The "nonfinite" verdict category is itself unrepresentable in
    strict JSON (``allow_nan=False``), and the sibling ``*_finite`` flags
    already encode that state, so null is unambiguous."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class HardFailArtifactError(RuntimeError):
    """Forensic artifact I/O failure (never masks the CMuonSafetyError)."""


def _fqn_safe(fqn: str) -> str:
    return fqn.replace(".", "_").replace("/", "_")


def _event_base_name(observations: int, rank: int, fqn: str, chunk_idx: int) -> str:
    return f"obs-{observations}-rank{rank}-{_fqn_safe(fqn)}-chunk{chunk_idx}"


def tensor_format_name() -> str:
    """The tensor serialization format this environment will use
    ("safetensors" when the package is importable, else "torch_pt"). The
    metadata records it so the replay CLI always knows what to load."""
    try:
        import safetensors.torch  # type: ignore[import-untyped]  # noqa: F401

        return "safetensors"
    except Exception:  # noqa: BLE001 - format fallback is part of the contract
        return "torch_pt"


def _write_tensor_bytes(tensor: torch.Tensor) -> tuple[bytes, str, str]:
    """Serialize the exact input tensor. Returns (bytes, format, filename).

    safetensors when importable (the recommended artifact format), else
    torch.save. The format is recorded in the metadata so the replay CLI
    (and humans) always know what to load.
    """
    if tensor_format_name() == "safetensors":
        from safetensors.torch import save_file  # type: ignore[import-untyped]

        # safetensors.save_file has no byte-buffer API (it takes a filename
        # and returns None in the supported versions): serialize to a temp
        # file, read the bytes back, remove the temp file.
        fd, tmp_path = tempfile.mkstemp(suffix=".safetensors")
        os.close(fd)
        try:
            save_file(
                {"input": tensor.contiguous()}, tmp_path, metadata={"format": "hardfail"}
            )
            with open(tmp_path, "rb") as handle:
                return handle.read(), "safetensors", "input.safetensors"
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
    return _torch_save_bytes(tensor), "torch_pt", "input.pt"


def _torch_save_bytes(tensor: torch.Tensor) -> bytes:
    import io

    stream = io.BytesIO()
    torch.save({"input": tensor.contiguous()}, stream)
    return stream.getvalue()


def publish_hard_fail_artifact(
    *,
    root: Path | str,
    observations: int,
    rank: int,
    world_size: int,
    fqn: str,
    chunk_idx: int,
    role: str,
    owner: int,
    input_tensor: torch.Tensor,
    metadata: dict[str, object],
) -> Path:
    """Atomically publish one hard-fail event directory (owner rank only).

    Creates ``<root>/<event-base>/`` with the input tensor file and
    ``metadata.json``, published via temp-dir + fsync + atomic rename.
    Raises ``HardFailArtifactError`` on any I/O failure (the caller logs
    and proceeds to the original safety error).
    """
    if owner != rank:
        raise ValueError(
            "hard-fail input artifacts are owner-only; a non-owner rank "
            f"(rank {rank}) must not publish an input for owner {owner}"
        )
    return _publish_event_dir(
        root,
        observations,
        rank,
        fqn,
        chunk_idx,
        input_tensor=input_tensor,
        metadata=metadata,
        fill_local_path=False,
        error_label="hard-fail artifact",
    )


def _publish_event_dir(
    root: Path | str,
    observations: int,
    rank: int,
    fqn: str,
    chunk_idx: int,
    *,
    input_tensor: torch.Tensor,
    metadata: dict[str, object],
    fill_local_path: bool = False,
    error_label: str = "hard-fail capsule",
) -> Path:
    """Atomic event-dir publish shared by the F1 artifact and the F2
    minimal capsule: temp sibling dir + tensor file + ``metadata.json``,
    fsynced, published by an atomic rename, unique per
    (observation, rank, fqn, chunk) with a ``-r2``/``-r3`` suffix on a
    repeated failure. Old events are never clobbered. Raises
    ``HardFailArtifactError`` on any I/O failure."""
    root_path = Path(root)
    base = _event_base_name(observations, rank, fqn, chunk_idx)
    event_dir = root_path / base
    seq = 1
    while event_dir.exists() or event_dir.is_symlink():
        # Repeated failure (crash loop): never overwrite an older event.
        seq += 1
        event_dir = root_path / f"{base}-r{seq}"
    tmp = root_path / f".{base}.tmp-{os.getpid()}-{uuid.uuid4().hex[:12]}"
    try:
        root_path.mkdir(parents=True, exist_ok=True)
        tmp.mkdir()
        tensor_bytes, _tensor_format, tensor_name = _write_tensor_bytes(input_tensor)
        # Non-finite verdict values (the "nonfinite" category itself) are
        # not representable in strict JSON: they become null. The sibling
        # *_finite flags already encode that state, so null is unambiguous.
        meta_dict = dict(metadata)
        if fill_local_path:
            # The event dir name is chosen before the temp dir is created;
            # the FileExistsError retry below can still shift the final
            # name by one -rN slot — the directory listing (and the
            # F2 mirror.json) remain authoritative in that race.
            meta_dict["local_artifact_path"] = str(event_dir)
        meta_bytes = (
            json.dumps(
                _json_safe(meta_dict), indent=1, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode()
        _fsync_write(tmp / tensor_name, tensor_bytes)
        _fsync_write(tmp / "metadata.json", meta_bytes)
        _fsync_dir(tmp)
        try:
            os.rename(tmp, event_dir)
        except FileExistsError:
            # Concurrent publish raced us (same seq); retry the next slot.
            seq += 1
            event_dir = root_path / f"{base}-r{seq}"
            while event_dir.exists() or event_dir.is_symlink():
                seq += 1
                event_dir = root_path / f"{base}-r{seq}"
            os.rename(tmp, event_dir)
        _fsync_dir(root_path)
    except Exception as exc:
        _cleanup(tmp)
        raise HardFailArtifactError(
            f"{error_label} publish failed for {fqn}#chunk{chunk_idx}: {exc!r}"
        ) from exc
    return event_dir


def _fsync_write(path: Path, data: bytes) -> None:
    with open(path, "xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _cleanup(tmp: Path) -> None:
    try:
        for child in tmp.iterdir() if tmp.exists() else ():
            child.unlink()
        if tmp.exists():
            tmp.rmdir()
    except OSError:
        pass


def build_hard_fail_metadata(
    *,
    observations: int,
    this_rank: int,
    world_size: int,
    fqn: str,
    chunk_idx: int,
    role: str,
    owner: int,
    input_tensor: torch.Tensor,
    alpha: float,
    ns_steps: int,
    ns_coefficients: tuple[float, float, float],
    eps: float,
    lr: float,
    target_delta_rms: float,
    ceiling: float,
    rescue_floor: float,
    bf16_delta_rms: float | None,
    original_fp32_delta_rms: float | None,
    original_fp32_finite: bool | None,
    fp32_failure_reason: str | None,
    bf16_failure_name: str,
    failure_message: str,
    tensor_sha256: str,
    tensor_format: str,
    diagnostic_bf16: dict[str, object] | None,
    diagnostic_fp32: dict[str, object] | None,
    forensic_trace_error: str | None,
) -> dict[str, object]:
    """Assemble the artifact metadata (spec §6 fields + original/diagnostic
    separation + trace error slot)."""
    inp = input_tensor.contiguous()
    tf = inp.float()
    return {
        "schema": HARD_FAIL_ARTIFACT_SCHEMA,
        "observations": observations,
        "this_rank": this_rank,
        "owner": owner,
        "world_size": world_size,
        "fqn": fqn,
        "chunk": chunk_idx,
        "role": role,
        "shape": [int(s) for s in tuple(inp.shape)],
        "numel": int(math.prod(tuple(inp.shape))),
        "dtype": str(inp.dtype),
        "contiguous": bool(inp.is_contiguous()),
        "input_rms": float(tf.pow(2).mean().sqrt().item()),
        "input_frobenius_norm": float(tf.norm().item()),
        "input_abs_max": float(tf.abs().max().item()),
        "input_finite": bool(torch.isfinite(tf).all().item()),
        "alpha": float(alpha),
        "ns_steps": int(ns_steps),
        "ns_coefficients": [float(v) for v in ns_coefficients],
        "eps": float(eps),
        "lr": float(lr),
        "target_delta_rms": float(target_delta_rms),
        "ceiling": float(ceiling),
        "rescue_floor": float(rescue_floor),
        # ORIGINAL production verdict values (what the step compared):
        "bf16_delta_rms": (
            None if bf16_delta_rms is None else float(bf16_delta_rms)
        ),
        "original_fp32_delta_rms": (
            None if original_fp32_delta_rms is None else float(original_fp32_delta_rms)
        ),
        "original_fp32_finite": (
            None if original_fp32_finite is None else bool(original_fp32_finite)
        ),
        # Spec-named aliases of the original values:
        "fp32_delta_rms": (
            None if original_fp32_delta_rms is None else float(original_fp32_delta_rms)
        ),
        "fp32_finite": None if original_fp32_finite is None else bool(original_fp32_finite),
        "fp32_failure_reason": fp32_failure_reason,
        "bf16_failure": bf16_failure_name,
        "failure_message": failure_message,
        # Artifact identity:
        "tensor_sha256": tensor_sha256,
        "tensor_format": tensor_format,
        "wall_clock_unix_seconds": time.time(),
        # DIAGNOSTIC replay (second NS pass over the saved clone):
        "diagnostic_replay_bf16": diagnostic_bf16,
        "diagnostic_replay_fp32": diagnostic_fp32,
        "diagnostic_replay_bf16_delta_rms": _final_delta_rms(diagnostic_bf16),
        "diagnostic_replay_fp32_delta_rms": _final_delta_rms(diagnostic_fp32),
        "diagnostic_replay_fp32_finite": _final_delta_finite(diagnostic_fp32),
        "forensic_trace_error": forensic_trace_error,
    }


def _final_delta_rms(diag: dict[str, object] | None) -> float | None:
    if not diag:
        return None
    final = diag.get("final")
    if not isinstance(final, dict):
        return None
    value = final.get("delta_rms")
    return None if value is None else float(value)  # type: ignore[arg-type]


def _final_delta_finite(diag: dict[str, object] | None) -> bool | None:
    if not diag:
        return None
    final = diag.get("final")
    if not isinstance(final, dict):
        return None
    value = final.get("delta_finite")
    return None if value is None else bool(value)  # type: ignore[arg-type]


# ======================================================================
# F2: FAST MINIMAL HARD-FAIL CAPSULE (local-first, teardown-safe)
# ======================================================================
#
# The F1 publish ran, BEFORE writing a single byte, two full op-exact
# CPU NS replays (BF16 + FP32) over the failing input (tens of seconds
# for 2560x2560). In the 2-rank production topology the non-owner rank
# raises first, torch-elastic tears down the worker group, and the owner
# rank is SIGTERM/SIGKILLed mid-replay: the exact input was lost (the
# 112105 event). F2 reorders the failure critical path to:
#
#   verdict settled -> freeze exact input (device clone) -> ONE
#   device->CPU transfer -> serialize + sha -> minimal metadata ->
#   LOCAL atomic capsule publish (durable) -> best-effort shared mirror
#   -> original CMuonSafetyError
#
# No CPU NS replay, no trace, no SVD, and no tensor statistics beyond
# the four existing O(n) scalar reductions required by the metadata are
# performed before the local capsule is durable. All diagnostics (NS
# replay, per-iteration trace, spectrum / effective-rank analysis) are
# OFFLINE via ``dev-tools/cmuon_hardfail_enrich.py``.

#: Local-first durable emergency root for the minimal capsule. On the
#: verified production host ``/sakuramoon-runtime`` is the local overlay
#: filesystem (716G free) while ``/root/private_data`` is the 50G-quota
#: NFS share — the emergency root therefore sits on local disk next to
#: the shared forensic tree. Isolated tests MUST redirect it (never
#: touch this path).
DEFAULT_EMERGENCY_CAPSULE_ROOT = "/sakuramoon-runtime/cmuon-f1-emergency"

#: Schema tag for the minimal capsule metadata (F2).
MINIMAL_CAPSULE_SCHEMA = "sakuramoon.cmuon_minimal_hardfail_capsule.v1"


def build_minimal_capsule_metadata(
    *,
    observations: int,
    this_rank: int,
    world_size: int,
    fqn: str,
    chunk_idx: int,
    role: str,
    owner: int,
    run_id: str | None,
    hostname: str,
    pid: int,
    process_steps: int,
    last_successful_update: int | None,
    attempted_update: int | None,
    checkpoint_source: str | None,
    input_tensor: torch.Tensor,
    alpha: float,
    ns_steps: int,
    ns_coefficients: tuple[float, ...],
    eps: float,
    lr: float,
    target_delta_rms: float,
    ceiling: float,
    rescue_floor: float,
    bf16_delta_rms: float | None,
    original_fp32_delta_rms: float | None,
    original_fp32_finite: bool | None,
    fp32_failure_reason: str | None,
    bf16_failure_name: str,
    failure_message: str,
    tensor_sha256: str,
    tensor_format: str,
    shared_mirror_root: str,
) -> dict[str, object]:
    """Assemble the MINIMAL capsule metadata (F2).

    Carries the event identity (run / host / process / update numbers),
    the exact-input description (shape / dtype / sha / format), the
    ORIGINAL production verdict values (the BF16 and FP32 delta rms +
    finiteness + reason the step actually compared), and the NS
    parameters the offline replay / enrichment need. Deliberately NO
    diagnostic replay, trace, spectrum, or rank statistics: those are
    computed OFFLINE by ``dev-tools/cmuon_hardfail_enrich.py`` and never
    in the production failure critical path.

    ``local_artifact_path`` is filled by the publish helper with the
    chosen event-dir name (and re-asserted by ``mirror.json`` after the
    mirror attempt).
    """
    inp = input_tensor.contiguous()
    tf = inp.float()
    # ``input_rms`` is the NS-input RMS (the ``u_t_rms`` of the legacy
    # guard-forensic record); both names are recorded so offline tooling
    # can use either.
    input_rms = float(tf.pow(2).mean().sqrt().item())
    return {
        "schema": MINIMAL_CAPSULE_SCHEMA,
        # Event identity:
        "observations": observations,
        "this_rank": this_rank,
        "owner": owner,
        "world_size": world_size,
        "run_id": run_id,
        "hostname": hostname,
        "pid": pid,
        "process_steps": process_steps,
        "last_successful_update": last_successful_update,
        "attempted_update": attempted_update,
        "checkpoint_source": checkpoint_source,
        # Failing NS input identity:
        "fqn": fqn,
        "chunk": chunk_idx,
        "role": role,
        "shape": [int(s) for s in tuple(inp.shape)],
        "numel": int(math.prod(tuple(inp.shape))),
        "dtype": str(inp.dtype),
        "contiguous": bool(inp.is_contiguous()),
        # Existing scalars required to write the metadata (four O(n) CPU
        # reductions over the saved clone; nothing heavier):
        "input_rms": input_rms,
        "u_t_rms": input_rms,
        "input_frobenius_norm": float(tf.norm().item()),
        "input_abs_max": float(tf.abs().max().item()),
        "input_finite": bool(torch.isfinite(tf).all().item()),
        # NS parameters (offline replay / enrichment inputs):
        "alpha": float(alpha),
        "ns_steps": int(ns_steps),
        "ns_coefficients": [float(v) for v in ns_coefficients],
        "eps": float(eps),
        "lr": float(lr),
        "target_delta_rms": float(target_delta_rms),
        "ceiling": float(ceiling),
        "rescue_floor": float(rescue_floor),
        # ORIGINAL production verdict values (what the step compared):
        "bf16_delta_rms": (
            None if bf16_delta_rms is None else float(bf16_delta_rms)
        ),
        "original_fp32_delta_rms": (
            None if original_fp32_delta_rms is None else float(original_fp32_delta_rms)
        ),
        "original_fp32_finite": (
            None if original_fp32_finite is None else bool(original_fp32_finite)
        ),
        # Spec-named aliases of the original values:
        "fp32_delta_rms": (
            None if original_fp32_delta_rms is None else float(original_fp32_delta_rms)
        ),
        "fp32_finite": (
            None if original_fp32_finite is None else bool(original_fp32_finite)
        ),
        "fp32_failure_reason": fp32_failure_reason,
        "bf16_failure": bf16_failure_name,
        "failure_message": failure_message,
        # Artifact identity:
        "tensor_sha256": tensor_sha256,
        "tensor_format": tensor_format,
        "shared_mirror_root": shared_mirror_root,
        "wall_clock_unix_seconds": time.time(),
    }


def publish_minimal_capsule(
    *,
    root: Path | str,
    observations: int,
    rank: int,
    world_size: int,
    fqn: str,
    chunk_idx: int,
    role: str,
    owner: int,
    input_tensor: torch.Tensor,
    metadata: dict[str, object],
) -> Path:
    """Publish the minimal LOCAL-first capsule (owner rank only, F2).

    This is the ONLY write in the production failure critical path: one
    exact-input tensor file + one ``metadata.json``, fsynced, atomically
    published under the local emergency root. CPU NS replay, traces, and
    SVD must happen OFFLINE (``dev-tools/cmuon_hardfail_enrich.py``).
    Follow up with :func:`mirror_capsule` (best-effort shared root) and
    then raise the original ``CMuonSafetyError``.
    """
    if owner != rank:
        raise ValueError(
            "hard-fail capsules are owner-only; a non-owner rank "
            f"(rank {rank}) must not publish an input for owner {owner}"
        )
    return _publish_event_dir(
        root,
        observations,
        rank,
        fqn,
        chunk_idx,
        input_tensor=input_tensor,
        metadata=metadata,
        fill_local_path=True,
        error_label="minimal hard-fail capsule",
    )


def mirror_capsule(event_dir: Path, shared_root: Path | str) -> dict[str, object]:
    """Best-effort mirror of a published local capsule into the shared
    forensic root (F2).

    NEVER raises and never changes the outcome: the result
    ``{"status": "ok" | "failed", "shared_path": ..., "error": ...}`` is
    returned AND recorded as a small new ``mirror.json`` file inside the
    local (already durable) event dir. A mirror failure never affects the
    original ``CMuonSafetyError`` and never invalidates the local
    capsule as evidence.
    """
    shared_root_path = Path(shared_root)
    tmp: Path | None = None
    result: dict[str, object] = {
        "status": "failed",
        "shared_path": None,
        "error": None,
    }
    try:
        base = event_dir.name
        target = shared_root_path / base
        seq = 1
        while target.exists() or target.is_symlink():
            seq += 1
            target = shared_root_path / f"{base}-r{seq}"
        shared_root_path.mkdir(parents=True, exist_ok=True)
        tmp = shared_root_path / f".{base}.mirror-{os.getpid()}-{uuid.uuid4().hex[:12]}"
        tmp.mkdir()
        for name in sorted(os.listdir(event_dir)):
            src = event_dir / name
            if src.is_file():
                _fsync_write(tmp / name, src.read_bytes())
        _fsync_dir(tmp)
        try:
            os.rename(tmp, target)
        except FileExistsError:
            seq += 1
            target = shared_root_path / f"{base}-r{seq}"
            while target.exists() or target.is_symlink():
                seq += 1
                target = shared_root_path / f"{base}-r{seq}"
            os.rename(tmp, target)
        _fsync_dir(shared_root_path)
        tmp = None
        result = {"status": "ok", "shared_path": str(target), "error": None}
    except Exception as exc:  # noqa: BLE001 - the mirror is best-effort by
        # contract: it must NEVER raise (the local capsule is already
        # durable and stays the success evidence).
        if tmp is not None:
            _cleanup(tmp)
        result = {"status": "failed", "shared_path": None, "error": repr(exc)}
    # Record the mirror status inside the local (already durable) event
    # dir as a new small file; the capsule is complete without it.
    try:
        payload = dict(result)
        payload["local_artifact_path"] = str(event_dir)
        payload["mirrored_at_unix_seconds"] = time.time()
        _fsync_write(
            event_dir / "mirror.json",
            (
                json.dumps(
                    _json_safe(payload), indent=1, sort_keys=True, allow_nan=False
                )
                + "\n"
            ).encode(),
        )
    except OSError:
        pass  # advisory record; the capsule itself is already durable
    return result


def tensor_sha256(tensor: torch.Tensor) -> str:
    """SHA256 of the exact tensor bytes (same serialization as the saved
    file: contiguous, original dtype)."""
    return hashlib.sha256(
        _write_tensor_bytes(tensor)[0]
    ).hexdigest()


@dataclass(frozen=True)
class HardFailReplayComparison:
    """recorded-original vs diagnostic-replay comparison (replay CLI)."""

    field: str
    recorded: float | None
    replayed: float | None
    abs_diff: float | None
    rel_diff: float | None


def compare_recorded_vs_replayed(
    metadata: dict[str, object],
) -> list[HardFailReplayComparison]:
    """The recorded original values vs the diagnostic replay values, with
    abs/rel deltas (None where either side is absent)."""
    rows: list[HardFailReplayComparison] = []
    pairs = (
        (
            "fp32_delta_rms",
            metadata.get("fp32_delta_rms"),
            metadata.get("diagnostic_replay_fp32_delta_rms"),
        ),
        (
            "bf16_delta_rms",
            metadata.get("bf16_delta_rms"),
            metadata.get("diagnostic_replay_bf16_delta_rms"),
        ),
    )
    for name, recorded, replayed in pairs:
        rec = None if recorded is None else float(recorded)  # type: ignore[arg-type]
        rep = None if replayed is None else float(replayed)  # type: ignore[arg-type]
        if rec is None or rep is None:
            rows.append(HardFailReplayComparison(name, rec, rep, None, None))
            continue
        abs_diff = abs(rec - rep)
        rel_diff = None if rec == 0.0 else abs_diff / abs(rec)
        rows.append(HardFailReplayComparison(name, rec, rep, abs_diff, rel_diff))
    fin_rec = metadata.get("fp32_finite")
    fin_rep = metadata.get("diagnostic_replay_fp32_finite")
    rows.append(
        HardFailReplayComparison(
            "fp32_finite",
            None if fin_rec is None else float(bool(fin_rec)),
            None if fin_rep is None else float(bool(fin_rep)),
            None,
            None,
        )
    )
    return rows


__all__ = [
    "DEFAULT_EMERGENCY_CAPSULE_ROOT",
    "DEFAULT_HARD_FAIL_ROOT",
    "FP32_REASON_ABOVE_CEILING",
    "FP32_REASON_BELOW_FLOOR",
    "FP32_REASON_NONFINITE",
    "HARD_FAIL_ARTIFACT_SCHEMA",
    "MINIMAL_CAPSULE_SCHEMA",
    "HardFailArtifactError",
    "HardFailReplayComparison",
    "build_hard_fail_metadata",
    "build_minimal_capsule_metadata",
    "classify_fp32_verdict",
    "compare_recorded_vs_replayed",
    "mirror_capsule",
    "publish_hard_fail_artifact",
    "publish_minimal_capsule",
    "tensor_format_name",
    "tensor_sha256",
]
