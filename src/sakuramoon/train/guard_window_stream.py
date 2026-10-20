"""Per-rank guard window telemetry stream (default off, opt-in).

Reads ONLY int/map public attributes of the guarded optimizer (never
``state_dict()`` — that would copy momenta/weights), takes a snapshot on
each trainer-successful update, seals one window every ``window_updates``
successful updates, and publishes one single-line JSONL record per rank per
window via the atomic no-clobber primitive (:mod:`guard_publish`).

Contract highlights (see ``docs/guard-window-owner-contract.md`` and the
Phase 2 design):

- one rank writes only its own session directory; the ``.claimed`` marker
  (O_EXCL) refuses session_id reuse for that rank without touching training;
- a snapshot or publication failure latches DISABLED for the whole session
  (stderr + traceback, NEVER raised into training): the stream never writes
  another formal record in this session (an explicit missing tail).
  Recovering later windows could falsely imply continuous comparable
  coverage.  Only a new session (new process + new session_id) starts
  recording again; the on-disk sequence shows the gap.
- only trainer-successful updates are observed (single hook next to the
  condition-diagnostics observer); failed attempts abort the run (fail
  closed) and therefore never enter a sealed window;
- ``close(run_outcome)`` writes at most one tail record
  (``run_outcome=clean|raised|unknown``) and never writes a ``complete``
  record; a disabled or un-baselined stream writes nothing (tail unknown).

``run_outcome`` is an OBSERVED LOOP-SCOPE value: it covers exactly the
training loop plus telemetry block that the production call site wraps. It
must NEVER be read as "the whole production session succeeded": later
cleanup (diagnostics close, checkpoint-verification check, data-stream
close) is out of scope, and the CLI only ever derives data-coverage
statements from the requested data, never session success.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
from sakuramoon.optim.fp32_rescue import HybridCMuonCanonicalNS4FP32Rescue
from sakuramoon.optim.guarded_canonical import (
    OWNER_MAPPING_VERSION,
    HybridCMuonGuardedCanonical,
    stable_owner,
)
from sakuramoon.telemetry.guard_publish import (
    PublicationStatus,
    publish_atomically,
    report_warning,
)
from sakuramoon.telemetry.metrics import DurableJsonlSink

if TYPE_CHECKING:
    from collections.abc import Callable

    from sakuramoon.train.runtime import SuccessfulTrainingObservation

SCHEMA_VERSION = 1

_R_SCALARS = ("observations", "skip_total")
_O_SCALARS = (
    "bf16_attempts",
    "bf16_safety_failures",
    "fp32_attempts",
    "fp32_rescues",
    "fp32_rescue_failures",
)
_ROLE_MAPS = ("skip_by_role", "rescue_by_role")
_FQN_MAP = "skip_by_fqn"

_RUN_OUTCOMES = ("clean", "raised", "unknown")
_LAUNCH_KINDS = ("fresh_bootstrap", "user_resume")

# Real candidate names, bound from the ACTUAL optimizer type (rescue is a
# subclass, so it is checked first).  The field presence of any counter
# never decides the variant: a missing rescue field is a schema violation,
# not "actually the base variant".
_VARIANT_BINDINGS = (
    (HybridCMuonCanonicalNS4FP32Rescue, "hybrid_cmuon_canonical_ns4_fp32_rescue"),
    (HybridCMuonGuardedCanonical, "hybrid_cmuon_guarded_canonical"),
)


def _optimizer_variant(optimizer: object) -> str:
    for cls, name in _VARIANT_BINDINGS:
        if isinstance(optimizer, cls):  # type: ignore[arg-type]
            return name
    raise CMuonSafetyError(
        "guard_window optimizer type not bound to a known guarded variant: "
        f"{type(optimizer).__name__}"
    )


def _stderr(message: str, error: BaseException | None = None) -> None:
    """Best-effort stderr diagnostic; must never change the training flow."""
    try:
        print(f"[guard_window] {message}", file=sys.stderr, flush=True)
        if error is not None:
            traceback.print_exception(error, file=sys.stderr)
    except Exception:  # noqa: BLE001,S110 - diagnostics must never change training
        pass


def _validate_session_component(session_id: str) -> None:
    """Minimal path-component validation for ``session_id`` (local contract)."""
    if not session_id or len(session_id) > 120:
        raise CMuonSafetyError(
            f"guard_window session_id must be 1..120 chars, got {session_id!r}"
        )
    for bad in ("/", "\\", "\x00", ".."):
        if bad in session_id:
            raise CMuonSafetyError(
                f"guard_window session_id contains illegal path component {bad!r}"
            )


@dataclass(frozen=True)
class GuardWindowIdentity:
    """Fixed identity of one session for one rank (never reset mid-session)."""

    run_id: str
    session_id: str
    rank: int
    world_size: int
    resume_source_checkpoint: str  # "fresh-start" or the checkpoint name
    resume_successful_updates: int
    # Explicit launch kind: production always loads a checkpoint object even
    # on a fresh start (bootstrap save+restore), so the mere presence of a
    # RestoredSingleGpuCheckpoint must NOT be used to infer "user resume".
    launch_kind: str  # "fresh_bootstrap" | "user_resume"


class GuardWindowRecorder:
    """One instance per rank; bound to its own session/rank directory."""

    def __init__(
        self,
        *,
        artifact_dir: Path,
        identity: GuardWindowIdentity,
        optimizer: object,
        window_updates: int,
        last_published_checkpoint: Callable[[], str | None] | None = None,
    ) -> None:
        if window_updates < 1:
            raise CMuonSafetyError(
                f"guard_window window_updates must be >= 1, got {window_updates}"
            )
        if not isinstance(optimizer, HybridCMuonGuardedCanonical):
            raise CMuonSafetyError(
                "guard_window requires the guarded HybridCMuon optimizer; "
                f"got {type(optimizer).__name__}"
            )
        _validate_session_component(identity.session_id)
        if identity.resume_successful_updates < 0:
            raise CMuonSafetyError(
                "guard_window resume_successful_updates must be >= 0"
            )
        if identity.launch_kind not in _LAUNCH_KINDS:
            raise CMuonSafetyError(
                f"guard_window launch_kind must be one of {_LAUNCH_KINDS}, "
                f"got {identity.launch_kind!r}"
            )

        self._rank_dir = (
            artifact_dir
            / "guard_window"
            / f"session-{identity.session_id}"
            / f"rank-{identity.rank}"
        )
        self._temp_dir = self._rank_dir / ".tmp"
        self._identity = identity
        self._optimizer = optimizer
        self._window_updates = window_updates
        self._last_published = last_published_checkpoint
        # Explicit variant identity for the record/merger contract; never
        # inferred from which fields happen to exist on the instance.
        self._optimizer_variant = _optimizer_variant(optimizer)

        # Per-rank exclusive session claim: O_EXCL marker; an existing claim
        # means this session_id was already used on this storage -> refuse to
        # record (training is not touched).  Cross-storage reuse is the
        # caller's guarantee, not something we can detect.  Any IO failure
        # here (mkdir/open/write/close) DISABLES the session; only the
        # config-level checks above raise at construction.
        self._disabled = False
        claim = self._rank_dir / ".claimed"
        try:
            self._rank_dir.mkdir(parents=True, exist_ok=True)
            flags: int = os.O_WRONLY | os.O_CREAT | os.O_EXCL
            for flag_name in ("O_NOFOLLOW", "O_CLOEXEC"):  # POSIX-only flags
                flags |= getattr(os, flag_name, 0)  # pyright: ignore[reportArgumentType]
            fd = os.open(claim, flags, 0o600)
            try:
                os.write(
                    fd,
                    json.dumps(
                        {
                            "rank": identity.rank,
                            "world_size": identity.world_size,
                            "session_id": identity.session_id,
                            "resume_source": identity.resume_source_checkpoint,
                            "resume_successful_updates": identity.resume_successful_updates,
                        },
                        separators=(",", ":"),
                    ).encode(),
                )
            finally:
                os.close(fd)
        except FileExistsError:
            report_warning(
                f"guard_window session {identity.session_id!r} already claimed on rank "
                f"{identity.rank}; refusing to record (start a new session_id)"
            )
            self._disabled = True
        except Exception as error:  # noqa: BLE001 - IO failure disables, never raises
            _stderr(
                "guard_window session claim failed; disabling this rank's stream", error
            )
            self._disabled = True

        # Routing captured from the REAL optimizer (not self-declared): full
        # spec structure incl. parameter shape, both the per-chunk semantic
        # ``roles`` and the canonical counting-bucket ``role`` (rescue/skip
        # role maps use the latter).  The owner partition is recomputed with
        # stable_owner.
        self._routing: list[dict[str, object]] = []
        self._owner_partition: dict[str, list[str]] = {
            str(r): [] for r in range(identity.world_size)
        }
        known_roles: set[str] = set()
        known_fqns: set[str] = set()
        for spec in optimizer.routing.cmuon_specs:  # pyright: ignore[reportUnknownMemberType]
            self._routing.append(
                {
                    "fqn": spec.name,
                    "shape": tuple(spec.parameter.shape),
                    "chunk_count": spec.chunk_count,
                    "chunk_dim": spec.chunk_dim,
                    "roles": list(spec.roles),
                    "role": spec.role,
                }
            )
            known_roles.add(spec.role)
            known_fqns.add(spec.name)
            for chunk in range(spec.chunk_count):
                owner = stable_owner(spec.name, chunk, identity.world_size)
                self._owner_partition[str(owner)].append(f"{spec.name}#{chunk}")
        self._known_roles = frozenset(known_roles)
        self._known_fqns = frozenset(known_fqns)
        # The rescue field set is bound from the REAL type, never from
        # hasattr: a base instance carrying stray O attributes does not
        # become a rescue record.
        self._has_rescue = isinstance(optimizer, HybridCMuonCanonicalNS4FP32Rescue)

        # Baseline: a malformed counter surface or IO fault at construction
        # DISABLES the session (stderr + traceback) instead of raising into
        # the training launch; only the config-level checks above raise.
        self._baseline_source = (
            "restored_rank0"
            if identity.resume_source_checkpoint != "fresh-start"
            else "fresh"
        )
        try:
            self._baseline = self._snapshot()
        except Exception as error:  # noqa: BLE001 - snapshot fault disables, never raises
            _stderr(
                "guard_window baseline snapshot failed; disabling this rank's stream",
                error,
            )
            self._disabled = True
            self._baseline = None
        first_start = identity.resume_successful_updates + 1
        self._window_start = first_start
        self._window_end = first_start + window_updates - 1
        self._last_snapshot: dict[str, object] | None = None
        self._last_update: int | None = None

    # ------------------------------------------------------------------ #
    # snapshot (int/map reads only)                                       #
    # ------------------------------------------------------------------ #

    def _read_int(self, name: str) -> int:
        value = getattr(self._optimizer, name)
        if type(value) is not int or value < 0:
            raise ValueError(f"{name} must be a nonnegative int, got {value!r}")
        return value

    def _read_map(self, name: str, domain: frozenset[str]) -> dict[str, int]:
        raw_value = getattr(self._optimizer, name)
        if not isinstance(raw_value, dict):
            raise TypeError(f"{name} must be a dict, got {type(raw_value).__name__}")
        raw: dict[object, object] = cast("dict[object, object]", raw_value)
        out: dict[str, int] = {
            key: 0 for key in domain
        }  # sparse known keys normalize to 0
        for key, value in raw.items():
            if type(key) is not str or key not in domain:
                raise ValueError(
                    f"{name} contains unknown key {key!r} (illegal domain)"
                )
            if type(value) is not int or value < 0:
                raise ValueError(
                    f"{name}[{key!r}] must be a nonnegative int, got {value!r}"
                )
            out[key] = value
        return out

    def _snapshot(self) -> dict[str, object]:
        snap: dict[str, object] = {name: self._read_int(name) for name in _R_SCALARS}
        snap["skip_by_role"] = self._read_map("skip_by_role", self._known_roles)
        snap["skip_by_fqn"] = self._read_map("skip_by_fqn", self._known_fqns)
        if self._has_rescue:
            for name in _O_SCALARS:
                snap[name] = self._read_int(name)
            snap["rescue_by_role"] = self._read_map("rescue_by_role", self._known_roles)
        return snap

    # ------------------------------------------------------------------ #
    # window bookkeeping                                                  #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _check_monotonic(prev: dict[str, object], cur: dict[str, object]) -> str | None:
        """Compare a snapshot against the PREVIOUS valid snapshot (or the
        baseline for the first one).  Any scalar decrease, or any decrease
        on any key of a normalized map, returns a reason string; otherwise
        None.  Maps are compared after known-sparse normalization (missing
        key = 0), so a key that vanished between snapshots is a decrease.
        """
        for name, prev_value in prev.items():
            if type(prev_value) is int:
                cur_value = cur.get(name)
                if type(cur_value) is not int or cur_value < prev_value:
                    return f"scalar {name!r} decreased {prev_value} -> {cur_value!r}"
            else:
                prev_map = cast("dict[str, int]", prev_value)
                cur_map_raw = cur.get(name)
                if type(cur_map_raw) is not dict:
                    return f"map {name!r} missing or not a dict in current snapshot"
                cur_map = cast("dict[str, int]", cur_map_raw)
                for key, prev_count in prev_map.items():
                    cur_count = cur_map.get(key, 0)
                    if type(cur_count) is not int or cur_count < prev_count:
                        return f"map {name!r}[{key!r}] decreased {prev_count} -> {cur_count!r}"
        return None

    def _disable(self, why: str, error: BaseException | None = None) -> None:
        """Latch the session DISABLED (root ruling: no per-window recovery).

        The shared DurableJsonlSink can leak one FD per incomplete
        init/close; recovering after such faults window by window would leak
        without bound.  The latch is for the whole session: no further
        formal record is written (an explicit missing tail), and only a new
        session (new process + new session_id) records again.  Never raises.
        """
        self._disabled = True
        self._baseline = None  # never reuse a stale baseline across the gap
        _stderr(f"guard_window {why}; disabling this rank's session stream", error)

    @staticmethod
    def _delta(
        base: dict[str, object], cur: dict[str, object]
    ) -> tuple[dict[str, object], dict[str, str]]:
        delta: dict[str, object] = {}
        status: dict[str, str] = {}

        def _as_int_map(value: dict[object, object], field: str) -> dict[str, int]:
            out: dict[str, int] = {}
            for k, v in value.items():
                if type(v) is not int:
                    raise ValueError(f"map value for {field!r} must be int, got {v!r}")
                out[str(k)] = v
            return out

        for key, c in cur.items():
            b = base.get(key)
            if b is None:
                delta[key] = None
                status[key] = "unknown"
                continue
            if isinstance(c, dict) and isinstance(b, dict):
                cur_map = _as_int_map(cast("dict[object, object]", c), f"delta:{key}")
                base_map = _as_int_map(cast("dict[object, object]", b), f"delta:{key}")
                diff = {k: cur_map[k] - base_map.get(k, 0) for k in cur_map}
                delta[key] = diff
                status[key] = "rollback" if any(v < 0 for v in diff.values()) else "ok"
            else:
                c_obj: object = cast("object", c)  # pyright: ignore[reportUnnecessaryCast]
                b_obj: object = cast("object", b)  # pyright: ignore[reportUnnecessaryCast]
                if type(c_obj) is not int:
                    raise ValueError(
                        "current counter must be int for this scalar field"
                    )
                if type(b_obj) is not int:
                    raise ValueError(
                        "baseline counter must be int for this scalar field"
                    )
                c_int: int = c_obj
                b_int: int = b_obj
                d = c_int - b_int
                delta[key] = d
                status[key] = "rollback" if d < 0 else "ok"
        return delta, status

    def _absolute_history(self) -> dict[str, str]:
        resumed = self._baseline_source == "restored_rank0"
        history = {
            "observations": "rank_provenance",
            "skip_total": "rank_provenance",
            "skip_by_role": "rank_provenance",
            "skip_by_fqn": "rank_provenance",
        }
        if self._has_rescue:
            for name in (*_O_SCALARS, "rescue_by_role"):
                history[name] = "rank0_owner_only" if resumed else "local"
        return history

    def _build_record(
        self,
        start: int,
        end: int,
        baseline: dict[str, object],
        absolute: dict[str, object],
        run_outcome: str | None,
    ) -> dict[str, object]:
        identity = self._identity
        delta, status = self._delta(baseline, absolute)
        last_published = (
            self._last_published() if self._last_published is not None else None
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": identity.run_id,
            "session_id": identity.session_id,
            "rank": identity.rank,
            "world_size": identity.world_size,
            "window_id": f"{identity.session_id}:u{start}-{end}:c{identity.resume_source_checkpoint}",
            "start_update": start,
            "end_update": end,
            "complete": (end - start + 1) == self._window_updates,
            "run_outcome": run_outcome,
            "routing": self._routing,
            "owner_partition": {k: sorted(v) for k, v in self._owner_partition.items()},
            "launch": {
                "resume_source_checkpoint": identity.resume_source_checkpoint,
                "resume_successful_updates": identity.resume_successful_updates,
                "launch_kind": identity.launch_kind,
                "optimizer": self._optimizer_variant,
                "owner_mapping_version": OWNER_MAPPING_VERSION,
                "last_published_checkpoint": last_published,
            },
            "baseline": {"source": self._baseline_source, "counts": baseline},
            "absolute": absolute,
            "delta": delta,
            "delta_status": status,
            "delta_coverage": "complete",
            "absolute_history": self._absolute_history(),
        }

    def _seal(
        self,
        end: int,
        baseline: dict[str, object],
        absolute: dict[str, object],
        run_outcome: str | None,
    ) -> None:
        record = self._build_record(
            self._window_start, end, baseline, absolute, run_outcome
        )
        target = self._rank_dir / f"guard-window-u{self._window_start}-{end}.jsonl"

        def prepare(temp: Path) -> None:
            # The sink owns serialization and closes even when write fails;
            # both fsync and close still precede the atomic link.
            with DurableJsonlSink(temp, fsync_every_records=1) as sink:
                sink.write(record)

        result = publish_atomically(target, self._temp_dir, prepare)
        if result.status is PublicationStatus.TARGET_EXISTS:
            self._disable(
                f"window u{self._window_start}-{end} already exists (duplicate/claim collision)"
            )
            return
        self._baseline = absolute
        self._window_start = end + 1
        self._window_end = self._window_start + self._window_updates - 1

    # ------------------------------------------------------------------ #
    # public API (hook + close)                                           #
    # ------------------------------------------------------------------ #

    def maybe_record(self, observation: SuccessfulTrainingObservation) -> None:
        """Successful-update hook.  Never raises into training."""
        if self._disabled:
            return
        update_no = observation.loop.update.state.successful_updates
        # Update-number continuity: the successful-update hook fires exactly
        # once per successful update, which advances successful_updates by
        # exactly one.  A duplicate, rewind, or skip is explicitly
        # incomparable -> session disable (never a defensive return that
        # keeps sealing later).
        if self._last_update is None:
            if update_no != self._window_start:
                self._disable(
                    f"first observed update {update_no} != window start "
                    f"{self._window_start} (skipped or rewound successful updates)"
                )
                return
        elif update_no != self._last_update + 1:
            self._disable(
                f"update counter not consecutive: {self._last_update} -> {update_no}"
            )
            return
        try:
            snapshot = self._snapshot()
        except Exception as error:  # noqa: BLE001 - disable, never raise
            self._disable(f"snapshot failed at update {update_no}", error)
            return
        # Monotonicity against the PREVIOUS valid snapshot (baseline for the
        # first one): any mid-window counter regression is explicitly
        # incomparable and latches the session; it must never be masked by
        # an end-baseline delta at seal time.
        prev = (
            self._last_snapshot if self._last_snapshot is not None else self._baseline
        )
        if prev is not None:
            reason = self._check_monotonic(prev, snapshot)
            if reason is not None:
                self._disable(f"counter regression at update {update_no}: {reason}")
                return
        self._last_snapshot = snapshot
        self._last_update = update_no
        if update_no == self._window_end:
            baseline = self._baseline
            if baseline is None:
                return  # defensive: cannot happen (baseline set at construction)
            self._baseline = None  # cleared; _seal either re-baselines or disables
            try:
                self._seal(update_no, baseline, snapshot, None)
            except Exception as error:  # noqa: BLE001 - disable, never raise
                self._disable(f"seal failed (u{self._window_start}-{update_no})", error)

    def close(self, run_outcome: str) -> None:
        """End-of-run tail seal.  Never raises; never writes ``complete``."""
        if run_outcome not in _RUN_OUTCOMES:
            report_warning(
                f"guard_window close: invalid run_outcome {run_outcome!r}; using 'unknown'"
            )
            run_outcome = "unknown"
        if self._disabled:
            return
        try:
            if (
                self._baseline is None
                or self._last_snapshot is None
                or self._last_update is None
            ):
                return  # tail unknown: no file, no claim
            if self._last_update < self._window_start:
                return  # nothing unsealed
            baseline = self._baseline
            absolute = self._last_snapshot
            self._baseline = None
            self._seal(self._last_update, baseline, absolute, run_outcome)
        except Exception as error:  # noqa: BLE001 - never change training outcome
            self._disable(f"tail seal failed (update {self._last_update})", error)

    @property
    def disabled(self) -> bool:
        return self._disabled


__all__ = [
    "SCHEMA_VERSION",
    "GuardWindowIdentity",
    "GuardWindowRecorder",
]
