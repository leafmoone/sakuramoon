"""Experimental Hybrid CMuon optimizer backend (dev/verification only).

This module adds an OPTIONAL Hybrid CMuon backend to SakuraMoon training.
It does NOT change the default torchao AdamW8bit path: configs that do not
select a hybrid optimizer name build the exact same AdamW8bit as before.

Algorithm (paper semantics, matching torch.optim.Muon as the reference):
  1. momentum on the raw gradient:  B_t = mu*B_{t-1} + (1-mu)*g_t
  2. Nesterov update:               u_t = (1-mu)*g_t + mu*B_t
  3. split u_t into semantic chunks (for fused tensors)
  4. per-chunk Newton-Schulz orthogonalization (quintic, always BF16)
  5. per-chunk Moonlight scaling:    alpha = lr * 0.2 * sqrt(max(d_out, d_in))
     (optionally alpha *= sqrt(N_chunk) when chunk_rescale_sqrt_n is on)
  6. concatenate the scaled orthogonal chunks back to the original layout
  7. decoupled weight decay + in-place parameter update

The native torch.optim.Muon is used as the numerical reference in the unit
tests; this module does not import its private helpers (it replicates the
public semantics with plain torch ops so it runs on the HCU without any
CUDA-only / Triton-only kernel).
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal, cast

import torch
from torch import nn
from torch.optim import Optimizer
from torchao.optim import AdamW8bit  # pyright: ignore[reportMissingTypeStubs]

from sakuramoon.optim.adamw8bit import IsolatedAdamW8bit
from sakuramoon.optim.groups import (
    ParameterAudit,
    ParameterSpec,
    audit_trainable_parameters,
)
from sakuramoon.optim.stochastic_rounding import StochasticRoundingRNG

if TYPE_CHECKING:
    from sakuramoon.optim.cmuon_forensic import ForensicConfig, ForensicMonitor

MomentumDtype = Literal["bfloat16", "float32"]

# Newton-Schulz quintic coefficients (Keller Jordan's Muon; == the paper).
DEFAULT_NS_COEFFICIENTS: tuple[float, float, float] = (3.4445, -4.7750, 2.0315)
DEFAULT_NS_STEPS = 5
DEFAULT_MOMENTUM = 0.95
DEFAULT_EPS = 1e-7

# Canonical per-role Newton-Schulz-depth keys. One key per semantic role.
# A fused tensor's semantic chunks share the role's ns_steps (v1): the two
# in_proj chunks (gate/up) both use "ffn_in", the six AdaLN chunks all use
# "adaln_shared". This is the stable vocabulary shared with the config schema
# ([optimizer.cmuon_ns]) and the checkpoint manifest (cmuon_ns_steps map).
CMUON_ROLES: tuple[str, ...] = (
    "attention_q",
    "attention_k",
    "attention_v",
    "attention_content_gate",
    "attention_out",
    "ffn_in",
    "ffn_down",
    "adaln_shared",
)


def _default_ns_by_role() -> dict[str, int]:
    """Canonical map: every role -> DEFAULT_NS_STEPS (backward-compatible default)."""
    return {role: DEFAULT_NS_STEPS for role in CMUON_ROLES}


def resolve_ns_map(
    ns_steps_by_role: Mapping[str, int] | None,
    ns_steps: int | None,
) -> dict[str, int]:
    """Build the canonical per-role NS-depth map (every role in CMUON_ROLES).

    - If ``ns_steps_by_role`` is given, it is a partial/complete override map;
      any role it omits falls back to ``ns_steps`` (or DEFAULT_NS_STEPS).
    - If only the scalar ``ns_steps`` is given, every role gets that value
      (backward compatible with the old global ``cmuon_ns_steps``).
    Unknown roles or out-of-range values raise ValueError.
    """
    base = ns_steps if ns_steps is not None else DEFAULT_NS_STEPS
    result = {role: base for role in CMUON_ROLES}
    if ns_steps_by_role is not None:
        unknown = set(ns_steps_by_role) - set(CMUON_ROLES)
        if unknown:
            raise ValueError(f"unknown cmuon ns roles: {sorted(unknown)}")
        for role, steps in ns_steps_by_role.items():
            if not isinstance(steps, int) or not 1 <= steps <= 99:
                raise ValueError(
                    f"cmuon ns_steps for {role} must be an int in [1, 99], got {steps!r}"
                )
            result[role] = steps
    return result


def _momentum_dtype(name: MomentumDtype) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"cmuon_momentum_dtype must be bfloat16 or float32, got {name!r}")


def cmuon_zeroth_power_bf16(
    grad: torch.Tensor,
    ns_steps: int = DEFAULT_NS_STEPS,
    ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization in BF16 (production path).

    Bit-identical to the historical ``cmuon_zeroth_power``: the input is
    cast to BF16, the Frobenius normalization, gram matmuls and quintic
    iterations all run in BF16. Tall matrices are transposed to wide form
    before iteration and transposed back afterwards, so the output always
    has the input's shape. This is the path known to have convergence-
    boundary bit chaos on near-zero-signal inputs (S1/S1-b, D1 round) —
    kept as the fast production path with the post-NS safety ceiling in
    front of it; see ``cmuon_zeroth_power_fp32`` for the rescue path.
    """
    if ns_steps <= 0 or ns_steps >= 100:
        raise ValueError("ns_steps must be in [1, 99]")
    if grad.ndim != 2:
        raise ValueError("cmuon input must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must be a 3-tuple")
    a, b, c = ns_coefficients
    ortho = grad.bfloat16()
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T
    # Normalize the spectral norm to <= 1 (Frobenius norm, clamped by eps).
    ortho = ortho / ortho.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        # gram_update = b*gram + c*(gram @ gram)
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        # ortho = a*ortho + gram_update @ ortho
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


def cmuon_zeroth_power(
    grad: torch.Tensor,
    ns_steps: int = DEFAULT_NS_STEPS,
    ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Alias of the BF16 production path (unchanged semantics)."""
    return cmuon_zeroth_power_bf16(grad, ns_steps, ns_coefficients, eps)


def cmuon_zeroth_power_fp32(
    grad: torch.Tensor,
    ns_steps: int = DEFAULT_NS_STEPS,
    ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS,
    eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Quintic Newton-Schulz orthogonalization in pure FP32 (rescue path).

    Same algorithm and coefficients as the BF16 production path, but the
    entire computation — input cast, Frobenius normalization, gram matmuls
    and all quintic iterations — stays in FP32. There is NO BF16 cast
    anywhere inside this function (the BF16 path's convergence-boundary
    bit chaos, measured at ~20% catastrophic branch rate on pathological
    near-zero-signal inputs, comes from BF16 rounding inside the NS
    iteration; FP32 has ~24x the mantissa and the same iteration is
    deterministically stable on identical inputs).

    The result is FP32 with the input's shape (tall matrices are
    transposed to wide form and back, exactly like the BF16 path).
    Callers decide the final parameter-dtype rounding; this function never
    rounds to BF16 itself.
    """
    if ns_steps <= 0 or ns_steps >= 100:
        raise ValueError("ns_steps must be in [1, 99]")
    if grad.ndim != 2:
        raise ValueError("cmuon input must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("ns_coefficients must be a 3-tuple")
    a, b, c = ns_coefficients
    ortho = grad.float()
    if ortho.dtype is not torch.float32:
        raise TypeError(
            f"cmuon_zeroth_power_fp32 requires an fp32-working input, got {ortho.dtype}"
        )
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T
    # FP32 Frobenius normalization (same clamp semantics as the BF16 path).
    ortho = ortho / ortho.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        # gram_update = b*gram + c*(gram @ gram) — all FP32
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        # ortho = a*ortho + gram_update @ ortho — all FP32
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
    if transposed:
        ortho = ortho.T
    return ortho


def cmuon_zeroth_power_traced(
    grad: torch.Tensor,
    ns_steps: int,
    coefficients: tuple[float, float, float],
    eps: float,
    trace: list[torch.Tensor],
) -> torch.Tensor:
    """Forensic variant of ``cmuon_zeroth_power``: bit-identical arithmetic
    (same ops in the same order; the trace reads are side-effect-free) that
    appends the post-iteration fp32 Frobenius norm of the working matrix to
    ``trace`` after every Newton-Schulz iteration. Used by the fail-closed
    forensic step to answer "which NS iteration blew up, for which tensor".
    """
    if ns_steps <= 0 or ns_steps >= 100:
        raise ValueError("ns_steps must be in [1, 99]")
    if grad.ndim != 2:
        raise ValueError("cmuon input must be a 2D matrix")
    a, b, c = coefficients
    ortho = grad.bfloat16()
    transposed = ortho.size(0) > ortho.size(1)
    if transposed:
        ortho = ortho.T
    ortho = ortho / ortho.norm().clamp(min=eps)
    for _ in range(ns_steps):
        gram = ortho @ ortho.T
        gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
        ortho = torch.addmm(ortho, gram_update, ortho, beta=a)
        trace.append(ortho.float().norm())
    if transposed:
        ortho = ortho.T
    return ortho


def cmuon_moonlight_alpha(
    rows: int,
    cols: int,
    lr: float,
    rescale_sqrt_n: int = 1,
) -> float:
    """Moonlight LR scaling: lr * 0.2 * sqrt(max(d_out, d_in)), per chunk shape.

    Optionally multiplies by sqrt(rescale_sqrt_n) when the independent
    chunk_rescale_sqrt_n switch is on (rescale_sqrt_n = the chunk count).
    This is intentionally separate from chunking itself.
    """
    if rows <= 0 or cols <= 0:
        raise ValueError("chunk dimensions must be positive")
    if rescale_sqrt_n < 1:
        raise ValueError("rescale_sqrt_n must be >= 1")
    alpha = lr * 0.2 * math.sqrt(max(rows, cols))
    if rescale_sqrt_n > 1:
        alpha *= math.sqrt(rescale_sqrt_n)
    return alpha


@dataclass(frozen=True)
class CMuonChunkSpec:
    """Semantic chunk routing for one fused 2D parameter."""

    name: str
    parameter: nn.Parameter
    weight_decay: float
    chunk_count: int
    chunk_dim: int
    roles: tuple[str, ...]
    role: str  # canonical per-role NS-depth key (see CMUON_ROLES)

    def chunk_size(self) -> int:
        extent = self.parameter.shape[self.chunk_dim]
        if extent % self.chunk_count != 0:
            raise ValueError(
                f"chunk dim {self.chunk_dim} size {extent} of {self.name} is not "
                f"divisible by chunk_count {self.chunk_count}"
            )
        return extent // self.chunk_count


@dataclass(frozen=True)
class CMuonConfig:
    """Hybrid CMuon algorithm settings (independent of the AdamW policy)."""

    lr: float
    momentum: float = DEFAULT_MOMENTUM
    nesterov: bool = True
    # Canonical per-role Newton-Schulz depth map: every key in CMUON_ROLES ->
    # an int in [1, 99]. Different semantic roles may use different depths
    # (per-spec NS). The default (all roles -> DEFAULT_NS_STEPS) keeps the old
    # global behavior. Build it with resolve_ns_map() from a scalar or a
    # partial [optimizer.cmuon_ns] override map.
    ns_steps_by_role: Mapping[str, int] = field(default_factory=_default_ns_by_role)
    ns_coefficients: tuple[float, float, float] = DEFAULT_NS_COEFFICIENTS
    eps: float = DEFAULT_EPS
    momentum_dtype: MomentumDtype = "bfloat16"
    chunk_rescale_sqrt_n: bool = False
    qkv_group_rescale: bool = False

    def __post_init__(self) -> None:
        if type(self.lr) is not float or not math.isfinite(self.lr) or self.lr <= 0.0:
            raise ValueError("cmuon lr must be a positive finite float")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("cmuon momentum must be in [0, 1)")
        if self.qkv_group_rescale:
            # v1: GQA Q/K/V are physically separate (no fused QKV chunk); the
            # sqrt(3) rescale is reserved and must stay off.
            raise ValueError("qkv_group_rescale must be false in v1")
        _momentum_dtype(self.momentum_dtype)
        for role in CMUON_ROLES:
            if role not in self.ns_steps_by_role:
                raise ValueError(f"cmuon ns_steps_by_role missing role: {role}")
            steps = self.ns_steps_by_role[role]
            if not isinstance(steps, int) or not 1 <= steps <= 99:
                raise ValueError(
                    f"cmuon ns_steps for {role} must be an int in [1, 99], got {steps!r}"
                )

    def ns_steps_for_role(self, role: str) -> int:
        if role not in self.ns_steps_by_role:
            raise ValueError(f"unknown cmuon role: {role}")
        return self.ns_steps_by_role[role]

    def canonical_ns_map(self) -> dict[str, int]:
        """Per-role NS map keyed by CMUON_ROLES (stored in the ckpt manifest)."""
        return {role: self.ns_steps_by_role[role] for role in CMUON_ROLES}


@dataclass(frozen=True)
class NSTelemetrySample:
    """One telemetry reading for a representative role (every N updates)."""

    role: str
    ns_output_rms: float
    applied_delta_rms: float
    nonfinite_count: int


class NSSafetyTelemetry:
    """Low-overhead Newton-Schulz safety telemetry.

    Samples a small set of REpresentative params (one per canonical role) and
    accumulates their NS-output / applied-delta statistics ON THE DEVICE
    (no per-step CPU sync). The device accumulators are read to CPU only every
    ``log_every_n`` steps (one batched sync), so the per-step overhead is a
    handful of device reductions and the CPU sync is amortized over N steps.

    Per representative role it records:
      - ns_output_rms:     RMS of the NS output (before Moonlight scaling)
      - applied_delta_rms: RMS of the applied parameter delta (-alpha * NS)
      - nonfinite_count:   count of nonfinite values in the NS output / delta

    It is OPT-IN (the default optimizer has no telemetry). It is meant to be
    wired into a long run to catch a slow NS instability (e.g. a spectrum
    drift that pushes a shape's NS over its convergence boundary, as the ns5
    k/v/AdaLN blow-up) without the cost of per-param per-step CPU syncs.
    """

    def __init__(
        self,
        representative: Mapping[str, CMuonChunkSpec],
        *,
        device,
        log_every_n: int = 100,
        logger=None,
        update_offset: int = 0,
    ) -> None:
        self.roles: tuple[str, ...] = tuple(representative)
        self._device = device
        self.log_every_n = max(1, int(log_every_n))
        self._logger = logger
        # The absolute update number this telemetry instance was constructed
        # at (e.g. the resume update). Log lines report
        # update_offset + relative steps, so a resumed run keeps logging the
        # real update numbers. Settable after construction.
        self.update_offset = int(update_offset)
        # device accumulators: [ns_sq, ns_n, delta_sq, nonfinite] per role
        self._acc = {
            role: torch.zeros(4, device=device, dtype=torch.float32)
            for role in self.roles
        }
        self._steps = 0

    def wants(self, role: str) -> bool:
        return role in self._acc

    def record(
        self, role: str, ns_output: torch.Tensor, applied_delta: torch.Tensor
    ) -> None:
        """Device-side accumulation (no CPU sync). Called once per
        representative role per step, during the CMuon update."""
        acc = self._acc.get(role)
        if acc is None:
            return
        q = ns_output.float()
        d = applied_delta.float()
        acc[0] += q.pow(2).sum()
        acc[1] += q.numel()
        acc[2] += d.pow(2).sum()
        acc[3] += (~torch.isfinite(q) | ~torch.isfinite(d)).sum()

    def step(self) -> list[NSTelemetrySample] | None:
        """Advance the step counter. Every log_every_n steps, read the
        accumulators to CPU (one batched sync) and return the samples;
        otherwise return None (no sync)."""
        self._steps += 1
        if self._steps % self.log_every_n != 0:
            return None
        stacked = torch.stack([self._acc[r] for r in self.roles]).detach().cpu()
        samples: list[NSTelemetrySample] = []
        for i, role in enumerate(self.roles):
            ns_sq, ns_n, delta_sq, nonfinite = stacked[i].tolist()
            ns_rms = math.sqrt(ns_sq / ns_n) if ns_n > 0 else 0.0
            delta_rms = math.sqrt(delta_sq / ns_n) if ns_n > 0 else 0.0
            samples.append(NSTelemetrySample(role, ns_rms, delta_rms, int(nonfinite)))
        for r in self.roles:
            self._acc[r].zero_()
        if self._logger is not None:
            for s in samples:
                self._logger(
                    f"[cmuon-ns-telemetry] update={self._steps + self.update_offset} "
                    f"role={s.role} "
                    f"ns_rms={s.ns_output_rms:.3e} delta_rms={s.applied_delta_rms:.3e} "
                    f"nonfinite={s.nonfinite_count}"
                )
        return samples


def select_representative_specs(
    routing: CMuonRouting, roles: Sequence[str] | None = None
) -> dict[str, CMuonChunkSpec]:
    """Pick one representative spec per canonical role (the first matching).

    Used to build a low-overhead NSSafetyTelemetry over a small subset instead
    of all 141 CMuon params.
    """
    if roles is None:
        roles = CMUON_ROLES
    rep: dict[str, CMuonChunkSpec] = {}
    for spec in routing.cmuon_specs:
        if spec.role in roles and spec.role not in rep:
            rep[spec.role] = spec
    return {role: rep[role] for role in roles if role in rep}


def build_ns_safety_telemetry(
    routing: CMuonRouting,
    *,
    device,
    log_every_n: int = 100,
    logger=None,
    roles: Sequence[str] | None = None,
    update_offset: int = 0,
) -> NSSafetyTelemetry:
    """Convenience: one representative spec per role -> NSSafetyTelemetry."""
    return NSSafetyTelemetry(
        select_representative_specs(routing, roles),
        device=device,
        log_every_n=log_every_n,
        logger=logger,
        update_offset=update_offset,
    )


def _cmuon_allowlist() -> tuple[tuple[re.Pattern[str], int, tuple[str, ...], str], ...]:
    # (FQN template, chunk_count, per-chunk roles, canonical role).
    # chunk_dim is always 0 (the output dim). The canonical role selects the
    # per-role Newton-Schulz depth (see CMUON_ROLES / [optimizer.cmuon_ns]).
    slot = r"dit\.blocks\.slot_\d+"
    return (
        (re.compile(rf"^{slot}\.attention\.q_proj\.weight$"), 1, ("q",), "attention_q"),
        (re.compile(rf"^{slot}\.attention\.k_proj\.weight$"), 1, ("k",), "attention_k"),
        (re.compile(rf"^{slot}\.attention\.v_proj\.weight$"), 1, ("v",), "attention_v"),
        (
            re.compile(rf"^{slot}\.attention\.content_gate\.weight$"),
            1,
            ("content_gate",),
            "attention_content_gate",
        ),
        (
            re.compile(rf"^{slot}\.attention\.out_proj\.weight$"),
            1,
            ("out_proj",),
            "attention_out",
        ),
        (re.compile(rf"^{slot}\.mlp\.in_proj\.weight$"), 2, ("gate", "up"), "ffn_in"),
        (re.compile(rf"^{slot}\.mlp\.down_proj\.weight$"), 1, ("down",), "ffn_down"),
        (
            re.compile(r"^dit\.conditioner\.shared_block_projection\.weight$"),
            6,
            (
                "attention_scale",
                "attention_shift",
                "attention_gate",
                "mlp_scale",
                "mlp_shift",
                "mlp_gate",
            ),
            "adaln_shared",
        ),
    )


@dataclass(frozen=True)
class CMuonRouting:
    """The disjoint+complete split of the trainable params into CMuon / AdamW."""

    full_audit: ParameterAudit
    cmuon_specs: tuple[CMuonChunkSpec, ...]
    adamw_specs: tuple[ParameterSpec, ...]

    @property
    def cmuon_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.cmuon_specs)

    @property
    def adamw_names(self) -> tuple[str, ...]:
        return tuple(spec.name for spec in self.adamw_specs)

    def routing_manifest(self) -> dict[str, object]:
        return {
            "cmuon": [
                {
                    "name": spec.name,
                    "shape": list(spec.parameter.shape),
                    "dtype": str(spec.parameter.dtype),
                    "chunk_count": spec.chunk_count,
                    "chunk_dim": spec.chunk_dim,
                    "roles": list(spec.roles),
                    "role": spec.role,
                    "weight_decay": spec.weight_decay,
                }
                for spec in self.cmuon_specs
            ],
            "adamw": [
                {
                    "name": spec.name,
                    "group": spec.group,
                    "weight_decay": spec.weight_decay,
                }
                for spec in self.adamw_specs
            ],
            "counts": {
                "total": len(self.full_audit.specs),
                "cmuon": len(self.cmuon_specs),
                "adamw": len(self.adamw_specs),
            },
        }


def route_cmuon_parameters(
    module: nn.Module,
    *,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
    exclude_roles: Sequence[str] = (),
) -> CMuonRouting:
    """Route the trainable params into the CMuon allowlist and the AdamW8bit rest.

    Uses the canonical-FQN allowlist (never an ndim==2 heuristic). Asserts the
    split is disjoint and complete (CMuon + AdamW == all trainable params).

    ``exclude_roles`` (forensic routing ablations only): canonical roles whose
    allowlisted params are deliberately routed back to the AdamW8bit fallback
    instead of CMuon (e.g. the safe-routing forensic candidate excludes
    attention_k / attention_v / adaln_shared). The split stays complete.
    """
    unknown = set(exclude_roles) - set(CMUON_ROLES)
    if unknown:
        raise ValueError(f"unknown cmuon exclude roles: {sorted(unknown)}")
    excluded = set(exclude_roles)
    full_audit = audit_trainable_parameters(
        module,
        matrix_weight_decay=matrix_weight_decay,
        sensitive_weight_decay=sensitive_weight_decay,
    )
    allowlist = _cmuon_allowlist()
    cmuon_specs: list[CMuonChunkSpec] = []
    adamw_specs: list[ParameterSpec] = []
    for spec in full_audit.specs:
        matched = None
        for pattern, chunk_count, roles, role in allowlist:
            if pattern.match(spec.name):
                matched = (chunk_count, roles, role)
                break
        if matched is None or matched[2] in excluded:
            adamw_specs.append(spec)
            continue
        chunk_count, roles, role = matched
        param = spec.parameter
        if param.ndim != 2:
            raise ValueError(
                f"cmuon parameter must be 2D, got {param.ndim}D: {spec.name}"
            )
        cmuon_specs.append(
            CMuonChunkSpec(
                name=spec.name,
                parameter=param,
                weight_decay=spec.weight_decay,
                chunk_count=chunk_count,
                chunk_dim=0,
                roles=roles,
                role=role,
            )
        )
    # disjoint + complete
    cmuon_names = {spec.name for spec in cmuon_specs}
    adamw_names = {spec.name for spec in adamw_specs}
    if cmuon_names & adamw_names:
        raise ValueError(
            "cmuon/adamw routing overlap: "
            + ", ".join(sorted(cmuon_names & adamw_names))
        )
    if len(cmuon_specs) + len(adamw_specs) != len(full_audit.specs):
        raise ValueError("cmuon/adamw routing is not complete")
    return CMuonRouting(
        full_audit=full_audit,
        cmuon_specs=tuple(cmuon_specs),
        adamw_specs=tuple(adamw_specs),
    )


def _cmuon_update(
    param: nn.Parameter,
    grad: torch.Tensor,
    buf: torch.Tensor,
    spec: CMuonChunkSpec,
    cfg: CMuonConfig,
    *,
    ns_telemetry: NSSafetyTelemetry | None = None,
) -> torch.Tensor:
    """One CMuon update for a single parameter (in-place on param).

    Returns the update tensor (for RMS telemetry), in the parameter's dtype.
    The in-place parameter update runs under torch.no_grad() (standard
    optimizer-step semantics; the param is a leaf that requires grad).
    """
    with torch.no_grad():
        return _cmuon_update_impl(
            param, grad, buf, spec, cfg, ns_telemetry=ns_telemetry
        )


def _cmuon_update_impl(
    param: nn.Parameter,
    grad: torch.Tensor,
    buf: torch.Tensor,
    spec: CMuonChunkSpec,
    cfg: CMuonConfig,
    *,
    ns_telemetry: NSSafetyTelemetry | None = None,
) -> torch.Tensor:
    mu = cfg.momentum
    # 1. momentum on the raw gradient, in the configured momentum dtype.
    grad_md = grad.to(buf.dtype)
    buf.lerp_(grad_md, 1.0 - mu)  # buf = mu*buf + (1-mu)*grad_md
    # 2. Nesterov update (in the momentum dtype).
    if cfg.nesterov:
        update = grad_md.lerp(buf, mu)  # (1-mu)*grad + mu*buf
    else:
        update = buf
    # 3. split into semantic chunks along chunk_dim.
    chunk_size = spec.chunk_size()
    if spec.chunk_count == 1:
        chunks = (update,)
    else:
        chunks = tuple(update.split(chunk_size, dim=spec.chunk_dim))
    # 4. per-chunk Newton-Schulz + per-chunk Moonlight scaling.
    # The NS depth is per-role (per-spec): all chunks of this fused tensor
    # share spec.role's depth (v1; e.g. the two in_proj chunks both use
    # "ffn_in", the six AdaLN chunks all use "adaln_shared").
    ns_steps = cfg.ns_steps_for_role(spec.role)
    rescale_sqrt_n = spec.chunk_count if cfg.chunk_rescale_sqrt_n else 1
    want_tel = ns_telemetry is not None and ns_telemetry.wants(spec.role)
    ortho_chunks: list[torch.Tensor] = []
    ns_chunks: list[torch.Tensor] | None = [] if want_tel else None
    for chunk in chunks:
        ns = cmuon_zeroth_power(chunk, ns_steps, cfg.ns_coefficients, cfg.eps)
        rows, cols = chunk.shape
        alpha = cmuon_moonlight_alpha(rows, cols, cfg.lr, rescale_sqrt_n)
        if want_tel:
            assert ns_chunks is not None
            ns_chunks.append(ns)
        # Scale in BF16 (the NS dtype) exactly like the native Muon
        # (param.add_(ns, alpha=-adjusted_lr)); the in-place add_ below
        # upcasts to the parameter dtype as needed.
        ortho_chunks.append((-alpha) * ns)
    update_ortho = (
        torch.cat(ortho_chunks, dim=spec.chunk_dim)
        if len(ortho_chunks) > 1
        else ortho_chunks[0]
    )
    if want_tel:
        assert ns_chunks is not None
        ns_full = (
            torch.cat(ns_chunks, dim=spec.chunk_dim)
            if len(ns_chunks) > 1
            else ns_chunks[0]
        )
        ns_telemetry.record(spec.role, ns_full, update_ortho)
    # 5. decoupled weight decay + in-place update.
    if spec.weight_decay != 0.0:
        param.mul_(1.0 - cfg.lr * spec.weight_decay)
    param.add_(update_ortho)
    return update_ortho


def _cmuon_update_phase1(
    param: nn.Parameter,
    grad: torch.Tensor,
    buf: torch.Tensor,
    spec: CMuonChunkSpec,
    cfg: CMuonConfig,
    *,
    ns_telemetry: NSSafetyTelemetry | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    """Forensic phase 1: compute the momentum candidate, Nesterov update,
    per-chunk NS outputs and the applied delta for one parameter WITHOUT
    writing the momentum buffer or the parameter.

    Element-wise this is the exact arithmetic of ``_cmuon_update_impl``
    (same lerp formulas, same NS ops); only the write timing differs.
    Returns (buf_candidate, nesterov, ns_full, delta, ns_trace) where
    ns_trace holds the per-NS-iteration fp32 norms (device tensors).
    """
    mu = cfg.momentum
    with torch.no_grad():
        grad_md = grad.to(buf.dtype)
        buf_candidate = buf.lerp(grad_md, 1.0 - mu)
        if cfg.nesterov:
            nesterov = grad_md.lerp(buf_candidate, mu)
        else:
            nesterov = buf_candidate
        chunk_size = spec.chunk_size()
        if spec.chunk_count == 1:
            chunks = (nesterov,)
        else:
            chunks = tuple(nesterov.split(chunk_size, dim=spec.chunk_dim))
        ns_steps = cfg.ns_steps_for_role(spec.role)
        rescale_sqrt_n = spec.chunk_count if cfg.chunk_rescale_sqrt_n else 1
        ortho_chunks: list[torch.Tensor] = []
        ns_chunks: list[torch.Tensor] = []
        ns_trace: list[torch.Tensor] = []
        for chunk in chunks:
            ns = cmuon_zeroth_power_traced(
                chunk, ns_steps, cfg.ns_coefficients, cfg.eps, ns_trace
            )
            rows, cols = chunk.shape
            alpha = cmuon_moonlight_alpha(rows, cols, cfg.lr, rescale_sqrt_n)
            ns_chunks.append(ns)
            ortho_chunks.append((-alpha) * ns)
        delta = (
            torch.cat(ortho_chunks, dim=spec.chunk_dim)
            if len(ortho_chunks) > 1
            else ortho_chunks[0]
        )
        ns_full = (
            torch.cat(ns_chunks, dim=spec.chunk_dim)
            if len(ns_chunks) > 1
            else ns_chunks[0]
        )
        if ns_telemetry is not None and ns_telemetry.wants(spec.role):
            ns_telemetry.record(spec.role, ns_full, delta)
        del param
    return buf_candidate, nesterov, ns_full, delta, ns_trace


def _cmuon_update_phase2(
    param: nn.Parameter,
    buf: torch.Tensor,
    buf_candidate: torch.Tensor,
    delta: torch.Tensor,
    spec: CMuonChunkSpec,
    cfg: CMuonConfig,
) -> None:
    """Forensic phase 2: commit the momentum buffer and apply the (already
    safety-checked) parameter update. Same order as the original path
    (weight decay first, then the in-place add)."""
    with torch.no_grad():
        buf.copy_(buf_candidate)
        if spec.weight_decay != 0.0:
            param.mul_(1.0 - cfg.lr * spec.weight_decay)
        param.add_(delta)


class HybridCMuon:
    """Hybrid optimizer: torchao AdamW8bit for the non-allowlisted params +
    chunked CMuon for the semantic allowlist.

    Exposes the same interface as IsolatedAdamW8bit so the training loop,
    scheduler, and checkpoint path treat it uniformly:
      .step() / .zero_grad(set_to_none) / .state_dict() / .load_state_dict()
      .optimizer (the inner AdamW8bit, for accelerate.prepare)
      .audit (the FULL parameter audit, for optimizer-coverage validation)
      .sr_rng (the isolated stochastic-rounding RNG for the AdamW part)
    """

    def __init__(
        self,
        *,
        routing: CMuonRouting,
        cfg: CMuonConfig,
        adamw: IsolatedAdamW8bit,
        sr_rng: StochasticRoundingRNG,
        ns_telemetry: NSSafetyTelemetry | None = None,
        forensic: ForensicMonitor | None = None,
    ) -> None:
        self.routing = routing
        self.cfg = cfg
        self.adamw = adamw
        self.audit = routing.full_audit
        self.sr_rng = sr_rng
        # Optional low-overhead NS safety telemetry (opt-in; default off keeps
        # the update path bit-identical). See NSSafetyTelemetry.
        self.ns_telemetry = ns_telemetry
        # Optional fail-closed forensic monitor (root-cause round). When set,
        # the CMuon part runs the two-phase step: compute+validate all
        # candidates first, commit only when every spec passes. Arithmetic is
        # element-wise identical to the single-phase path.
        self.forensic = forensic
        self._forensic_step0_checked = False
        # The inner torch optimizer exposed to accelerate.prepare is the
        # AdamW8bit (it owns the AdamW params). The CMuon params are updated
        # in-place here; their gradients are synced by DDP on the model.
        self.optimizer = adamw.optimizer
        self._momenta: dict[nn.Parameter, torch.Tensor] = {}
        # Eagerly allocate the momentum buffers so the state is present from
        # step 0 (deterministic checkpoint resume).
        for spec in routing.cmuon_specs:
            self._momenta[spec.parameter] = torch.zeros_like(
                spec.parameter, dtype=_momentum_dtype(cfg.momentum_dtype)
            )
        # AdamW8bit -> Hybrid transition record (set by _load_adamw_transition).
        self.transition_from_adamw8bit = False
        self.transition_preserved_adamw_params = 0
        self.transition_dropped_cmuon_params = 0

    # -- interface ---------------------------------------------------------

    def _sync_learning_rate(self) -> None:
        """Keep the CMuon Moonlight scale on the governed learning rate.

        The production LR scheduler (and checkpoint restore) update only the
        inner torch optimizer's parameter groups. The CMuon part scales its
        updates with ``cfg.lr`` (Moonlight) and its weight-decay term, so it
        must observe the same scheduled rate. Sync from the inner groups at
        every step (a no-op float read; the groups always share one lr).
        Reads ``self.optimizer`` (the attribute production rebinds after
        accelerator.prepare) so scheduler writes and this read always agree.
        """
        groups = self.optimizer.param_groups
        if not groups:
            return
        raw = groups[0].get("lr")
        if type(raw) is torch.Tensor:
            lr = float(raw.item())
        elif type(raw) is float:
            lr = raw
        else:
            return
        if math.isfinite(lr) and lr > 0.0 and lr != self.cfg.lr:
            self.cfg = replace(self.cfg, lr=lr)

    def _validate_finite_gradients(self) -> None:
        gradients = [
            spec.parameter.grad
            for spec in self.audit.specs
            if spec.parameter.grad is not None
        ]
        if not gradients:
            return
        device = gradients[0].device
        if any(g.device != device for g in gradients):
            raise ValueError("all gradients must share one device")
        finite = torch.ones((), device=device, dtype=torch.bool)
        for g in gradients:
            finite.logical_and_(torch.isfinite(g).all())
        if not bool(finite.item()):
            raise FloatingPointError("hybrid optimizer received a nonfinite gradient")

    def step(self) -> None:
        self._sync_learning_rate()
        self._validate_finite_gradients()
        # AdamW part (its params), under the isolated SR RNG. Route through
        # self.optimizer (the attribute production rebinds after
        # accelerator.prepare) so scheduler writes and this step always see
        # the same torch optimizer, exactly like IsolatedAdamW8bit.
        self.sr_rng.run_step(self.optimizer.step)
        # CMuon part.
        if self.forensic is not None:
            self._cmuon_step_all_forensic()
        else:
            self._cmuon_step_all()
        # Optional telemetry: advance + (every N steps) batched read. No-op
        # (and no sync) when telemetry is disabled or off-cycle.
        if self.ns_telemetry is not None:
            self.ns_telemetry.step()

    def _cmuon_step_all_forensic(self) -> None:
        """Two-phase fail-closed CMuon step (forensic runs only).

        Phase 1 computes every spec's momentum candidate / Nesterov / NS /
        applied delta and records full per-spec statistics + rank
        fingerprints; ``collect_and_compare`` (one batched CPU sync + small
        all_gathers + one all_reduce) then decides. Phase 2 commits momentum
        and parameters only when the verdict is clean. If the guard trips,
        EVERY rank raises CMuonSafetyError at the same step, after the main
        rank dumps the crash state (the offending spec's phase-1 tensors are
        stashed for the dump).
        """
        mon = self.forensic
        if mon is None:  # defensive: step() only routes here when set
            raise RuntimeError("forensic step called without a forensic monitor")
        if not self._forensic_step0_checked:
            mon.check_initial_momenta(self._momenta)
            self._forensic_step0_checked = True
        stashed: list[tuple[int, dict[str, torch.Tensor]]] = []
        for i, spec in enumerate(self.routing.cmuon_specs):
            grad = spec.parameter.grad
            if grad is None:
                mon.record_spec(
                    i, grad=None, momentum=None, nesterov=None, ns=None, delta=None
                )
                continue
            buf = self._momenta[spec.parameter]
            buf_cand, nesterov, ns_full, delta, ns_trace = _cmuon_update_phase1(
                spec.parameter,
                grad,
                buf,
                spec,
                self.cfg,
                ns_telemetry=self.ns_telemetry,
            )
            mon.record_spec(
                i,
                grad=grad,
                momentum=buf_cand,
                nesterov=nesterov,
                ns=ns_full,
                delta=delta,
                ns_trace=ns_trace,
            )
            stashed.append(
                (
                    i,
                    {
                        "grad": grad.detach(),
                        "momentum": buf_cand.detach(),
                        "nesterov": nesterov.detach(),
                        "ns_output": ns_full.detach(),
                        "delta": delta.detach(),
                    },
                )
            )
        mon.collect_and_compare(self.cfg.lr, stashed=stashed)
        # Phase 2: commit (only reached when the verdict is clean; a trip
        # raised inside collect_and_compare on every rank).
        for i, tensors in stashed:
            spec = self.routing.cmuon_specs[i]
            _cmuon_update_phase2(
                spec.parameter,
                self._momenta[spec.parameter],
                tensors["momentum"],
                tensors["delta"],
                spec,
                self.cfg,
            )
        mon.record_param_after()

    def _cmuon_step_all(self) -> None:
        for spec in self.routing.cmuon_specs:
            param = spec.parameter
            grad = param.grad
            if grad is None:
                continue
            buf = self._momenta[param]
            _cmuon_update(
                param, grad, buf, spec, self.cfg, ns_telemetry=self.ns_telemetry
            )

    def zero_grad(self, *, set_to_none: bool) -> None:
        if not set_to_none:
            raise ValueError("hybrid optimizer zero_grad requires set_to_none=True")
        self.adamw.zero_grad(set_to_none=True)
        for spec in self.routing.cmuon_specs:
            spec.parameter.grad = None

    # -- state -------------------------------------------------------------

    def _cmuon_state(self) -> dict[str, object]:
        return {
            "momenta": {
                spec.name: self._momenta[spec.parameter].detach().cpu()
                for spec in self.routing.cmuon_specs
            },
            "ns_steps": self.cfg.canonical_ns_map(),
            "ns_coefficients": list(self.cfg.ns_coefficients),
            "momentum": self.cfg.momentum,
            "nesterov": self.cfg.nesterov,
            "eps": self.cfg.eps,
            "momentum_dtype": self.cfg.momentum_dtype,
            "chunk_rescale_sqrt_n": self.cfg.chunk_rescale_sqrt_n,
            "qkv_group_rescale": self.cfg.qkv_group_rescale,
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "optimizer": self.adamw.optimizer.state_dict(),
            "sr_rng": self.sr_rng.state_dict(),
            "cmuon": self._cmuon_state(),
            "routing": self.routing.routing_manifest(),
            "transition": {
                "from_adamw8bit": self.transition_from_adamw8bit,
                "preserved_adamw_params": self.transition_preserved_adamw_params,
                "dropped_cmuon_params": self.transition_dropped_cmuon_params,
                "note": (
                    "AdamW8bit->Hybrid: AdamW state preserved per-FQN, CMuon "
                    "momentum from zero, no fake 2nd-moment->Muon conversion"
                    if self.transition_from_adamw8bit
                    else "native hybrid state"
                ),
            },
            "hybrid_cmuon_schema_version": 1,
        }

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        if not isinstance(state_dict, dict):
            raise TypeError("hybrid optimizer state must be a mapping")
        if "guard" in state_dict:
            # A guarded canonical checkpoint must not be silently downgraded
            # into the (retired) unguarded candidate.
            raise ValueError(
                "optimizer state carries a guard section: it belongs to the "
                "guarded canonical candidate and cannot be loaded into the "
                "unguarded hybrid CMuon"
            )
        if "cmuon" in state_dict:
            # Hybrid -> Hybrid: state-exact (AdamW state + CMuon momentum).
            self.adamw.load_state_dict(
                {
                    "optimizer": state_dict.get("optimizer"),
                    "sr_rng": state_dict.get("sr_rng"),
                }
            )
            cmuon_state = state_dict.get("cmuon")
            if not isinstance(cmuon_state, dict):
                raise TypeError("hybrid CMuon state must be a mapping")
            self._load_cmuon_state(cmuon_state)
            self.transition_from_adamw8bit = False
            return
        # AdamW8bit -> Hybrid transition: the checkpoint is a pure AdamW8bit
        # IsolatedAdamW8bit state_dict ("optimizer" + "sr_rng", no "cmuon").
        # Preserve the AdamW state per-FQN for the 148 AdamW params, discard the
        # 141 CMuon params' AdamW moments (NO fake 2nd-moment -> Muon conversion),
        # keep the CMuon momentum at zero, and record the transition.
        self._load_adamw_transition(state_dict)

    def _load_adamw_transition(self, state_dict: dict[str, object]) -> None:
        optimizer_state = state_dict.get("optimizer")
        sr_state = state_dict.get("sr_rng")
        if not isinstance(optimizer_state, dict):
            raise TypeError("AdamW->Hybrid transition requires the optimizer state")
        if not isinstance(sr_state, dict):
            raise TypeError("AdamW->Hybrid transition requires the SR RNG state")
        # Map each FQN to the baseline's torch state id. build_adamw8bit builds
        # param_groups as [matrix_decay group, sensitive_no_decay group], each
        # in name-sorted order, so torch assigns state ids by FLATTENED group
        # order: decay ids 0..len(decay)-1, then sensitive ids
        # len(decay)..N-1. (The full audit is name-sorted ACROSS both groups,
        # which is a different order — never use its index as a state id.)
        name_to_baseline_id: dict[str, int] = {}
        for rank, spec in enumerate(self.audit.decay):
            name_to_baseline_id[spec.name] = rank
        offset = len(self.audit.decay)
        for rank, spec in enumerate(self.audit.sensitive):
            name_to_baseline_id[spec.name] = offset + rank
        # We set the inner AdamW8bit's live state directly (keyed by parameter
        # objects) rather than via load_state_dict, because the baseline covers
        # all params (multiple groups) while the inner covers only the AdamW
        # subset — load_state_dict's per-group size check rejects the
        # mismatched group structure.
        saved_state = optimizer_state.get("state")
        if not isinstance(saved_state, dict):
            raise TypeError("AdamW->Hybrid transition: optimizer state missing 'state'")
        inner_state = self.adamw.optimizer.state
        preserved = 0
        for spec in self.routing.adamw_specs:
            baseline_id = name_to_baseline_id.get(spec.name)
            if baseline_id is None:
                raise ValueError(
                    f"AdamW->Hybrid transition: {spec.name} not in the baseline audit"
                )
            if baseline_id in saved_state:
                inner_state[spec.parameter] = saved_state[baseline_id]
                preserved += 1
        # Restore the quantized moments onto the parameter HCU.
        self.adamw._move_quantized_state_to_parameter_devices()
        self.sr_rng.load_state_dict(sr_state)
        # CMuon momentum stays at zero (initialized in __init__); no conversion.
        self.transition_from_adamw8bit = True
        self.transition_preserved_adamw_params = preserved
        self.transition_dropped_cmuon_params = len(self.routing.cmuon_specs)

    def _load_cmuon_state(self, cmuon_state: dict[str, object]) -> None:
        meta_keys = (
            "ns_steps",
            "ns_coefficients",
            "momentum",
            "nesterov",
            "eps",
            "momentum_dtype",
            "chunk_rescale_sqrt_n",
            "qkv_group_rescale",
        )
        for key in meta_keys:
            if key not in cmuon_state:
                raise ValueError(f"hybrid CMuon state missing metadata key: {key}")
        if cmuon_state["momentum_dtype"] != self.cfg.momentum_dtype:
            raise ValueError(
                "hybrid CMuon momentum dtype mismatch: "
                f"checkpoint={cmuon_state['momentum_dtype']!r} "
                f"runtime={self.cfg.momentum_dtype!r}"
            )
        # Per-role NS map: canonical dict keyed by CMUON_ROLES. Legacy checkpoints
        # (pre per-spec NS) store a scalar int, treated as every-role-equal so a
        # scalar-5 checkpoint matches a config where all roles resolve to 5.
        saved_ns = cmuon_state["ns_steps"]
        if isinstance(saved_ns, dict):
            saved_ns_map: dict[str, int] = {str(k): int(v) for k, v in saved_ns.items()}
        elif isinstance(saved_ns, int):
            saved_ns_map = {role: saved_ns for role in CMUON_ROLES}
        else:
            raise TypeError(
                "hybrid CMuon ns_steps must be a role->int mapping "
                f"(or a legacy int), got {type(saved_ns).__name__}"
            )
        runtime_ns_map = self.cfg.canonical_ns_map()
        if saved_ns_map != runtime_ns_map:
            diff = {
                role: (saved_ns_map.get(role), runtime_ns_map[role])
                for role in CMUON_ROLES
                if saved_ns_map.get(role) != runtime_ns_map[role]
            }
            raise ValueError(
                "hybrid CMuon per-role ns_steps mismatch (optimizer-state semantic "
                f"incompatibility): {diff}"
            )
        if cmuon_state["chunk_rescale_sqrt_n"] != self.cfg.chunk_rescale_sqrt_n:
            raise ValueError("hybrid CMuon chunk_rescale_sqrt_n mismatch")
        if cmuon_state["qkv_group_rescale"] != self.cfg.qkv_group_rescale:
            raise ValueError("hybrid CMuon qkv_group_rescale mismatch")
        momenta = cmuon_state.get("momenta")
        if not isinstance(momenta, dict):
            raise TypeError("hybrid CMuon momenta must be a mapping")
        for spec in self.routing.cmuon_specs:
            tensor = momenta.get(spec.name)
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"hybrid CMuon momentum missing/invalid for {spec.name}"
                )
            if tuple(tensor.shape) != tuple(spec.parameter.shape):
                raise ValueError(
                    f"hybrid CMuon momentum shape mismatch for {spec.name}"
                )
            target = self._momenta[spec.parameter]
            target.copy_(tensor.to(dtype=target.dtype, device=target.device))

    def optimizer_state_bytes(self) -> int:
        """Theoretical persistent optimizer-state bytes (AdamW + CMuon)."""
        adamw_bytes = 0
        for spec in self.adamw.audit.specs:
            state = self.adamw.optimizer.state.get(spec.parameter)
            if not state:
                continue
            for name, value in state.items():
                if name == "step":
                    continue
                if torch.is_tensor(value):
                    adamw_bytes += value.numel() * value.element_size()
        cmuon_bytes = sum(
            buf.numel() * buf.element_size() for buf in self._momenta.values()
        )
        return adamw_bytes + cmuon_bytes


def _build_adamw8bit_for_specs(
    specs: tuple[ParameterSpec, ...],
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    block_size: int,
    bf16_stochastic_round: bool,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
    sr_seed: int,
) -> IsolatedAdamW8bit:
    """Construct the locked-policy AdamW8bit over an explicit spec subset.

    Replicates build_adamw8bit's validation exactly (locked policy +
    quantizability + single CUDA device) but builds the parameter groups from
    the given specs instead of re-auditing a module. The returned
    IsolatedAdamW8bit's audit covers only these specs.
    """
    if (
        type(lr) is not float
        or not math.isfinite(lr)
        or lr <= 0.0
        or betas != (0.9, 0.95)
        or eps != 1e-8
        or block_size != 256
        or not bf16_stochastic_round
    ):
        raise ValueError("optimizer settings differ from the locked policy")
    if not specs:
        raise ValueError("hybrid AdamW8bit subset is empty")
    audit = ParameterAudit(specs=specs)
    fallback = [
        spec.name
        for spec in audit.decay
        if spec.parameter.numel() < 4096 or spec.parameter.numel() % block_size != 0
    ]
    if fallback:
        raise ValueError(
            "BF16 decay parameters would use non-quantized optimizer state: "
            + ", ".join(fallback)
        )
    devices = {spec.parameter.device for spec in specs}
    if len(devices) != 1:
        raise ValueError("all trainable parameters must share one device")
    device = next(iter(devices))
    if device.type != "cuda":
        raise ValueError("TorchAO AdamW8bit production optimizer requires CUDA")
    parameter_groups = [
        {
            "params": [spec.parameter for spec in audit.decay],
            "param_names": [spec.name for spec in audit.decay],
            "weight_decay": matrix_weight_decay,
            "group_name": "matrix_decay",
        },
        {
            "params": [spec.parameter for spec in audit.sensitive],
            "param_names": [spec.name for spec in audit.sensitive],
            "weight_decay": sensitive_weight_decay,
            "group_name": "sensitive_no_decay",
        },
    ]
    optimizer = cast(
        Optimizer,
        AdamW8bit(
            parameter_groups,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=0.0,
            amsgrad=False,
            block_size=block_size,
            bf16_stochastic_round=bf16_stochastic_round,
        ),
    )
    return IsolatedAdamW8bit(
        optimizer=optimizer,
        audit=audit,
        sr_rng=StochasticRoundingRNG.seeded(device, sr_seed),
    )


def build_hybrid_cmuon(
    module: nn.Module,
    *,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    block_size: int,
    bf16_stochastic_round: bool,
    matrix_weight_decay: float,
    sensitive_weight_decay: float,
    sr_seed: int,
    ns_steps: int = DEFAULT_NS_STEPS,
    ns_steps_by_role: Mapping[str, int] | None = None,
    momentum_dtype: MomentumDtype = "bfloat16",
    chunk_rescale_sqrt_n: bool = False,
    ns_telemetry: NSSafetyTelemetry | None = None,
    ns_telemetry_enabled: bool = False,
    ns_telemetry_log_every_n: int = 100,
    ns_telemetry_update_offset: int = 0,
    ns_telemetry_logger: Callable[[str], None] | None = None,
    ns_telemetry_roles: Sequence[str] | None = None,
    exclude_roles: Sequence[str] = (),
    forensic: ForensicConfig | None = None,
) -> HybridCMuon:
    """Build the hybrid optimizer (AdamW8bit for the rest + CMuon allowlist).

    The AdamW8bit part reuses the exact locked policy (same validation as
    build_adamw8bit) restricted to the non-CMuon params. The CMuon part covers
    the semantic allowlist.

    Newton-Schulz depth: pass ``ns_steps_by_role`` for a per-role (per-spec)
    canonical map, or the scalar ``ns_steps`` for a uniform depth (backward
    compatible). A partial ``ns_steps_by_role`` fills omitted roles from
    ``ns_steps`` (or DEFAULT_NS_STEPS).

    ``ns_telemetry`` is an optional low-overhead safety telemetry (off by
    default; see NSSafetyTelemetry / build_ns_safety_telemetry). When
    ``ns_telemetry_enabled`` is set and no explicit ``ns_telemetry`` is given,
    one is built here from the routing (one representative spec per canonical
    role) using ``ns_telemetry_log_every_n`` / ``ns_telemetry_update_offset`` /
    ``ns_telemetry_logger``.
    """
    routing = route_cmuon_parameters(
        module,
        matrix_weight_decay=matrix_weight_decay,
        sensitive_weight_decay=sensitive_weight_decay,
        exclude_roles=exclude_roles,
    )
    canonical_ns = resolve_ns_map(ns_steps_by_role, ns_steps)
    cfg = CMuonConfig(
        lr=lr,
        ns_steps_by_role=canonical_ns,
        momentum_dtype=momentum_dtype,
        chunk_rescale_sqrt_n=chunk_rescale_sqrt_n,
    )
    adamw = _build_adamw8bit_for_specs(
        routing.adamw_specs,
        lr=lr,
        betas=betas,
        eps=eps,
        block_size=block_size,
        bf16_stochastic_round=bf16_stochastic_round,
        matrix_weight_decay=matrix_weight_decay,
        sensitive_weight_decay=sensitive_weight_decay,
        sr_seed=sr_seed,
    )
    if ns_telemetry is None and ns_telemetry_enabled:
        # Opt-in safety telemetry: one representative spec per canonical role,
        # device-side accumulation, one batched CPU sync every log_every_n.
        device = routing.cmuon_specs[0].parameter.device
        ns_telemetry = build_ns_safety_telemetry(
            routing,
            device=device,
            log_every_n=ns_telemetry_log_every_n,
            logger=ns_telemetry_logger,
            update_offset=ns_telemetry_update_offset,
            roles=ns_telemetry_roles,
        )
    if forensic is not None:
        from sakuramoon.optim.cmuon_forensic import ForensicMonitor

        device = routing.cmuon_specs[0].parameter.device
        forensic_monitor = ForensicMonitor(
            routing,
            cfg,
            forensic,
            device=device,
            logger=ns_telemetry_logger,
            update_offset=ns_telemetry_update_offset,
        )
    else:
        forensic_monitor = None
    return HybridCMuon(
        routing=routing,
        cfg=cfg,
        adamw=adamw,
        sr_rng=adamw.sr_rng,
        ns_telemetry=ns_telemetry,
        forensic=forensic_monitor,
    )


__all__ = [
    "CMUON_ROLES",
    "CMuonChunkSpec",
    "CMuonConfig",
    "CMuonRouting",
    "HybridCMuon",
    "NSSafetyTelemetry",
    "NSTelemetrySample",
    "build_hybrid_cmuon",
    "build_ns_safety_telemetry",
    "cmuon_moonlight_alpha",
    "cmuon_zeroth_power",
    "cmuon_zeroth_power_traced",
    "resolve_ns_map",
    "route_cmuon_parameters",
    "select_representative_specs",
]
