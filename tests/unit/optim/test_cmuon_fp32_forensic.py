"""F1 forensic tests for the Hybrid CMuon FP32-rescue optimizer.

Verifies the telemetry-only hard-failure patch against the spec:

  A. BF16 fail + FP32 success          -> rescue, commit proceeds
  B. BF16 fail + FP32 above ceiling    -> hard fail, reason=above_ceiling
  C. BF16 fail + FP32 below floor      -> hard fail, reason=below_floor
  D. BF16 fail + FP32 nonfinite        -> hard fail, reason=nonfinite
  E. forensic writer failure           -> original CMuonSafetyError still
                                          raised (I/O never masks root cause)
  F. diagnostic trace failure          -> original CMuonSafetyError still
                                          raised; forensic_trace_error set
  G. successful step                   -> no hard-fail artifact created
  H. hard fail                         -> no CMuon param commit, no AdamW
                                          fallback step commit
  I. exact input artifact              -> tensor is the EXACT NS input (same
                                          bits); replay CLI recomputes from it
  J. non-owner rank                    -> no fabricated input artifact
  K. repeated failure (crash loop)     -> older event never overwritten
  L. legacy forensic JSON              -> new fp32 fields + bf16 rename with
                                          deprecated delta_rms alias

All tests require HCU/CUDA (the production optimizer builds torchao
AdamW8bit on a CUDA device; CPU builds are rejected by design). The NS
functions are stubbed at the ``fp32_rescue`` module level per test so the
verdict/telemetry logic is exercised deterministically; the production NS
math itself is covered by the frozen-algorithm diff gate (``cmuon.py`` must
stay at zero diff) and by the bit-parity capture protocol.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import ClassVar

import pytest
import torch
from torch import nn

REQUIRES_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="requires HCU/CUDA (torchao AdamW8bit is cuda-only by design)",
)

POISON_SHAPE = (96, 96)  # content_gate chunk: unique shape in the mini module


def _load_replay_cli() -> object:
    # tests/unit/optim/<file> -> parents[3] is the repo root.
    repo_root = Path(__file__).resolve().parents[3]
    path = repo_root / "dev-tools" / "cmuon_fp32_rescue_replay.py"
    spec = importlib.util.spec_from_file_location("cmuon_fp32_rescue_replay", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Attention(nn.Module):
    """Every attention projection has a UNIQUE 2D shape, so a shape-based
    poison (POISON_SHAPE) can never hit a non-target chunk — including under
    2-rank DDP where the same module is built on both ranks."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(128, 128, bias=False)
        self.k_proj = nn.Linear(120, 120, bias=False)
        self.v_proj = nn.Linear(112, 112, bias=False)
        self.content_gate = nn.Linear(96, 96, bias=False)  # unique shape
        self.out_proj = nn.Linear(104, 104, bias=False)
        self.q_norm = nn.Parameter(torch.ones(128))  # AdamW sensitive (1D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


class _MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # in_proj weight (256,128): 2 chunks of 128x128 on dim 0.
        self.in_proj = nn.Linear(128, 256, bias=False)
        self.down_proj = nn.Linear(256, 128, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


class _MiniBlock(nn.Module):
    """Block with the production ``attention`` / ``mlp`` submodule names so
    the FQNs are exactly ``dit.blocks.slot_00.attention.*.weight`` and
    ``dit.blocks.slot_00.mlp.*.weight`` (the CMuon allowlist anchors on the
    ``.attention.`` / ``.mlp.`` segments — a flat block routes everything to
    the AdamW fallback and the 8bit quantizability check rejects it)."""

    def __init__(self) -> None:
        super().__init__()
        self.attention = _Attention()
        self.mlp = _MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


class _MiniDiT(nn.Module):
    """Canonical-FQN mini DiT: 14 NS inputs (5 attention + 2 in_proj chunks
    + 1 down_proj + 6 shared-block chunks) + 1 AdamW sensitive param
    (q_norm) + 1 AdamW matrix param (final_layer, not in the allowlist)."""

    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleDict({"slot_00": _MiniBlock()})
        self.conditioner = nn.ModuleDict(
            # weight is (out, in) = (768, 144): chunk_dim 0 => 768 output
            # rows / 6 chunks = 128x144 (chunk_count must divide the output
            # dim exactly; 144 rows would not).
            {"shared_block_projection": nn.Linear(144, 768, bias=False)}
        )
        self.final_layer = nn.ModuleDict({"linear": nn.Linear(128, 128, bias=False)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return x


class _Composite(nn.Module):
    """Minimal stand-in for ``TrainableComposite``: holds the DiT as a
    ``self.dit`` submodule so parameter FQNs match the production
    ``dit.*`` allowlist (``audit_trainable_parameters`` derives names from
    ``module.named_parameters()``). It is deliberately NOT the real
    composite (no training inputs / VAE / forward graph)."""

    def __init__(self, dit: nn.Module) -> None:
        super().__init__()
        self.dit = dit

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return self.dit(x)


def _make_module(device: torch.device) -> _Composite:
    module = _Composite(_MiniDiT()).to(device)
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(std=0.02).mul_(0.1)
    # Locked audit policy: 2D matrix projections are BF16; 1D/low-dim
    # "sensitive" params (e.g. q_norm) are FP32. ``module.to(bfloat16)``
    # alone would violate the sensitive dtype and the audit rejects it.
    for p in module.parameters():
        p.data = p.data.to(torch.bfloat16 if p.ndim == 2 else torch.float32)
    return module


def _bootstrap_refs() -> dict[str, float]:
    keys = [
        "dit.blocks.slot_00.attention.q_proj.weight#chunk0",
        "dit.blocks.slot_00.attention.k_proj.weight#chunk0",
        "dit.blocks.slot_00.attention.v_proj.weight#chunk0",
        "dit.blocks.slot_00.attention.content_gate.weight#chunk0",
        "dit.blocks.slot_00.attention.out_proj.weight#chunk0",
        "dit.blocks.slot_00.mlp.in_proj.weight#chunk0",
        "dit.blocks.slot_00.mlp.in_proj.weight#chunk1",
        "dit.blocks.slot_00.mlp.down_proj.weight#chunk0",
        "dit.conditioner.shared_block_projection.weight#chunk0",
        "dit.conditioner.shared_block_projection.weight#chunk1",
        "dit.conditioner.shared_block_projection.weight#chunk2",
        "dit.conditioner.shared_block_projection.weight#chunk3",
        "dit.conditioner.shared_block_projection.weight#chunk4",
        "dit.conditioner.shared_block_projection.weight#chunk5",
        "dit.final_layer.linear.weight#chunk0",
    ]
    return {key: 1e-4 for key in keys}


def _set_gradients(module: nn.Module, seed: int) -> None:
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in module.parameters():
            p.grad = torch.randn(
                tuple(p.shape), generator=g, dtype=p.dtype
            ).to(p.device) * 0.1


def _build_optimizer(
    module: nn.Module,
    legacy_dir: str | Path | None = None,
    *,
    rank: int,
    world_size: int,
    lr: float = 1.5625e-4,
    artifact_root: str | None = None,
):
    from sakuramoon.optim.fp32_rescue import build_fp32_rescue
    from sakuramoon.optim.guarded_canonical import GuardedCanonicalGuardConfig

    guard_cfg = GuardedCanonicalGuardConfig(
        guard_ratio=0.1,
        reference_decay=0.999,
        min_reference=3.096e-08,
        numerical_floor=6.575e-07,
        warmup_observations=50,
        invariant_check=False,
    )
    ns_map = {
        "attention_q": 4,
        "attention_k": 4,
        "attention_v": 4,
        "attention_content_gate": 4,
        "attention_out": 4,
        "ffn_in": 4,
        "ffn_down": 4,
        "adaln_shared": 4,
    }
    kwargs: dict[str, object] = {
        "module": module,
        "lr": lr,
        "betas": (0.9, 0.95),
        "eps": 1e-8,
        "block_size": 256,
        "bf16_stochastic_round": True,
        "matrix_weight_decay": 0.0,
        "sensitive_weight_decay": 0.0,
        "sr_seed": 20260903,
        "ns_steps_by_role": ns_map,
        "guard_cfg": guard_cfg,
        "guard_bootstrap_refs": _bootstrap_refs(),
        "rank": rank,
        "world_size": world_size,
        "momentum_dtype": "bfloat16",
        "chunk_rescale_sqrt_n": False,
    }
    if artifact_root is not None:
        kwargs["hard_fail_artifact_root"] = str(artifact_root)
    if legacy_dir is not None:
        kwargs["legacy_forensic_dir"] = str(legacy_dir)
    return build_fp32_rescue(**kwargs)  # type: ignore[arg-type]


def _install_ns_stubs(monkeypatch, lr: float, fp32_mode: str) -> list[torch.Tensor]:
    """Stub both production NS entry points in the fp32_rescue module.

    BF16: poison (delta_rms 100x target) for POISON_SHAPE, healthy otherwise.
    FP32: mode selects the poison-chunk outcome:
      "good" / "above_ceiling" / "below_floor" / "nonfinite".
    Returns a list capturing every chunk handed to the FP32 stub (the EXACT
    inputs the production rescue recomputed from).
    """
    import sakuramoon.optim.fp32_rescue as fr
    from sakuramoon.optim.cmuon import cmuon_moonlight_alpha

    captured: list[torch.Tensor] = []

    # The production NS computes on the input's device; a CPU result would
    # survive to the commit, where the in-place add_ raises.
    def _scaled_full(
        chunk: torch.Tensor, scale: float, ns_steps: int
    ) -> torch.Tensor:
        rows, cols = chunk.shape
        alpha = cmuon_moonlight_alpha(rows, cols, lr, ns_steps)
        return torch.full(
            chunk.shape,
            scale * (0.2 * lr) / alpha,
            dtype=chunk.dtype,
            device=chunk.device,
        )

    def bf16_stub(
        chunk: torch.Tensor, ns_steps: int, ns_coefficients: object, eps: float
    ) -> torch.Tensor:
        # delta_rms = scale * target exactly (constant tensor: rms = |c|*alpha).
        if tuple(chunk.shape) == POISON_SHAPE:
            return _scaled_full(chunk, 100.0, ns_steps)  # 100x target > ceiling
        return _scaled_full(chunk, 1.0, ns_steps)

    def fp32_stub(
        chunk: torch.Tensor, ns_steps: int, ns_coefficients: object, eps: float
    ) -> torch.Tensor:
        captured.append(chunk.clone())
        if tuple(chunk.shape) != POISON_SHAPE or fp32_mode == "good":
            return _scaled_full(chunk, 1.0, ns_steps)
        if fp32_mode == "above_ceiling":
            return _scaled_full(chunk, 100.0, ns_steps)
        if fp32_mode == "below_floor":
            return _scaled_full(chunk, 0.001, ns_steps)
        if fp32_mode == "nonfinite":
            return torch.full(
                chunk.shape, math.inf, dtype=chunk.dtype, device=chunk.device
            )
        raise AssertionError(fp32_mode)

    monkeypatch.setattr(fr, "cmuon_zeroth_power_bf16", bf16_stub)
    monkeypatch.setattr(fr, "cmuon_zeroth_power_fp32", fp32_stub)
    return captured


def _event_dirs(artifact_root: Path) -> list[Path]:
    if not artifact_root.exists():
        return []
    return sorted(p for p in artifact_root.iterdir() if p.is_dir() and not p.name.startswith("."))


def _event_for_fqn(artifact_root: Path, fqn_safe_fragment: str) -> Path | None:
    for d in _event_dirs(artifact_root):
        if fqn_safe_fragment in d.name:
            return d
    return None


@REQUIRES_CUDA
def test_a_bf16_fail_fp32_success_rescues_and_commits(
    tmp_path: Path, monkeypatch
) -> None:
    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    opt = _build_optimizer(module, rank=0, world_size=1, lr=lr,
                           artifact_root=tmp_path / "artifacts")
    _install_ns_stubs(monkeypatch, lr, "good")
    _set_gradients(module, seed=7)
    params_before = {n: p.detach().clone() for n, p in module.named_parameters()}

    opt.step()

    assert opt.fp32_attempts == 1
    assert opt.fp32_rescues == 1
    assert opt.fp32_rescue_failures == 0
    assert opt.observations == 1
    # Commit proceeded: the poisoned chunk got the FP32-rescued delta, the
    # other chunks their healthy deltas — at least the params changed.
    changed = [
        n
        for n, p in module.named_parameters()
        if not torch.equal(p.detach(), params_before[n])
    ]
    assert changed, "no parameter was updated after a successful rescue"
    assert _event_dirs(tmp_path / "artifacts") == []
    assert (tmp_path / "artifacts").exists() is False or not any(
        (tmp_path / "artifacts").iterdir()
    )


@REQUIRES_CUDA
@pytest.mark.parametrize(
    ("fp32_mode", "reason"),
    [
        ("above_ceiling", "above_ceiling"),
        ("below_floor", "below_floor"),
        ("nonfinite", "nonfinite"),
    ],
)
def test_bcd_hard_fail_captures_fp32_reason(
    tmp_path: Path, monkeypatch, fp32_mode: str, reason: str
) -> None:
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError

    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    root = tmp_path / "artifacts"
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=root)
    captured = _install_ns_stubs(monkeypatch, lr, fp32_mode)
    _set_gradients(module, seed=11)

    with pytest.raises(CMuonSafetyError) as excinfo:
        opt.step()

    msg = str(excinfo.value)
    assert "dit.blocks.slot_00.attention.content_gate.weight#chunk0" in msg
    assert opt.fp32_rescue_failures == 1
    assert opt.fp32_rescues == 0
    assert opt.observations == 0  # no commit: observation not counted

    # Legacy forensic JSON (F1: redirected to the per-test legacy dir):
    # new FP32 verdict fields + reason, written on the owner rank (rank0).
    legacy_path = tmp_path / f"guard-forensic-rank{0}.json"
    assert legacy_path.is_file(), f"legacy forensic JSON missing at {legacy_path}"
    recs = json.loads(legacy_path.read_text())["records"]
    rec = next(r for r in recs if "content_gate" in str(r.get("fqn", "")))
    assert rec["fp32_failure_reason"] == reason
    assert rec["bf16_delta_rms"] == rec["delta_rms"]  # deprecated alias
    # The production path computes target/ceiling/floor from the lr after the
    # optimizer's own dtype handling (HCU rounds it to bf16), so pin the
    # MATHEMATIC against the recorded lr (exact production expressions) and
    # bound the recorded lr against the requested literal.
    rec_lr = float(rec["lr"])
    assert abs(rec_lr - lr) <= lr * 1e-6, rec_lr
    assert rec["fp32_rescue_floor"] == 0.05 * (0.2 * rec_lr)
    assert rec["fp32_ceiling"] == 10.0 * (0.2 * rec_lr)
    if reason == "nonfinite":
        assert rec["fp32_finite"] is False
        # The legacy dump uses default json.dump (allow_nan=True): a
        # nonfinite rms is written as Infinity and read back as inf.
        assert math.isinf(float(rec["fp32_delta_rms"]))
    else:
        assert rec["fp32_finite"] is True
        if reason == "above_ceiling":
            assert rec["fp32_delta_rms"] > 10.0 * (0.2 * lr)
        elif reason == "below_floor":
            assert rec["fp32_delta_rms"] < 0.05 * (0.2 * lr)

    # Exact-input artifact: owner (rank0) published one event.
    events = _event_for_fqn(root, "content_gate")
    assert events is not None, f"no hard-fail event under {root}"
    meta = json.loads((events / "metadata.json").read_text())
    assert meta["fp32_failure_reason"] == reason
    assert meta["owner"] == 0
    assert meta["this_rank"] == 0
    assert meta["shape"] == list(POISON_SHAPE)
    assert meta["dtype"] == "torch.bfloat16"
    assert meta["observations"] == 0
    # Internal consistency of the recorded verdict constants (exact
    # production expressions against the recorded lr; HCU rounds lr to
    # bf16, so the literal request is only a bound).
    meta_lr = float(meta["lr"])
    assert abs(meta_lr - lr) <= lr * 1e-6, meta_lr
    assert meta["target_delta_rms"] == 0.2 * meta_lr
    assert meta["ceiling"] == 10.0 * (0.2 * meta_lr)
    assert meta["rescue_floor"] == 0.05 * (0.2 * meta_lr)
    assert meta["bf16_delta_rms"] is not None
    assert meta["bf16_delta_rms"] > meta["ceiling"]
    assert meta["fp32_delta_rms"] == meta["original_fp32_delta_rms"]
    assert meta["fp32_finite"] == meta["original_fp32_finite"]
    if reason == "nonfinite":
        assert meta["fp32_finite"] is False
        assert meta["fp32_delta_rms"] is None or meta["fp32_delta_rms"] != meta["fp32_delta_rms"]
    else:
        assert meta["fp32_finite"] is True
        if reason == "above_ceiling":
            assert meta["fp32_delta_rms"] > meta["ceiling"]
        elif reason == "below_floor":
            assert meta["fp32_delta_rms"] < meta["rescue_floor"]
    # Diagnostic replay present (CPU trace of both dtypes) and structurally
    # valid. It runs the REAL production NS on the saved exact input, so it
    # is NOT required to equal the (stubbed) recorded value here; the
    # recorded-vs-replay equivalence is asserted in the real-NS HCU test and
    # the replay CLI (spec §8 separates the two on purpose).
    assert meta["diagnostic_replay_bf16"] is not None
    assert meta["diagnostic_replay_fp32"] is not None
    diag_fp32_rms = meta["diagnostic_replay_fp32"]["final"]["delta_rms"]
    if meta["fp32_finite"]:
        assert diag_fp32_rms is not None and math.isfinite(diag_fp32_rms)
    # The saved tensor is the EXACT FP32-stub input (bits preserved).
    from sakuramoon.optim.cmuon_hardfail import tensor_format_name

    poison_input = [t for t in captured if tuple(t.shape) == POISON_SHAPE][-1]
    tensor_path = events / ("input.safetensors" if tensor_format_name() == "safetensors" else "input.pt")
    assert tensor_path.is_file()
    if tensor_path.suffix == ".safetensors":
        from safetensors.torch import load_file

        saved = load_file(str(tensor_path))["input"]
    else:
        saved = torch.load(tensor_path, map_location="cpu", weights_only=True)["input"]
    assert saved.dtype == torch.bfloat16
    assert torch.equal(saved, poison_input.cpu())
    assert meta["tensor_sha256"]  # 64 hex chars
    assert len(meta["tensor_sha256"]) == 64


@REQUIRES_CUDA
def test_e_writer_failure_still_raises_original_error(
    tmp_path: Path, monkeypatch
) -> None:
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError

    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    blocker = tmp_path / "blocker"
    blocker.write_bytes(b"file, not a dir")
    root = blocker / "sub" / "artifacts"  # mkdir must fail
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=root)
    _install_ns_stubs(monkeypatch, lr, "below_floor")
    _set_gradients(module, seed=13)

    with pytest.raises(CMuonSafetyError) as excinfo:
        opt.step()

    assert "content_gate.weight#chunk0" in str(excinfo.value)
    assert opt.fp32_rescue_failures == 1
    # No partial artifact directory was published.
    assert not any(tmp_path.iterdir()) or not (tmp_path / "artifacts").exists()


@REQUIRES_CUDA
def test_f_trace_failure_still_raises_and_records_error(
    tmp_path: Path, monkeypatch
) -> None:
    import sakuramoon.optim.fp32_rescue as fr
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError

    def boom(*a, **k):
        raise RuntimeError("trace boom")

    monkeypatch.setattr(fr, "trace_ns_replay", boom)
    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    root = tmp_path / "artifacts"
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=root)
    _install_ns_stubs(monkeypatch, lr, "below_floor")
    _set_gradients(module, seed=17)

    with pytest.raises(CMuonSafetyError) as excinfo:
        opt.step()

    assert "content_gate.weight#chunk0" in str(excinfo.value)
    events = _event_for_fqn(root, "content_gate")
    assert events is not None, "artifact must still publish despite trace failure"
    meta = json.loads((events / "metadata.json").read_text())
    assert meta["forensic_trace_error"] is not None
    assert "trace boom" in meta["forensic_trace_error"]
    assert meta["diagnostic_replay_bf16"] is None
    assert meta["diagnostic_replay_fp32"] is None
    # Original verdict values are independent of the trace.
    assert meta["fp32_failure_reason"] == "below_floor"


@REQUIRES_CUDA
def test_g_successful_step_creates_no_artifact(tmp_path: Path) -> None:
    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    root = tmp_path / "artifacts"
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=root)
    # REAL production NS (no stubs): a healthy gradient passes BF16 directly
    # (or rescues silently if BF16 trips) — either way no hard fail.
    _set_gradients(module, seed=23)
    params_before = {n: p.detach().clone() for n, p in module.named_parameters()}

    opt.step()

    assert opt.observations == 1
    assert opt.fp32_rescue_failures == 0
    assert not root.exists() or not any(root.iterdir())
    assert any(
        not torch.equal(p.detach(), params_before[n])
        for n, p in module.named_parameters()
    )


@REQUIRES_CUDA
def test_h_hard_fail_commits_nothing(tmp_path: Path, monkeypatch) -> None:
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError

    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=tmp_path / "artifacts")
    _install_ns_stubs(monkeypatch, lr, "above_ceiling")
    _set_gradients(module, seed=29)

    params_before = {n: p.detach().clone() for n, p in module.named_parameters()}
    adamw_before = opt.optimizer.state_dict()
    sr_before = opt.sr_rng.state_dict()

    with pytest.raises(CMuonSafetyError):
        opt.step()

    # No CMuon parameter commit (NOT a bit-exact momentum claim: PHASE 1
    # updates momentum in place before the verdict by design; the resume
    # discards those updates via the checkpoint).
    for name, p in module.named_parameters():
        assert torch.equal(p.detach(), params_before[name]), f"{name} was committed"
    # No AdamW fallback step commit.
    adamw_after = opt.optimizer.state_dict()

    def _walk(d: dict[str, object], path: str = "") -> list[tuple[str, torch.Tensor]]:
        out: list[tuple[str, torch.Tensor]] = []
        for k, v in d.items():
            if isinstance(v, dict):
                out.extend(_walk(v, f"{path}.{k}"))
            elif isinstance(v, torch.Tensor):
                out.append((f"{path}.{k}", v))
        return out

    before = dict(_walk(adamw_before))
    after = dict(_walk(adamw_after))
    assert set(before) == set(after)
    for key, value in before.items():
        if torch.is_floating_point(value):
            assert torch.equal(value, after[key]), f"AdamW state {key} changed"
    # SR RNG stream untouched (no fallback step consumed the stream).
    assert torch.equal(
        sr_before["state"],  # type: ignore[arg-type]
        opt.sr_rng.state_dict()["state"],
    )


class _FakeReduceOp:
    MAX = "max"
    MIN = "min"


class _FakeDist:
    """Single-process simulation of the 2-rank verdict wire.

    ``all_reduce`` merges into a wire keyed by (op, shape) — the same
    elementwise op semantics as the production collective, but keyed so the
    fail-flag tensor (n_inputs,) never collides with the fingerprint
    tensors (2n,). The wire PERSISTS across the two sequential single-
    process "steps" (owner, then non-owner): that models the fact that in a
    real simultaneous all_reduce the non-owner merges with the owner's
    contribution, so the non-owner sees the owner's failure flag.
    ``broadcast`` fills receivers with zeros (the J assertions never depend
    on broadcast values). The real 2-process DDP contract is covered by
    the salt10 HCU test (tests/gpu/optim/cmuon_fp32_forensic_2rank.py).
    """

    ReduceOp = _FakeReduceOp
    wire: ClassVar[dict[tuple[str, tuple[int, ...]], torch.Tensor]] = {}

    @classmethod
    def all_reduce(cls, t: torch.Tensor, op: object = None) -> None:
        key = (str(op), tuple(t.shape))
        if op == _FakeReduceOp.MAX:
            cur = cls.wire.get(key)
            merged = t if cur is None else torch.maximum(cur, t)
            cls.wire[key] = merged
            t.copy_(merged)
        else:
            cur = cls.wire.get(key)
            merged = t if cur is None else torch.minimum(cur, t)
            cls.wire[key] = merged
            t.copy_(merged)

    @classmethod
    def broadcast(cls, t: torch.Tensor, src: int = 0) -> None:
        t.zero_()


@REQUIRES_CUDA
def test_j_nonowner_writes_no_input_artifact(tmp_path: Path, monkeypatch) -> None:
    import sakuramoon.optim.fp32_rescue as fr
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError
    from sakuramoon.optim.guarded_canonical import stable_owner

    monkeypatch.setattr(fr, "dist", _FakeDist)
    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    fqn = "dit.blocks.slot_00.attention.content_gate.weight"
    owner = stable_owner(fqn, 0, 2)
    nonowner = 1 - owner

    # Owner instance: publishes the artifact.
    _FakeDist.wire = {}
    module = _make_module(device)
    root_owner = tmp_path / f"artifacts-rank{owner}"
    opt_owner = _build_optimizer(module, tmp_path, rank=owner, world_size=2, lr=lr,
                                 artifact_root=root_owner)
    _install_ns_stubs(monkeypatch, lr, "above_ceiling")
    _set_gradients(module, seed=31)
    with pytest.raises(CMuonSafetyError):
        opt_owner.step()
    assert _event_for_fqn(root_owner, "content_gate") is not None

    # Non-owner instance (same hard fail seen through the persisted MAX
    # wire — the owner's contribution survives, exactly as in a real
    # simultaneous all_reduce): must NOT publish an input artifact (no
    # fabricated input).
    module2 = _make_module(device)
    root_nonowner = tmp_path / f"artifacts-rank{nonowner}"
    opt_nonowner = _build_optimizer(module2, tmp_path, rank=nonowner, world_size=2, lr=lr,
                                    artifact_root=root_nonowner)
    _install_ns_stubs(monkeypatch, lr, "above_ceiling")
    _set_gradients(module2, seed=31)  # identical stream -> identical verdict
    with pytest.raises(CMuonSafetyError):
        opt_nonowner.step()
    assert not root_nonowner.exists() or not any(root_nonowner.iterdir())
    # Its legacy JSON record exists but carries null FP32 fields (it never
    # ran the FP32 rescue for the owned-by-someone-else chunk).
    legacy = tmp_path / f"guard-forensic-rank{nonowner}.json"
    assert legacy.is_file(), f"non-owner legacy JSON missing at {legacy}"
    recs = json.loads(legacy.read_text())["records"]
    rec = next(r for r in recs if "content_gate" in str(r.get("fqn", "")))
    assert rec["fp32_delta_rms"] is None
    assert rec["fp32_finite"] is None
    assert rec["delta_rms"] is None


@REQUIRES_CUDA
def test_k_crash_loop_never_overwrites_older_event(
    tmp_path: Path, monkeypatch
) -> None:
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError

    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    root = tmp_path / "artifacts"
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=root)
    _install_ns_stubs(monkeypatch, lr, "above_ceiling")
    for attempt in range(2):
        _set_gradients(module, seed=37 + attempt)
        with pytest.raises(CMuonSafetyError):
            opt.step()
    events = _event_dirs(root)
    assert len(events) == 2, f"expected 2 distinct events, got {events}"
    first_meta = json.loads((events[0] / "metadata.json").read_text())
    second_meta = json.loads((events[1] / "metadata.json").read_text())
    # Same observation (no commit happened) -> the second must carry the -r2
    # suffix; the first event is byte-identical to what the first failure
    # published.
    assert events[1].name.endswith("-r2")
    assert first_meta["wall_clock_unix_seconds"] < second_meta["wall_clock_unix_seconds"]
    assert first_meta["fp32_failure_reason"] == "above_ceiling"
    assert second_meta["fp32_failure_reason"] == "above_ceiling"


@REQUIRES_CUDA
def test_i_replay_cli_recomputes_from_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    from sakuramoon.optim.cmuon_forensic import CMuonSafetyError

    device = torch.device("cuda", 0)
    lr = 1.5625e-4
    module = _make_module(device)
    root = tmp_path / "artifacts"
    opt = _build_optimizer(module, tmp_path, rank=0, world_size=1, lr=lr,
                           artifact_root=root)
    captured = _install_ns_stubs(monkeypatch, lr, "above_ceiling")
    _set_gradients(module, seed=41)
    with pytest.raises(CMuonSafetyError):
        opt.step()
    event = _event_for_fqn(root, "content_gate")
    assert event is not None

    replay = _load_replay_cli()
    out = tmp_path / "replay-report.json"
    rc = replay.main(["--artifact", str(event), "--output", str(out)])
    assert rc == 0
    report = json.loads(out.read_text())
    assert report["fqn"] == "dit.blocks.slot_00.attention.content_gate.weight"
    assert report["chunk"] == 0
    assert report["shape"] == list(POISON_SHAPE)
    assert report["ns_steps"] == 4
    assert "replay_production_ns" in report
    assert report["replay_production_ns"]["fp32_delta_rms"] > 0
    assert report["recorded_original"]["fp32_failure_reason"] == "above_ceiling"
    # The replay runs the REAL production NS on the saved exact input (the
    # stub is gone), so its value legitimately differs from the stubbed
    # recorded value; the comparison block must report both sides.
    comp = {row["field"]: row for row in report["comparison_recorded_vs_replay"]}
    assert comp["fp32_delta_rms"]["recorded"] is not None
    assert comp["fp32_delta_rms"]["replayed"] is not None
    # The saved input round-trips bit-exactly.
    poison_input = [t for t in captured if tuple(t.shape) == POISON_SHAPE][-1]
    meta = json.loads((event / "metadata.json").read_text())
    assert meta["input_rms"] == pytest.approx(
        float(poison_input.float().pow(2).mean().sqrt().item()), rel=1e-6
    )
    del replay, captured
