"""Phase 5 spec-18 lambda=0 one-update baseline parity (HCU) — HARD GATE.

Constructs one no-iREPA source checkpoint N (production-shape composite,
two real updates, RAW checkpoint).  Then:

* Arm A (legacy / no-iREPA): continues from update N with one update N+1.
* Arm B (migrated iREPA): real ``migrate_irepa_checkpoint`` of the same
  source checkpoint, production resume via ``load_raw_checkpoint`` into a
  fresh v4 composite + the PRODUCTION optimizer class
  (``hybrid_cmuon_canonical_ns4_fp32_rescue``, built from the real
  ``train_g1_fp32_rescue_r1.toml`` — guard config + per-(FQN,chunk)
  bootstrap references included), one update N+1 at
  ``lambda(N+1) == exact zero`` with the FULL teacher/projector/cosine
  graph still running (no-skip contract).

The chain itself (composite/optimizer/batch construction, migration,
resume, updates, and every comparison) lives in ``s18_chain.py`` and is
shared with the 2-rank DDP smoke (``irepa_ddp_lambda_zero_smoke.py``).

Controls (identical for both arms): exact same batch, timestep, noise,
(no dropout: all production dropout rates are 0), train RNG (no forward
RNG consumed), optimizer state (Arm B resumes the saved state bit-exact),
and optimizer SR RNG (saved and restored in the checkpoint).

HCU determinism facts (measured on this backend, DTK torch 2.9.0+das,
salt13 2x BW): bf16 ``A@B`` matmul, bf16 reductions and the DiT
dense_sdpa forward/backward are bit-deterministic; bf16
``torch.addmm`` (the fused GEMM the BF16 Newton-Schulz iteration uses)
is NON-deterministic across calls for identical inputs on every
production chunk shape (``use_deterministic_algorithms`` does not fix
it); the FP32 NS path is bit-deterministic on every production chunk
shape.  The production 2-rank flow is unaffected by this because the
owner rank computes each NS once and broadcasts it; a single-rank
two-arm comparison necessarily performs two independent NS calls of the
same update and therefore cannot be bit-exact on the raw BF16-first
path.

Two gates, one chain:

1. ``test_lambda_zero_one_update_parity`` (PRIMARY, spec-18 bit exact):
   the NS entry point is replaced, for the duration of the chain, by the
   deterministic FP32-NS computation with a single BF16 rounding at the
   update boundary (exactly the production rescue staging, and measured
   bit-deterministic on every production chunk shape).  Under this
   deterministically-controllable test environment EVERY comparison in
   the spec is bit exact: all pre-existing parameters, all pre-existing
   CMuon momenta, all pre-existing AdamW state, guard references and
   rescue counters, SR RNG at resume, MAIN JLT loss bit exact,
   TOTAL == MAIN bit exact, counters equal.

2. ``test_lambda_zero_production_ns_behavior`` (spec-17 HCU leg):
   the UNPATCHED production optimizer (raw BF16-first NS) runs the same
   chain.  Every deterministic component stays bit exact (losses,
   AdamW params/state, CMuon momenta, guard references, counters); the
   NS-affected CMuon parameter VALUES are verified by tolerance +
   finiteness (cross-call addmm non-determinism is ulp-level; a
   checkpoint/state error would show as O(1) relative divergence), and
   no safety failure may occur.  HCU non-determinism must not mask a
   checkpoint error.

Only allowed to differ (both gates): the new projector params/state
(including the shared SR stream advancing by the projector's own draw
after the update), iREPA persistent metadata, iREPA telemetry/timing,
wallclock, and the teacher/projector compute itself.

Model: the production composite (d=2560, depth 20, 20Q/5KV, head_dim
128, BF16) with the production TextConditioner and ConditionTokenEncoder
— a small model must not fake this parity (spec 20), and the production
``text.*`` decay FQNs sort AFTER ``irepa_alignment.*``: this is exactly
the SR-consumption ordering hazard that spec 11 (AdamW SR RNG audit)
requires the migration to keep out of the old-parameter stream
(projector appended after every existing AdamW parameter, never
interleaved).

If the primary gate fails: PHASE 5 FAIL.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import s18_chain
import torch

DEVICE = torch.device("cuda", 0)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        not s18_chain.teacher_asset_available(),
        reason="PE-Spatial-B16-512 teacher asset is not present in this checkout",
    ),
]


def test_lambda_zero_one_update_parity(tmp_path: Path) -> None:
    """PRIMARY gate (spec 18): bit-exact lambda=0 one-update parity in a
    deterministically-controllable test environment (deterministic NS)."""

    result = s18_chain.run_chain(
        tmp_path, deterministic_ns=True, device=DEVICE, rank=0, world_size=1
    )
    s18_chain.assert_deterministic_parts(result)
    s18_chain.assert_primary_gate(result)


def test_lambda_zero_production_ns_behavior(tmp_path: Path) -> None:
    """Spec-17 HCU leg: the UNPATCHED production optimizer (raw BF16-first
    NS) — every deterministic component bit exact, NS-affected CMuon
    parameter values within tolerance, no safety failure."""

    result = s18_chain.run_chain(
        tmp_path, deterministic_ns=False, device=DEVICE, rank=0, world_size=1
    )
    s18_chain.assert_deterministic_parts(result)
    report = s18_chain.production_ns_gate_report(result)
    print(
        f"[production-ns behavior] worst CMuon rel-rms = "
        f"{cast(float, report['worst_rel_rms']):.3e} "
        f"({cast(str, report['worst_param'])}); "
        f"rescue asymmetry: A={report['rescue_a']} B={report['rescue_b']}"
    )
