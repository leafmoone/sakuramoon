# iREPA Phase 4 — COPY / delivery report

Status: **STOPPED at the user quality gate.** No push. No amend/squash of P1/P2/P3.
Date: 2026-09-03
Branch: `iprea` (local), parent `14c9cee` (iREPA phase 3 evidence).

## What was delivered

A single functional commit on `iprea`:

```
iREPA phase 4: integrate representation alignment training graph
```

It integrates the representation-alignment training graph end-to-end:

- **slot_08 stable capture** — the tapped stable block's output (image span,
  row-major `T=H*W`), read at an eager outer-loop point in both the dense and
  packed DiT paths.
- **projector** — `IRepaAlignment` (locked bf16 MixedPrecisionConv2d → 768),
  wired into the DDP `TrainableComposite`.
- **FP32 spatial z-score** teacher target + **FP32 per-sample cosine**
  alignment loss.
- **successful-update λ schedule** (stateless, deterministic).
- **JLT/iREPA combined objective** (`main + λ·irepa`, no λ=0 skip).
- **telemetry v11** (2 timing phases, 6 iREPA fields, MAIN-JLT-only t-bins,
  bit-exact fp32 split re-derivation).
- Full regression on salt10 DCU.

## Verification (salt10, DTK HCU)

| gate | result |
|------|--------|
| ruff `src tests` | All checks passed |
| pyright NEW-in-src (vs dev) | **0** |
| frozen check (flow/cmuon/guarded/fp32/save) | CLEAN |
| new P4 unit tests | 59 passed |
| new P4 GPU training-graph tests | 13 passed |
| full `pytest tests/` iprea-P4 | 2 failed, 992 passed, 2 skipped |
| full `pytest tests/` dev BASE | 2 failed, 786 passed, 2 skipped |
| **parity** | **iprea-only = 0, base-only = 0 → ZERO NEW FAILURES** |

The 2 shared failures are pre-existing on dev (non-blocking): the Qwen asset
gap on salt10, and dev's own `host_metadata` exception-contract bug. No
skip/xfail was added.

## Constraints discovered during the gate (recorded for the record)

1. **PackedDiT is locked to the production FA4 config** (d=2560, 20Q/5KV,
   head_dim=128, bfloat16) — a small-model dense-vs-packed parity is impossible;
   the parity test uses the full production model.
2. **slot 8 is a G1 growth slot** (active only at depth 20/24, not the base
   depth 16) — the capture validates the tap slot against the active slots.
3. **pyright on salt10 does not resolve `pytest`** (a pre-existing environment
   quirk, present in the dev baseline too); the gate metric is NEW-in-src, and
   P4's production src is clean.
4. **torch.compile namespace pollution** — the compile-stability GPU test
   injects dynamo `__compiled_fn_*` artifacts that a later unit test asserts
   on; the GPU file carries an autouse fixture to isolate them (no production
   change).

## NOT done (intentionally, per standing orders)

- **No push** of `iprea` (the 2 P3 commits `91d340f`/`14c9cee` and the new P4
  commit stay local).
- **No amend/squash** of the P1/P2/P3 commits.
- No follow-up asset work (Qwen weights on salt10) or the dev host_metadata
  fix (those are separate, non-blocking).

## Files in the commit

new src: `objective/irepa.py`, `train/irepa_diagnostics.py`
new tests: `gpu/irepa/test_irepa_training_graph.py`,
`unit/objective/test_irepa_objective.py`, `unit/test_irepa_metrics_v11.py`,
`unit/train/test_irepa_lambda_binding.py`, `unit/train/test_irepa_shadow_audit.py`,
`unit/train/test_irepa_zero_impact.py`
modified: `config/assembly.py`, `model/dit.py`, `telemetry/metrics.py`,
`telemetry/observer.py`, `train/loop.py`, `train/production.py`,
`train/runtime.py`, `train/step.py`, `unit/test_g1_telemetry.py`,
`unit/test_telemetry_spatial.py`, `unit/test_telemetry_t_bins.py`,
`unit/train/test_irepa_composite.py`, `unit/train/test_irepa_production_gate.py`
reports: `irepa-phase4-training-graph-audit.md`,
`irepa-phase4-training-graph-audit.json`
