# P2-Prep — Remote Runbook (design / implement / test only, NO launch)

Scope gate: this phase does **not** start Full M4, does **not** enter Stage II,
and does not retrain Phase I-P. It prepares the three P2 items so that the M4
decision (first exposure, default 6M) can be executed immediately after approval.

Base: remote branch **`p2-prep` off `5a92ce1`** (tag `p1formal-final-20260829`).
All P2 code below is additive; nothing in this phase changes training
semantics of the frozen p1formal model.

## 1. Files (7, mirrored under `deploy/p2prep/`)

| file | role |
|---|---|
| `src/anime_sr/train/ema_sample.py` | P2-① `SampleEMA` (sample-rate-based EMA, fp32 shadow, bf16-safe) |
| `src/anime_sr/train/ckpt_v2.py` | P2-② production checkpoint v2 schema (`save_v2`/`load_v2`, RNG snapshot, provenance) |
| `src/anime_sr/model/window_attention_sdpa.py` | P2-③ `WindowAttentionSDPA` + `sdpa_variant` (bit-safe repeat mode + native-GQA A/B twin) |
| `tests/test_p2_ema.py` | EMA unit tests (recurrence, dtypes, state round-trip, DDP-unwrap) |
| `tests/test_p2_ckpt_v2.py` | ckpt v2 unit tests (round-trip, v1-legacy, v2→v1 compat, RNG, atomicity) |
| `tests/test_p2_sdpa_parity.py` | parity suite (fp32 fwd/bwd/trajectory hard gates, bf16 rel-L2 report, `P2_SDPA_REPORT` dump) |
| `tools/bench_attention_backends.py` | backend benchmark (explicit vs `sdpa_rep` vs `sdpa_native`, phase split, peak-mem) |

Apply on the remote branch (paths are repo-relative, `cp -r` or `git apply`
both work):

```bash
git checkout 5a92ce1 -b p2-prep
# copy the 7 files from deploy/p2prep/{src,tests,tools}/... into the repo tree
source /opt/dtk-26.04/env.sh
export LD_LIBRARY_PATH=/opt/dtk-26.04/lib:/opt/dtk-26.04/hip/lib:/opt/hyhal/lib
export PYTHONPATH=src OMP_NUM_THREADS=2
```

DTK torch-2.9 notes:
- `ckpt_v2.load_v2` calls `torch.load(..., weights_only=False)` **explicitly**
  (the v1 loader on DTK torch 2.9 requires it; the ckpt embeds numpy RNG state
  → the v1 path already runs `weights_only=False` too — keep both consistent).
- No torchvision / skimage / scipy / lpips on DTK; P2 items do not need them.
- Python is `/usr/local/bin/python3.11`.

## 2. Parity acceptance (P2-③, gate order)

Run on HCU (DTK torch 2.9) — the local RTX5090 numbers below are smoke only.

| gate | criterion | kind |
|---|---|---|
| fp32 forward (exact / padded / global, repeat mode) | max_abs < 1e-4 CPU, < 1e-3 CUDA | **hard** |
| fp32 backward (q/k/v/o proj + input grads) | max_abs < 1e-4 | **hard** |
| 20-step fixed-seed AdamW trajectory | loss gap < 1e-3, param-drift cosine > 0.999 | **hard** |
| bf16 relative-L2 (exact / padded) | rel_l2 < 0.05 | report (soft) |
| native GQA (Hq vs Hkv, un-repeated) | backend-acceptance report | report (A/B target) |

The **default production path is `sdpa_rep`** (repeat_interleave k/v →
bit-safe). `sdpa_native` is the benchmark A/B twin only.

Known dtype flow (verified): the parent's fp32 RoPE tables upcast q/k to fp32
even inside bf16 modules while v keeps the module dtype; the parent computes
logits/softmax in fp32 and casts back before the final V matmul. The SDPA
core mirrors it: SDPA runs in q's dtype (v cast up), output cast back to
v's dtype. SDPA's fp32 final matmul is *slightly* more accurate than the
parent's bf16 one — this is captured by the bf16 rel-L2 report, not a gate.

Known backend limitation (local finding, re-verify on HCU): the SDPA
dispatcher rejects GQA head-count mismatch (Hq=4 vs Hkv=2) on the local
torch 2.11+cu128 build in **both** fp32 (falls back to a non-GQA backend)
and bf16 (flash territory, still rejected) → native GQA is
attempt-and-report on any host; the repeat mode is always available.

Local smoke (RTX5090, torch 2.11+cu128, py3.14): fp32 max_abs ≤ 6e-7,
bwd ≤ 6e-7, 20-step trajectory gap **0.0** (bit-exact path), bf16 rel_l2
≈ 0.0034; 25/25 unit tests pass.

## 3. Benchmark protocol (HCU)

Model stage table at 1024 HR (input latent 64×64), from `config/base.toml`
`[model.uflow]` (frozen plan §7.2; `qk_norm=true`, `rope=continuous-2d`,
`window_shift_pattern=normal-shifted`):

| stage | grid | dim | q_heads | kv_heads | attention | depth |
|---|---|---|---|---|---|---|
| 0 | 64×64 | 384 | 6 | 2 | window-8 | 4 |
| 1 | 32×32 | 512 | 8 | 2 | window-8 | 6 |
| 2 | 16×16 | 768 | 12 | 4 | global | 8 |
| 3 | 32×32 | 512 | 8 | 2 | window-8 | 6 |
| 4 | 64×64 | 384 | 6 | 2 | window-8 | 4 |

Bench commands (bf16, one per config; `--iters 200 --warmup 20` unless the
HCU clock throttle demands more):

```bash
python tools/bench_attention_backends.py --H 64 --W 64 --dim 384 --heads 6 --kv 2 --window 8 --dtype bf16 --iters 200 --warmup 20 --out p2-bench-s0.json
python tools/bench_attention_backends.py --H 32 --W 32 --dim 512 --heads 8 --kv 2 --window 8 --dtype bf16 --iters 200 --warmup 20 --out p2-bench-s1.json
python tools/bench_attention_backends.py --H 16 --W 16 --dim 768 --heads 12 --kv 4 --dtype bf16 --iters 200 --warmup 20 --out p2-bench-s2.json
```

(Stages 3/4 reuse the s1/s0 shapes; run them too if cheap. For the M4
mixed-resolution case — **only if** the 512/768/1024 mixing decision is made
— add the 512 (32/16/8/16/32) and 768 (48/24/12/24/48) tables; 12-wide
stages are window-padded at runtime, covered by the padded-bucket parity.)

hy-smi: capture HCU utilization + power in a parallel shell over each run
window (the bench JSON carries `ms_per_step`, `it_per_s`, `peak_mb`, and the
pre/core/out-proj phase split per backend × shift; hy-smi supplies the
hardware side). Record both shift=0 and shift=1 (the report splits
`shift0`/`shift1`).

Deliverable numbers: per-stage throughput (explicit vs `sdpa_rep` vs
`sdpa_native`) + peak-mem delta → feeds the 6M/10M/16M wall-clock estimate
(for the 6M default: steps = 6e6 / (bs·world) at the p1formal cadence).

## 4. Trainer integration design (design-only this phase)

- **EMA (P2-①)**: instantiate `SampleEMA(model, decay, ref_samples)` when
  `[train.ema] enabled=true`; call `ema.update(model, n_samples)` immediately
  after `opt.step()` (`latent_flow.py` L1071), where `n_samples` = the
  optimizer-sample count consumed by that step. DDP: per-rank identical
  recurrences stay bit-identical; the rank-0 copy is what goes into the
  checkpoint (documented in `ema_sample.py`).
- **ckpt v2 (P2-②)**: replace the `_save_ckpt` calls at `latent_flow.py`
  L1117 (step ckpts) and L1142 (`latest.pt`) with
  `save_v2(path, step=..., model=..., opt=..., ema=ema_or_None,
  scalars=..., exposure=..., provenance=make_provenance(...),
  capture_rng=True)`. On resume (`_load_ckpt` call site L841):
  `load_v2` auto-detects v1 legacy files (returns `meta["legacy"]=True` and
  `ema=None` semantics) — a v1 checkpoint resumes with exactly today's
  semantics, so **no retrain risk** and no forced conversion.
- **exposure cursor**: `{"index": int, "cycle": int, "per_cycle": int}`
  stored in the v2 section and restored on resume (P2 groundwork for the
  mixed-exposure sampler; inert for single-exposure M4).
- **config keys** (training params live only in `config/*.toml`):
  `[train.ema]` (`enabled`, `decay`, `ref_samples`) and
  `[train.ckpt]` (`version = 2`, `capture_rng = true`).
- **integration acceptance** (at M4 launch time, not now): 200-step
  parent-vs-P2 run → loss curves overlap < 1e-3; v2→v1 loader compat
  re-confirmed on DTK (unit test already covers it on CPU).

## 5. Do-not list

No Full M4 (6M/10M/16M), no Stage II, no mixed-resolution sampler launch,
no backbone/loss/optimizer/hyperparam changes, no retrain of Phase I-P,
no git commit of weights/data (P2 files stay untracked in the repo tree;
deploy/ mirror only).

## 6. Decision gate

Everything above ends at the user decision on Full M4 (first exposure
default 6M). The RGB quality report (p1formal `latest.pt`, 1-step
Faithful / 4-step Experimental ruling, ratio_4_1=1.0022) is the input to
that decision, together with the M4 wall-clock derived from §3.
