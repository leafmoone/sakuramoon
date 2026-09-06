# SakuraMoon iREPA Token-Spatial Layout Fix

Date: 2026-09-06 · Branch: `fix/irepa-token-spatial-layout` · Author: DSH session (salt13)

> **Reconstruction note (09-06 evening).** The salt13 machine was unexpectedly
> released after the fix was committed locally there (`c3e00ed`). This branch
> was reconstructed on the same BASE `cd32bfc…` from local staging files plus
> the session record; `src/sakuramoon/model/irepa.py` in the reconstruction is
> **byte-identical** to the pre-release version (sha256 `f8d0856f…573f87`
> verified after LF normalization). The three raw evidence logs under
> `reports/irepa-layout-fix/` (RED log, full-pytest log, restart-audit log)
> lived only on the salt13 filesystem and are unrecoverable after the release;
> their key figures are all quoted in this report's body, and the files
> themselves are not part of this commit. All other verification claims
> (RED/GREEN, HCU, full-pytest classification, pyright/ruff, restart audit)
> were executed on salt13 before the release as documented below.

## 1. Identity

| Field | Value |
|---|---|
| Hostname | `crdnotebook-2095693311250268161-salt13-93952` |
| Fix worktree | `/sakuramoon-runtime/sakuramoon-irepa-layout-fix` (single writer: this session) |
| Baseline source | `cd32bfc1df741c7b90ab921b83c716567bb88400` (branch `iprea-phase6b-canary`, approved source commit) |
| Branch | `fix/irepa-token-spatial-layout` created from the exact full SHA; no rebase/merge from dev |
| Frozen historical worktree | `/sakuramoon-runtime/sakuramoon-iprea-phase6b` — untouched (still clean @ cd32bfc) |
| Python / torch | 3.11.9 / 2.9.0+das (DTK venv `/sakuramoon-runtime/sakuramoon-dtk-venv`, `/opt/dtk/env.sh` sourced) |
| Devices | 2× idle DCU (HCA 68.7 GB each), verified idle before all GPU runs |
| BASE `irepa.py` sha256 | `5196ae66576972c2d3949560bb37140a1fc2a7ee6db9082c4b5d12e91b77eab7` |
| FIXED `irepa.py` sha256 | `f8d0856f6cdf09b72516222d90ef69b4e77d0c0c8976438144f13eae6b573f87` |
| Module import path (verified) | `PYTHONPATH=src …import sakuramoon.model.irepa → __file__ = …/sakuramoon-irepa-layout-fix/src/sakuramoon/model/irepa.py` |

## 2. Root cause

- **input_layout**: `TrainableComposite._project_student_capture` hands `IRepaAlignment.forward` a **token-major `[B, T, D]`** tensor with `T = H*W` row-major (`t = y*W + x`). Dense path passes `[B, T, D]` through; packed path does `capture.reshape(batch, H*W, D)` on the flat per-sample row-major image span (`PackedDiT.forward_packed_tapped` docstring: "flat in per-sample row-major sample order").
- **expected_index**: the Conv2d input must be channel-first with `spatial[b, c, y, x] == image_hidden[b, y*W + x, c]` — each token keeps its spatial position and its feature channel. The teacher side agrees: the frozen PE-Spatial ViT flattens patch features `[B, E, H, W] → [B, H*W, E]` row-major (`pe_vision.py: x.permute(0,2,3,1).reshape(batch, -1, width)`), `spatial_zscore_target` normalizes over the token axis only, and `irepa_alignment_loss` is a token-aligned cosine on equal-shaped `[B, T, 768]`.
- **old_expression** (`irepa.py` L148-150, BASE): `image_hidden.reshape(batch, width, height, grid_width)` on the flat `[B, T, D]` storage. Flat index `p = i*H*W + j*W + k` of the new `[B, D, H, W]` view maps to token `t = p // D` and channel `c = p % D` of the source — **tokens and channels are mixed**. Concretely at production shape (D=2560, H=W=16): a 3×3 receptive window at fixed (j,k) over i=0..9 reads *token 0's channels 0..255* — the "neighbor" the 3×3 conv saw was the adjacent **feature channel**, not the adjacent **spatial token**. The output side (`features.flatten(2).transpose(1,2)`) was row-major-correct; the corruption is entirely in the conv-input construction.
- **fixed_expression** (the only runtime-semantic change in this commit):
  ```python
  spatial = image_hidden.transpose(1, 2).reshape(batch, width, height, grid_width)
  features = self.projector(spatial)
  return features.flatten(2).transpose(1, 2)
  ```
  `transpose(1,2)` then `reshape` realizes exactly `spatial[b, c, y, x] == image_hidden[b, y*W+x, c]` (non-contiguous reshape forces a copy — correct and cheap). No `.contiguous()` needed; the backend received contiguous input in all tested runs.
- **upstream_compensation**: **none**. Verified read-only: (a) `_project_student_capture` (train/step.py L273-326) performs no spatial transform; (b) `forward_tapped` / `forward_packed_tapped` capture the joint image span in row-major order (docstring + grid validation `T == H*W`); (c) `spatial_zscore_target` / `irepa_alignment_loss` (objective/irepa.py) preserve token order. The contract `spatial[b,c,y,x] == image_hidden[b,y*W+x,c]` is the one both student and teacher sides actually use, and the old code violated it on every forward at every λ>0.
- **old_test_self_reference**: `test_forward_square_grid_matches_conv2d_reference` (tests/unit/model/test_irepa_alignment.py) built its "reference" as `irepa.projector(image_hidden.reshape(2, INPUT_WIDTH, height, width))` — the *same wrong reshape* — so it passed on the buggy implementation and could never detect the layout bug. Corrected in this commit to an index-tensor reference `spatial[b,c,y,x] = image_hidden[b, y*W+x, c]`.
- **RUNTIME_ATTRIBUTION = CONFIRMED** (the historical eff1000 run imported the buggy code):
  1. `launch-eff1000.sh` sets `PROJECT_ROOT=/sakuramoon-runtime/sakuramoon-iprea-phase6b` and execs `${PROJECT_ROOT}/scripts/training_stack.sh`, which exports `PYTHONPATH="${PROJECT_ROOT}/src"` (L136) → module path resolves to the frozen worktree's `irepa.py`.
  2. Frozen worktree: HEAD `cd32bfc…` clean (verified twice this session); `irepa.py` mtime `2026-09-04 22:29:48` predates the run window; `__pycache__` pre-dates the run.
  3. Run telemetry `artifacts/metrics.jsonl` carries `irepa_cosine_mean / irepa_lambda / irepa_loss / irepa_projector_grad_norm / irepa_weighted_loss` from update 113601 → the iREPA enabled path (exactly this code) was live throughout the run.
  4. Fresh import check in an identical env: `PYTHONPATH=src` resolves `sakuramoon.model.irepa.__file__` to the worktree file (sha `5196ae66…` at BASE).

## 3. Proof

### 3.1 RED (BASE, pre-fix) — independent index oracle

New test file `tests/unit/model/test_irepa_spatial_layout.py` — the expected spatial mapping is constructed **directly from the index definition** (nested loops for small grids; index tensors for production width), never via the production expression. Selector convolutions (single active 0/1 kernel tap, zero bias) on deterministic 0/1 BF16 pulses make layout assertions **exact** (no tolerance to hide behind); FP32 is used only for the one arbitrary-weight reference derivation.

RED evidence (BASE @ cd32bfc, full log: `reports/irepa-layout-fix/red-baseline-cd32bfc.txt`, sha256 `42df898616e575a935a39f891756064e4f9de789ce2322c1bd9521b0f47198cc`): **8/8 failed** — all with the expected index/gradient assertion failures (exact `torch.equal` mismatches; the float reference showed 96.5% mismatched elements, max abs 1.68). No import/device errors.

Coverage (all on the real `IRepaAlignment` module, locked contract out=768/kernel=3/stride=1/padding=1):

| # | Test | Contract exercised |
|---|---|---|
| A1 | `test_center_tap_reads_exact_token_channel_square` | square grid, token/channel encoding (two (ci,co) pairs) |
| A2 | `test_center_tap_reads_exact_token_channel_non_square` | non-square 3×7 (kills transposed `x*H+y` interpretations), multi-batch |
| A3 | `test_center_tap_multi_batch_isolation` | 3 batches, independent per-batch index mapping + explicit oracle-vs-input cross-check |
| B | `test_center_tap_non_contiguous_input` | strided `[B,T,D]` view (token-stride 2D), same index relation |
| D | (A1-A3 center tap) | each output token reads **its own** input token's chosen channel |
| E | `test_neighbor_tap_reads_spatial_neighbor_with_zero_padding` | taps (0,0) and (2,2): reads the true spatial neighbor token, zero-padding at the boundary — not the adjacent channel |
| F | `test_gradient_lands_on_exact_input_token_channel` | loss on one output token/channel → input grad exactly `tokens[b0,t0,ci]` on the read token/channel only, other batch exactly zero; weight grad mirrors the full index read-map `wgrad[7,ci,dy,dx] = tokens[b0,(y0+dy-1)*W+(x0+dx-1),ci]`; bias grad exactly the probed channel |
| G | `test_production_width_2560_index_contract` | D=2560, square 16×16, exact 0/1 index contract |
| H | `test_random_weight_matches_independent_reference` | arbitrary bounded weights: conv on the index-built spatial reference (FP32), torch BF16 tolerance (rtol=1.6e-2); a layout error is O(1), far above tolerance |

### 3.2 GREEN (fixed)

- All 8 layout/gradient tests **PASS** (exact assertions exact; H uses the framework BF16 tolerance).
- Corrected `test_forward_square_grid_matches_conv2d_reference` **PASS** (index-built reference; same tolerance rationale).
- `test_mixed_precision_conv.py` 11/11 PASS.
- Verified directly: fixed module output is **bit-identical** to `projector(index-built spatial)` (0 mismatched of 393216) — the residual 244/393216 1-ULP differences seen when comparing two *separate* conv invocations are backend reduction nondeterminism (measured max abs 3.9e-3), which is why H and the corrected reference test use the BF16 tolerance. Layout errors remain O(1).

### 3.3 HCU validation

`HCU_validation = PASS` — on the isolated idle DCUs (verified: 2× 68.4 GB free, no training/pytest processes): `tests/gpu/irepa/` (teacher geometry + reference parity + iREPA training graph + checkpoint save) and `tests/gpu/data/test_pipeline_encoders.py` all PASS with the fixed code (70 passed / 5 known-fail in the subset run — see §4.4).

### 3.4 Production width

`production_width_2560 = PASS` — test G (exact index contract at D=2560, 16×16) + parameter identity §4.3.

## 4. Regression

### 4.1 Targeted suites (fixed worktree, CPU unit + DTK)

| Suite | Result |
|---|---|
| tests/unit/model/test_irepa_spatial_layout.py (new) | 8/8 PASS |
| tests/unit/model/test_irepa_alignment.py (incl. corrected reference test) | 10/10 PASS |
| tests/unit/model/test_mixed_precision_conv.py | 11/11 PASS |
| tests/unit/train/test_irepa_composite.py + test_irepa_zero_impact.py (tapped capture / dense-packed bridge, composite) | PASS |
| tests/unit/checkpoint/test_irepa_artifact.py | PASS |
| tests/unit/optim/test_irepa_optimizer_routing.py | PASS |
| tests/unit/objective/test_irepa_objective.py | PASS |
| targeted total | 42/42 PASS |

### 4.2 lambda0 zero-contribution (criterion met exactly)

`test_enabled_lambda_zero_is_bit_identical_to_disabled` (PASS) asserts, with a finite auxiliary graph in place:
- weighted auxiliary term kept in the graph as `0.0 * per_sample` → **exactly zero** contribution to the loss;
- **every legacy (main-path) gradient bit-identical** (`torch.equal` per parameter) to the disabled (alignment=None) run; predictions and main loss bit-identical;
- projector graph reachable but **grad exactly zero** (`torch.equal(grad, zeros)`), `irepa_projector_grad_norm == 0.0`.

Scope note (per gate): this proves the λ=0 *gradient contribution* is exactly zero for the update under test (plain-SGD adapter, no momentum); it is not a claim that auxiliary parameters never move under momentum/weight decay at λ>0 — at λ>0 the auxiliary gradients are legitimately nonzero by design (the fix changes their *routing*, not their existence).

### 4.3 Parameter identity (pre/post, both trees)

| FQN | shape | dtype | numel |
|---|---|---|---|
| `projector.weight` | (768, 2560, 3, 3) | bfloat16 | 17,694,720 |
| `projector.bias` | (768,) | float32 | 768 |

Byte-identical FQN/shape/dtype/numel between BASE and FIXED worktrees. **Projector total numel = 2560·768·3·3 + 768 = 17,695,488** (computed, not inherited). Locked contract unchanged: in=2560, out=768, kernel=3, stride=1, padding=1, dilation=1, groups=1, bias=True, schema v1. BF16 input dtype guard unchanged (TypeError otherwise).

### 4.4 Full pytest (final candidate) + same-env baseline classification

Final candidate, full `tests/` (unit+gpu): **6 failed, 1003 passed, 6 skipped, 44 errors** (1240 s). Log: `reports/irepa-layout-fix/full-pytest-fix-final.txt`.

All 50 non-passes are **not attributable to the fix**:

| Group | Count | Attribution |
|---|---|---|
| `tests/gpu/irepa/test_pe_spatial_{geometry,reference}.py` errors | 44 | Environment: the fresh worktree had no `model/` assets; the fail-closed asset validator (`assets/pe_spatial.py`) rejects symlinked assets ("escapes the repository root"). With real (hardlinked) assets in place, **all 44 PASS** (fix worktree subset run, §3.3). |
| `tests/gpu/data/test_pipeline_encoders.py::test_real_pipeline_qwen_and_mage_encode_one_batch` | 1 | Same missing-asset environment; **PASSES** with assets in place. |
| `tests/gpu/fa4/test_varlen_attention.py::…[host_metadata]` | 1 | Known pre-existing P5 baseline bug (recorded in project memory); fails **identically on clean cd32bfc** same-env baseline. |
| `tests/unit/optim/test_cmuon_fp32_forensic.py` (4 tests) | 4 | Known F2 forensic semantic difference (tests expect old hard-fail semantics); fail **identically on clean cd32bfc** same-env baseline. |

Baseline attribution run (clean cd32bfc worktree `/sakuramoon-runtime/sakuramoon-layout-baseline`, same venv/assets): **the 5 failing tests fail identically** — `diff` of failure sets: **identical**. Net: **zero new failures, zero new errors attributable to the fix.**

### 4.5 Static gates

- **ruff** (venv `ruff~=0.16`): `All checks passed!` on all changed files (`irepa.py`, `test_irepa_alignment.py`, `test_irepa_spatial_layout.py`, `diag_restart_readiness.py`).
- **pyright** (strict, `include = ["src/sakuramoon", "tests"]`, same env, explicit scope): baseline clean cd32bfc = 4939 diagnostics; fix worktree = **4939** (delta 0; rule-level multiset diff ignoring line numbers: 0 new, 0 gone). The new test file was written to add no diagnostics (Optional-narrowing for `projector.weight/bias/grad`, deterministic weights instead of `torch.manual_seed`).

## 5. Historical experiment scope (append-only)

The historical eff1000 experiment (113600 → 114600, ckpts 113400/113600/114600 lineage, hub, logs, W&B, boards, reports) is **preserved as-is**. Because the runtime source of that experiment (RUNTIME_ATTRIBUTION = CONFIRMED, §2) contained this layout bug:

- The old experiments remain valid **stability evidence of the OLD (buggy-layout) implementation**: run/save/resume/eval mechanics, gate behavior, telemetry consistency.
- They are **not** effectiveness evidence for correctly-laid-out iREPA, and they cannot prove the iREPA method valid or invalid. `IREPA_EFFECTIVENESS_1000U = STABLE_INCONCLUSIVE` stands, now with the added fact that the 1000u aux gradients were computed on a scrambled conv input.
- **"checkpoint 能加载 ≠ 辅助目标语义兼容"**: the 113600/114600 checkpoints load fine and remain loadable, but resuming from them as a *continuous experiment of the fixed implementation* is **not allowed** — the trunk has already absorbed wrong auxiliary gradients; resetting the projector alone does not undo that.
- Future fixed-implementation training must use: **new code hash (this branch), new run_id / W&B identity, new output/ckpt paths, an independent timeline.** "Only 200u at final lambda" must not be used as a long-horizon verdict.

## 6. Restart readiness (read-only verification, §8)

Candidates: original no-iREPA `/sakuramoon-runtime/p6b-source/ckpt_113400_raw-113400-update-cadence` and migrated `/sakuramoon-runtime/p6b/migrated/ckpt_113400_irepa`.

Audit: `scripts/diag_restart_readiness.py` (read-only, in-memory only). Evidence: `reports/irepa-layout-fix/restart-readiness-audit.txt` — **17/17 PASS**:

- trunk model shards **bit-identical** original vs migrated (2,088,736,024 + 1,104,529,088 bytes);
- RNG state (`rank-0.safetensors`, `optimizer_sr.safetensors`) **bit-identical** (mtimes = original save time, 22:23);
- `trainer_state.json` identical: `successful_updates == attempted_updates == 113400` → **zero training-update history** after the save;
- optimizer: CMuon FQN set **unchanged** (141 params); AdamW set = source + **exactly** `irepa_alignment.projector.{weight,bias}`; **all 148 source AdamW state entries bit-identical** after id renumbering (incl. torchao 8-bit `OptimState8bit` payloads compared at the `codes/scale/qmap` level); CMuon / sr_rng / transition blocks bit-identical;
- projector FQNs present in groups with **no AdamW state entry** (pristine lazy init — no update ever applied);
- migrated projector shard == deterministic reconstruction from `migration_seed=20260904` via `IRepaAlignment(2560)` init, **bit-exact** (weight BF16 + bias FP32);
- `irepa_state.json`: `source_update=113400`, `start_successful_update=113401` → **lambda anchor binds exactly 0.0 at the first update 113401** (schedule contract: u==start ⇒ λ exactly 0.0).

`RESTART_READINESS = PASS` for the migrated ckpt_113400 (the original no-iREPA ckpt passes every trunk/RNG/trainer_state check; the migrated one adds the projector in its approved initial state). No migration, no state reset, no new checkpoint, no training was started in this round.

## 7. Verdict

```
== SakuraMoon iREPA Token-Spatial Layout Fix ==
IDENTITY: salt13 worktree /sakuramoon-runtime/sakuramoon-irepa-layout-fix; branch fix/irepa-token-spatial-layout @ cd32bfc1df741c7b90ab921b83c716567bb88400; frozen iprea-phase6b worktree untouched
ROOT_CAUSE: bare image_hidden.reshape(B,D,H,W) on token-major [B,T,D] mixed tokens with feature channels (3x3 conv read adjacent feature channels, not adjacent spatial tokens); expected_index spatial[b,c,y,x]==image_hidden[b,y*W+x,c]; fixed_expression image_hidden.transpose(1,2).reshape(B,D,H,W); upstream_compensation NONE; old test self-referential (same wrong reshape as reference)
PROOF: RED 8/8 index/gradient assertion failures on BASE (evidence red-baseline-cd32bfc.txt); GREEN 8/8 + corrected reference PASS; HCU_validation PASS (idle DCU, gpu/irepa all green); production_width_2560 PASS (exact index contract + numel 17,695,488); gradient_landing PASS (input/weight/bias grads mirror the exact index read-map; other batches exactly zero)
REGRESSION: targeted 42/42 PASS (alignment/conv/composite/bridge/artifact/optimizer/lambda0); lambda0 exact-zero criteria met (bit-identical main path, exact zero aux grad); parameter identity byte-identical pre/post; ruff zero new; pyright 4939->4939 delta 0; full pytest 6F/44E all attributed to environment or pre-existing baseline (identical failure set on clean cd32bfc same-env); zero new failures
HISTORICAL_EXPERIMENT: eff1000 (113600->114600) = stability evidence of the OLD buggy-layout implementation only; NOT effectiveness evidence for correctly-laid-out iREPA; resume from 113600/114600 as continuous fixed-implementation experiment NOT allowed; projector reset alone does not undo absorbed wrong aux gradients
RESTART_READINESS: PASS (migrated ckpt_113400_irepa: 17/17 audit checks; trunk+RNG+optimizer bit-identical to pristine original; projector pristine lazy init == deterministic seed-20260904 init; lambda anchor first update 113401; zero update history). Future fixed training: new code hash, new run_id/W&B identity, new output/ckpt paths, independent timeline
VERDICT: IREPA_LAYOUT_FIX = PASS
READY_FOR_SEPARATE_RETRAINING_GATE = YES
NEXT = STOP WAIT_FOR_USER_REVIEW
```
