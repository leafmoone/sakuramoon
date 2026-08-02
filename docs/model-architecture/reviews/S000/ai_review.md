# S000 AI/model correctness self-review

Reviewer authority: main agent, under the user's no-agent direction. This is not an
independent review.

## Verdict

PASS for the bounded synthetic single-GPU engineering scope only.

## Findings

No blocking AI/model-correctness issue remains in the implemented scope.

- The final checked TOML fixes one visible CUDA device, world size 1, S0 depth 16,
  resolution 256, local batch 1, dense SDPA reference, two total successful updates,
  growth alpha 1.0, and observation boundary 0.95. Unknown or changed semantics fail
  schema validation; the CLI exposes no training-semantic override.
- The actual local Qwen and Mage-VAE run before the actual text/style conditioning and
  native 16-layer dense DiT. The report contains all required forward, JLT loss,
  backward, clip, TorchAO optimizer, and zero-grad phases for updates 1 and 2.
- The raw checkpoint binds successful update 1 and the exact resolved config. A fresh
  process restores update 1 and advances exactly to update 2; it does not use a PMA,
  model-only artifact, latest-checkpoint fallback, or data-service cursor sidecar.
- Production `sakuramoon.cli.train` remains unchanged and gated. The runner and report
  force `formal_s000=false` and all production capacity/quality/unlock claims false.

## Residual gates

The local tar content is synthetic and the run is two updates on one RTX 5090. It
cannot validate real-data distribution, training quality, FID/IS, evaluator identity,
maximum batch, sustained throughput, 1,000-update stability, DDP/NCCL, four-rank state,
or any formal stage. Formal S000/S001 and `P060-P067` remain blocked by the five
recorded prerequisites.

## Independent production-readiness review (2026-08-02)

Reviewer authority: independent `s000_ai_reviewer`. No GPU command was run and no
formal-stage evidence is inferred from the retained bounded engineering smokes.

### Verdict

BLOCKED. The production training lifecycle has the expected fail-closed fresh/raw
resume structure, but the current tree is not ready for a user S001 launch. In addition
to the explicitly recorded configuration/runtime blockers, the formal evaluator has
three model-correctness defects that must be remediated before it can publish governed
stage-end evidence.

### Findings

1. **S000-AI-001 (blocking): stage-end checkpoint comparison is optional.**
   `eval/runner.py::preflight_evaluator` accepts any nonempty checkpoint selection and
   does not require the governed `raw`/`pma`/`accepted` set. The runner merely skips a
   comparison when the set is incomplete, so a raw-only `--stage-end` run can publish
   `fid_formal`, `is_formal`, and manual-quality artifacts. It also does not verify that
   the PMA artifact is the PMA-10 window ending at the selected raw trigger update.
   This conflicts with T052's raw-latest/PMA-10/accepted same-protocol comparison.
2. **S000-AI-002 (blocking): objective provenance is asserted, not verified.**
   `cli/eval.py::_checkpoint_selection` silently assigns `strict_jlt` to raw, PMA, and
   accepted roles, and preflight copies that value into artifacts without inspecting a
   raw `resolved_config.toml` or PMA/release source provenance. A pre-fix artifact can
   therefore be labeled strict JLT. Current decisions allow old-objective weights only
   as explicitly marked `pre_fix` model-only inference and require strict objective
   identity for governed evaluation.
3. **S000-AI-003 (blocking): the formal t=0 state is sampled in BF16.**
   `eval/generate.py::CheckpointGenerator.generate` creates each initial-noise tensor as
   BF16 and only then lets the solver cast it to FP32. This irreversibly quantizes the
   initial solver state while generation metadata reports `state_dtype=float32` and the
   confirmed reference contract requires an FP32 state throughout Heun-50/final-Euler.
4. **S000-AI-GATE-001 (explicit unresolved contract): governed prompt conditions are
   not executable.** Any nonempty `PromptCase.conditions` produces
   `PROMPT_CONDITION_CONTRACT_UNRESOLVED`. This is an appropriate fail-closed response
   because the flat condition-to-caption-field semantics are not governed, but it means
   an identity-complete condition manifest still cannot execute tag-control evaluation.
   This gate must remain explicit; condition semantics must not be invented in code.

### Training-path assessment

The reviewed production path runs static checks before CUDA/service/bootstrap, accepts
only fresh start or one canonical absolute raw `COMPLETE` directory, restores raw model,
optimizer, RNG, config, dependency, and parameter identities before connecting to the
data service, and advances scheduler/checkpoint counters only after a successful update.
The JLT loss, FP32 clip, TorchAO update, telemetry learning-rate capture, durable raw
readback, and fresh-process N-to-N+1 structure are internally consistent in the reviewed
scope. No additional AI/model defect was found in that path.

Launch remains independently blocked by 59 unresolved bindings in each of
`train_s0.toml` and `eval.toml`, plus
`S0_WARMUP_FUNCTION_UNRESOLVED`, `S0_PASS_INDEX_OWNERSHIP_UNRESOLVED`,
`S0_LIVE_READY_QUEUE_DEPTH_UNBOUND`, and `S0_DIT_FLOPS_OBSERVATION_UNBOUND`.
The capacity sweep, formal evaluator run, and S001 were not run and cannot be closed by
the synthetic bounded evidence.

### CPU/static evidence

- Focused S000 train/eval/config selector: 245 passed, 17 warnings.
- Ruff on the affected train/eval/config paths: passed.
- Pyright on the affected train/eval/config paths: 0 errors, 0 warnings.
- Archive-free unresolved-binding inspection: `train_s0.toml=59`, `eval.toml=59`.

## Independent post-remediation rereview (2026-08-02)

Reviewer authority: independent `s000_ai_reviewer`. This rereview is append-only and
does not replace the earlier findings. No GPU, formal-stage, or long-running command
was run.

### Verdict

BLOCKED. `S000-AI-001`, `S000-AI-002`, and `S000-AI-003` are resolved, but the
checkpoint-driven stage-end evaluator still cannot execute the governed older
accepted-release comparison. The unresolved prompt-condition contract and the
production configuration/runtime bindings independently remain hard gates.

### Finding status

- **S000-AI-001: RESOLVED.** Preflight now requires exactly one each of `raw`, `pma`,
  and `accepted` for stage-end evaluation, and execution independently rejects a
  forged incomplete stage-end plan. PMA provenance requires exactly ten unique,
  strictly increasing same-lineage sources ending at the selected raw trigger, with
  matching stage/world-size/resolution/active-slot topology and raw alpha 1.0.
- **S000-AI-002: RESOLVED.** Raw strict-JLT provenance now comes from the checksummed
  checkpoint `resolved_config.toml`; PMA provenance is derived from its verified source
  chain; a CLI-asserted strict-JLT model-only artifact is rejected. Explicit `pre_fix`
  model-only inference remains isolated from governed raw/PMA/accepted evaluation.
- **S000-AI-003: RESOLVED.** Per-case seeded Gaussian noise is now created directly in
  FP32, and the reference Heun-50/final-Euler solver retains its FP32 state contract.
- **S000-AI-004 (blocking): an older accepted release cannot satisfy executable
  provenance.** T052 explicitly permits the accepted comparison checkpoint to be
  older than the evaluation trigger, and `CheckpointRef` supports a release artifact
  in the accepted role. However, `_validate_release_strict_jlt` requires the accepted
  release's recorded source identity to equal the *currently selected* PMA identity.
  The current PMA is itself required to end at the raw checkpoint for the exact trigger
  update, while `save_release` requires a release update to equal its PMA source update.
  Consequently an accepted release can pass only at the current trigger update, not at
  an older accepted update. The stage-end test avoids this path by using an older raw as
  `accepted`, so it does not cover the governed release case. Remediation must verify
  the accepted release against its own historical PMA provenance without conflating it
  with the current comparison PMA, and add a preflight test with current raw/PMA at
  update N and an accepted release sourced from a verified PMA at M where M < N.

### Remaining explicit gates

`S000-AI-GATE-001` remains correctly fail-closed: every stage-end plan receives
`STAGE_END_PROMPT_CONDITION_CONTRACT_UNRESOLVED`, so no formal artifact can be
published while prompt-condition semantics are ungoverned. Merged `train_s0.toml` and
`eval.toml` each retain 60 unresolved bindings (43 benchmark and 17 required), plus
the five non-TOML blockers for warmup semantics, pass-index ownership, formal prompt
conditions, live ready-queue depth, and actual DiT FLOPs observation. Bounded synthetic
evidence does not release formal FID/IS, S001, sustained training, or four-GPU gates.

### CPU/static evidence

- Evaluator unit suite: 119 passed, 17 warnings.
- Focused readiness/evaluation/config contracts: 78 passed, 17 warnings.
- Ruff on evaluator implementation, CLI, and evaluator tests: passed.
- Pyright on evaluator implementation, CLI, and evaluator tests: 0 errors, 0 warnings.
- `git diff --check` on the reviewed evaluator/report paths: passed.
- Archive-free merged-config inspection through `unresolved_config_bindings`:
  `train_s0.toml=60` and `eval.toml=60`, each split 43 benchmark plus 17 required.

## Older accepted-release post-remediation rereview (2026-08-02)

Reviewer authority: independent `s000_ai_reviewer`. This section appends to, and does
not rewrite, either prior review. No implementation/configuration file was modified by
the reviewer, and no GPU, formal-stage, or long-running command was run.

### Verdict

PASS for the `S000-AI-004` remediation and the bounded evaluator CPU/control-plane
scope. No remaining implementation defect was found in the reviewed checkpoint
provenance chain. Overall S000 remains BLOCKED by the already recorded governed
prompt-condition contract, 60 merged configuration bindings, and five non-TOML
readiness blockers; this verdict does not authorize formal evaluation or S001.

### Finding status

- **S000-AI-001: RESOLVED, unchanged.** Stage-end still requires exactly current
  `raw`, current `pma`, and `accepted` roles, and current PMA remains exactly anchored
  to the raw checkpoint at the trigger update.
- **S000-AI-002: RESOLVED, unchanged.** Strict-JLT provenance remains derived from
  checksummed raw config and PMA source lineage rather than from a CLI assertion;
  explicit `pre_fix` remains restricted to model-only inference.
- **S000-AI-003: RESOLVED, unchanged.** Initial Gaussian state remains directly
  sampled in FP32 for the FP32 reference solver.
- **S000-AI-004: RESOLVED.** The CLI now accepts a separate canonical absolute
  `--accepted-source-pma` and binds it only to exactly one accepted checkpoint. An
  accepted release without that path fails with
  `ACCEPTED_RELEASE_SOURCE_PMA_REQUIRED`; supplying it for a non-release accepted raw
  artifact also fails. Preflight reads the selected source PMA as its own complete
  checkpoint, so its `pma_sources.json` is covered by the checkpoint payload-set,
  symlink, size, and SHA-256 checks before provenance is evaluated. It then requires
  PMA kind, exactly ten unique strictly increasing raw identities, output update equal
  to the window tail, common config/dependency/parameter-schema lineage, no update
  newer than the current raw trigger, and the current raw's stage/world-size/
  resolution/active-slot topology. The release's checksummed `release_source.json`
  must name that exact independently verified PMA; release update and lineage must
  match it and `automatic_release` remains false. Crucially, only the current PMA uses
  the exact-anchor rule. The accepted source PMA may end at M < N, so a current raw and
  current PMA at N can be compared with an accepted release sourced at M.

The stage-end integration test now uses current raw/PMA update 10 plus accepted release
and source PMA update 9. After both current and historical PMA chains and release source
identity are accepted, its sole blocker is the pre-existing
`STAGE_END_PROMPT_CONDITION_CONTRACT_UNRESOLVED`. That is the correct fail-closed result
until the governed prompt-condition mapping exists; bounded tests do not publish or
stand in for formal stage-end evidence.

### CPU/static evidence

- Evaluator unit suite: 119 passed, 17 warnings.
- PMA/release checkpoint policy suite: 20 passed, 17 warnings.
- Ruff on evaluator implementation, CLI, and evaluator tests: passed.
- Pyright on evaluator implementation, CLI, and evaluator tests: 0 errors, 0 warnings.
