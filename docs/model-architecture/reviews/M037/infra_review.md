# M037 independent Infra/performance review

Reviewer: independent agent `/root/m037_independent_review`

Date: 2026-08-01

Reviewed implementation: commit
`c7b2e90a293aaaddb39015d0ab4c86f6b7c9af39` as present in the current
committed tree. Concurrent unrelated K001 worktree changes were excluded. No
`reference/` code was read, imported, or executed.

## Verdict

**PASS.** No Infra/performance finding blocks M037 review closure.

## Findings

No blocking or non-blocking Infra finding was identified.

- M037 retains the pre-existing vectorized bucket implementation: one timestep
  comparison, one complementary mask, and four scalar sum/count reductions. It
  changes the comparison value from the historical inline `0.5` to the explicit
  locked `0.95` input; tensor count, asymptotic work, and allocation shape do not
  change.
- The added locked-float validation operates on the resolved Python float. It
  performs no tensor-to-host transfer, CUDA synchronization, per-sample loop, or
  backend fallback. The observation masks are created after the loss vector and
  cannot alter optimizer work.
- The production runtime forwards the resolved value directly into the objective.
  There is no silent default or silent downgrade to the prior boundary.
- This is an observability contract revision, not a performance optimization.
  A before/after performance artifact would be misleading; focused CPU/one-GPU
  correctness and synchronization review are proportionate evidence.

## Independent validation

- Focused objective/config CPU contracts: `91 passed`.
- Objective/sampling/config/checkpoint CPU regressions: `184 passed`, with 17
  pre-existing dependency deprecation warnings.
- One-GPU objective smoke: `1 passed in 3.52s` on NVIDIA GeForce RTX 5090.
- Independent CPU and GPU loss/gradient comparisons each reported maximum
  absolute difference `0`; the GPU comparison used BF16 inputs and the FP32
  objective path.
- Targeted Ruff and strict Pyright passed.
- Trace contracts: `40 passed`; live trace verification passed at `237/237`.
- `git diff --check` passed before review evidence was written.

## Remaining boundaries

- No production end-to-end throughput, memory profile, decoder cost, checkpoint
  amortization, or data-ready-wait claim is closed here. Those remain with
  T050/T051/T053 and later stage evidence.
- No DDP, NCCL, four-GPU, long-run, formal-stage, FID, or IS evidence was run or
  inferred. The one-GPU result cannot close any four-GPU performance gate.
