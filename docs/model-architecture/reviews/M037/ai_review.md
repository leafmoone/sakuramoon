# M037 independent AI/model correctness review

Reviewer: independent agent `/root/m037_independent_review`

Date: 2026-08-01

Reviewed implementation: commit
`c7b2e90a293aaaddb39015d0ab4c86f6b7c9af39` as present in the current
committed tree. Concurrent unrelated K001 worktree changes were excluded. No
`reference/` code was read, imported, or executed.

## Verdict

**PASS.** No AI/model-correctness finding blocks M037 review closure.

## Findings

No blocking or non-blocking correctness finding was identified.

- The M037 objective diff changes only the observation mask boundary. The strict
  JLT prediction/target x-to-v conversion, FP32 shared clamp, squared error,
  feature-then-sample reduction, and returned full-batch
  `loss=per_sample.mean()` are unchanged. Neither bucket sum nor count feeds the
  returned loss.
- An independent CPU calculation over timesteps on both sides of `0.95` matched
  the implementation's loss, per-sample losses, and prediction gradients with
  maximum absolute difference `0`. The same BF16-input/FP32-objective comparison
  on one RTX 5090 matched loss and gradients with maximum absolute difference
  `0`.
- The half-open classification is exact: `t<0.95` is high noise and `t>=0.95`
  is low noise. A batch containing `0.94` and `0.95` produced one sample in each
  bucket while retaining the unchanged full-batch mean.
- M037 does not modify `guided_velocity` or any sampler implementation. The
  existing contracts continue to prove conditional and unconditional x-predictions
  are independently converted before CFG and that the fixed Euler/Heun profiles
  retain their solver, NFE, endpoint, and FP32-state identities.
- `logging.noise_observation_boundary` is a required exact TOML float fixed to
  `0.95`; missing, wrongly typed, unknown, and drifting configuration remains a
  schema hard failure. The production single-GPU runtime passes the resolved
  value into every objective call, whose own locked-value check rejects drift.
- Historical identity is preserved: the changelog still records the prior
  `t=0.5` observation-only decision and appends the `0.95` revision. Existing
  history was not rewritten into a claim that upstream JLT defines either
  observation threshold.

## Independent validation

- Focused objective/config CPU contracts: `91 passed`.
- Objective/sampling/config/checkpoint CPU regressions: `184 passed`, with 17
  pre-existing dependency deprecation warnings.
- One-GPU objective forward, FP32 loss, backward, SGD update, CFG, and Heun smoke:
  `1 passed` on NVIDIA GeForce RTX 5090.
- Targeted Ruff: passed.
- Targeted strict Pyright: `0 errors, 0 warnings`.
- Trace contracts: `40 passed`; live verifier: `237/237` source requirements,
  107 production modules, and 271 runtime config keys.

## Remaining boundaries

- The GPU evidence is a bounded component smoke, not a full production
  data/Qwen/VAE/conditioning/DiT/loss/checkpoint training step. T050 retains that
  integration boundary.
- T041 retains global-sample-mean and all DDP/multi-GPU correctness. No four-GPU,
  NCCL, long-run, formal-stage, generation-quality, FID, or IS conclusion is
  inferred from M037.
