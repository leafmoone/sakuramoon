# T040 AI review

Status: PASS after independent remediation rereview; no blocking findings.

Initial findings:

- Role assignment accepted arbitrary FP32 matrices and unknown BF16 ranked tensors.
- The audit omitted text/style branches and did not lock the full composite schema.
- The 1,000-step canary compared final training loss rather than held-out validation EMA.
- State restore did not prove serialized next-step equality.
- Corrupt SR RNG dtype/shape was accepted until the next optimizer step.

Independent rereview confirmed that the role-negative tests, 239-FQN full-composite schema, held-out validation EMA canary, serialized bitwise next-step test, SR load validation and nonfinite pre-step failure close the initial findings. The reviewer reran the 27 CPU tests, Ruff, strict Pyright and `git diff --check`; all passed. The existing eight RTX 5090 tests and their evidence were reviewed without using them to claim four-rank correctness.

Four-rank model/moment/step/SR equality and strict distributed global-mean equivalence remain pending T041.
