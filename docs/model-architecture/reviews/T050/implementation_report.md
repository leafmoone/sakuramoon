# T050 implementation report

`SingleGpuTrainingLoop` creates one poisoned `SingleGpuStep` per attempted update,
consumes exactly the configured microbatch count, and advances scheduler and periodic
checkpoint callbacks only after the optimizer succeeds. Interrupted input or loss
construction now counts one failed attempt and clears all gradients accumulated by
earlier microbatches. Scheduler/checkpoint failures preserve the already successful
optimizer state and terminate immediately.

The preflight orchestrator requires every fixed category exactly once and exposes no
force parameter. Reports and diagnostic bundles omit exception messages. Diagnostic
publication collisions, dangling symlinks, and write failures are fail-closed; when
publication fails, an `ExceptionGroup` preserves both the training error and the
diagnostic error.

This is the CPU control plane, not an assembled training executable. Concrete checks
and the real full-chain single-GPU smoke remain pending until strict production config
can construct T022/T023 and C002 without inventing undecided values.
