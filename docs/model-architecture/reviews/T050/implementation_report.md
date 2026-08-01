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

The trainer-side runtime now accepts only typed `TrainingBatch` iterators assembled by
D025's governed production factory, transfers one batch to CUDA, runs local frozen
Qwen and Mage-VAE, constructs strict JLT state/noise/timesteps, and forwards the
trainable conditioning/DiT composite. It exposes no independent batch size, worker,
queue, pin-memory, drop-last, pipeline, or data-client control. The loop requires an
explicit checkpoint-cadence anchor matching restored successful-update state,
supports forced stage/growth reasons, and advances the wall/update anchor only after
a durable checkpoint callback succeeds.

The command entry remains deliberately fail-closed because the strict config does not
yet expose every Text and Style constructor choice; it does not hard-code the values
used by bounded engineering tests. The runtime assembly API itself was exercised with
real local Qwen/Mage-VAE, the 16-layer PackedDiT, JLT loss, backward, FP32 clip, and
one TorchAO update on an RTX 5090. A separate real AF_UNIX D024 service test exercised
two persistent workers and normal-exhaustion ACKs.
