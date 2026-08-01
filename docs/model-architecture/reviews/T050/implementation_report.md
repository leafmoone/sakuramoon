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

The resolved S0 stage enablement, topology, depth slots, resolution, growth-ramp
presence, alpha, zero budget origin, planned budget, remaining work, and checkpoint
cadence are bound to the restored RAW state during mandatory preflight and again
before the runtime consumes a batch. The configured attention backend is separately
bound to the assembled DiT artifact; the dense SDPA adapter delegates the same
artifact/model metadata contract without changing the model package.

Production preflight executes backward for all 17 image and 8 text shape probes,
including the real empty-condition framing/mask contract, and clears probe gradients
without advancing optimizer durable state. CPU data wait and checkpoint publication
use monotonic wall-clock timing; asynchronous CUDA phases remain on `PhaseTimer` and
do not introduce data/checkpoint device synchronizations.

Every cadence RAW is registered as pending by the concrete publisher. The runtime
then performs a complete RAW readback, checks the exact config/dependency/parameter
identity and trainer/growth/stage-budget/cadence state, and only then requests
retention. Readback, identity, or state failure issues no retention. The old
caller-supplied fresh/qualified retention capability was removed. A spawned
post-training integration process independently restores the durable RAW before
connecting to the data service and executes the next update; this validation is not
in the per-checkpoint hot path.

Fresh resume loads frozen Qwen and Mage-VAE before RAW restore, leaving checkpoint
restore as the last model-assembly operation that mutates training RNG state. A
mandatory preflight or report-publication failure closes the bound D025 stream exactly
once; if cleanup also fails, the primary and close errors are both retained. A
successful preflight keeps the stream live for the training invocation.

The command entry remains deliberately fail-closed pending C002's production binding
of the already-confirmed Text and Style decisions into strict config, overlays, and
assembly; it does not promote bounded engineering-test values into production. The
runtime assembly API itself was exercised with
real local Qwen/Mage-VAE, the 16-layer PackedDiT, JLT loss, backward, FP32 clip, and
one TorchAO update on an RTX 5090. The real AF_UNIX D024 integration exercised two
spawned persistent workers, durable checkpoint readback/retention, replay after
restart, and the post-restore next update.

The final fresh-resume rerun first exposed an observation race: update 1 restored and
update 2 completed, both worker leases were active/replayed, but the test closed the
stream immediately after its first batch and observed only one tokenizer marker. The
test now records DataLoader worker ID and PID, requires exact IDs `{0,1}`, two distinct
child PIDs, and both issued leases behind a bounded progress barrier, and keeps the
stream open through update 2. It neither drains nor ACKs an unused second batch. The
strengthened worker contract passed repeatedly on the RTX 5090. One later rerun
stopped before service readiness under the old 15-second boolean barrier; it did not
enter preflight or training. Startup now distinguishes early child exit from bounded
timeout and deterministically stops an unready child. The final frozen full-chain run
was 2 passed in 275.56 seconds, restored update 1 to update 2, and observed exact
worker IDs `{0,1}` with four distinct child PIDs across the initial and resume runs.
