# T051 implementation report

`TrainingMetric` schema v2 admits no free-form fields and validates every numeric
value, explicit high/low-noise sample counts, all 12 dropout hit counters, and the
complete fixed coarse+detailed phase vocabulary. Empty noise buckets use loss `0.0`
with count `0`, so absence is not confused with an observed zero loss. The W&B
projection contains numbers only. `MetricsPublisher` writes the local JSONL record
before placing it on the remote queue. Post-clip norm may not exceed pre-clip norm,
matching T050's exact `post = pre * coefficient` contract.

`DurableJsonlSink` uses append-only, no-follow file opening, mode 0600, full-write
handling, explicit fsync cadence, tail fsync on close, and directory fsync on creation.
`AsyncWandbSink` catches remote failures in its worker and writes only exception type,
successful update, and numeric metrics to a durable retry queue. Replay deletes the
queue only after every record uploads successfully.

`PhaseTimer` uses `perf_counter_ns` for CPU and paired CUDA events for GPU. Event pairs
are collected only when `query()` reports completion; no phase boundary calls
`synchronize()`. NVTX ranges remain optional and balanced through exceptions.

`AsyncTrainingMetricObserver` consumes T050's detached
`SuccessfulTrainingObservation` through a type-only import, so telemetry introduces
no runtime dependency cycle. A bounded worker waits for CUDA events using `query()`,
then aggregates total/high/low loss, bucket counts, gradient clipping, timestep
summary, tokens, dropout hits, memory, data wait, checkpoint time, and phase events.
It requires explicit FLOPs, samples/s, ready-queue depth, and supplemental phase facts;
it never guesses them. Records accepted before the first background publication error
are still attempted in FIFO order while the first error is retained for `close()`.
Queue exhaustion, event timeout, conversion errors, and local durability failures are
surfaced rather than silently dropping a successful-update record.

Existing JSONL files without exact mode 0600 are rejected. Retry replay additionally
rejects non-finite numeric payloads and a successful-update identity mismatch while
retaining the original queue on failure. C002 owns final production construction and
lifecycle ordering; this T051 change does not edit config or CLI paths.
