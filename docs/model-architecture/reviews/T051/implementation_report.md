# T051 implementation report

`TrainingMetric` admits no free-form fields and validates every numeric value, all 12
dropout hit counters, and a fixed coarse+detailed phase vocabulary. The W&B projection
contains numbers only. `MetricsPublisher` writes the local JSONL record before placing
it on the remote queue.

`DurableJsonlSink` uses append-only, no-follow file opening, mode 0600, full-write
handling, explicit fsync cadence, tail fsync on close, and directory fsync on creation.
`AsyncWandbSink` catches remote failures in its worker and writes only exception type,
successful update, and numeric metrics to a durable retry queue. Replay deletes the
queue only after every record uploads successfully.

`PhaseTimer` uses `perf_counter_ns` for CPU and paired CUDA events for GPU. Event pairs
are collected only when `query()` reports completion; no phase boundary calls
`synchronize()`. NVTX ranges remain optional and balanced through exceptions.
