# T051 Infra/performance review

Status: implementation-agent self-review PASS for implemented CPU and prior 1GPU
scope; independent package review and four-GPU measurement pending.

Local metrics precede asynchronous remote submission. Network exception messages are
not persisted, and a failing upload is converted into a durable retry record without
changing training controls. Retry replay is lossless on failure. CUDA timing queues
events and polls completion rather than synchronizing each phase.

The T050 adapter uses a bounded queue and a bounded event-query wait. Tensor scalar
extraction happens only after the recorded CUDA events report ready; the training
callback captures only the timestamp and explicit context. Queue saturation and local
durability failure are visible hard errors, never silent metric loss. Once accepted,
queued records are attempted even if an earlier publication fails; `close()` drains
the bounded queue and then propagates the retained first error. Existing retry files
must be private regular files, and non-finite retry values cannot reach W&B.

The focused single-GPU median overhead was 0.8057%. It does not establish full-chain,
DDP, checkpoint, or four-GPU overhead and was not rerun during this CPU-only change;
those measurements remain pending.
