# D015 implementation report

The pipeline reads prepared local tar shards with the locked WebDataset library,
parses metadata once, excludes validation IDs before image decode, and composes the
D013 image and D014 caption contracts. Collate groups by target image shape and Qwen
dense length, keeps EOT padding masked, and preserves all structured indices and audit
metadata. Persistent worker and ready-batch counts are explicit and incompatible
queue budgets fail instead of being adjusted.

The Data package audit found that expected D013 image rejections aborted iteration.
The pipeline now catches only the typed `ImageRejected` boundary and skips that sample.
Two synthetic real-tar contracts exercise no-upscale and retention rejection followed
by a valid sample, proving that iteration continues without weakening other failures.

A synthetic real-tar 1/2/3-worker sweep found no duplicate or missing sample IDs. A
real local-tokenizer RTX 5090 smoke passed the serialized CPU pipeline, Qwen seven-state
forward, and Mage posterior-mean encode with both frozen models resident together.

Independent follow-up review found that the normal worker loader did not enter the
D012 lease path, local-only paths were enforced only by type hints, rejection reasons
were discarded, and collate accepted an unrelated padding ID. The corrected public
entry point drains each prepared shard under `SingleProcessShardCoordinator.lease`, so
normal exhaustion marks it complete while early failure leaves it active for replay.
Direct DataLoader worker iteration on an unmanaged base pipeline hard-fails, so callers
cannot silently select the old bypass path.
Pipeline construction accepts only unique absolute local regular files and strict
stage/pass/seed/boolean fields. An explicit observer receives each typed image rejection,
and every `PipelineSample` carries the framing padding ID that collate requires to agree
across the batch.

The D012 schema permits one active shard, whereas WebDataset's persistent workers split
at shard granularity. The durable entry point therefore rejects `worker_count != 1`
instead of silently changing it or claiming the earlier 1/2/3 mechanics sweep proves
resume. Production durable multi-worker design and throughput remain pending.

A committed GPU test now reproduces one real local tar through pipeline/collate, local
Qwen seven-state inference and Mage posterior-mean encode. It passed in 16.69 seconds;
no backward, optimizer update, training canary or multi-GPU path ran.
