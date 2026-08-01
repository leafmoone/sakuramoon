# D025 implementation report

Status: explicit-spawn remediation and real service-to-worker CPU verification
passed; independent DATA rereview pending.

D025 introduces one governed production-data composition module. It owns the real
ModelScope nested metadata projection, caption field parser, strict validation-ID
manifest load, and the derivation of all DataLoader controls from resolved TOML. It
does not import or own ModelScope transport, cache publication/eviction, service
state, or checkpoint state.

The factory and generated pipeline now hard fail unless every worker-visible field
serializes through the explicit spawn pickler. The returned production batch source
is a factory-issued process-local accepted handle rather than a plain iterator. Its
identity binds the resolved-config digest, exact batch/worker/ready/pin/drop controls,
manifest digest, service-session digest, and a unique 256-bit factory identity. It
cannot be constructed with caller authority, serialized to another process, or used
after crossing a PID boundary.

The boundary preserves the D021 rule that release comes only from the trusted
service-issued `ShardRecord`: the adapter projects only ID, dimensions, and caption
availability, while `WebDatasetPipeline` still passes the original JSON to the
caption parser. The configured validation artifact must be canonical JSONL with an
exact configured digest and exactly 2,000 sorted unique positive IDs before a
production pipeline factory can be built.

A real independent AF_UNIX service process, real tar decoding, and two spawned
DataLoader workers completed four service leases with one ACK per normal shard
exhaustion. Parent early-close and worker decode-failure cases emitted no ACK, left
two active leases, and a fresh service replayed exactly two shards/two samples from
their starts. Test-only D024 consumer lambdas were replaced with equivalent
module-level callables so adjacency now exercises the same explicit-spawn contract.

Verification results are recorded separately in `test_report.json`. Production
manifest generation, the approximately 11M-ID scan, cold-cache/server-backed-storage
performance, exact real-parser GPU rerun, long training, and multi-GPU gates remain
pending or blocked and are not inferred from this task. T050 must separately require
this accepted handle at its production entry; D025 does not modify T050 files.
