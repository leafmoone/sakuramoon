# D010 Infra/performance review

Status: canonical builder expansion passed main-agent Infra review after one durability
finding was remediated. Fresh independent rereview is unavailable after two direct
agent-start failures.

The entry-count check is bounded by the already materialized remote listing and does
not alter download streaming or memory complexity. Redirect credential stripping,
bounded retries, `.partial` cleanup, byte/SHA verification, and atomic publication
remain covered by synthetic transport tests.

No live network, `.env`, model asset, GPU, DDP/NCCL path, long run, or performance
placeholder was used. Production network/access and throughput evidence remains
pending. The final conclusion is main-agent remediation acceptance because direct
independent re-review startup was unavailable.

Main-agent review found that the no-clobber hard link became visible before temporary
unlink and parent fsync, but the generic error path did not remove it when either later
operation failed. The publisher now records link visibility, rolls the final name back
on every later `OSError`, removes its unique temporary and best-effort fsyncs the parent.
An injected second-fsync failure leaves neither final nor temporary.

Remote listing remains bounded by explicit page size/page limit and produces the exact
inventory required to build the canonical manifest. Shard bodies remain streamed in
configured chunks; neither builder nor review downloads the dataset. Live access,
network throughput and production inventory scale remain pending. The 57 targeted CPU
contracts passed in 9.70 seconds; Ruff and strict Pyright passed. No performance
artifact was generated because D010 is not a performance task.

Two direct fresh-review starts failed with `agent thread limit reached`; this is a
main-agent remediation conclusion, not an independent PASS.
