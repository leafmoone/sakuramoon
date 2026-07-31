# D015 Infra/performance review

Status: independent follow-up review returned CHANGES_REQUIRED; local-only, durable
lease and strict-input findings have main-agent remediation, while independent rereview
remains pending.

The added exception branch is local to CPU image preparation and does not add queues,
caches, synchronization, model work, disk reads, or network calls. Continued iteration
uses the existing bounded WebDataset and collate path.

Cold-cache throughput, worker/queue sweep, ready-wait, RSS/swap, real rejection counts,
and production NVMe behavior remain pending. The earlier CPU remediation ran no new
GPU work; the later committed selector runs one local 1GPU inference batch. No
DDP/NCCL, multi-GPU path, training long run, or placeholder performance artifact was
used.

The public durable batch path now holds one D012 lease until the corresponding loader
is exhausted; early exception or generator close leaves the shard active. URL strings,
relative/non-file paths, duplicate local paths, truthy non-booleans and padding-ID drift
hard-fail. The rejection observer is caller-supplied so a process-safe telemetry sink can
aggregate worker events without introducing a cache or model dependency.

The serial durable path intentionally rejects `worker_count != 1` because the current
D012 state has one active shard and WebDataset splits workers by shard. The earlier
1/2/3 sweep is only loader mechanics evidence. Production multi-worker durability,
cold-cache throughput and queue/RSS behavior remain blocked/pending. A reproducible
one-batch GPU selector passed, but it is not a throughput benchmark or training run.
