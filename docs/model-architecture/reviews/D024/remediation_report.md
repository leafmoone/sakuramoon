# D024 independent-review remediation

Base implementation commit: `b387c5a7d6470ec71b18cee15338e5773e1bd699`

Review source: independent agent `/root/data_d024_package_review`

Status: implementation and main-agent validation complete; same-reviewer rereview
pending. D010-D023 remained PASS in the review.

## Findings and closure

1. The singleton lock was derived from configurable `mainset_path`, so two services
   could own the same cache under different mainset names. Ownership is now locked at
   `<cache-root>/.sakuramoon-data-service.lock`; the cache root is the shared mutation
   domain for partial cleanup, download, publication, LRU, and eviction.
2. Socket stale probing and cleanup occurred before service ownership, and every
   failed server unlinked the common path in `finally`. The server now acquires cache
   ownership before touching the socket. It records the inode after successful bind
   and removes the path only when the current socket still has that exact inode.
   A bind loser or ownership loser cannot remove the winner's endpoint.
3. The original CUDA smoke used an in-memory client. The new RTX 5090 contract starts
   `DataServiceServer` in an independent spawned process, connects through the real
   `DataServiceClient`, delegates all health/lease/ACK traffic over AF_UNIX, reuses two
   persistent workers, transfers four batches to CUDA, ACKs all four shards, and
   verifies the service atomically rotated to a new all-pending mainset. A test-only
   four-lease admission wrapper bounds the otherwise continuous mainset service; it
   does not synthesize or intercept protocol responses.

## Scope boundary

No production network, cold-cache duration sweep, long run, formal NVMe benchmark,
DDP/NCCL, multi-GPU validation, 1,000-step canary, or formal stage was run. Production
throughput, ready-wait, RSS/swap, quota, and fully-cached control gates remain blocked
or pending exactly as in the original D024 evidence.
