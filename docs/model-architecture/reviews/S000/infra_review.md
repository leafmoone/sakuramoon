# S000 Infra/performance self-review

Reviewer authority: main agent, under the user's no-agent direction. This is not an
independent review.

## Verdict

PASS for bounded process, resource, publication, and fresh-recovery mechanics only.

## Findings

No blocking Infra issue remains in the implemented engineering scope.

- The service, two DataLoader workers, and fresh-resume worker are spawned processes
  with bounded startup/request/shutdown/fresh-process timeouts. The final run left no
  matching child process or AF_UNIX socket; the GPU returned to 1 MiB used and idle.
- The runner refuses an occupied socket and an existing output root. Runtime evidence,
  raw checkpoint, and fresh result use no-clobber publication. The expected zero-byte
  ownership lock remains as a non-active filesystem identity.
- The raw checkpoint contains 5,143,061,370 payload bytes and exact `COMPLETE` content.
  The fresh service recorded two replayed shards/two replayed samples before update 2.
- Parent model/optimizer references are deleted before garbage collection and CUDA
  cache release, avoiding retention of the initial model allocator cache while the
  fresh process runs.

## Performance boundary

The observed peak allocation of 11,981,415,936 bytes is not a capacity result. There
was no warmup/measured benchmark window, max-batch search, throughput statistic,
same-backend repeat distribution, cold-cache run, long run, profiler claim, or four-GPU
measurement. No `perf_baseline.json` or `perf_after.json` is warranted. Production
throughput, memory, quality, formal S000/S001, and all multi-GPU gates remain blocked.
