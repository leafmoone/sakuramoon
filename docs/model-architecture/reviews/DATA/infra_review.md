# Data package Infra/performance audit

Scope: D010-D015 CPU code plus the existing D015 one-GPU engineering smoke. No live
network/NVMe sweep, `.env`, `reference/`, DDP/NCCL, multi-GPU path, or training long run
was used.

Initial result: FAIL for the correctness/failure-boundary findings listed in the AI
audit. None requires a performance redesign. The bounded WebDataset pending-fragment
map, persistent-worker queue budget, streaming shard writer, and startup-only manifest
checks remain outside GPU model hot paths.

Cold-cache two-hour throughput, ready-wait, RSS/swap, disk-full, concurrent coordinator,
and production rejection metrics remain pending milestone evidence and are not inferred
from synthetic tests or the D015 first-kernel smoke. Each finding is remediated in its
own atomic task commit. Direct independent re-review startup was unavailable; final
per-task conclusions therefore identify main-agent remediation acceptance.
