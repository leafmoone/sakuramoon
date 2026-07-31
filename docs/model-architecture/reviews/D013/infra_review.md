# D013 Infra/performance review

Status: streaming scan/report CPU code passed main-agent Infra review after one
durability finding was remediated. Fresh independent rereview is unavailable after two
direct agent-start failures.

The new validation is constant-time and runs once per image assignment before routing.
It adds no allocation, synchronization, network, disk, model, or GPU work and does not
change the bucket-selection complexity.

Production CPU throughput, rejection counts, 100k decode coverage, and downstream VAE
timing remain pending. No GPU, DDP/NCCL, multi-GPU path, training long run, or
placeholder performance artifact was used. Final package rereview remains pending.

The new runners are O(samples) time and O(bucket-count) memory, with the 100k decode
path retaining only counters. Evidence validation is O(bucket-count) over the fixed 17
shapes and does not enter per-pixel processing.

Main-agent review found that parent-directory fsync failure after the no-clobber hard
link returned an error but left the final report visible. Publication now tracks link
success, removes only its destination on a later `OSError`, then best-effort fsyncs the
parent. A fault-injected contract proves final and temporary names are absent. The
no-clobber property remains a single-writer operational boundary, not an adversarial
filesystem or multi-process coordination claim.

The 39 targeted CPU contracts passed in 1.23 seconds; Ruff and strict Pyright passed.
Real CPU throughput, dataset distributions, decode latency, production 100k coverage,
and Mage-VAE timing remain pending; no performance placeholder was created. This is a
main-agent conclusion, not an independent PASS.
