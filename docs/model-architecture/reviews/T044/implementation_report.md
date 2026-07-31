# T044 implementation report

T044 replaces only the production raw continuation contract. Raw manifests,
trainer state and growth state use schema v2, while model-only, PMA and release
artifacts retain schema v1. `RawCheckpointState` now contains only trainer
counters and canonical stage/growth progress; checkpoint code has no import or
field binding to shard, mainset, lease, cache, replay, prefetch or queue state.

`save_raw_checkpoint` requires keyword-only resolved-config bytes. Publication
accepts only nonempty UTF-8 TOML whose exact SHA-256 matches the checkpoint
identity, writes it as a checksummed sidecar and preserves the existing
temporary-directory, fsync, COMPLETE and atomic-rename protocol. The loader
requires the exact v2 raw sidecar set, validates the resolved config and all
payload checksums, and rejects raw schema v1, legacy `data_state.json`, opaque
sidecars, unknown payloads hidden under `model/`, missing or malformed config and
config-identity mismatch before any model, optimizer or RNG mutation. TorchAO
state remains weights-only loaded.

PMA requires exact v2 raw sources including the resolved config but continues
to publish a schema-v1 non-resumable PMA. Stage transition preserves trainer and
growth state without resetting, carrying or interpreting any data cursor; its
request, CLI and durable plan no longer accept the superseded pass/seed controls.

AI/model self-check covers exact fixed-input fresh-process continuation:
production output, per-sample and mean loss, all 239 canonical-FQN gradients,
clip coefficient, updated parameters, TorchAO optimizer state, training RNG and
optimizer-SR RNG match the uninterrupted path. A separate real AF_UNIX service
client contract shows that a resumed client leases the service's current row
without sending a historical cursor or requiring the next live batch to match.

Infra self-check covers exact sidecar whitelisting, checksum-before-apply failure
ordering, full 16-layer 5.14 GB raw save/load, bounded one-shard-at-a-time model
I/O inherited from T042 and unchanged atomic publication. No performance
improvement is claimed. Four-rank restore, DDP/NCCL behavior, formal NVMe timing,
long runs and stage canaries remain pending or blocked by the task scope.
