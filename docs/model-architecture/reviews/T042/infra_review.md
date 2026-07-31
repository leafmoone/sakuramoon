# T042 Infra review

Status: prior raw/model-only independent PASS remains valid. The PMA/release/policy
remediation passed main-agent infrastructure review; fresh independent rereview is
unavailable after two direct agent-start failures.

## Publication and failure behavior

The writer creates a uniquely named, task-owned temporary directory under the
destination root and never writes into an existing final checkpoint. Every
payload file is flushed and fsynced. The nested `model`, `train_state/rng`, and
`train_state` directories are fsynced before the temporary root; `COMPLETE` is
written and fsynced before the temporary directory is atomically renamed; the
destination root is fsynced after the rename. A failed write removes its own
temporary directory without touching an older complete checkpoint. A failure
of the final parent-directory fsync rolls the new final name back and reports
failure. The targeted tests cover unpublished temporary discovery, injected
write cleanup, preservation of an older complete checkpoint, and parent-fsync
rollback.

The implementation is intentionally a single-writer, single-rank contract. It
does not contain checkpoint cadence, retention, stage/growth scheduling,
barriers, collectives, or rank aggregation. The only training RNG payload is
`rank-0.safetensors`. This is appropriate for the current scope and is not
evidence for concurrent writers or four-rank publication.

## Integrity and restore ordering

Before changing model, optimizer, or global RNG state, raw restore validates:

- a real checkpoint directory, exact `complete\n` marker, no symlinks, and the
  manifest-to-physical-tree file set;
- every payload byte size and streaming SHA-256, full expected identity and raw
  kind;
- self-described architecture plus every canonical FQN, Safetensors dtype,
  shape, shard mapping, and declared tensor byte count;
- canonical optimizer groups and parameter-schema hash, all TorchAO state
  fields/classes/shapes/dtypes/block metadata, and per-parameter step bounds;
- strict trainer/data/growth JSON and both rank and optimizer-SR RNG tensors.

`optimizer.pt` is deserialized with `torch.load(..., weights_only=True)` only.
The real RTX 5090 test restored TorchAO `OptimState8bit`, including lazy and
lagging per-parameter state, into a fresh optimizer and matched the next update
and isolated SR RNG exactly. Sidecar corruption/missing-file tests establish
that checksum rejection precedes all model, optimizer, and RNG changes. Raw
resume rejects model-only kind, while the copied raw `model/` directory remains
self-describing and loads without training sidecars.

## Sharding, memory, disk, and timing

Model tensors are traversed by sorted canonical FQN. Save materializes at most
one model shard of contiguous CPU tensor copies at a time; load validates shard
headers and then loads one shard at a time. The configured model-shard ceiling
is 2,147,483,648 bytes, with a 1 MiB header reserve and a post-write physical
size check. Existing full-composite evidence records 239 FQNs, two model
shards, and a largest shard of 2,145,315,208 bytes, 2,168,440 bytes below the
2 GiB ceiling.

The TorchAO optimizer sidecar is intentionally monolithic and is resident in
CPU memory during safe load. The complete full-composite payload was
5,143,055,405 bytes. This artifact size is bounded relative to the documented
approximately 120 GB target host, but peak host RSS has not been measured and
must be included in later target preflight/benchmark evidence; this review does
not infer a memory result from artifact bytes alone.

The recorded 20.81 s save and 6.79 s load were produced on temporary overlay
storage, not NVMe. The current workspace is an NFSv3 mount with
`local_lock=none`, total size 429,496,729,600 bytes and 387,657,498,624 bytes
available at review time. Three payloads of the observed full size require at
least 15,429,166,215 bytes before filesystem and manifest overhead, but no
space is reserved beyond the future cache high-water mark. Therefore neither
the current free-space snapshot nor the overlay timing closes the required
real-NVMe capacity, durability, atomic-rename/fsync, or timing gates.

## Independent verification

Commands run by this reviewer:

```text
uv run ruff check src/sakuramoon/checkpoint src/sakuramoon/conditioning/style_resampler.py src/sakuramoon/conditioning/text_mixer.py src/sakuramoon/model/dit.py tests/unit/checkpoint tests/gpu/checkpoint
uv run pyright src/sakuramoon/checkpoint src/sakuramoon/conditioning/style_resampler.py src/sakuramoon/conditioning/text_mixer.py src/sakuramoon/model/dit.py tests/unit/checkpoint tests/gpu/checkpoint
uv run pytest -q tests/unit/checkpoint
LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libnvidia-ml.so.580.105.08 uv run pytest -q tests/gpu/checkpoint/test_raw_checkpoint.py
uv run python tools/verify_traceability.py
git diff --check
```

Results: Ruff passed; Pyright reported 0 errors; 21 CPU checkpoint tests passed
in 6.69 s; 9 targeted one-GPU tests passed in 27.70 s; traceability verification
passed for 221 requirements and 64 production modules; `git diff --check`
passed. The GPU review used one RTX 5090 and did not rerun the 5.14 GB
full-composite test because the existing shape/size evidence was inspected and
the smaller suite directly re-exercised safe TorchAO restore and failure
ordering without a long run.

The default NVML library link currently points to 610.43.02 while the loaded
driver is 580.105.08; default `nvidia-smi` fails with a driver/library mismatch.
The targeted GPU process used the installed matching 580.105.08 library only
as an explicit review-time preload. This workaround is not a production
preflight waiver: the default NVML mismatch remains a hard environment blocker.

## Pending boundaries

Four-rank state, barriers, concurrent publication, DDP/NCCL restore, and
all-rank identity remain pending until 4x RTX 5090 hardware is available.
Operational cadence/retention integration, actual NFS/NVMe durability and performance,
measured host RSS, cache-high-water checkpoint reservation, fault-injection
matrices, and formal stage canaries remain pending. No long training was run,
and this PASS must not be used to close any of those gates. The later
PMA/release/cadence/retention expansion is covered only by the main-agent remediation
review below, not by this independent PASS.

## Expansion remediation review

The initial retention implementation called full checkpoint validation while planning
and applying deletions, rereading and hashing every payload twice. At the observed
5.14 GB raw size this amplified checkpoint I/O solely for retention selection. The new
metadata validator reads strict manifests and COMPLETE markers, enforces canonical raw
directory names, exact file/directory sets, no symlinks and manifest-declared payload
sizes, while full checksum verification remains mandatory for resume, PMA and model
load. A monkeypatched contract makes the load-path SHA function fail if retention calls
it; planning and application still complete and preserve the newest two raws.

Apply recomputes the full plan before any deletion and rejects policy/root/identity or
physical-tree drift. This fail-closed behavior prevents deletion from a stale or forged
plan. It does not claim safety against concurrent writers or malicious filesystem
replacement; checkpoint retention remains an operational single-writer boundary.

The complete checkpoint CPU suite passed 33 tests in 8.06 seconds. Eleven real RTX
5090 tests, including TorchAO restore and real-composite PMA fresh-load, passed in 43.27
seconds with the matching installed NVML library explicitly preloaded. No long run or
formal NFS/NVMe performance test was executed. Two direct independent-review agent
starts failed with `agent thread limit reached`; this section records main-agent review
only.
