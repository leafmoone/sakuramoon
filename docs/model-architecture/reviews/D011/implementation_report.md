# D011 implementation report

D011 validates the five fields needed by the locked validation protocol and retains the
source mapping without inventing a complete Danbooru schema. `MetadataFieldMapping`
has no defaults: the caller supplies the four raw field names, while `release` comes
only from the immutable D010 `ShardRecord`. Raw metadata cannot override it.

`validation.py` rejects duplicate IDs, groups records by the locked three-part stratum,
selects exactly 2,000 IDs deterministically, serializes canonical JSONL, and removes
those IDs from the training stream. `ValidationSelection` itself rejects non-2,000,
duplicate, non-positive, or unsorted IDs and non-integer seeds, so downstream code
cannot bypass selector invariants by manually constructing a weak selection.

The bundle writer consumes the selected samples in exact manifest order and streams
their path-safe, sorted members into deterministic USTAR. It publishes
`validation_manifest.jsonl` and `validation.tar` together through a temporary sibling
directory with file/directory fsync, atomic rename, parent fsync, and no-clobber checks.
Missing, extra, or out-of-order IDs fail and remove the temporary bundle.

A read-only schema inspection confirmed that local supporting data has
`id/width/height` but does not define the locked release/caption-availability mapping;
no value was guessed. The production aspect-bucket mapping remains owned by D013 and
is supplied through a small callback. No database, identity registry, capability
object, distributed coordination, or external compatibility layer was added.

The earlier Data package review found and accepted a weak bucket-key check. Main-agent
review of the later explicit mapping and bundle publisher found that failure of the
parent fsync after directory rename left the final bundle visible even though the API
reported failure. Publication now tracks rename completion, removes the task's fixed
manifest/tar/final directory on later `OSError`, and best-effort fsyncs that rollback.
Cleanup failures are isolated so one unlink cannot mask the stable publication error.

The corrected implementation passed 32 targeted CPU contracts, Ruff and strict
Pyright. Production field names, full scans and real validation payloads remain
pending. Two direct fresh-review starts failed with `agent thread limit reached`; this
is main-agent remediation acceptance, not an independent PASS.
