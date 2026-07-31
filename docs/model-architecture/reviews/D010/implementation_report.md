# D010 implementation report

D010 implements the remote WebDataset boundary and a fail-closed production manifest
builder. A canonical JSON manifest binds the fixed ModelScope repository and commit
revision to every shard's path, release, bytes, SHA-256, and sample count. The HTTPS
transport lists the revision, compares an existing manifest exactly, and streams one
requested shard into a local `.partial` file while computing its digest. A matching
shard is published with `os.replace`; a mismatch fails.

Manifest construction does not infer dataset facts. The remote enumeration contributes
only normalized tar path, positive bytes, and lowercase SHA-256. A separate canonical
build inventory contributes explicit path, release, positive sample count, license, and
access terms. The inventory is bound by caller-supplied SHA-256 and the same fixed
repo/revision as config. Missing, extra, or duplicate paths on either side fail before
publication.

CLI build mode requires explicit inventory path, inventory SHA-256, and output path;
there is no credential argument. Output must match the config-bound manifest path. The
canonical manifest is written to an exclusive temporary file, file-fsynced, atomically
hard-linked without replacement, temporary-unlinked, and parent-directory-fsynced.
Existing destinations are preserved and reported with a stable structured error.

The implementation deliberately omits the discarded local-asset machinery and does not add resume, cache coordination, hostile filesystem defenses, or a second transport abstraction. It does not access `reference/`, local model weights, `.env`, or production dataset payloads.

Production dataset enumeration and an actual network smoke require the chosen immutable
dataset revision, explicit production release/sample inventory, and valid token at
runtime. Those inputs are not inferred by code.

The Data package review found that comparing remote entries as a set could hide an
identical duplicate. The remediated validator requires both exact entry count and exact
`(path, bytes, SHA-256)` equality, with a negative contract for duplicate listing
records. Direct independent re-review startup was unavailable; the main agent completed
remediation acceptance without claiming an independent final pass.

Main-agent review of the subsequent builder expansion found a publication durability
gap: after `os.link()` exposed the final manifest, failure while unlinking the temporary
or fsyncing the parent returned an error without removing the final name. The publisher
now tracks whether the link became visible, removes that final on any later `OSError`,
and best-effort fsyncs the rollback. A parent-fsync fault contract proves that both the
final name and unique temporary are absent after failure.

The corrected expansion passed 57 targeted CPU contracts plus Ruff and strict Pyright.
Two direct attempts to start a fresh independent reviewer failed with
`agent thread limit reached`; per user direction work continued without agents. This
is main-agent remediation acceptance, not a fresh independent PASS.
