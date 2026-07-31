# R001 Infra/asset-boundary review

Status: the original independent Infra review requested one evidence correction. That
correction passed main-agent acceptance; no fresh independent rereview is claimed.

The original blocker was attribution: `production_modules=13` and `195 passed` came
from a shared dirty worktree, not immutable commit
`664fda71faed5e5d7d26d5fd06754af1a20b721f`. The implementation report and test
report now retain only the clean-commit values (`production_modules=12`, `187 passed`)
for that snapshot. No runtime or asset-boundary code changed to resolve the finding.

Current main-agent checks used only Git paths/indexed content and did not read `.env`.
Ten representative root-level secret/model/DB/data/cache/reference/checkpoint/W&B/
profile/artifact paths matched the ignore policy; four `src/sakuramoon` and manifest
negative controls remained trackable. The tracked-path scan found no forbidden root
asset directory or weight/database extension. A high-confidence private-key/cloud-key
scan over the Git index found no match. The current index contained 425 tracked paths
before these two review files were added.

The traceability unit suite passed 36 tests in 14.43 seconds and the live registry
verified 221 requirements against 221 source nodes, 16 archive files, 96 production
modules and 247 runtime config keys. The repository-wide CPU regression immediately
preceding this review passed 535 tests with 5 skips in 53.32 seconds. R001 is not a
performance task and no performance placeholder was generated.

The original independent Infra verdict remains `changes_required` in the historical
record. Main-agent review accepts the corrected evidence and current repository
boundary. Two direct independent-review starts failed with `agent thread limit
reached`, so this file does not represent a fresh independent PASS.
