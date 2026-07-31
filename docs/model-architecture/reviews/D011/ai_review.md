# D011 AI/model correctness review

Status: explicit mapping and validation bundle expansion passed main-agent AI/model
review. Fresh independent rereview is unavailable after two direct agent-start
failures.

Deterministic SHA-256 ranking, exact 2,000-ID cardinality, global duplicate rejection,
capacity-proportional stratum allocation, canonical JSONL, and pre-shuffle exclusion
remain correct. The package audit found only that the injected aspect bucket key could
be whitespace or a wrong type. That boundary is now strict and has negative contracts.

Production metadata field mapping, global uniqueness, validation shard publication,
and full zero-leak evidence remain pending. No production fact was inferred. Direct
independent re-review startup was unavailable, so this PASS is main-agent remediation
acceptance.

The explicit raw-field mapping has no defaults, uses the immutable D010 shard for
release, and retains the original raw mapping for D014. `ValidationSelection` enforces
exactly 2,000 sorted unique positive IDs, while the writer consumes samples in that
exact order. Canonical JSONL and deterministic tar membership agree on the same IDs;
exclusion removes the selected set before downstream processing.

No CPU implementation correctness blocker remains. Production field values, the 11M
uniqueness scan, real validation payloads and full training zero-leak evidence remain
pending and are not inferred from synthetic rows. This main-agent conclusion does not
replace fresh independent review.
