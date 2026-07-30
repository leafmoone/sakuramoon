# D010 AI/model correctness review

Status: PENDING independent review of canonical builder expansion.

The Data package audit found that set equality allowed an identical duplicate remote
shard entry to pass the immutable inventory contract. Validation now requires the
remote tar count and exact `(path, bytes, SHA-256)` content to match the manifest. A
negative test proves duplicate remote entries fail.

The fixed source/revision binding, canonical manifest, per-shard sample counts, and
streamed byte/SHA verification remain unchanged. Production enumeration is still
pending and no dataset facts were invented. Direct independent re-review startup was
unavailable, so this PASS is explicitly main-agent remediation acceptance.

That conclusion predates the canonical build inventory, merge contract, and no-clobber
CLI publication path. Those additions require fresh independent review before this file
may record a current PASS. Production enumeration remains pending and must not be
inferred from synthetic tests.
