# D010 AI/model correctness review

Status: canonical builder expansion passed main-agent AI/model review. Fresh
independent rereview is unavailable after two direct agent-start failures.

The Data package audit found that set equality allowed an identical duplicate remote
shard entry to pass the immutable inventory contract. Validation now requires the
remote tar count and exact `(path, bytes, SHA-256)` content to match the manifest. A
negative test proves duplicate remote entries fail.

The fixed source/revision binding, canonical manifest, per-shard sample counts, and
streamed byte/SHA verification remain unchanged. Production enumeration is still
pending and no dataset facts were invented. Direct independent re-review startup was
unavailable, so this PASS is explicitly main-agent remediation acceptance.

The canonical build inventory contributes only explicit release/sample/license/access
facts; ModelScope listing contributes only path/bytes/SHA. Exact path equality and
duplicate rejection prevent either source from silently adding or dropping shards.
Canonical encoding, caller-supplied inventory digest, fixed repo/revision binding and
strict source equality prevent runtime inference of production dataset facts.

No AI/model correctness blocker remains in the CPU implementation. Production
enumeration, chosen immutable revision, release/sample inventory and real access remain
pending and are not inferred from synthetic HTTP tests. This main-agent conclusion does
not replace a fresh independent review.
