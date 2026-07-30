# D012 implementation report

D012 adds an ordinary single-process cache and JSON shard-state store. The cache considers only paths from the immutable D010 manifest, calls D010 for verified fetch and publication, and removes oldest unprotected cached shards only when an incoming shard would exceed the explicit high watermark. The state store atomically replaces a small JSON file containing completed paths, one active path, and replay counters.

On restart, an active shard remains active and increments replayed shard/sample counters. The coordinator refuses a different shard until the active shard is replayed, while completed shards return without a fetch. No prefetch queue or shuffle-buffer state is serialized.

The Data package review found that path-only state could be reused with a different
immutable manifest and silently skip new shards sharing the same names. State schema v2
now records the canonical manifest SHA-256 and validates it before accepting completed,
active, or replay counters. Same-path/different-revision and legacy-unbound-schema
negative contracts prove both failure modes. Independent CPU rereview passed the
implementation, targeted tests, static checks, and traceability boundary.
