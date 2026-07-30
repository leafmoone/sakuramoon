# D010 implementation report

D010 implements the minimal remote WebDataset boundary. A canonical JSON manifest binds the fixed ModelScope repository and commit revision to each shard's path, release, bytes, SHA-256, and sample count. The HTTPS transport lists the revision, compares the tar inventory exactly, and streams one requested shard into a local `.partial` file while computing its digest. A matching file is published with `os.replace`; a mismatch fails.

The implementation deliberately omits the discarded local-asset machinery and does not add resume, cache coordination, hostile filesystem defenses, or a second transport abstraction. It does not access `reference/`, local model weights, `.env`, or production dataset payloads.

Production dataset enumeration and an actual network smoke require the chosen immutable dataset revision and valid token at runtime. Those inputs are not inferred by code.
