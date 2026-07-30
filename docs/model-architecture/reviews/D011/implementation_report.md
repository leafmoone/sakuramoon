# D011 implementation report

D011 adds two ordinary in-memory modules. `metadata.py` validates five fields that are already needed by the locked validation protocol and retains the source mapping without inventing a complete Danbooru schema. `validation.py` rejects duplicate IDs, groups records by the locked three-part stratum, selects exactly 2,000 IDs deterministically, serializes canonical JSONL, and removes those IDs from the training stream.

The implementation does not read production metadata or data payloads. It does not add a database, identity registry, capability object, filesystem race defense, distributed coordination, or external compatibility layer. The production aspect-bucket mapping remains owned by D013 and is supplied through a small callback.
