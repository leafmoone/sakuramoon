# D015 implementation report

The pipeline reads prepared local tar shards with the locked WebDataset library,
parses metadata once, excludes validation IDs before image decode, and composes the
D013 image and D014 caption contracts. Collate groups by target image shape and Qwen
dense length, keeps EOT padding masked, and preserves all structured indices and audit
metadata. Persistent worker and ready-batch counts are explicit and incompatible
queue budgets fail instead of being adjusted.

A synthetic real-tar 1/2/3-worker sweep found no duplicate or missing sample IDs. A
real local-tokenizer RTX 5090 smoke passed the serialized CPU pipeline, Qwen seven-state
forward, and Mage posterior-mean encode with both frozen models resident together.
