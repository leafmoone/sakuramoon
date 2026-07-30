# D015 AI/model correctness review

Status: PASS after remediation acceptance; independent package rereview pending.

Expected D013 `no_upscale` and `retention` outcomes now skip only the affected sample
instead of terminating WebDataset iteration. Real-tar contracts prove both reasons and
continued delivery of a later valid sample. Decode, metadata, caption, tokenizer, and
untyped processing failures remain hard errors.

Production metadata mapping, the eleven non-global dropout values, rejection counts,
full validation zero-leak scan, and stable end-to-end training evidence remain blocked
or pending. The existing one-GPU smoke is retained only as engineering evidence.
