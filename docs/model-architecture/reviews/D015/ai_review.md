# D015 AI/model correctness review

Status: independent follow-up review returned CHANGES_REQUIRED; all AI/model findings
have main-agent remediation and targeted tests, while independent rereview remains
pending.

Expected D013 `no_upscale` and `retention` outcomes now skip only the affected sample
instead of terminating WebDataset iteration. Real-tar contracts prove both reasons and
continued delivery of a later valid sample. Decode, metadata, caption, tokenizer, and
untyped processing failures remain hard errors.

Production metadata mapping, the eleven non-global dropout values, rejection counts,
full validation zero-leak scan, and stable end-to-end training evidence remain blocked
or pending. The existing one-GPU smoke is retained only as engineering evidence.

Follow-up review found that rejection reasons were indistinguishable from validation
exclusion and collate padding was not bound to the Qwen framing. The pipeline now sends
only typed `no_upscale`/`retention` reasons to an explicit observer; validation exclusion
still occurs earlier and emits no rejection. Each accepted sample carries the measured
framing padding ID, and collate rejects any mismatch before tensor construction.

The committed 1GPU selector reproduces real tar decoding, Qwen and Mage inference with
the approved shapes and finite frozen outputs. It remains engineering evidence: real
rejection distribution, production metadata, zero-leak and stable training evidence
are not inferred. No independent post-fix PASS is recorded yet.
