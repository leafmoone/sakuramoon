# D014 implementation report

D014 adds a direct structured caption planner and serializer. The planner accepts canonical tag IDs from upstream, performs deterministic keyed dropout/shuffle/selection, and never assigns a default to an unresolved probability. The serializer builds the Qwen sequence from four structural segments, so main and Artist token indices are known during construction and Artist cannot causally affect earlier main tokens.

The serializer reserves the fixed suffix and complete Artist tags before fitting the main body into the 512-token condition budget. It removes one whole NL field first and then whole low-priority tags. A sole Artist tag that cannot fit is a hard failure rather than a partial token or silent null-style fallback.

The Data package audit found that tail-only Artist trimming could discard a valid later
source when the first source was individually oversized. Artist reservation now first
filters complete sources that cannot individually fit, preserves the original order of
all fitting sources, then trims only complete trailing sources if their combined segment
is still over budget. All-unfit and unique-unfit inputs remain hard failures.

The prepared local tokenizer confirmed the 34/5 framing counts, padding ID 248044,
and segment/whole-string token equivalence. A 100,000-seed scan measured the fixed
all-condition dropout at 10.09%; no unresolved probability was assigned.

Main-agent review hardened the remaining public boundaries without changing caption
semantics. `Tag` now requires exact strings and a trim-stable canonical ID so candidate
deletion cannot be bypassed by boundary whitespace. `build_caption_plan` rejects
boolean, non-integer, and negative seeds. The framing contract requires a non-negative
exact-integer padding ID, and every tokenizer result must be a non-negative exact
integer before it can enter Qwen or collate.

The corrected scope passed 42 targeted contracts and the 197-test Data/text integration
suite. Two direct fresh-review starts failed with `agent thread limit reached`; the
current conclusion is main-agent acceptance, not an independent PASS. The eleven
non-global dropout values remain unassigned and continue to block production config.
