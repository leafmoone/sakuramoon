# D014 AI/model correctness review

Status: CPU caption/serializer code passed main-agent AI/model review after strict
canonical-ID and RNG findings were remediated. Fresh independent rereview is unavailable
after two direct agent-start failures.

The package audit found that an oversized first Artist could block a valid later style
source. The serializer now evaluates every Artist at a complete tag boundary, retains
fitting sources in deterministic order, and reserves at least one valid source whenever
one exists. Unique or all-oversized Artist inputs still hard-fail instead of splitting a
tag or silently switching to null style.

The eleven non-global dropout probabilities, production metadata mapping, non-global
dropout distributions, and truncation distribution remain blocked or pending. No smoke
probability was promoted to a production decision.

Main-agent review confirmed the fixed category order, exact separators, one available
NL branch, candidate deletion across the four tag categories, all-condition 0.10 path,
complete-boundary truncation, Artist reservation and structural indices. It also found
that a whitespace-padded canonical ID could avoid a canonical candidate match and that
the public planner accepted non-integer/negative seeds. Both boundaries now hard-fail,
with explicit negative contracts.

The real local tokenizer still proves 34/5 framing and segment equivalence. No CPU
correctness blocker remains in the implemented D014 scope. Production metadata values,
the undecided dropout probabilities and their distributions remain blocked/pending;
this main-agent conclusion does not replace fresh independent review.
