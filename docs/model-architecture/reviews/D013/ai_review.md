# D013 AI/model correctness review

Status: streaming scan/report CPU code passed main-agent AI/model review after evidence
constructor and crop-seed findings were remediated. Fresh independent rereview is
unavailable after two direct agent-start failures.

The package audit found that a NaN crop-retention threshold made the comparison fail
open. The runtime assignment boundary now rejects non-finite, non-float, and
out-of-range thresholds before routing. Exact 17-shape generation, transpose closure,
no-upscale selection, cover resize, inclusive retention, and deterministic crop
semantics remain unchanged.

Main-agent review of the expansion found that callers could directly construct scan
reports with non-integer counters or duplicate/non-canonical bucket evidence. The
dataclass boundaries now reconstruct every `BucketShape`, require unique canonical
ordering, validate exact field types, and recompute all totals and rates. This prevents
malformed evidence objects from reaching canonical publication. Crop execution also
requires an exact integer seed, excluding a boolean that Python otherwise treats as an
integer.

The exact-count manifest scan, fixed 100k decode scan, inclusive 0.1% threshold,
diagnostic hard failure, EXIF/RGB processing and deterministic crop contracts have no
remaining CPU correctness blocker in this scope. The production metadata scan, real
100k decoded check, retention distribution and VAE reconstruction quality gates remain
pending and are not inferred from synthetic observations. This is a main-agent
conclusion and does not replace fresh independent review.
