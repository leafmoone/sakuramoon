# D013 implementation report

D013 implements two direct CPU modules. `buckets.py` derives the locked 17-shape family from the strict configuration, scales it proportionally, and returns either a complete cover-resize assignment or a scan-friendly rejection reason. `image_ops.py` applies Pillow EXIF transpose and RGB conversion, executes the assignment with Lanczos resize, and uses a caller-provided seed for the crop offsets.

The implementation performs no fallback routing: if no target fits without upscale, or the selected nearest-aspect cover crop retains less than the configured threshold, the sample is rejected. It does not access dataset payloads outside synthetic tests or add service/registry abstractions.

The Data package audit found that a non-finite direct-call threshold could fail open.
`assign_bucket` now validates the threshold before bucket eligibility or retention
comparison and accepts only exact finite floats in `[0, 1]`. Negative contracts cover
NaN, positive and negative infinity, both range violations, integers, and booleans.

The scan expansion adds a bounded full-manifest assignment runner and a fixed 100,000
observation decode-dimension runner. The manifest runner requires the streamed record
count to equal the caller-provided D010 aggregate, reports assigned/no-upscale/retention
counts plus every canonical bucket count, and stores no sample rows. The decode runner
uses observations produced after the same EXIF/RGB normalization as training, accepts
exactly 100 mismatches, and raises `DimensionMismatchError` with the rejected report at
101 mismatches.

`write_image_scan_report` emits canonical JSON through a unique sibling temporary file,
file fsync, atomic no-clobber hard link, temporary unlink, and parent-directory fsync.
Main-agent review found that a failure of the final parent fsync left the hard-linked
destination visible even though publication reported failure. The writer now tracks
publication, removes that destination on a later `OSError`, best-effort fsyncs the
rollback, and isolates temporary/final cleanup failures so they cannot mask the stable
`ImageScanError`.

The evidence dataclasses also now reject forged or malformed direct construction:
bucket sample counts require positive exact-integer dimensions and non-negative exact
integer counts; bucket reports require exact integer totals, a non-empty unique tuple
in canonical aspect/height/width order, and consistent accounting. Dimension reports
require exact counter/float/bool field types before recomputing their acceptance
invariants. `resize_and_crop` accepts only an exact integer seed, excluding booleans.

No production data was scanned and no report artifact was generated for this ordinary
implementation task. Two direct fresh-review starts failed with
`agent thread limit reached`; the current conclusion is main-agent remediation
acceptance, not an independent PASS.
