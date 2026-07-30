# T052 implementation report

Strict evaluation config now requires the prompt manifest path/hash, evaluator GPU and
training-pause plan, feature extractor version, and preprocess hash in addition to the
existing FID real-stat identity and IS splits. `build_evaluation_jobs` verifies the
actual prompt-manifest hash and derives deterministic trend/formal job IDs from the
resolved configuration, checkpoint, sample count, and successful update.

FID aggregation uses CPU float64 sample statistics and a symmetric PSD square-root
identity; IS uses finite nonnegative probabilities, float32-compatible normalization
tolerance, and deterministic equal splits. Artifact publication is no-clobber and
fsynced, records wall/GPU/pause cost, and cannot set automatic release. Manual quality
dimensions and raw/PMA-10/accepted comparison are independent of FID/IS.

The implementation consumes features/probabilities from an explicitly locked external
extractor; it does not download or disguise a synthetic extractor as production.
