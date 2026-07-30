# T020 package-review scope

Review local-only checkpoint loading, exact Mage weight compatibility, posterior-mean
semantics, frozen eval/inference behavior, 128-channel shape, and absence of
download/fallback/reference imports. Also review required external metric injection,
metric identity fields, tensor/metric validation, cohort hash and duplicate preflight,
exact 2,000 closure, report recomputation, canonical per-observation artifact, fsync,
cleanup, and no-clobber publication. Real 1GPU encode/decode and quality evidence are
not closed by CPU tests.
