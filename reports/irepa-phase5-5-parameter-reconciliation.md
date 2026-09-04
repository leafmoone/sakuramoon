# iREPA Phase 5.5 — Parameter Count Reconciliation (P5.5 evidence)

Date: 09-04 (read-only measured reconciliation; salt13 + historical salt1
audit record)
Phase5 head: 4571d75b006c77e628a693fa7038dece47556db2
Verdict: PASS

## Purpose

The Phase 5 audit documents reported the source topology as 287/141/146
(total / CMuon / AdamW) and the migrated topology as 289/141/148. Phase 5.5
established, by read-only rebuild-and-compare of the actual S18 fixture
against the historical production topology, that 287/146 were stale draft
figures present only in the P5 report documents — not in any spec, source
file, or test. This file fixes the record and anchors the evidence.

## Measured topologies (all rebuild-based, read-only)

| topology | total | CMuon | AdamW |
|---|---|---|---|
| historical production (salt1 audit record) | 289 | 141 | 148 |
| REAL production rebuild | 289 | 141 | 148 |
| S18 source-topology rebuild (actual fixture) | 289 | 141 | 148 |
| migrated (S18 + iREPA) | 291 | 141 | 150 |

## FQN reconciliation

- COMMON (REAL ∩ S18) = 289
- REAL_ONLY = 0
- S18_ONLY = 0
- For all 289 COMMON FQNs, every structural attribute compared:
  - shape mismatch = 0
  - dtype mismatch = 0
  - audit group mismatch = 0
  - route mismatch = 0 (CMuon role / AdamW route)
  - weight decay mismatch = 0
  - owner mismatch = 0

## Migration audit (migrate_irepa_checkpoint.py)

- count-hardcodes in the migration path = 0 (no literal 289/291/141/148/150
  gates; the migration is FQN-driven)
- metadata-only synthetic production dry-run (source update test anchor
  111500): PASS — the plan delta is exactly the 2 projector FQNs and nothing
  else
- added FQNs: irepa_alignment.projector.weight, irepa_alignment.projector.bias
  (both routed AdamW; the CMuon set is unchanged at 141, identical FQN set)

## Root cause of the stale documentation

287/146 existed only in the Phase 5 report documents
(irepa-phase5-checkpoint-migration-audit.md and the untracked
irepa-phase5-copy-report.md) as draft figures carried over before the final
S18 fixture was pinned. No spec, source file, or test hardcodes them. The
S18 fixture actually used in the accepted parity run was the full
production-equivalent 289/141/148 topology; the parity conclusions (bit-exact
old params, CMuon momenta, AdamW state, guard/rescue state, MAIN/TOTAL
losses) therefore cover the full production parameter set. The Phase 5
conclusion is NOT weakened — coverage was 289 measured, not 287 claimed.

## What this reconciliation does NOT claim

- No real live production checkpoint was read, migrated, or modified.
- No production training was started; no salt11/G1 state was touched.

## Verdict

P5.5 = PASS. Phase 5 conclusions stand with corrected coverage numbers.
Phase 6 has no parameter-coverage gap between the S18 fixture and the
production topology.
