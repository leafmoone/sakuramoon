# T045 implementation report

T045 upgrades only production raw continuation artifacts from schema v2 to v3.
The v3 trainer document strictly persists the absolute stage start and terminal
successful-update budget plus the checkpoint cadence that is committed by the
same raw publication. The cadence uses Unix wall-clock seconds, retains the
locked 1,000-update/6-hour policy values, and rejects clock rollback rather than
resetting or substituting a process-local monotonic anchor.

Cross-field validation requires the raw identity update, trainer successful
update and committed cadence update to agree. The trainer update must be inside
the persisted stage interval. Active growth begins at that exact stage origin
and cannot extend beyond the terminal budget. Unknown or missing nested fields,
policy drift, invalid numeric types and raw schema v1/v2 fail closed. Model-only,
PMA and release artifacts remain schema v1, and no data-service state is added.

The checkpoint unit suite exercises strict serialization, restart-equivalent
wall cadence, clock rollback, stage/cadence drift, legacy manifest rejection and
the adjacent PMA/retention contracts. Targeted GPU checkpoint tests now carry the
same v3 state for raw round-trip, full S0 publication and growth migration. No
long training, DDP/NCCL, formal stage or multi-GPU claim is made.

The growth migration API requires an explicit post-transition cadence and passes
it into the durable target state; it no longer reuses an implicit process-local
timestamp from the source checkpoint.

Independent AI/model and Infra review remain pending. The T042 and T044 task and
review files are historical evidence and were not modified.
