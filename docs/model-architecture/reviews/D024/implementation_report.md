# D024 implementation report

D024 moves production data ownership behind an independently launched local service.
The service is the only process that imports the ModelScope transport and mutates
download partials, verified cache files, reservations, eviction state, mainset state,
or replay counters. The trainer imports a lightweight AF_UNIX client and hands only
service-issued immutable local descriptors to the existing two persistent workers.

The durable unit is now a full-manifest `PersistentMainset`. A sibling temporary file,
file fsync, atomic replace, and parent-directory fsync publish every activation,
completion, replay-counter update, and generation rollover. A loaded table must match
the immutable manifest, exact worker count, full unique permutation, contiguous exact
ordinals, bounded active rows, and a status prefix with no non-pending row after the
first pending ordinal. The service generates order and identity from system randomness;
no trainer stage, pass, seed, checkpoint, or resume input participates.

On restart, all active rows are counted exactly once for that recovery attempt and
fully fetched with the complete active set protected before any pending lookahead is
scheduled. Download reservations include manifest bytes in quota decisions, verified
lookahead and IPC channels are bounded, activation precedes lease publication, and only
a matching normal-exhaustion ACK completes one row. Final completion atomically
publishes a new full random mainset only after all supply state is drained.

The prior D024 snapshot/checkpoint coupling was removed. D024 does not alter the T042
checkpoint schema or claim T044 completion. It also preserves trusted metadata,
validation exclusion, deterministic caption/image processing, and typed collate output.

Production network/NVMe performance remains outside the accepted implementation
evidence because no production immutable manifest or task-authorized long cold-cache
sweep was available. No placeholder performance artifact was created.
