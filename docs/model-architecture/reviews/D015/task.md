# D015 review scope

D015 closes the Data implementation package. Review validation-before-decode,
WebDataset worker shard splitting, one image decode/serializer call, deterministic
RNG identity, homogeneous image/text buckets, EOT padding masks, structured indices,
bounded persistent-worker prefetch, shard replay ordering, and absence of cross-batch
model-state caches.

Do not attribute the unresolved production metadata mapping/dropout values, full
validation scan, cold-cache throughput, ready-wait, RSS, rejection distribution, or
stable end-to-end train-step performance to this task.
