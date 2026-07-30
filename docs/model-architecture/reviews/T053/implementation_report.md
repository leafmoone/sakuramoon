# T053 implementation report

The remediated harness no longer accepts caller-constructed benchmark samples.
`run_benchmark` drives a successful-update adapter and directly measures wall time,
CUDA event spans, one measured-window CUDA peak-memory interval, process-tree RSS
high-water plus bounded pinned/swap peaks, and bytes under returned checkpoint paths.
Warmup and measured windows synchronize only at their boundaries; per-update
synchronization does not serialize the workload.

`SingleGpuStepBenchmarkAdapter` connects the harness to the existing fail-closed
training primitive, executes accumulation/backward/clip/optimizer/zero-grad, advances
the real successful-update state, invokes the scheduler, and times actual checkpoint
callbacks. DiT and loss must be timed separately by every measured microbatch; Qwen,
VAE, and conditioning remain caller-recorded at their real boundaries. GPU phases use
delayed CUDA events, while data and checkpoint host wall time use the monotonic clock.

Fairness uses a normalized workload identity plus explicit implementation variants.
Only regional compile and attention backend are permitted variant config keys. Exact
typed sample-ID and shape streams are accumulated across warmup and measured updates
and must match the corresponding identity artifacts, so a different iterator cannot
reuse an identity. The full resolved config, source commit, build hash, backend, and
enabled features remain in the variant record.

Comparison uses measured-window throughput and guards p95/p99 plus CUDA allocated,
CUDA reserved, host RSS, and pinned RAM disclosures. Regional compile remains disabled
in current runtime config. A future positive decision additionally requires file/SHA,
world-size, checkpoint, resolved-config, source, and build identities for correctness,
4GPU DDP, and 4GPU resume evidence.

Trace SHA validation is streaming. Trace range/count is bound to the benchmark plan,
identity, `profile_trace_updates`, and exact successful-update range. Trace metrics are
parsed from the PyTorch Chrome trace captured inside the same measured loop rather
than supplied by the caller. Report, samples, hotspot decisions, comparison, compile
gate, and trace index writers use unique temporary files, fsync, and no-clobber hard
links. External Nsight smokes return an unbound smoke artifact, and the formal trace
index rejects Nsys/NCU entries until a report marker/range/kernel importer exists.

The installed Nsight Systems 2025.1.1 collector was exercised through the smoke
collector and generated the expected `.nsys-rep` without creating formal benchmark
evidence. Nsight Compute 2025.1.1 reached a real CUDA matmul but the host denied
performance-counter access with
`ERR_NVGPUCTRPERM`; the collector remains fail-closed and no NCU result is claimed.

Production 16/20/24-layer benchmark integration is intentionally not claimed because
the production T022/T023 configuration remains blocked and no long training run was
authorized.
