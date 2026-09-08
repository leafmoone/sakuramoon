#!/usr/bin/env python3
"""MBS corrected canary: long compile-saturation smoke (FORWARD ONLY).

Fix-review sections 17-24.  Drives the REAL corrected production path on
one HCU with the exact v2 canary config and checks that the governed
torch.compile recompile ceiling (64, from config
``kernels.torch_compile_recompile_limit``) survives the corrected mirror's
variable packed physical row count over a LONG run:

  ProductionPipelineFactory.from_config(v2 canary)
    -> factory.batches(lease client over local real shards)
       -> spawned persistent DataLoader workers
          -> pipeline._with_local_shards -> collate_samples
  -> real U118100 DiT composite (read-only inference artifact)
     + local Qwen3.5-2B + local Mage-VAE
  -> compile_packed_dit_blocks(recompile_limit=<config value 64>)
     (the exact production install path from production.py)
  -> SingleGpuBatchRuntime.measure over >= 64 real batches (bounded
     extend to 128), SAME compiled module throughout, under
     torch.no_grad().

FORWARD ONLY: no backward, no optimizer object, no scheduler step, no
checkpoint write, no publisher, no training-state mutation.  grad must be
None before and after every batch; parameter/buffer checksums must be
bit-identical at the end.

Telemetry contract:
  - TORCH_LOGS must equal "recompiles" in the process environment BEFORE
    python import (dynamo recompile telemetry armed before torch import);
  - runtime readback: dynamo recompile_limit == 64 and
    fail_on_recompile_limit_hit is True after the install;
    accumulated_recompile_limit is recorded and must not change;
  - recompiles are counted in-process via torch._dynamo.utils.counters
    (per-batch deltas) AND from the redirected stderr recompile log
    (entries, guard reasons, max compile index).

Hard gates (any failure -> VERDICT FAIL, exit 1):
  - distinct physical packed sequence counts >= 9 (prefer >= 12; no
    fabricated shapes -- real data only);
  - no FailOnRecompileLimitHit, no "recompile_limit reached",
    no accumulated-limit failure, no eager fallback, no traceback;
  - per-batch MBS mirror invariants (severe_vertical = sev2to4 + ge4;
    eligible == severe_vertical; selected == eligible; applied ==
    selected; degraded == 0; extra_views == applied;
    physical == logical + applied) and aggregate applied > 0;
  - finite losses for every batch;
  - grad None before/after; parameter checksums unchanged; no
    optimizer object; U118100 source manifest and U118200 control
    manifest byte-identical; v2 output root stays empty.

Verdict fields in the evidence JSON:
  SATURATION_COVERAGE = SUFFICIENT | INSUFFICIENT (distinct >= 9)
  CANARY_READY        = TRUE  only when every gate passes
                        (even TRUE does NOT authorize training; the
                        corrected canary starts only on a later GO)

Usage (come7), from a launcher that sources the workload env:
  TORCH_LOGS=recompiles OMP_NUM_THREADS=32 MKL_NUM_THREADS=32 \
    <venv>/bin/python scripts/mbs_compile_saturation_smoke.py \
      --repo <worktree> --stderr-log <stderr.log> ...
Exit codes: 0 = PASS, 1 = FAIL, 2 = operational error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Fail-closed environment contract: checked BEFORE torch is imported.
# ---------------------------------------------------------------------------
if os.environ.get("TORCH_LOGS") != "recompiles":
    print(
        "OPERATIONAL FAIL: TORCH_LOGS must equal 'recompiles' in the "
        "process environment before python import (fix-review s17)",
        file=sys.stderr,
    )
    raise SystemExit(2)

import torch
from torch._dynamo import config as dynamo_config
from torch._dynamo import utils as dynamo_utils
from transformers import AutoTokenizer

from sakuramoon.assets import require_local_qwen, require_local_vae
from sakuramoon.checkpoint.load import load_inference_artifact
from sakuramoon.checkpoint.schema import CheckpointIdentity
from sakuramoon.config.load import load_config
from sakuramoon.data.camera_viewport import CameraMirrorCounts
from sakuramoon.data.manifest import DatasetManifest
from sakuramoon.data.production import ProductionPipelineFactory
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
)
from sakuramoon.data.service_protocol import (
    DataServiceSessionIdentity,
    ShardLeaseDescriptor,
)
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.train.runtime import (
    SingleGpuBatchRuntime,
    compile_packed_dit_blocks,
)
from sakuramoon.train.step import TrainableComposite

CKPT_ID = "raw-118100-update-cadence"
CKPT_UPDATE = 118100

MIRROR_FIELDS = (
    "logical_samples",
    "vertical_applied",
    "mirror_eligible",
    "mirror_selected",
    "mirror_applied",
    "severity_lt2",
    "severity_2to4",
    "severity_ge4",
    "original_start",
    "original_center",
    "original_end",
    "mirror_start",
    "mirror_center",
    "mirror_end",
    "mirror_extra_views",
    "physical_views",
)


def _add_mirror(
    a: CameraMirrorCounts | None, b: CameraMirrorCounts
) -> CameraMirrorCounts:
    if a is None:
        return b
    return CameraMirrorCounts(
        **{name: getattr(a, name) + getattr(b, name) for name in MIRROR_FIELDS}
    )  # pyright: ignore[reportCallIssue]


class _CycleLeaseClient:
    """Local stand-in for the data service: cycles real local shards.

    Same queue semantics as the validated data-only exposure smoke
    (scripts/mbs_worker_exposure_smoke.py): a shard path is only
    re-leased after its previous lease was acknowledged; the
    trainer-facing stream is infinite by design.
    """

    def __init__(
        self,
        records: list[Any],
        root: Path,
        dataset_id: str,
        worker_count: int,
    ) -> None:
        self.identity = DataServiceSessionIdentity(
            dataset_id=dataset_id, worker_count=worker_count
        )
        self._records = list(records)
        if not self._records:
            raise SystemExit("saturation smoke requires at least one local shard")
        self._root = root
        self._leased: set[int] = set()
        self._active_paths: set[str] = set()
        self._issued = 0

    def health(self) -> bool:
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if worker_id in self._leased:
            return None
        for offset in range(len(self._records)):
            index = self._issued + offset
            record = self._records[index % len(self._records)]
            if record.path in self._active_paths:
                continue
            self._issued = index + 1
            self._leased.add(worker_id)
            self._active_paths.add(record.path)
            return ShardLeaseDescriptor(
                lease_id=f"sat-{index:06d}",
                worker_id=worker_id,
                cycle_index=index // len(self._records),
                state_revision=1,
                record=record,
                local_path=self._root / record.path,
            )
        return None

    def acknowledge(self, descriptor: ShardLeaseDescriptor) -> None:
        self._leased.discard(descriptor.worker_id)
        self._active_paths.discard(descriptor.record.path)


def _reject_sample(reason: str) -> None:
    # Module-level (not a lambda): the factory must be spawn-picklable.
    print(f"[saturation-smoke] reject {reason}", flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dir_fingerprint(path: Path) -> dict:
    if not path.is_dir():
        return {"exists": False}
    files: dict[str, int] = {}
    for item in sorted(path.rglob("*")):
        if item.is_file():
            files[str(item.relative_to(path))] = item.stat().st_size
    return {"exists": True, "file_count": len(files), "sizes": files}


def _checksums(module: torch.nn.Module) -> dict[str, dict[str, Any]]:
    table: dict[str, dict[str, Any]] = {}
    for name, tensor in list(module.named_parameters()) + list(
        module.named_buffers()
    ):
        cpu = tensor.detach().to("cpu")
        digest = hashlib.sha256(
            cpu.to(torch.float32).numpy().tobytes()
        ).hexdigest()
        table[name] = {
            "sha256": digest,
            "numel": int(tensor.numel()),
            "grad_is_none": tensor.grad is None,
        }
    return table


def _checksum_equal(
    before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]
) -> bool:
    if set(before) != set(after):
        return False
    for key, value in before.items():
        other = after[key]
        if value["sha256"] != other["sha256"] or value["numel"] != other["numel"]:
            return False
    return True


def _stats_snapshot() -> dict[str, int]:
    stats = dynamo_utils.counters.get("stats", {})
    return {key: int(value) for key, value in stats.items()}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument(
        "--config", default="train_g1_camera_v2_p25_mirror_v2_canary.toml"
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=Path("/sakuramoon-runtime"),
        help="directory containing model/{qwen_3.5_2B,vae,clip} (real dir)",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "/sakuramoon-runtime/output_model/g1_camera_v2_p25/"
            "ckpt_118100_raw-118100-update-cadence"
        ),
    )
    parser.add_argument(
        "--control-checkpoint",
        type=Path,
        default=Path(
            "/sakuramoon-runtime/output_model/g1_camera_v2_p25_mirror/"
            "ckpt_118200_raw-118200-update-cadence"
        ),
    )
    parser.add_argument(
        "--v2-output-root",
        type=Path,
        default=Path("/sakuramoon-runtime/output_model/g1_camera_v2_p25_mirror_v2"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("/sakuramoon-runtime/data/dataset-manifest-danbooru-v2.json"),
    )
    parser.add_argument(
        "--shard-root", type=Path, default=Path("/sakuramoon-runtime/cache/data")
    )
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="qwen model dir for the tokenizer (default <model-root>/model/qwen_3.5_2B)",
    )
    parser.add_argument("--batch-count", type=int, default=64)
    parser.add_argument("--extend-to", type=int, default=128)
    parser.add_argument("--min-distinct", type=int, default=9)
    parser.add_argument(
        "--stderr-log",
        type=Path,
        required=True,
        help="path of the redirected process stderr (TORCH_LOGS=recompiles sink)",
    )
    parser.add_argument(
        "--evidence-out",
        type=Path,
        default=Path(
            "/sakuramoon-runtime/mbs-recompile-fix/compile-saturation-evidence.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.time()
    results: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    recompile_limit_hit = False
    hard_fail = False

    def check(name: str, ok: bool, detail: Any = None) -> bool:
        results.append(
            {
                "check": name,
                "status": "PASS" if ok else "FAIL",
                "detail": detail,
            }
        )
        if not ok:
            print(f"CHECK FAIL: {name}: {detail!r}", flush=True)
        return ok

    if not torch.cuda.is_available():
        print("OPERATIONAL FAIL: no CUDA/HCU device", file=sys.stderr)
        return 2
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    generator = torch.cuda.default_generators[0]
    details["device"] = torch.cuda.get_device_name(0)
    details["torch"] = torch.__version__
    details["generator_initial_seed"] = int(generator.initial_seed())

    # ------------------------------------------------------------------
    # Exact v2 canary config; kernel contract readback.
    # ------------------------------------------------------------------
    loaded = load_config(
        Path(args.config),
        config_root=args.repo / "config",
        validate_secrets=False,
    )
    config = loaded.config
    import tomllib

    kernels = tomllib.loads(loaded.resolved_toml)["kernels"]
    details["resolved_kernels"] = kernels
    check(
        "config_is_v2_canary_compile_on",
        kernels["torch_compile_enabled"] is True
        and kernels["torch_compile_backend"] == "inductor"
        and kernels["torch_compile_mode"] == "max-autotune-no-cudagraphs"
        and kernels["torch_compile_dynamic"] is True,
        kernels,
    )
    configured_limit = kernels["torch_compile_recompile_limit"]
    check(
        "config_recompile_limit_is_sixty_four",
        configured_limit == 64,
        configured_limit,
    )
    check(
        "config_object_carries_limit",
        config.kernels.torch_compile_recompile_limit == configured_limit,
        config.kernels.torch_compile_recompile_limit,
    )

    # ------------------------------------------------------------------
    # Immutable evidence fingerprints (before anything runs).
    # ------------------------------------------------------------------
    src_manifest_sha = _sha256_file(args.checkpoint / "manifest.json")
    ctrl_manifest_sha = _sha256_file(args.control_checkpoint / "manifest.json")
    ckpt_fingerprint = _dir_fingerprint(args.checkpoint)
    v2_output_before = _dir_fingerprint(args.v2_output_root)
    details["immutable_before"] = {
        "source_manifest_sha256": src_manifest_sha,
        "control_manifest_sha256": ctrl_manifest_sha,
        "source_ckpt_file_count": ckpt_fingerprint.get("file_count"),
        "v2_output_root": v2_output_before,
    }

    # ------------------------------------------------------------------
    # Models: real U118100 DiT composite (read-only) + local Qwen + VAE.
    # ------------------------------------------------------------------
    require_local_qwen(args.model_root)
    require_local_vae(args.model_root)
    qwen_attention_backend = config.kernels.qwen_attention_backend
    details["qwen_attention_backend"] = qwen_attention_backend
    t0 = time.time()
    qwen_runtime = load_local_qwen(
        args.model_root, device, attention_backend=qwen_attention_backend
    )
    encoder = qwen_runtime.encoder
    vae = load_local_mage_vae(args.model_root, device)
    composite = load_inference_artifact(
        args.checkpoint,
        CheckpointIdentity(checkpoint_id=CKPT_ID, update=CKPT_UPDATE),
        device=device,
    )
    if not isinstance(composite, TrainableComposite):
        raise TypeError("inference artifact is not the trainable composite")
    composite.eval()
    details["model_load_seconds"] = round(time.time() - t0, 2)
    details["parameter_counts"] = {
        "composite": len(dict(composite.named_parameters())),
        "qwen": len(dict(encoder.named_parameters())),
        "vae": len(dict(vae.named_parameters())),
    }

    before_composite = _checksums(composite)
    before_qwen = _checksums(encoder)
    before_vae = _checksums(vae)
    check(
        "grad_none_before",
        all(v["grad_is_none"] for v in before_composite.values())
        and all(v["grad_is_none"] for v in before_qwen.values())
        and all(v["grad_is_none"] for v in before_vae.values()),
        None,
    )
    details["no_optimizer_object_used"] = (
        "no optimizer/scheduler/publisher constructed in this smoke "
        "(forward-only contract; no .step() reachable in this process)"
    )

    # ------------------------------------------------------------------
    # Dynamo global-state baseline + production compile install.
    # ------------------------------------------------------------------
    dynamo_before = {
        "recompile_limit": dynamo_config.recompile_limit,
        "fail_on_recompile_limit_hit": dynamo_config.fail_on_recompile_limit_hit,
        "accumulated_recompile_limit": dynamo_config.accumulated_recompile_limit,
        "suppress_errors": dynamo_config.suppress_errors,
    }
    stats_before = _stats_snapshot()
    details["dynamo_before"] = dynamo_before

    t0 = time.time()
    compiled_blocks = compile_packed_dit_blocks(
        composite,
        backend=config.kernels.torch_compile_backend,
        mode=config.kernels.torch_compile_mode,
        dynamic=config.kernels.torch_compile_dynamic,
        # the exact production binding (production.py): the config value,
        # not a constant
        recompile_limit=config.kernels.torch_compile_recompile_limit,
    )
    install_seconds = time.time() - t0

    check(
        "runtime_readback_recompile_limit_64",
        dynamo_config.recompile_limit == 64,
        dynamo_config.recompile_limit,
    )
    check(
        "runtime_readback_fail_on_limit_true",
        dynamo_config.fail_on_recompile_limit_hit is True,
        dynamo_config.fail_on_recompile_limit_hit,
    )
    check(
        "runtime_readback_suppress_errors_false",
        dynamo_config.suppress_errors is False,
        dynamo_config.suppress_errors,
    )
    check(
        "accumulated_recompile_limit_untouched",
        dynamo_config.accumulated_recompile_limit
        == dynamo_before["accumulated_recompile_limit"],
        {
            "before": dynamo_before["accumulated_recompile_limit"],
            "after_install": dynamo_config.accumulated_recompile_limit,
        },
    )
    check(
        "all_blocks_compiled_after_install",
        all(
            getattr(block, "_compiled_call_impl", None) is not None
            for block in composite.dit.blocks.values()
        ),
        {"compiled_blocks": len(compiled_blocks)},
    )
    details["compile_install_seconds"] = round(install_seconds, 3)

    # ------------------------------------------------------------------
    # Forward-only runtime (production order: compile first, then runtime).
    # growth_alpha: the U118100 stage-relative update count (60100) is far
    # beyond the growth schedule's max_updates (5000), so the schedule has
    # converged to 1.0 (matches the launch-readiness compile smoke).
    # ------------------------------------------------------------------
    runtime = SingleGpuBatchRuntime(
        qwen=encoder,
        vae=vae,
        composite=composite,
        device=device,
        generator=generator,
        p_mean=config.timestep.p_mean,
        p_std=config.timestep.p_std,
        noise_scale=config.timestep.noise_scale,
        t_eps=config.timestep.t_eps,
        noise_observation_boundary=config.logging.noise_observation_boundary,
        growth_alpha=1.0,
        torch_compile_enabled=config.kernels.torch_compile_enabled,
        torch_compile_backend=config.kernels.torch_compile_backend,
        torch_compile_mode=config.kernels.torch_compile_mode,
        torch_compile_dynamic=config.kernels.torch_compile_dynamic,
    )
    details["runtime_hyperparameters"] = {
        "p_mean": config.timestep.p_mean,
        "p_std": config.timestep.p_std,
        "noise_scale": config.timestep.noise_scale,
        "t_eps": config.timestep.t_eps,
        "noise_observation_boundary": config.logging.noise_observation_boundary,
        "growth_alpha": 1.0,
        "no_grad": True,
    }

    # DiT capture: the packed physical sequence count the compiled blocks
    # actually see (len(main_token_lengths)); the DiT forward is eager, so
    # wrapping it cannot perturb the per-block compiled regions.
    dit_captures: list[int] = []
    original_dit_forward = composite.dit.forward

    def capturing_dit_forward(inputs: Any, *args: Any, **kwargs: Any) -> Any:
        result = original_dit_forward(inputs, *args, **kwargs)
        try:
            # inputs = per-view latents tuple (one tensor per physical view);
            # args[2] = main_token_lengths: the per-view packed token count
            # tuple (positional arg 4 of DiT.forward, from
            # TrainableCompositeInputs.main_token_lengths).  Its length is
            # exactly the dynamo guard that drove the Phase B recompile
            # storm (variable physical row count per corrected-mirror batch).
            latent_count = len(inputs)
            token_len = len(args[2]) if len(args) >= 3 else -1
            dit_captures.append((latent_count, token_len))
        except Exception:  # noqa: BLE001
            dit_captures.append((-1, -1))
        return result

    composite.dit.forward = capturing_dit_forward  # type: ignore[method-assign]

    # ------------------------------------------------------------------
    # Real production data path over local real shards.
    # ------------------------------------------------------------------
    manifest = DatasetManifest.model_validate(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    local_records = [
        record
        for record in manifest.shards
        if (args.shard_root / record.path).is_file()
        and (args.shard_root / record.path).stat().st_size == record.bytes
    ]
    records = local_records[: args.shards]
    if not records:
        print("OPERATIONAL FAIL: no local shards match the manifest", file=sys.stderr)
        return 2
    details["shards"] = [record.path for record in records]

    model_dir = args.model or (args.model_root / "model" / "qwen_3.5_2B")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    pad_token_id = tokenizer.pad_token_id
    if type(pad_token_id) is not int:
        print("OPERATIONAL FAIL: qwen tokenizer padding identity missing", file=sys.stderr)
        return 2
    factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=args.repo,
        tokenizer=tokenizer,
        framing=FramingContract(
            EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, pad_token_id
        ),
        rejection_observer=_reject_sample,
    )
    client = _CycleLeaseClient(
        records,
        args.shard_root,
        dataset_id=manifest.dataset_id,
        worker_count=config.data.cache.persistent_workers_per_rank,
    )
    stream = factory.batches(client)

    # ------------------------------------------------------------------
    # The long forward-only loop.
    # ------------------------------------------------------------------
    batch_records: list[dict[str, Any]] = []
    agg: CameraMirrorCounts | None = None
    physical_counts: list[int] = []
    batches = 0
    target = args.batch_count
    extended = False

    try:
        for batch in stream:
            idx = batches
            cm = batch.camera_mirror
            if cm is None:
                check("batch_has_camera_mirror_counts", False, {"batch": idx})
                hard_fail = True
                break
            # per-batch MBS invariants (hard, every batch)
            severe = cm.severity_2to4 + cm.severity_ge4
            batch_ok = (
                cm.mirror_eligible == severe
                and cm.mirror_selected == cm.mirror_eligible
                and cm.mirror_applied == cm.mirror_selected
                and (cm.mirror_selected - cm.mirror_applied) == 0
                and cm.mirror_extra_views == cm.mirror_applied
                and cm.physical_views == cm.logical_samples + cm.mirror_applied
            )
            if not batch_ok:
                check(f"batch_{idx}_mirror_invariants", False, cm)

            dit_captures.clear()
            stats_prev = _stats_snapshot()
            t0 = time.time()
            try:
                with torch.no_grad():
                    measurement = runtime.measure(batch)
            except Exception as error:  # noqa: BLE001
                if "FailOnRecompileLimitHit" in type(error).__name__ or (
                    "recompile_limit" in str(error)
                ):
                    recompile_limit_hit = True
                check(
                    "no_compile_or_runtime_exception",
                    False,
                    f"batch {idx}: {type(error).__name__}: {error}",
                )
                hard_fail = True
                break
            dt = time.time() - t0

            capture = dit_captures[0] if dit_captures else (-1, -1)
            physical_latents, physical_tokens = capture
            physical = physical_tokens
            if (
                physical != cm.physical_views
                or physical_latents != cm.physical_views
            ):
                check(
                    f"batch_{idx}_dit_sequence_matches_mirror_table",
                    False,
                    {
                        "latent_views": physical_latents,
                        "main_token_lengths_len": physical_tokens,
                        "mirror_table": cm.physical_views,
                    },
                )
            physical_counts.append(physical)

            loss = measurement.per_sample_loss.float()
            finite = bool(torch.isfinite(loss).all())
            if not finite:
                check(f"batch_{idx}_loss_finite", False, loss.tolist())

            stats_now = _stats_snapshot()
            delta = {
                key: stats_now.get(key, 0) - stats_prev.get(key, 0)
                for key in set(stats_now) | set(stats_prev)
                if stats_now.get(key, 0) != stats_prev.get(key, 0)
            }

            # no eager fallback: every block still compiled; grad still None
            still_compiled = all(
                getattr(block, "_compiled_call_impl", None) is not None
                for block in composite.dit.blocks.values()
            )
            grad_any = any(p.grad is not None for p in composite.parameters())

            batch_records.append(
                {
                    "batch": idx,
                    "logical": cm.logical_samples,
                    "applied": cm.mirror_applied,
                    "physical": physical,
                    "loss_mean": float(loss.mean()),
                    "loss_max": float(loss.max()),
                    "loss_finite": finite,
                    "measure_seconds": round(dt, 3),
                    "dynamo_counter_delta": delta,
                    "still_compiled": still_compiled,
                    "grad_none": not grad_any,
                }
            )
            agg = _add_mirror(agg, cm)
            batches += 1
            if batches % 4 == 0 or batches == 1:
                print(
                    f"[saturation-smoke] batch {batches}: physical={physical} "
                    f"applied={cm.mirror_applied} "
                    f"distinct={len(set(physical_counts))} "
                    f"dt={dt:.2f}s delta={delta}",
                    flush=True,
                )

            if batches >= target:
                if (
                    len(set(physical_counts)) < args.min_distinct
                    and not extended
                    and batches < args.extend_to
                ):
                    extended = True
                    print(
                        f"[saturation-smoke] distinct "
                        f"{len(set(physical_counts))} < {args.min_distinct} "
                        f"at {batches} batches; extending to {args.extend_to}",
                        flush=True,
                    )
                    continue
                break
    finally:
        stream.close()

    if not batch_records and not hard_fail and not recompile_limit_hit:
        print("OPERATIONAL FAIL: no batches produced", file=sys.stderr)
        return 2

    distinct_counts = sorted(set(physical_counts))
    details["physical_sequence_counts"] = {
        "observed": physical_counts,
        "distinct": distinct_counts,
        "distinct_count": len(distinct_counts),
    }
    saturation_coverage = (
        "SUFFICIENT" if len(distinct_counts) >= args.min_distinct else "INSUFFICIENT"
    )
    check(
        "saturation_coverage_min_distinct",
        len(distinct_counts) >= args.min_distinct,
        {"distinct": distinct_counts, "min": args.min_distinct},
    )

    # ------------------------------------------------------------------
    # Aggregate MBS invariants.
    # ------------------------------------------------------------------
    if agg is not None:
        severe_total = agg.severity_2to4 + agg.severity_ge4
        aggregate_gates = {
            "severe_vertical_gt_0": agg.mirror_applied > 0,
            "eligible_eq_severe_vertical": agg.mirror_eligible == severe_total,
            "selected_eq_eligible": agg.mirror_selected == agg.mirror_eligible,
            "applied_eq_selected": agg.mirror_applied == agg.mirror_selected,
            "degraded_zero": agg.mirror_selected - agg.mirror_applied == 0,
            "extra_views_eq_applied": agg.mirror_extra_views == agg.mirror_applied,
            "physical_eq_logical_plus_applied": agg.physical_views
            == agg.logical_samples + agg.mirror_applied,
            "physical_gt_logical": agg.physical_views > agg.logical_samples,
        }
        for name, ok in aggregate_gates.items():
            check(f"aggregate_{name}", ok, None)
        details["aggregate_mirror"] = {
            name: int(getattr(agg, name)) for name in MIRROR_FIELDS
        }

    # ------------------------------------------------------------------
    # Telemetry: in-process counters + redirected stderr recompile log.
    # ------------------------------------------------------------------
    stats_after = _stats_snapshot()
    details["dynamo_counters"] = {
        "before": stats_before,
        "after": stats_after,
        "delta": {
            key: stats_after.get(key, 0) - stats_before.get(key, 0)
            for key in set(stats_after) | set(stats_before)
            if stats_after.get(key, 0) != stats_before.get(key, 0)
        },
    }
    details["dynamo_after"] = {
        "recompile_limit": dynamo_config.recompile_limit,
        "fail_on_recompile_limit_hit": dynamo_config.fail_on_recompile_limit_hit,
        "accumulated_recompile_limit": dynamo_config.accumulated_recompile_limit,
    }
    check(
        "accumulated_recompile_limit_unchanged_after_run",
        dynamo_config.accumulated_recompile_limit
        == dynamo_before["accumulated_recompile_limit"],
        details["dynamo_after"]["accumulated_recompile_limit"],
    )

    stderr_text = ""
    if args.stderr_log.is_file():
        stderr_text = args.stderr_log.read_text(encoding="utf-8", errors="replace")
    recompile_entries = [
        line.strip()
        for line in stderr_text.splitlines()
        if "recompil" in line.lower()
    ]
    guard_lines = [
        line.strip()
        for line in stderr_text.splitlines()
        if "guard" in line.lower()
    ]
    limit_reached_lines = [
        line.strip()
        for line in stderr_text.splitlines()
        if "recompile_limit reached" in line
    ]
    eager_fallback_lines = [
        line.strip()
        for line in stderr_text.splitlines()
        if "fell back to eager" in line.lower()
        or "torchdynamo fallback" in line.lower()
        or "TORCHDYNAMO_FALLBACK" in line
    ]
    traceback_lines = [
        line.strip() for line in stderr_text.splitlines() if "Traceback" in line
    ]
    graph_ids = [
        int(token)
        for line in recompile_entries + guard_lines
        for token in line.replace("graph_id:", " graph_id ").split()
        if token.isdigit()
    ]
    details["recompile_log"] = {
        "entry_count": len(recompile_entries),
        "guard_line_count": len(guard_lines),
        "max_graph_id_observed": max(graph_ids) if graph_ids else None,
        "entries_tail": recompile_entries[-40:],
        "guard_reasons_tail": guard_lines[-40:],
    }
    check(
        "no_recompile_limit_reached",
        not limit_reached_lines and not recompile_limit_hit,
        limit_reached_lines[:5],
    )
    check(
        "no_eager_fallback_markers",
        not eager_fallback_lines,
        eager_fallback_lines[:5],
    )
    check(
        "no_traceback_in_stderr",
        not traceback_lines,
        traceback_lines[:5],
    )
    check(
        "every_batch_still_compiled",
        all(record["still_compiled"] for record in batch_records),
        None,
    )
    check(
        "every_batch_grad_none",
        all(record["grad_none"] for record in batch_records),
        None,
    )
    check(
        "every_batch_loss_finite",
        all(record["loss_finite"] for record in batch_records),
        None,
    )

    # ------------------------------------------------------------------
    # Mutation / side-effect audit.
    # ------------------------------------------------------------------
    after_composite = _checksums(composite)
    after_qwen = _checksums(encoder)
    after_vae = _checksums(vae)
    check(
        "no_composite_parameter_or_buffer_mutation",
        _checksum_equal(before_composite, after_composite),
        {
            "tensors_checked": len(before_composite),
            "grad_none_after": all(v["grad_is_none"] for v in after_composite.values()),
        },
    )
    check(
        "no_qwen_parameter_mutation",
        _checksum_equal(before_qwen, after_qwen),
        {"tensors_checked": len(before_qwen)},
    )
    check(
        "no_vae_parameter_mutation",
        _checksum_equal(before_vae, after_vae),
        {"tensors_checked": len(before_vae)},
    )
    src_manifest_sha_after = _sha256_file(args.checkpoint / "manifest.json")
    ctrl_manifest_sha_after = _sha256_file(args.control_checkpoint / "manifest.json")
    ckpt_fingerprint_after = _dir_fingerprint(args.checkpoint)
    v2_output_after = _dir_fingerprint(args.v2_output_root)
    check(
        "source_and_control_manifests_unchanged",
        src_manifest_sha_after == src_manifest_sha
        and ctrl_manifest_sha_after == ctrl_manifest_sha,
        {
            "source_sha256": src_manifest_sha_after,
            "control_sha256": ctrl_manifest_sha_after,
        },
    )
    check(
        "no_checkpoint_write",
        ckpt_fingerprint_after == ckpt_fingerprint,
        ckpt_fingerprint_after.get("file_count"),
    )
    check(
        "v2_output_root_stays_empty",
        v2_output_after == v2_output_before,
        v2_output_after,
    )

    # ------------------------------------------------------------------
    # Report.
    # ------------------------------------------------------------------
    all_pass = all(item["status"] == "PASS" for item in results)
    canary_ready = bool(all_pass) and saturation_coverage == "SUFFICIENT"
    report = {
        "smoke": "mbs_compile_saturation_smoke",
        "section": "fix review sections 17-24 (long forward-only saturation)",
        "config": args.config,
        "config_recompile_limit": configured_limit,
        "device": details["device"],
        "torch": details["torch"],
        "peak_memory_GiB": round(
            torch.cuda.max_memory_allocated(device) / 1024**3, 3
        ),
        "batches_run": batches,
        "extended": extended,
        "SATURATION_COVERAGE": saturation_coverage,
        "distinct_physical_sequence_counts": distinct_counts,
        "CANARY_READY": canary_ready,
        "recompile_limit_hit": recompile_limit_hit,
        "total_seconds": round(time.time() - started, 3),
        "checks": results,
        "details": details,
        "batches": batch_records,
        "verdict": "PASS" if all_pass else "FAIL",
    }
    # CANARY_READY even TRUE does NOT authorize training; the corrected
    # canary starts only on a later explicit GO.
    args.evidence_out.parent.mkdir(parents=True, exist_ok=True)
    args.evidence_out.write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, default=str), flush=True)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
