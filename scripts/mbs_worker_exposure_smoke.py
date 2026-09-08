"""MBS worker-clone fix: real service-worker exposure smoke (data-only).

Drives the exact production data path on the runtime host with the v2
canary config:

  ProductionPipelineFactory.from_config(v2 canary)
    -> factory.batches(lease_client)
       -> ConfiguredDataLoader / iter_service_batches
          -> _PersistentShardDataset (spawned DataLoader workers)
             -> pipeline._with_local_shards
             -> shard_pipeline._iter_paths
             -> collate_samples

No Qwen/VAE forward, no backward, no optimizer, no checkpoint write.
Leases are served by a local client over pre-downloaded real
webdataset_danbooru_v2 shards (the repository's own
ModelScopeDatasetTransport), cycling shards across queue rounds exactly
like the data service (the trainer-facing stream is infinite by design;
the smoke stops early and closes the stream deterministically).

Verdict gates (spec section 11):

  severe_vertical = severity_2to4 + severity_ge4 > 0
  eligible == severe_vertical
  selected == eligible
  applied == selected
  degraded == 0
  extra_views == applied
  physical_views == logical_samples + applied
  physical_views > logical_samples

Exit code 0 = PASS, 1 = FAIL, 2 = operational error.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from transformers import AutoTokenizer

from sakuramoon.config import load_config
from sakuramoon.data.camera_viewport import (
    CameraMirrorCounts,
)
from sakuramoon.data.manifest import DatasetManifest
from sakuramoon.data.modelscope import ModelScopeDatasetTransport
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


class _Writer:
    def __init__(self, path: Path) -> None:
        self._handle = path.open("wb")

    def write(self, payload: bytes, /) -> int:
        return self._handle.write(payload)

    def close(self) -> None:
        self._handle.close()


def _download_missing(
    transport: ModelScopeDatasetTransport,
    manifest: DatasetManifest,
    records: list[Any],
    root: Path,
) -> None:
    by_path = {shard.path: shard for shard in manifest.shards}
    for record in records:
        target = root / record.path
        if target.is_file() and target.stat().st_size == record.bytes:
            print(f"[smoke] have {record.path}", flush=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[smoke] downloading {record.path} ({record.bytes} bytes)",
            flush=True,
        )
        writer = _Writer(target)
        try:
            transport.download(manifest, by_path[record.path], writer)
        finally:
            writer.close()
        size = target.stat().st_size
        if size != record.bytes:
            raise SystemExit(
                f"shard {record.path}: downloaded {size} bytes, "
                f"expected {record.bytes}"
            )
        print(f"[smoke] verified {record.path} ({size} bytes)", flush=True)


class _CycleLeaseClient:
    """Local stand-in for the data service: cycles real shards forever.

    The trainer-facing stream is infinite by design (queue rounds / epoch
    rollover), so ``health()`` never reports exhaustion and ``lease()``
    cycles the shard list with an advancing cycle_index -- exactly the
    data service's queue semantics.
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
            raise SystemExit("smoke requires at least one local shard")
        self._root = root
        self._leased: set[int] = set()
        self._active_paths: set[str] = set()
        self._issued = 0

    def health(self) -> bool:
        return False

    def lease(self, worker_id: int) -> ShardLeaseDescriptor | None:
        if worker_id in self._leased:
            return None
        # Queue semantics: a shard path is only re-leased after its previous
        # lease was acknowledged (the collate layer rejects duplicate active
        # shards). With fewer shards than workers the remaining workers wait
        # for an ack.
        for offset in range(len(self._records)):
            index = self._issued + offset
            record = self._records[index % len(self._records)]
            if record.path in self._active_paths:
                continue
            self._issued = index + 1
            self._leased.add(worker_id)
            self._active_paths.add(record.path)
            return ShardLeaseDescriptor(
                lease_id=f"smoke-{index:06d}",
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
    print(f"[smoke] reject {reason}", flush=True)


def _add_mirror(
    a: CameraMirrorCounts | None, b: CameraMirrorCounts
) -> CameraMirrorCounts:
    fields = (
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
    if a is None:
        return b
    return CameraMirrorCounts(
        **{name: getattr(a, name) + getattr(b, name) for name in fields}
    )  # pyright: ignore[reportCallIssue]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", default="train_g1_camera_v2_p25_mirror_v2_canary.toml")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=3)
    parser.add_argument("--min-logical", type=int, default=400)
    parser.add_argument("--min-severe", type=int, default=20)
    parser.add_argument("--max-logical", type=int, default=4000)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args(argv)

    manifest = DatasetManifest.model_validate(
        json.loads(args.manifest.read_text(encoding="utf-8"))
    )
    records = list(manifest.shards)[: args.shards]

    loaded = load_config(
        Path(args.config),
        config_root=args.repo / "config",
        validate_secrets=False,
    )
    config = loaded.config
    transport = ModelScopeDatasetTransport.from_token_environment(
        "MODELSCOPE_API_TOKEN",
        cast(Any, config.data.transport),
    )
    _download_missing(transport, manifest, records, args.shard_root)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    pad_token_id = tokenizer.pad_token_id
    if type(pad_token_id) is not int:
        print(
            "FAIL: qwen tokenizer padding identity is unavailable", file=sys.stderr
        )
        return 2
    factory = ProductionPipelineFactory.from_config(
        config,
        repository_root=args.repo,
        tokenizer=tokenizer,
        framing=FramingContract(
            EXPECTED_PREFIX_TOKENS,
            EXPECTED_SUFFIX_TOKENS,
            pad_token_id,
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
    agg: CameraMirrorCounts | None = None
    camera_selected = 0
    camera_applied = 0
    batches = 0
    try:
        for batch in stream:
            if batch.camera_mirror is None:
                print(
                    "FAIL: worker-produced batch has no camera_mirror counts",
                    file=sys.stderr,
                )
                return 1
            agg = _add_mirror(agg, batch.camera_mirror)
            camera_selected += batch.camera_viewport.selected
            camera_applied += batch.camera_viewport.applied
            batches += 1
            severe = agg.severity_2to4 + agg.severity_ge4
            # Spec: >= min-logical rows OR severe_vertical >= min-severe,
            # whichever occurs first; max-logical is the safety bound.
            if (
                agg.logical_samples >= args.min_logical
                or severe >= args.min_severe
                or agg.logical_samples >= args.max_logical
            ):
                break
    finally:
        stream.close()

    if agg is None:
        print("FAIL: no batches produced", file=sys.stderr)
        return 2

    severe_vertical = agg.severity_2to4 + agg.severity_ge4
    degraded = agg.mirror_selected - agg.mirror_applied
    gates = {
        "severe_vertical_gt_0": severe_vertical > 0,
        "eligible_eq_severe_vertical": agg.mirror_eligible == severe_vertical,
        "selected_eq_eligible": agg.mirror_selected == agg.mirror_eligible,
        "applied_eq_selected": agg.mirror_applied == agg.mirror_selected,
        "degraded_zero": degraded == 0,
        "extra_views_eq_applied": agg.mirror_extra_views == agg.mirror_applied,
        "physical_eq_logical_plus_applied": agg.physical_views
        == agg.logical_samples + agg.mirror_applied,
        "physical_gt_logical": agg.physical_views > agg.logical_samples,
    }
    result = {
        "schema": "mbs_worker_exposure_smoke_v1",
        "config": args.config,
        "batches_consumed": batches,
        "logical": agg.logical_samples,
        "camera_selected": camera_selected,
        "camera_applied": camera_applied,
        "vertical_applied": agg.vertical_applied,
        "severity_lt2": agg.severity_lt2,
        "severity_2to4": agg.severity_2to4,
        "severity_ge4": agg.severity_ge4,
        "severe_vertical": severe_vertical,
        "eligible": agg.mirror_eligible,
        "selected": agg.mirror_selected,
        "applied": agg.mirror_applied,
        "degraded": degraded,
        "extra_views": agg.mirror_extra_views,
        "physical_views": agg.physical_views,
        "gates": gates,
        "verdict": "PASS" if all(gates.values()) else "FAIL",
    }
    print(json.dumps(result, indent=1))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
