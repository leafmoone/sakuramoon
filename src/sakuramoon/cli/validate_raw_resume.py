"""Run isolated synthetic real-optimizer updates from a RAW copy and save a new RAW.

Run this command again in a fresh launcher/process for fresh-resume validation.
No production data service, W&B, evaluation, retention pruning, or original write.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import torch
from torch import nn

from sakuramoon.config import load_config
from sakuramoon.config.assembly import build_trainable_composite_from_config
from sakuramoon.data.serialize import (
    EXPECTED_PREFIX_TOKENS,
    EXPECTED_SUFFIX_TOKENS,
    FramingContract,
)
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen
from sakuramoon.optim.dtk import configure_tunableop
from sakuramoon.perf.harness import _topology  # pyright: ignore[reportPrivateUsage]
from sakuramoon.perf.raw_resume import (
    advance_synthetic,
    isolate_optimizer_diagnostics,
    validate_paths,
    validate_resume_config,
)
from sakuramoon.perf.synthetic import SyntheticBatchSource
from sakuramoon.train import production
from sakuramoon.train.runtime import (
    compile_packed_dit_blocks,
    require_distributed_forward_module,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--resume", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--checkpoint-id", required=True)
    args = parser.parse_args(argv)
    from sakuramoon.checkpoint.schema import CheckpointIdentity
    CheckpointIdentity(args.checkpoint_id, 0)
    source, destination = validate_paths(args.resume, args.output_root)
    loaded = load_config(args.config, config_root=args.config_root, environment={}, validate_secrets=False)
    source_state = validate_resume_config(loaded, source, args.updates)
    config = loaded.config
    if config.distributed.world_size == 1 and (os.environ.get("WORLD_SIZE", "1") != "1" or os.environ.get("RANK", "0") != "0" or os.environ.get("LOCAL_RANK", "0") != "0"):
        raise ValueError("single-rank config cannot run under a multi-rank launcher")
    rank, world = _topology(config)  # pyright: ignore[reportPrivateUsage]
    if not torch.cuda.is_available():
        raise RuntimeError("actual RAW optimizer continuation requires the configured CUDA/DCU runtime")
    accelerator: Any = None
    if world > 1:
        from accelerate import Accelerator  # pyright: ignore[reportMissingTypeStubs]
        from accelerate.utils import (  # pyright: ignore[reportMissingTypeStubs]
            DistributedDataParallelKwargs,
            InitProcessGroupKwargs,
        )
        accelerator = Accelerator(mixed_precision="no", kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True), InitProcessGroupKwargs(timeout=timedelta(minutes=15))])
        if accelerator.num_processes != world:
            raise ValueError("launcher topology differs from config")
        device = cast(torch.device, accelerator.device)
    else:
        device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    if accelerator is not None:
        accelerator.wait_for_everyone()
    configure_tunableop(
        args.root, str(destination / "runtime" / f"rank-{rank}"),
        enabled=config.kernels.tunableop_enabled, tuning=config.kernels.tunableop_tuning,
        record_untuned=config.kernels.tunableop_record_untuned,
        max_tuning_duration_ms=config.kernels.tunableop_max_tuning_duration_ms,
    )
    module = build_trainable_composite_from_config(config, device=device, new_slot_ids=source_state.growth.new_slot_ids)
    optimizer = production._build_optimizer(config, module, rank=rank, world_size=world)  # pyright: ignore[reportPrivateUsage]
    isolate_optimizer_diagnostics(optimizer, destination)
    # Frozen assets and synthetic input preparation happen BEFORE restoring RNG.
    qwen = load_local_qwen(args.root, device, attention_backend=config.kernels.qwen_attention_backend)
    vae = load_local_mage_vae(args.root, device)
    pad = qwen.tokenizer.pad_token_id
    if not isinstance(pad, int):
        raise TypeError("tokenizer pad token id is invalid")
    batches = SyntheticBatchSource(
        seed=config.run.seed + rank * 1_000_003,
        target_height=config.train.resolution, target_width=config.train.resolution,
        local_batch=config.train.local_batch, tokenizer=qwen.tokenizer,
        framing=FramingContract(EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, pad),
    ).pregenerate(args.updates * config.train.accumulation, pin=True)
    restored = production._restore_checkpoint(  # pyright: ignore[reportPrivateUsage]
        loaded, checkpoint=source, module=module, optimizer=optimizer,
        learning_rate_for_update=production._config_learning_rate,  # pyright: ignore[reportPrivateUsage]
    )
    forward: nn.Module | None = None
    no_sync: Any = None
    backward: Any = None
    barrier: Any = lambda: None
    if accelerator is not None:
        if rank > 0:
            seed = config.run.seed + rank * 1_000_003 + restored.state.trainer.successful_updates
            cast(Any, torch).manual_seed(seed)
            torch.cuda.default_generators[int(device.index or 0)].manual_seed(seed)
        forward, wrapped_optimizer = accelerator.prepare(module, optimizer.optimizer)
        optimizer.optimizer = wrapped_optimizer
        no_sync = require_distributed_forward_module(module, cast(nn.Module, forward)).no_sync
        backward, barrier = accelerator.backward, accelerator.wait_for_everyone
    if config.kernels.torch_compile_enabled:
        compile_packed_dit_blocks(module, backend=config.kernels.torch_compile_backend, mode=config.kernels.torch_compile_mode, dynamic=config.kernels.torch_compile_dynamic, emulate_precision_casts=config.kernels.torch_compile_emulate_precision_casts)
    runtime = production._runtime(  # pyright: ignore[reportPrivateUsage]
        config, qwen=qwen, vae=vae, module=module, forward_module=forward,
        restored=restored, device=device, repository_root=args.root, resume=source,
    )
    if rank == 0:
        print(f"pid={os.getpid()} source_update={restored.state.trainer.successful_updates} synthetic_RGB_and_captions=True world_size={world}", flush=True)
    advance_synthetic(loaded=loaded, restored=restored, runtime=runtime, batches=batches,
                      updates=args.updates, output_root=destination, checkpoint_id=args.checkpoint_id,
                      rank=rank, world_size=world, backward=backward, no_sync=no_sync, barrier=barrier)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
