"""Measure the intentional head architecture change using synthetic config-shaped inputs.

This diagnostic never reads checkpoints, datasets, credentials, or frozen assets.
It is not a full-model quality or training-stability evaluation.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import torch

from sakuramoon.config import load_config
from sakuramoon.model.output_head import FinalOutputHead


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-root", type=Path, default=Path("config"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--tokens", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args(argv)
    if args.tokens <= 0 or args.batch_size <= 0:
        parser.error("tokens and batch size must be positive")
    config = load_config(
        args.config, config_root=args.config_root, environment={}, validate_secrets=False
    ).config
    hidden, channels = config.model.dit.hidden_size, config.model.head.out_channels
    device = torch.device(args.device)
    dtype = getattr(torch, args.dtype)
    cast(Any, torch).manual_seed(1729)
    heads = [
        FinalOutputHead(
            hidden_size=hidden, out_channels=channels, norm_eps=config.model.dit.norm_eps,
            projection_dtype=torch.float32, weight_zero_init=True, bias_zero_init=True,
            final_output_norm=norm,
        ).to(device)
        for norm in ("rmsnorm", "none")
    ]
    old, new = heads
    with torch.no_grad():
        old.projection.weight.normal_(std=hidden ** -0.5)
        old.projection.bias.normal_(std=0.1)
        old.norm.weight.uniform_(0.5, 1.5)
    new.projection.load_state_dict(old.projection.state_dict(), strict=True)
    x = 3 * torch.randn(args.batch_size, args.tokens, hidden, device=device, dtype=dtype)
    scale = 0.1 * torch.randn(args.batch_size, hidden, device=device)
    shift = 0.1 * torch.randn(args.batch_size, hidden, device=device)
    target = torch.randn(args.batch_size, channels, 1, args.tokens, device=device)
    print(f"torch={torch.__version__}; device={device}; dtype={dtype}; hidden={hidden}; channels={channels}")
    print("Synthetic head-only diagnostic; nonzero projection and nonunit old gamma. Output equivalence is NOT expected.")
    for head in heads:
        local = [value.detach().clone().requires_grad_() for value in (x, scale, shift)]
        output = head(*local, (1, args.tokens))
        loss = (output - target).square().mean()
        gradients = torch.autograd.grad(loss, (*local, *head.parameters()))
        if not torch.isfinite(output).all() or not all(torch.isfinite(grad).all() for grad in gradients):
            raise FloatingPointError("head forward or backward is nonfinite")
        gradient_norm = torch.stack([grad.float().square().sum() for grad in gradients]).sum().sqrt()
        print(
            f"final_output_norm={head.final_output_norm}: "
            f"output_rms={float(output.detach().square().mean().sqrt()):.8g}; "
            f"loss={float(loss.detach()):.8g}; all_gradient_l2={float(gradient_norm):.8g}; "
            f"hidden_gradient_rms={float(gradients[0].float().square().mean().sqrt()):.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
