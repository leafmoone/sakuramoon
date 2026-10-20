"""Two-rank CPU/Gloo presence consensus through the real optimizer step.

Real CPU NS kernels, collectives and AdamW; explicit CPU AdamW/SR adapters from
index tests, NOT native TorchAO/SR or CUDA/NCCL acceptance. No kernel substitution.
Process-group timeout is 15s; the parent bounds each pair of children to 90s.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Any, cast

import pytest
import torch
import torch.distributed as dist
from test_missing_gradient_index import (
    MASKS,
    VARIANTS,
    assert_equal,
    assert_index,
    expected_index,
    make_optimizer,
    seed_gradients,
    snapshot,
)


def _step_with_collective_audit(opt: Any) -> list[tuple[Any, ...]]:
    calls: list[tuple[Any, ...]] = []
    # PyTorch's collective/tolist stubs are incomplete; keep Any at that boundary.
    real_reduce, real_broadcast = cast(Any, dist).all_reduce, cast(Any, dist).broadcast
    real_finite = opt._validate_finite_gradients

    def reduce(tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM, **kwargs: Any) -> Any:
        before = cast(Any, tensor).tolist()
        result = real_reduce(tensor, op=op, **kwargs)
        calls.append(("reduce", op, tensor.dtype, tuple(tensor.shape), before, cast(Any, tensor).tolist()))
        return result

    def broadcast(tensor: torch.Tensor, src: int, **kwargs: Any) -> Any:
        calls.append(("broadcast", src, tuple(tensor.shape), tensor.dtype))
        return real_broadcast(tensor, src=src, **kwargs)

    def finite() -> None:
        # This proves consensus is the first step operation, not only first NS op.
        assert len(calls) == 1 and calls[0][1] == dist.ReduceOp.SUM
        real_finite()

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dist, "all_reduce", reduce)
        patch.setattr(dist, "broadcast", broadcast)
        patch.setattr(opt, "_validate_finite_gradients", finite)
        opt.step()
    return calls


def _consistent(root: Path, variant: str, rank: int) -> None:
    opt = make_optimizer(root, variant, rank, 2)
    reference = make_optimizer(root / "reference", variant)
    route = opt.routing
    manifest = route.routing_manifest()
    # Retain state across all missing positions and a second changing-mask cycle.
    masks = (*MASKS, (1, 4), (), tuple(range(6)), tuple(range(6)))
    for step, missing in enumerate(masks):
        adam_none = step == len(masks) - 1
        seed_gradients(opt, missing, step=step, adam_none=adam_none)
        seed_gradients(reference, missing, step=step, adam_none=adam_none)
        assert_index(opt)
        expected = expected_index(opt)
        calls = _step_with_collective_audit(opt)
        # Single-rank reference performs the same untouched CPU NS arithmetic.
        reference.step()
        assert opt.routing is route
        assert_equal(route.routing_manifest(), manifest)
        assert_equal([s.parameter for s in opt.audit.specs],
                     [s.parameter for s in reference.audit.specs])
        assert_equal(list(opt._momenta.values()), list(reference._momenta.values()))
        assert opt._refs == reference._refs
        assert opt.observations == reference.observations == step + 1
        assert opt.sr_rng.calls == reference.sr_rng.calls == step + 1
        assert_equal(opt.optimizer.state_dict(), reference.optimizer.state_dict())
        presence = [int(s.parameter.grad is not None) for s in (
            *route.cmuon_specs, *route.adamw_specs,
        )]
        assert calls[0] == ("reduce", dist.ReduceOp.SUM, torch.int32, (8,),
                            presence, [2 * p for p in presence])
        broadcasts = [call for call in calls if call[0] == "broadcast"]
        assert broadcasts == [("broadcast", owner, shape, torch.bfloat16)
                              for _spec, _ci, shape, owner in expected]
        reductions = [call for call in calls if call[0] == "reduce"]
        if expected:
            assert [call[1] for call in reductions] == [
                dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN,
                dist.ReduceOp.MAX, dist.ReduceOp.MIN, dist.ReduceOp.MAX,
            ]
            assert reductions[1][3] == (len(expected),)
        else:
            assert len(calls) == 3  # presence plus full-routing parameter MIN/MAX
            assert [call[1] for call in reductions] == [
                dist.ReduceOp.SUM, dist.ReduceOp.MIN, dist.ReduceOp.MAX,
            ]
        # Invariants include every routed parameter in routing order, even when
        # its gradient is missing (including the all-None CMuon/AdamW step).
        with torch.no_grad():
            fingerprints = [float(value) for spec in route.cmuon_specs for value in (
                spec.parameter.float().square().mean().sqrt(),
                spec.parameter.float().abs().max(),
            )]
        for call, op in zip(reductions[-2:], (dist.ReduceOp.MIN, dist.ReduceOp.MAX)):
            assert call[1:4] == (op, torch.float32, (2 * len(route.cmuon_specs),))
            assert call[4] == call[5] == fingerprints
        if variant == "rescue":
            assert opt.fp32_attempts == 0
    # Both ranks independently passed exact single-rank numerical/state parity.
    assert opt.max_delta_rank_spread == opt.max_param_rank_diff == 0


def _mismatch(root: Path, variant: str, rank: int, kind: str) -> None:
    opt = make_optimizer(root, variant, rank, 2)
    seed_gradients(opt, ())
    opt.step()  # nonempty AdamW state, nonzero momentum/ref and SR counters
    seed_gradients(opt, (), step=1)
    targets: list[Any] = []
    if kind in ("cmuon", "both"):
        targets.append(opt.routing.cmuon_specs[0])
    if kind in ("adamw", "both"):
        targets.append(opt.routing.adamw_specs[0])
    for spec in targets:
        if rank == 0:
            spec.parameter.grad = None
    # Mismatch must win even over finite-gradient validation on another input.
    if rank == 1:
        opt.routing.cmuon_specs[1].parameter.grad.fill_(float("nan"))
    before = snapshot(opt)
    calls: list[tuple[Any, ...]] = []
    real_reduce = cast(Any, dist).all_reduce

    def reduce(tensor: torch.Tensor, op: Any = dist.ReduceOp.SUM, **kwargs: Any) -> Any:
        before_values = cast(Any, tensor).tolist()
        result = real_reduce(tensor, op=op, **kwargs)
        calls.append((op, tensor.dtype, tuple(tensor.shape), before_values, cast(Any, tensor).tolist()))
        return result

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("presence mismatch reached preparation, NS, broadcast or commit")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dist, "all_reduce", reduce)
        patch.setattr(dist, "broadcast", forbidden)
        patch.setattr(opt, "_validate_finite_gradients", forbidden)
        patch.setattr(opt, "_active_inputs", forbidden)
        patch.setattr(opt.sr_rng, "run_step", forbidden)
        with pytest.raises(RuntimeError, match="gradient presence differs across ranks") as error:
            opt.step()
    assert str(error.value).split(": ", 1)[1].split(", ") == [s.name for s in targets]
    assert_equal(snapshot(opt), before)  # weights, momentum, refs, AdamW, SR, ALL scalar counters
    assert len(calls) == 1
    assert calls[0][:3] == (dist.ReduceOp.SUM, torch.int32, (8,))
    counts = [2] * 8
    if kind in ("cmuon", "both"):
        counts[0] = 1
    if kind in ("adamw", "both"):
        counts[6] = 1
    assert calls[0][4] == counts
    assert not list(root.rglob("*.json"))  # no rescue/forensic artifact path reached


def _worker(rank: int, root: str, variant: str, kind: str) -> None:
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    path = Path(root)
    try:
        dist.init_process_group(
            "gloo", init_method=(path / "rendezvous").as_uri(), rank=rank,
            world_size=2, timeout=timedelta(seconds=15),
        )
        if kind == "consistent":
            _consistent(path / f"rank{rank}", variant, rank)
        else:
            _mismatch(path / f"rank{rank}", variant, rank, kind)
    except BaseException:
        (path / f"error-rank{rank}.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn(root: Path, variant: str, kind: str) -> None:
    context = mp.get_context("spawn")
    processes = [context.Process(target=_worker, args=(rank, str(root), variant, kind))
                 for rank in range(2)]
    deadline = time.monotonic() + 90
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        hung = [process.pid for process in processes if process.is_alive()]
        errors = "\n".join(path.read_text(encoding="utf-8") for path in root.glob("error-rank*.txt"))
        assert not hung, f"Gloo subprocess exceeded 90s: {hung}\n{errors}"
        assert all(process.exitcode == 0 for process in processes), errors
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.pid is not None:
                process.join(5)
                if process.is_alive():
                    process.kill()
                    process.join(5)


@pytest.mark.skipif(not dist.is_available() or not dist.is_gloo_available(), reason="Gloo unavailable")
@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("kind", ("consistent", "cmuon", "adamw", "both"))
def test_two_rank_presence_consensus(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str, kind: str,
) -> None:
    for name in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES"):
        monkeypatch.setenv(name, "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        monkeypatch.setenv(name, "2")
    if os.name == "nt":
        pytest.skip("POSIX CPU/Gloo subprocess coverage runs in CI/WSL")
    _spawn(tmp_path, variant, kind)
