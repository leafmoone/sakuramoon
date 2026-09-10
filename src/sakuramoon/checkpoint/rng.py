"""Safe per-rank RNG capture and restore using plain tensors.

P2-R contract: snapshots are validated in two layers so a rank may validate
ANOTHER rank's snapshot against that rank's device ordinal:

- structure: exact keys, dtypes, shapes, installable states (device-agnostic)
- device: the saved ``cuda_device_index`` must equal the caller-supplied
  ``expected_device_index`` (defaults to the current device, preserving the
  historical single-rank contract).
"""

from __future__ import annotations

import random
from typing import cast

import numpy as np
import torch
from numpy.typing import NDArray

from sakuramoon.checkpoint.schema import CheckpointError

_RNG_KEYS = {
    "cuda_device_index",
    "cuda_present",
    "numpy_cached_gaussian",
    "numpy_has_gauss",
    "numpy_keys",
    "numpy_position",
    "python_gauss",
    "python_has_gauss",
    "python_internal",
    "python_version",
    "torch_cpu_state",
    "torch_cuda_state",
}


def capture_rank_rng() -> dict[str, torch.Tensor]:
    python_state = random.getstate()
    numpy_state = cast(
        tuple[str, NDArray[np.uint32], int, int, float],
        np.random.get_state(),
    )
    if numpy_state[0] != "MT19937":
        raise RuntimeError("NumPy global RNG must use MT19937")
    python_gauss = python_state[2]
    cuda_present = torch.cuda.is_available()
    cuda_device = torch.cuda.current_device() if cuda_present else -1
    return {
        "cuda_device_index": torch.tensor(cuda_device, dtype=torch.int64),
        "cuda_present": torch.tensor(cuda_present, dtype=torch.bool),
        "numpy_cached_gaussian": torch.tensor(float(numpy_state[4]), dtype=torch.float64),
        "numpy_has_gauss": torch.tensor(bool(numpy_state[3]), dtype=torch.bool),
        "numpy_keys": torch.from_numpy(  # pyright: ignore[reportUnknownMemberType]
            numpy_state[1].astype(np.int64, copy=True)
        ),
        "numpy_position": torch.tensor(int(numpy_state[2]), dtype=torch.int64),
        "python_gauss": torch.tensor(0.0 if python_gauss is None else python_gauss, dtype=torch.float64),
        "python_has_gauss": torch.tensor(python_gauss is not None, dtype=torch.bool),
        "python_internal": torch.tensor(python_state[1], dtype=torch.int64),
        "python_version": torch.tensor(python_state[0], dtype=torch.int64),
        "torch_cpu_state": torch.get_rng_state(),
        "torch_cuda_state": (
            torch.cuda.get_rng_state(cuda_device)
            if cuda_present
            else torch.empty(0, dtype=torch.uint8)
        ),
    }


def validate_rank_rng(
    tensors: dict[str, torch.Tensor],
    *,
    expected_device_index: int | None = None,
) -> None:
    if set(tensors) != _RNG_KEYS:
        raise CheckpointError("rank RNG file has unknown or missing tensors")
    scalar_dtypes = {
        "cuda_device_index": torch.int64,
        "cuda_present": torch.bool,
        "numpy_cached_gaussian": torch.float64,
        "numpy_has_gauss": torch.bool,
        "numpy_position": torch.int64,
        "python_gauss": torch.float64,
        "python_has_gauss": torch.bool,
        "python_version": torch.int64,
    }
    if any(tensors[key].shape != () or tensors[key].dtype != dtype for key, dtype in scalar_dtypes.items()):
        raise CheckpointError("rank RNG scalar tensor is invalid")
    if tensors["python_internal"].dtype != torch.int64 or tensors["python_internal"].ndim != 1:
        raise CheckpointError("Python RNG state tensor is invalid")
    if tensors["numpy_keys"].dtype != torch.int64 or tensors["numpy_keys"].shape != (624,):
        raise CheckpointError("NumPy RNG state tensor is invalid")
    for key in ("torch_cpu_state", "torch_cuda_state"):
        if tensors[key].dtype != torch.uint8 or tensors[key].ndim != 1:
            raise CheckpointError("Torch RNG state tensor is invalid")
    cuda_present = bool(tensors["cuda_present"].item())
    if cuda_present != torch.cuda.is_available():
        raise CheckpointError("checkpoint CUDA RNG availability does not match")
    if expected_device_index is None:
        expected_device_index = torch.cuda.current_device() if cuda_present else -1
    if cuda_present:
        device_index = int(tensors["cuda_device_index"].item())
        if device_index != expected_device_index or tensors["torch_cuda_state"].numel() == 0:
            raise CheckpointError("checkpoint CUDA RNG device does not match")
    elif int(tensors["cuda_device_index"].item()) != -1 or tensors["torch_cuda_state"].numel() != 0:
        raise CheckpointError("CPU checkpoint has invalid CUDA RNG state")

    python_values = cast(
        list[int],
        tensors["python_internal"].tolist(),  # pyright: ignore[reportUnknownMemberType]
    )
    python_gauss = (
        float(tensors["python_gauss"].item())
        if bool(tensors["python_has_gauss"].item())
        else None
    )
    numpy_keys = tensors["numpy_keys"].numpy().astype(np.uint32, copy=True)
    try:
        python_probe = random.Random()
        python_probe.setstate(
            (
                int(tensors["python_version"].item()),
                tuple(python_values),
                python_gauss,
            )
        )
        numpy_probe = np.random.RandomState()  # pyright: ignore[reportPrivateUsage]
        numpy_probe.set_state(
            (
                "MT19937",
                numpy_keys,
                int(tensors["numpy_position"].item()),
                int(bool(tensors["numpy_has_gauss"].item())),
                float(tensors["numpy_cached_gaussian"].item()),
            )
        )
        torch.Generator(device="cpu").set_state(tensors["torch_cpu_state"])
        if cuda_present:
            torch.Generator(
                device=f"cuda:{int(tensors['cuda_device_index'].item())}"
            ).set_state(tensors["torch_cuda_state"])
    except (RuntimeError, TypeError, ValueError):
        raise CheckpointError("rank RNG state is not restorable") from None


def restore_rank_rng(
    tensors: dict[str, torch.Tensor],
    *,
    expected_device_index: int | None = None,
) -> None:
    validate_rank_rng(tensors, expected_device_index=expected_device_index)
    python_values = cast(
        list[int],
        tensors["python_internal"].tolist(),  # pyright: ignore[reportUnknownMemberType]
    )
    python_internal = tuple(python_values)
    python_gauss = (
        float(tensors["python_gauss"].item())
        if bool(tensors["python_has_gauss"].item())
        else None
    )
    random.setstate((int(tensors["python_version"].item()), python_internal, python_gauss))
    numpy_keys = tensors["numpy_keys"].numpy().astype(np.uint32, copy=True)
    numpy_state = (
        "MT19937",
        numpy_keys,
        int(tensors["numpy_position"].item()),
        int(bool(tensors["numpy_has_gauss"].item())),
        float(tensors["numpy_cached_gaussian"].item()),
    )
    np.random.set_state(numpy_state)
    torch.set_rng_state(tensors["torch_cpu_state"])
    if bool(tensors["cuda_present"].item()):
        torch.cuda.set_rng_state(
            tensors["torch_cuda_state"],
            int(tensors["cuda_device_index"].item()),
        )


class RankRngAnchor:
    """The rank-local training RNG anchor of one checkpoint (P2-R).

    ``mode`` is the recovery classification of the per-rank RNG:

    - ``"same_topology_exact"``: a versioned rank-set checkpoint whose
      SOURCE world size equals the resume world size; the rank's own saved
      snapshot restores bit-exactly.
    - ``"topology_changed"``: a versioned rank-set checkpoint whose source
      world size differs from the resume world size; rank0 may keep the
      saved rank-0 snapshot, every other rank uses the deterministic
      reseed.  NEVER historical-exact.
    - ``"legacy_rank0"``: a pre-rank-set checkpoint; the historical
      rank-0 saved snapshot.
    - ``"legacy"``: a pre-rank-set checkpoint, rank > 0; deterministic
      reseed.  NEVER historical-exact.

    ``tensors`` carries the exact snapshot (other modes leave it ``None``).
    The anchor is validated material only; production rebinds the training
    RNG from it at the FINAL bind point, after preflight.
    """

    __slots__ = ("mode", "source_world_size", "tensors")

    def __init__(
        self,
        mode: str,
        *,
        source_world_size: int,
        tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        if mode not in {
            "same_topology_exact",
            "topology_changed",
            "legacy_rank0",
            "legacy",
        }:
            raise ValueError(f"unknown rank RNG anchor mode: {mode!r}")
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "source_world_size", source_world_size)
        object.__setattr__(self, "tensors", tensors)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RankRngAnchor is immutable")


class RankRngBundle:
    """Immutable per-rank RNG snapshots for one checkpoint boundary (P2-R).

    Built from ``capture_rank_rng()`` on every rank and gathered to the
    publishing rank BEFORE the rank0-only write delegate; the save path only
    ever consumes this frozen value (never re-captures live generators).
    """

    def __init__(self, entries: tuple[tuple[int, dict[str, torch.Tensor]], ...]) -> None:
        if not entries or any(type(rank) is not int or rank < 0 for rank, _ in entries):
            raise ValueError("rank RNG bundle needs non-negative integer ranks")
        ranks = tuple(rank for rank, _ in entries)
        if len(set(ranks)) != len(ranks):
            raise ValueError("rank RNG bundle contains duplicate ranks")
        if any(not isinstance(tensors, dict) for _, tensors in entries):
            raise TypeError("rank RNG bundle entries must be capture_rank_rng mappings")
        object.__setattr__(self, "_entries", entries)

    @property
    def entries(self) -> tuple[tuple[int, dict[str, torch.Tensor]], ...]:
        return self._entries

    @property
    def ranks(self) -> tuple[int, ...]:
        return tuple(rank for rank, _ in self._entries)

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError("RankRngBundle is immutable")


__all__ = [
    "RankRngAnchor",
    "RankRngBundle",
    "capture_rank_rng",
    "restore_rank_rng",
    "validate_rank_rng",
]
