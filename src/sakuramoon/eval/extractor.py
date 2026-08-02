"""Locked local preprocessing, feature extraction, and real-stat loading."""

from __future__ import annotations

import hashlib
import io
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

import torch
from safetensors.torch import (
    load as load_safetensors,  # pyright: ignore[reportUnknownVariableType]
)

from sakuramoon.eval.metrics import FeatureStats


class ExtractorContractError(RuntimeError):
    """A local evaluator dependency does not satisfy its locked contract."""


@dataclass(frozen=True, slots=True)
class VerifiedLocalFile:
    path: Path
    sha256: str
    size: int

    def __post_init__(self) -> None:
        if (
            not self.path.is_absolute()
            or ".." in self.path.parts
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
            or type(self.size) is not int
            or self.size <= 0
        ):
            raise ValueError("verified evaluator file identity is invalid")


class _TensorModule(Protocol):
    def __call__(self, inputs: torch.Tensor) -> object: ...

    def eval(self) -> _TensorModule: ...


def _reject_symlink_components(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        if current.is_symlink():
            raise ExtractorContractError("evaluator identity path contains a symlink")


def _read_local_file(path: Path) -> tuple[bytes, str]:
    if not path.is_absolute() or ".." in path.parts:
        raise ExtractorContractError(
            "evaluator identity path must be a canonical absolute path"
        )
    _reject_symlink_components(path)
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    except OSError:
        raise ExtractorContractError("evaluator identity file cannot be opened") from None
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ExtractorContractError("evaluator identity must be a regular file")
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            chunks.append(chunk)
    except OSError:
        raise ExtractorContractError("evaluator identity file cannot be read") from None
    finally:
        os.close(descriptor)
    payload = b"".join(chunks)
    if not payload:
        raise ExtractorContractError("evaluator identity file must not be empty")
    return payload, digest.hexdigest()


def verify_local_file(path: Path, expected_sha256: str) -> VerifiedLocalFile:
    """Verify one explicit absolute regular file without following symlinks."""

    if len(expected_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in expected_sha256
    ):
        raise ExtractorContractError("evaluator identity SHA-256 is invalid")
    payload, observed = _read_local_file(path)
    if observed != expected_sha256:
        raise ExtractorContractError("evaluator identity SHA-256 mismatch")
    return VerifiedLocalFile(path=path, sha256=observed, size=len(payload))


def _verified_bytes(identity: VerifiedLocalFile) -> bytes:
    payload, observed = _read_local_file(identity.path)
    if observed != identity.sha256 or len(payload) != identity.size:
        raise ExtractorContractError("evaluator identity changed after preflight")
    return payload


def load_real_feature_stats(identity: VerifiedLocalFile) -> FeatureStats:
    """Load exact stats from the same bytes that satisfy the bound identity."""

    try:
        tensors = load_safetensors(_verified_bytes(identity))
    except ExtractorContractError:
        raise
    except Exception:  # noqa: BLE001 - normalize safe-loader diagnostics
        raise ExtractorContractError("real-stat file is not valid Safetensors") from None
    if set(tensors) != {"count", "covariance", "mean"}:
        raise ExtractorContractError("real-stat tensors are unknown or missing")
    count = tensors["count"]
    if count.dtype != torch.int64 or count.shape != ():
        raise ExtractorContractError("real-stat count must be an int64 scalar")
    value = int(count.item())
    try:
        return FeatureStats(
            count=value,
            mean=tensors["mean"],
            covariance=tensors["covariance"],
        )
    except ValueError:
        raise ExtractorContractError("real-stat tensor contract is invalid") from None


class TorchScriptFeatureExtractor:
    """Execute only the explicitly hash-bound local TorchScript pair."""

    def __init__(
        self,
        *,
        preprocess_file: VerifiedLocalFile,
        extractor_file: VerifiedLocalFile,
        device: torch.device,
    ) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise ExtractorContractError("formal feature extraction requires CUDA")
        preprocess_payload = _verified_bytes(preprocess_file)
        extractor_payload = _verified_bytes(extractor_file)
        try:
            preprocess = cast(
                _TensorModule,
                torch.jit.load(io.BytesIO(preprocess_payload), map_location=device),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            )
            extractor = cast(
                _TensorModule,
                torch.jit.load(io.BytesIO(extractor_payload), map_location=device),  # pyright: ignore[reportUnknownMemberType,reportUnknownArgumentType]
            )
        except Exception:  # noqa: BLE001 - normalize local module load diagnostics
            raise ExtractorContractError(
                "preprocess or feature extractor is not valid TorchScript"
            ) from None
        self.preprocess = preprocess.eval()
        self.extractor = extractor.eval()
        self.device = device

    @torch.inference_mode()
    def extract(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[1] != 3:
            raise ExtractorContractError("generated images must be uint8 [B,3,H,W]")
        inputs = images.to(self.device, dtype=torch.float32).div_(255.0)
        processed = self.preprocess(inputs)
        if not isinstance(processed, torch.Tensor):
            raise ExtractorContractError("preprocess must return one tensor")
        result = self.extractor(processed)
        result_tuple = cast(tuple[object, ...], result) if isinstance(result, tuple) else ()
        if (
            not isinstance(result, tuple)
            or len(result_tuple) != 2
            or not all(isinstance(item, torch.Tensor) for item in result_tuple)
        ):
            raise ExtractorContractError(
                "feature extractor must return (features, probabilities)"
            )
        features, probabilities = cast(tuple[torch.Tensor, torch.Tensor], result_tuple)
        batch = images.shape[0]
        if (
            features.ndim != 2
            or features.shape[0] != batch
            or features.shape[1] == 0
            or probabilities.ndim != 2
            or probabilities.shape[0] != batch
            or probabilities.shape[1] < 2
            or not features.is_floating_point()
            or not probabilities.is_floating_point()
        ):
            raise ExtractorContractError("feature extractor returned invalid shapes")
        features_cpu = features.detach().to(device="cpu", dtype=torch.float64)
        probabilities_cpu = probabilities.detach().to(
            device="cpu", dtype=torch.float64
        )
        if not bool(
            torch.isfinite(features_cpu).all().item()
            and torch.isfinite(probabilities_cpu).all().item()
        ):
            raise ExtractorContractError("feature extractor returned nonfinite values")
        if bool((probabilities_cpu < 0.0).any().item()) or not torch.allclose(
            probabilities_cpu.sum(dim=1),
            torch.ones(batch, dtype=torch.float64),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ExtractorContractError(
                "feature extractor probabilities must be normalized"
            )
        return features_cpu, probabilities_cpu


__all__ = [
    "ExtractorContractError",
    "TorchScriptFeatureExtractor",
    "VerifiedLocalFile",
    "load_real_feature_stats",
    "verify_local_file",
]
