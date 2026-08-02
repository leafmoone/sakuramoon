from __future__ import annotations

# pyright: reportPrivateUsage=false
import hashlib
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file  # pyright: ignore[reportUnknownVariableType]

from sakuramoon.eval import publisher as publisher_module
from sakuramoon.eval.extractor import (
    ExtractorContractError,
    RealStatsProvenance,
    TorchScriptFeatureExtractor,
    load_real_feature_stats,
    load_real_stats_provenance,
    real_stats_provenance_path,
    verify_local_file,
)
from sakuramoon.eval.publisher import (
    AtomicEvaluationPublisher,
    EvaluationPublicationError,
)


def test_verified_local_file_and_real_stats_are_hash_and_contract_bound(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "real-stats.safetensors"
    save_file(
        {
            "count": torch.tensor(4, dtype=torch.int64),
            "covariance": torch.eye(2, dtype=torch.float64),
            "mean": torch.tensor([1.0, 2.0], dtype=torch.float64),
        },
        stats_path,
    )
    expected = hashlib.sha256(stats_path.read_bytes()).hexdigest()

    verified = verify_local_file(stats_path)
    stats = load_real_feature_stats(verified)

    assert verified.size == stats_path.stat().st_size
    assert verified.sha256 == expected
    assert stats.count == 4
    torch.testing.assert_close(stats.mean, torch.tensor([1.0, 2.0], dtype=torch.float64))
    symlink = tmp_path / "real-stats-link.safetensors"
    symlink.symlink_to(stats_path)
    with pytest.raises(ExtractorContractError, match="symlink"):
        verify_local_file(symlink)
    with pytest.raises(ExtractorContractError, match="canonical absolute path"):
        verify_local_file(Path("real-stats.safetensors"))
    nested = tmp_path / "nested"
    nested.mkdir()
    with pytest.raises(ExtractorContractError, match="canonical absolute path"):
        verify_local_file(nested / ".." / stats_path.name)


def test_real_stats_reject_unknown_or_invalid_tensor_contract(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.safetensors"
    save_file({"mean": torch.zeros(2)}, unknown)
    unknown_identity = verify_local_file(unknown)
    with pytest.raises(ExtractorContractError, match="unknown or missing"):
        load_real_feature_stats(unknown_identity)

    invalid = tmp_path / "invalid.safetensors"
    save_file(
        {
            "count": torch.tensor(1.0),
            "covariance": torch.eye(2, dtype=torch.float64),
            "mean": torch.zeros(2, dtype=torch.float64),
        },
        invalid,
    )
    invalid_identity = verify_local_file(invalid)
    with pytest.raises(ExtractorContractError, match="int64 scalar"):
        load_real_feature_stats(invalid_identity)


def test_real_stats_loads_only_the_bytes_matching_preflight_identity(
    tmp_path: Path,
) -> None:
    path = tmp_path / "real-stats.safetensors"
    save_file(
        {
            "count": torch.tensor(2, dtype=torch.int64),
            "covariance": torch.eye(1, dtype=torch.float64),
            "mean": torch.zeros(1, dtype=torch.float64),
        },
        path,
    )
    identity = verify_local_file(path)
    path.write_bytes(b"changed-after-preflight")

    with pytest.raises(ExtractorContractError, match="changed after preflight"):
        load_real_feature_stats(identity)


def test_real_stats_metadata_binds_validation_and_preprocess_identity(
    tmp_path: Path,
) -> None:
    stats_path = tmp_path / "real-stats.safetensors"
    save_file(
        {
            "count": torch.tensor(4, dtype=torch.int64),
            "covariance": torch.eye(2, dtype=torch.float64),
            "mean": torch.zeros(2, dtype=torch.float64),
        },
        stats_path,
    )
    preprocess_path = tmp_path / "preprocess.pt"
    preprocess_path.write_bytes(b"governed-preprocess")
    extractor_path = tmp_path / "extractor.pt"
    extractor_path.write_bytes(b"governed-extractor")
    stats_file = verify_local_file(stats_path)
    preprocess_file = verify_local_file(preprocess_path)
    extractor_file = verify_local_file(extractor_path)
    provenance = RealStatsProvenance(
        selection_id="1" * 64,
        manifest_id="2" * 64,
        prompt_manifest_sha256="3" * 64,
        preprocess_sha256=preprocess_file.sha256,
        feature_extractor="inception",
        feature_extractor_version="locked-1",
        feature_extractor_sha256=extractor_file.sha256,
        real_stats_sha256=stats_file.sha256,
        sample_count=4,
    )
    metadata_path = real_stats_provenance_path(stats_path)
    metadata_path.write_bytes(provenance.canonical_bytes())
    metadata_file = verify_local_file(metadata_path)

    assert load_real_stats_provenance(
        metadata_file,
        real_stats_file=stats_file,
        selection_id="1" * 64,
        manifest_id="2" * 64,
        prompt_manifest_sha256="3" * 64,
        preprocess_file=preprocess_file,
        feature_extractor="inception",
        feature_extractor_version="locked-1",
        extractor_file=extractor_file,
        stats_count=4,
    ) == provenance

    with pytest.raises(ExtractorContractError, match="differs from validation"):
        load_real_stats_provenance(
            metadata_file,
            real_stats_file=stats_file,
            selection_id="9" * 64,
            manifest_id="2" * 64,
            prompt_manifest_sha256="3" * 64,
            preprocess_file=preprocess_file,
            feature_extractor="inception",
            feature_extractor_version="locked-1",
            extractor_file=extractor_file,
            stats_count=4,
        )


def test_torchscript_loader_and_output_contract_fail_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def invalid_load(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("invalid module")

    monkeypatch.setattr(torch.jit, "load", invalid_load)
    invalid_module = tmp_path / "invalid.pt"
    invalid_module.write_bytes(b"invalid-torchscript")
    module_identity = verify_local_file(invalid_module)
    with pytest.raises(ExtractorContractError, match="not valid TorchScript"):
        TorchScriptFeatureExtractor(
            preprocess_file=module_identity,
            extractor_file=module_identity,
            device=torch.device("cuda", 0),
        )

    class Identity:
        def __call__(self, inputs: torch.Tensor) -> torch.Tensor:
            return inputs

        def eval(self) -> Identity:
            return self

    class BadExtractor:
        def __call__(self, inputs: torch.Tensor) -> object:
            del inputs
            return torch.zeros(1)

        def eval(self) -> BadExtractor:
            return self

    extractor = object.__new__(TorchScriptFeatureExtractor)
    extractor.preprocess = Identity()
    extractor.extractor = BadExtractor()
    extractor.device = torch.device("cpu")
    with pytest.raises(ExtractorContractError, match="features, probabilities"):
        extractor.extract(torch.zeros((1, 3, 16, 16), dtype=torch.uint8))


def test_atomic_publisher_commits_complete_tree_and_prevents_clobber(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    publisher = AtomicEvaluationPublisher(output_root, "evaluation-test")
    publisher.write_json("nested/result.json", {"value": 1})
    _path, image_payload = publisher.write_png(
        "images/prompt.png", torch.zeros((3, 16, 16), dtype=torch.uint8)
    )

    with pytest.raises(FileExistsError):
        AtomicEvaluationPublisher(output_root, "evaluation-test")
    final = publisher.commit({"artifact_count": 2})

    assert final == output_root / "evaluation-test"
    assert (final / "COMPLETE").read_bytes() == b"complete\n"
    assert json.loads((final / "summary.json").read_text()) == {"artifact_count": 2}
    assert hashlib.sha256((final / "images/prompt.png").read_bytes()).digest() == (
        hashlib.sha256(image_payload).digest()
    )
    assert not (output_root / ".evaluation-test.incomplete").exists()
    with pytest.raises(FileExistsError):
        AtomicEvaluationPublisher(output_root, "evaluation-test")


def test_atomic_publisher_preserves_concurrent_destination_and_staging(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    publisher = AtomicEvaluationPublisher(output_root, "evaluation-race")
    publisher.write_bytes("artifact.bin", b"staged-artifact")
    original_publish_tree = publisher_module._publish_tree_noreplace

    def create_destination_then_publish(source: Path, destination: Path) -> None:
        destination.mkdir()
        (destination / "owner-marker").write_bytes(b"concurrent-owner")
        original_publish_tree(source, destination)

    monkeypatch.setattr(
        publisher_module, "_publish_tree_noreplace", create_destination_then_publish
    )

    with pytest.raises(FileExistsError):
        publisher.commit({"artifact_count": 1})

    assert (publisher.final_path / "owner-marker").read_bytes() == b"concurrent-owner"
    assert not (publisher.final_path / "COMPLETE").exists()
    assert (publisher.staging_path / "artifact.bin").read_bytes() == b"staged-artifact"
    assert (publisher.staging_path / "COMPLETE").read_bytes() == b"complete\n"
    assert json.loads((publisher.staging_path / "summary.json").read_text()) == {
        "artifact_count": 1
    }


def test_atomic_publisher_rejects_traversal_and_does_not_publish_early(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "artifacts"
    publisher = AtomicEvaluationPublisher(output_root, "evaluation-incomplete")

    with pytest.raises(EvaluationPublicationError, match="relative path"):
        publisher.write_bytes("../escape", b"invalid")

    assert not (output_root / "evaluation-incomplete").exists()
    assert (output_root / ".evaluation-incomplete.incomplete").is_dir()


def test_atomic_publisher_rejects_symlinked_output_ancestor(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)

    with pytest.raises(EvaluationPublicationError, match="symbolic link"):
        AtomicEvaluationPublisher(linked / "artifacts", "evaluation-test")


def test_atomic_publisher_rejects_noncanonical_output_root(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()

    with pytest.raises(EvaluationPublicationError, match="target"):
        AtomicEvaluationPublisher(
            nested / ".." / "artifacts", "evaluation-noncanonical"
        )


def test_atomic_publisher_rejects_unsafe_run_identity(tmp_path: Path) -> None:
    with pytest.raises(EvaluationPublicationError, match="target"):
        AtomicEvaluationPublisher(tmp_path, "../evaluation")
