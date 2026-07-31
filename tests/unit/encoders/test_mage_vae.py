from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from sakuramoon.encoders.mage_vae import FrozenMageVAE

_ROOT = Path(__file__).parents[3]
_LOCK = _ROOT / "docs/model-architecture/reviews/T020/mage_upstream_lock.json"
_MATRIX = _ROOT / "docs/model-architecture/reviews/T020/upstream_algorithm_matrix.md"


class _FakeMageVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            image.shape[0],
            128,
            image.shape[2] // 16,
            image.shape[3] // 16,
            dtype=image.dtype,
        )

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            latent.shape[0],
            3,
            latent.shape[2] * 16,
            latent.shape[3] * 16,
            dtype=latent.dtype,
        )


def test_wrapper_freezes_and_uses_mean_shape_contract() -> None:
    backend = _FakeMageVAE()
    vae = FrozenMageVAE(backend)
    image = torch.zeros(2, 3, 32, 48, dtype=torch.bfloat16, requires_grad=True)

    latent = vae.encode(image)
    reconstruction = vae.decode(latent)

    assert latent.shape == (2, 128, 2, 3)
    assert reconstruction.shape == image.shape
    assert latent.dtype == torch.bfloat16
    assert not latent.requires_grad
    assert not backend.weight.requires_grad
    assert not vae.training


def test_train_keeps_backend_in_eval_mode() -> None:
    backend = _FakeMageVAE()
    vae = FrozenMageVAE(backend)

    vae.train()

    assert not vae.training
    assert not backend.training


def test_mage_upstream_commit_license_and_contracts_are_immutable() -> None:
    document = json.loads(_LOCK.read_text(encoding="utf-8"))

    assert set(document) == {
        "schema_version",
        "task_id",
        "provenance_kind",
        "repository",
        "source",
        "license",
        "governance",
        "contracts",
    }
    assert document["schema_version"] == 1
    assert document["task_id"] == "T020"
    assert document["provenance_kind"] == "upstream_implementation_source"
    assert document["repository"] == {
        "name": "Microsoft Mage",
        "url": "https://github.com/microsoft/Mage.git",
        "commit": "8c94a0ac905167f40b05b09332b78752b7f9fbef",
        "git_tree": "73288529688298fc2934707d6b8bb39071810dc1",
    }
    assert document["source"] == {
        "path": "mage_flow/models/modules/mage_vae.py",
        "sha256": "64f4d7041003e416bc2f4fac5bbf8aabf2e7c798ad106682c34332ba347b0ef9",
    }
    assert document["license"] == {
        "spdx_id": "MIT",
        "copyright": "Copyright (c) 2026 Microsoft",
        "path": "LICENSE",
        "sha256": "275b4dd619de4e16a017b10d0beec72abbbbf14ee8a2fc68f8bdb398e821f623",
    }
    assert document["governance"] == {
        "immutable_commit_required": True,
        "runtime_dependency": False,
        "reference_repository": False,
        "local_model_identity": False,
        "local_model_file_hashing": False,
        "automatic_download": False,
    }
    assert set(document["contracts"]) == {
        "checkpoint_key_mapping",
        "posterior_mean_zero_timestep_encode",
        "zero_timestep_decode",
        "replicate_padded_patch_attention",
        "latent_shape_without_patch_packing",
    }

    implementation = (
        _ROOT / "src/sakuramoon/encoders/mage_vae.py"
    ).read_text(encoding="utf-8")
    for required in (
        document["repository"]["commit"],
        'encoder_prefix = "student.dconv_encoder."',
        'decoder_prefix = "pipeline."',
        "return moments[:, :128]",
        'mode="replicate"',
    ):
        assert required in implementation

    matrix = _MATRIX.read_text(encoding="utf-8")
    assert document["repository"]["commit"] in matrix
    assert matrix.count("| PASS |") == len(document["contracts"])
