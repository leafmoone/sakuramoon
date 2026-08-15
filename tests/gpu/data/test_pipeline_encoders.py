from __future__ import annotations

import io
import json
import os
import tarfile
from collections.abc import Mapping
from pathlib import Path

import pytest
import torch
from PIL import Image

from sakuramoon.config.schema import DataBucketsConfig
from sakuramoon.data.buckets import BucketShape, generate_base_buckets
from sakuramoon.data.caption import (
    CaptionDropoutProbabilities,
    CaptionFields,
    NlCandidates,
    NlDropoutProbabilities,
)
from sakuramoon.data.collate import collate_samples
from sakuramoon.data.manifest import ShardRecord
from sakuramoon.data.metadata import MetadataFieldMapping
from sakuramoon.data.pipeline import WebDatasetPipeline
from sakuramoon.data.production import (
    PRODUCTION_METADATA_FIELDS,
    adapt_modelscope_metadata,
    parse_modelscope_caption_fields,
)
from sakuramoon.data.serialize import FramingContract
from sakuramoon.encoders.mage_vae import load_local_mage_vae
from sakuramoon.encoders.qwen import load_local_qwen

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="a CUDA-compatible HCU is required"
)


def _empty_fields(_raw: Mapping[str, object]) -> CaptionFields:
    return CaptionFields(
        nsfw=(),
        character=(),
        copyright=(),
        general=(),
        artists=(),
        candidate_tags=frozenset(),
        nl=NlCandidates(None, None, None, None, None),
    )


def _probabilities() -> CaptionDropoutProbabilities:
    nl = NlDropoutProbabilities(0.0, 0.0, 0.0, 0.0, 0.0)
    return CaptionDropoutProbabilities(tag=0.0, candidate_source=0.0, nl=nl)


def _identity_metadata(
    raw: Mapping[str, object],
) -> Mapping[str, object]:
    return raw


def _write_shard(path: Path) -> None:
    image = io.BytesIO()
    Image.new("RGB", (512, 512), color=(10, 20, 30)).save(image, format="JPEG")
    metadata = json.dumps(
        {
            "id": 1,
            "release": "synthetic",
            "width": 512,
            "height": 512,
            "caption_available": False,
        }
    ).encode()
    with tarfile.open(path, "w") as archive:
        for extension, payload in (("json", metadata), ("jpg", image.getvalue())):
            info = tarfile.TarInfo(f"000001.{extension}")
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))


def test_real_pipeline_qwen_and_mage_encode_one_batch(tmp_path: Path) -> None:
    repository_root = Path(__file__).parents[3]
    device = torch.device("cuda", 0)
    qwen = load_local_qwen(
        repository_root,
        device,
        attention_backend="flash_attention_2",
    )
    vae = load_local_mage_vae(repository_root, device)
    shard = tmp_path / "sample.tar"
    _write_shard(shard)
    rejections: list[str] = []
    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(
            ShardRecord(
                path=shard.name,
                bytes=shard.stat().st_size,
            ),
        ),
        metadata_adapter=_identity_metadata,
        metadata_fields=MetadataFieldMapping(
            id_field="id",
        ),
        buckets=(BucketShape(512, 512),),
        min_crop_retention=0.8,
        probabilities=_probabilities(),
        style_condition_mode="artist_or_character",
        tokenizer=qwen.tokenizer,
        framing=FramingContract(34, 5, 248044),
        caption_fields_parser=_empty_fields,
        rejection_observer=rejections.append,
        base_seed=7,
        stage="S0",
        cycle_index=0,
    )
    sample = next(iter(pipeline))
    batch = collate_samples((sample,))

    torch.cuda.reset_peak_memory_stats(device)
    qwen_output = qwen.encoder(
        batch.input_ids.to(device),
        batch.attention_mask.to(device),
    )
    image = batch.images.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
    latent = vae.encode(image)

    assert rejections == []
    assert batch.images.shape == (1, 3, 512, 512)
    assert batch.input_ids.shape == (1, 98)
    assert qwen_output.hidden_states.shape == (1, 98, 7, 2048)
    assert latent.shape == (1, 128, 32, 32)
    assert torch.isfinite(qwen_output.hidden_states).all()
    assert torch.isfinite(latent).all()
    assert not qwen.encoder.training and not vae.training
    assert not any(parameter.requires_grad for parameter in qwen.encoder.parameters())
    assert not any(parameter.requires_grad for parameter in vae.parameters())
    assert torch.cuda.max_memory_allocated(device) > 0


def test_real_modelscope_shard_pipeline_qwen_and_mage() -> None:
    shard_value = os.environ.get("SAKURAMOON_REAL_SHARD_PATH")
    if shard_value is None:
        pytest.skip("SAKURAMOON_REAL_SHARD_PATH is not set")
    shard = Path(shard_value)
    repository_root = Path(__file__).parents[3]
    device = torch.device("cuda", 0)
    qwen = load_local_qwen(
        repository_root,
        device,
        attention_backend="flash_attention_2",
    )
    vae = load_local_mage_vae(repository_root, device)
    nl = NlDropoutProbabilities(0.3, 0.3, 0.3, 0.3, 0.3)
    probabilities = CaptionDropoutProbabilities(tag=0.1, candidate_source=0.3, nl=nl)
    buckets = generate_base_buckets(
        DataBucketsConfig(
            base_area_px=262144,
            quantum_px=32,
            min_short_edge_px=256,
            max_aspect_ratio=4.0,
            shape_count=17,
            transpose_closed=True,
        )
    )
    rejections: list[str] = []
    pipeline = WebDatasetPipeline(
        shard_paths=(shard,),
        shard_records=(
            ShardRecord(
                path="data/1_2024/shard-000000.tar",
                bytes=2146867200,
            ),
        ),
        metadata_adapter=adapt_modelscope_metadata,
        metadata_fields=PRODUCTION_METADATA_FIELDS,
        buckets=buckets,
        min_crop_retention=0.8,
        probabilities=probabilities,
        style_condition_mode="artist_or_character",
        tokenizer=qwen.tokenizer,
        framing=FramingContract(34, 5, 248044),
        caption_fields_parser=parse_modelscope_caption_fields,
        rejection_observer=rejections.append,
        base_seed=7,
        stage="S0",
        cycle_index=0,
    )
    sample = next(iter(pipeline))
    batch = collate_samples((sample,))

    qwen_output = qwen.encoder(
        batch.input_ids.to(device),
        batch.attention_mask.to(device),
    )
    image = batch.images.to(device=device, dtype=torch.bfloat16).div(127.5).sub(1.0)
    latent = vae.encode(image)

    assert sample.source_shard == "data/1_2024/shard-000000.tar"
    assert sample.sample_id > 0
    assert sample.audit.source_width > 0 and sample.audit.source_height > 0
    assert batch.images.shape[0] == 1
    assert qwen_output.hidden_states.shape[:2] == batch.input_ids.shape
    assert latent.shape[0:2] == (1, 128)
    assert torch.isfinite(qwen_output.hidden_states).all()
    assert torch.isfinite(latent).all()
