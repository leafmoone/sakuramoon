"""SakuraMoon Camera v2 — coordinate causal audit: shared helpers.

Read-only, forward-only audit. No training, no checkpoint writes, no source edits.
All production paths are imported and called unchanged; this module only wires them.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

import torch

# ---------------- fixed audit constants ----------------

MASTER_SEED = 20260906
T_QUANTILES: tuple[float, float, float, float] = (0.10, 0.35, 0.65, 0.90)
N_STRATA = 4
# Required arms: CORRECT, IDENTITY, OPPOSITE, SHUFFLED (spec s10).
# Optional interpretive arms: HALF (0.5x) and OVER (1.5x) interpolation (spec s11).
ARMS: tuple[str, ...] = ("CORRECT", "IDENTITY", "OPPOSITE", "SHUFFLED", "HALF", "OVER")
ARMS_REQUIRED: tuple[str, ...] = ("CORRECT", "IDENTITY", "OPPOSITE", "SHUFFLED")
SAME_ARM = "SAME"  # exact duplicate of CORRECT; numeric-zero harness check (spec s20)
RANDOM_ARM = "RANDOM"  # ordinary units only: seeded random geometry sensitivity check
T_EPS = 0.05
NOISE_OBSERVATION_BOUNDARY = 0.95
P_MEAN = -0.8
P_STD = 0.8
NOISE_SCALE = 1.0
NORM_EPS = 1e-8  # normalization floor: margin / max(L_correct, eps)

N_CAMERA_TARGET = 2048  # spec s5: >=1024 (2048 preferred)
N_ORDINARY_TARGET = 512  # spec s5: >=256
N_STRICT_TARGET = 256  # strict-identity ordinary subset for the zero-margin negative control
N_RANDOM_GEOM = 64  # ordinary 256x256 units receiving a random camera geometry (NC-3)

REPO = Path("/sakuramoon-runtime/sakuramoon-camera-v2-c2")
RUNTIME_ROOT = Path("/sakuramoon-runtime")
OUT = Path("/tmp/camera-coordinate-causal")
CONFIG_PATH = REPO / "config" / "train_g1_camera_v2_p25.toml"
CONFIG_ROOT = REPO / "config"
VALIDATION_DIR = (
    RUNTIME_ROOT / "data" / "validation-cohorts" / "s0-validation-50k-v1"
)
SELECTION_JSON = VALIDATION_DIR / "validation-selection.json"
SHARD_ROOT = VALIDATION_DIR / "shards"
VENV_PY = "/sakuramoon-runtime/sakuramoon-dtk-venv/bin/python"

CKPTS: dict[str, Path] = {
    "PRE": RUNTIME_ROOT / "output_model" / "g1" / "ckpt_116100_raw-116100-update-cadence",
    "MID": OUT / "ckpts" / "MID" / "ckpt_117100_raw-117100-update-cadence",
    "POST": (
        RUNTIME_ROOT
        / "output_model"
        / "g1_camera_v2_p25"
        / "ckpt_118100_raw-118100-update-cadence"
    ),
}

ENTRANCE_HEAD = "b2443af436b268fafb2cb6c05d724e3ab0d6042c"

# ---------------- deterministic numerics ----------------

# Acklam's rational approximation of the inverse standard-normal CDF
# (maximum relative error ~1.15e-9; verified in-script against known values).
_A = (-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02,
      1.383577518672690e02, -3.066479806614716e01, 2.506628277459239e00)
_B = (-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02,
      6.680131188771972e01, -1.328068155288572e01)
_C = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00,
      -2.549732539343734e00, 4.374664141464968e00, 2.938163982698783e00)
_D = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00,
      3.754408661907416e00)
_P_LO = 0.02425


def normal_cdf_inv(p: float) -> float:
    if not (0.0 < p < 1.0):
        raise ValueError("p must be in (0,1)")
    if p < _P_LO:
        q = math.sqrt(-2.0 * math.log(p))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return num / den
    if p > 1.0 - _P_LO:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        num = (((((_C[0] * q + _C[1]) * q + _C[2]) * q + _C[3]) * q + _C[4]) * q + _C[5])
        den = (((_D[0] * q + _D[1]) * q + _D[2]) * q + _D[3]) * q + 1.0
        return -num / den
    q = p - 0.5
    r = q * q
    num = (((((_A[0] * r + _A[1]) * r + _A[2]) * r + _A[3]) * r + _A[4]) * r + _A[5])
    den = (((((_B[0] * r + _B[1]) * r + _B[2]) * r + _B[3]) * r + _B[4]) * r + 1.0)
    return num * q / den


def jlt_quantile(q: float, p_mean: float = P_MEAN, p_std: float = P_STD) -> float:
    """t-quantile of the production JLT sampler: sigmoid(N(p_mean, p_std)).

    sample_jlt_timesteps (objective/flow.py) computes sigmoid(z*p_std + p_mean)
    for z ~ N(0,1); the q-quantile is therefore sigmoid(p_mean + p_std * Phi^-1(q)).
    """
    z = normal_cdf_inv(q)
    a = p_mean + p_std * z
    return 1.0 / (1.0 + math.exp(-a))


def t_values() -> tuple[float, float, float, float]:
    """The four deterministic timestep strata (10/35/65/90 percentiles of JLT)."""
    return tuple(jlt_quantile(q) for q in T_QUANTILES)


def eps_seed(unit_idx: int, stratum: int) -> int:
    """Deterministic per-(unit, stratum) noise seed (documented in provenance)."""
    return MASTER_SEED * 1_000_003 + unit_idx * 100 + stratum


def random_geom_seed(unit_idx: int) -> int:
    return MASTER_SEED * 1_000_003 + 900_000 + unit_idx


# ---------------- hashing ----------------


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tree_sha256(root: Path) -> str:
    """sha256 over the sorted (relative path, file sha256) list of root's files."""
    files: list[tuple[str, str]] = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            files.append((p.relative_to(root).as_posix(), sha256_file(p)))
    payload = json.dumps(files, sort_keys=True).encode("utf-8")
    return sha256_bytes(payload)


# ---------------- deterministic permutation ----------------


def derangement(n: int, seed: int = MASTER_SEED) -> list[int]:
    """Seeded uniform derangement (no fixed points) of 0..n-1."""
    if n < 2:
        raise ValueError("derangement requires n >= 2")
    rng = random.Random(seed)
    while True:
        perm = list(range(n))
        rng.shuffle(perm)
        if all(perm[i] != i for i in range(n)):
            return perm


def size_scale_aspect(height_px: int, width_px: int) -> tuple[float, float]:
    """Production size/aspect conditions (train/runtime.py::_size_conditions)."""
    size_scale = 0.5 * math.log2((height_px * width_px) / float(512 * 512))
    aspect = math.log2(width_px / height_px)
    return size_scale, aspect


# ---------------- JSON helpers ----------------


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[write] {path}", flush=True)


# ---------------- production pipeline wiring ----------------


def load_runtime_config():
    from sakuramoon.config import load_config

    return load_config(CONFIG_PATH, config_root=CONFIG_ROOT, validate_secrets=False)


def build_validation_pipeline(
    shard_local: Path,
    shard_rel: str,
    shard_bytes: int,
    config: Any,
    qwen: Any,
    padding_token_id: int,
    observer,
):
    """One-shard production pipeline, wired exactly like train/production.py
    (ProductionPipelineFactory.from_config + pipeline_for_lease)."""
    from sakuramoon.data.buckets import generate_base_buckets, scale_buckets
    from sakuramoon.data.caption import (
        CaptionDropoutProbabilities,
        NlDropoutProbabilities,
    )
    from sakuramoon.data.camera_viewport import CameraViewportPolicy
    from sakuramoon.data.manifest import ShardRecord
    from sakuramoon.data.pipeline import WebDatasetPipeline
    from sakuramoon.data.production import (
        PRODUCTION_METADATA_FIELDS,
        adapt_modelscope_metadata,
        parse_modelscope_caption_fields,
    )
    from sakuramoon.data.serialize import (
        EXPECTED_PREFIX_TOKENS,
        EXPECTED_SUFFIX_TOKENS,
        FramingContract,
    )
    from sakuramoon.data.spatial_crop import SpatialCropPolicy

    dropout = config.caption.dropout
    probabilities = CaptionDropoutProbabilities(
        condition_route=dropout.condition_route,
        condition_only=dropout.condition_only,
        tag=dropout.tag,
        candidate_source=dropout.candidate_source,
        nl=NlDropoutProbabilities(
            long_names=dropout.nl.long_names,
            long_no_names=dropout.nl.long_no_names,
            short_vibes=dropout.nl.short_vibes,
            nl2=dropout.nl.nl2,
            nl3=dropout.nl.nl3,
        ),
    )
    buckets = scale_buckets(
        generate_base_buckets(config.data.buckets), config.stage.resolution
    )
    spatial_policy = SpatialCropPolicy.from_config(
        config.data.spatial_crop,
        min_crop_retention=config.data.image.min_crop_retention,
    )
    camera_policy = (
        CameraViewportPolicy.from_config(config.data.camera_viewport)
        if config.data.camera_viewport is not None
        else None
    )
    return WebDatasetPipeline(
        shard_paths=(shard_local,),
        shard_records=(ShardRecord(path=shard_rel, bytes=shard_bytes),),
        metadata_adapter=adapt_modelscope_metadata,
        metadata_fields=PRODUCTION_METADATA_FIELDS,
        buckets=buckets,
        min_crop_retention=config.data.image.min_crop_retention,
        probabilities=probabilities,
        condition_mode=config.caption.condition_mode,
        tokenizer=qwen.tokenizer,
        framing=FramingContract(
            EXPECTED_PREFIX_TOKENS, EXPECTED_SUFFIX_TOKENS, padding_token_id
        ),
        caption_fields_parser=parse_modelscope_caption_fields,
        rejection_observer=observer,
        base_seed=config.run.seed,
        stage=config.stage.name,
        cycle_index=0,
        spatial_policy=spatial_policy,
        camera_policy=camera_policy,
        transparent_policy=config.data.transparent_background,
    )


def load_validation_shard_plan() -> list[dict[str, Any]]:
    doc = json.loads(SELECTION_JSON.read_text(encoding="utf-8"))
    plan: list[dict[str, Any]] = []
    for entry in doc["shards"]:
        rel = str(entry["path"])
        local = SHARD_ROOT / rel
        if not local.is_file():
            raise FileNotFoundError(f"validation shard missing locally: {local}")
        plan.append(
            {
                "rel": rel,
                "local": local,
                "bytes": int(entry["bytes"]),
                "local_sha256": sha256_file(local),
            }
        )
    return plan
