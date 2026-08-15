"""Batch deepghs classification and strict WebDataset metadata rewriting."""

from __future__ import annotations

import argparse
import copy
import io
import json
import math
import os
import sys
import tarfile
import time
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import onnxruntime as ort
from PIL import Image


class QualityPipelineError(RuntimeError):
    """The pipeline cannot safely continue."""


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    model: str
    labels: tuple[str, ...]
    size: int

    def directory(self, model_root: Path) -> Path:
        return model_root / self.repo.replace("/", "--") / self.model


AESTHETIC: Final = ModelSpec(
    "deepghs/anime_aesthetic",
    "swinv2pv3_v0_448_ls0.2_x",
    ("masterpiece", "best", "great", "good", "normal", "low", "worst"),
    448,
)
COMPLETENESS: Final = ModelSpec(
    "deepghs/anime_completeness",
    "caformer_s36_v2.2",
    ("polished", "rough", "monochrome"),
    384,
)
CORRUPTED: Final = ModelSpec(
    "deepghs/ai_image_corrupted",
    "caformer_s36_v0_sce",
    ("corrupted", "normal"),
    384,
)
CLASSIFICATION: Final = ModelSpec(
    "deepghs/anime_classification",
    "caformer_s36_v1.2_focal",
    ("3d", "bangumi", "comic", "illustration", "not_painting"),
    384,
)
MODEL_SPECS: Final = (AESTHETIC, COMPLETENESS, CORRUPTED, CLASSIFICATION)
TORCH_MODEL_ARGUMENTS: Final = {
    AESTHETIC.repo: (
        "swinv2_base_window8_256",
        {
            "img_size": 448,
            "window_size": 14,
            "act_layer": "gelu_tanh",
            "drop_path_rate": 0.4,
        },
    ),
    COMPLETENESS.repo: (
        "caformer_s36.sail_in22k_ft_in1k_384",
        {"drop_path_rate": 0.4},
    ),
    CORRUPTED.repo: ("caformer_s36.sail_in22k_ft_in1k_384", {}),
    CLASSIFICATION.repo: (
        "caformer_s36.sail_in22k_ft_in1k_384",
        {"drop_path_rate": 0.4},
    ),
}
TORCH_CHECKPOINT_NAMES: Final = {
    AESTHETIC.repo: "hf-hub:SmilingWolf/wd-swinv2-tagger-v3",
    COMPLETENESS.repo: "caformer_s36.sail_in22k_ft_in1k_384",
    CORRUPTED.repo: "caformer_s36.sail_in22k_ft_in1k_384",
    CLASSIFICATION.repo: "caformer_s36.sail_in22k_ft_in1k_384",
}
IMAGE_SUFFIXES: Final = frozenset({".jpg", ".jpeg", ".png", ".webp"})
DB_COLUMNS: Final = (
    "quality",
    "anime_completeness",
    "ai_image_corrupted",
    "anime_classification",
)


@dataclass(frozen=True)
class Tags:
    sample_id: int
    quality: str
    anime_completeness: str
    ai_image_corrupted: str | None
    anime_classification: str

    def json_fields(self) -> dict[str, str]:
        fields = {
            "quality": self.quality,
            "anime_completeness": self.anime_completeness,
            "anime_classification": self.anime_classification,
        }
        if self.ai_image_corrupted is not None:
            fields["ai_image_corrupted"] = self.ai_image_corrupted
        return fields

    def result_record(self) -> dict[str, object]:
        return {
            "id": self.sample_id,
            "quality": self.quality,
            "anime_completeness": self.anime_completeness,
            "ai_image_corrupted": self.ai_image_corrupted,
            "anime_classification": self.anime_classification,
        }


@dataclass
class _TarPair:
    image_info: tarfile.TarInfo
    image_bytes: bytes
    json_info: tarfile.TarInfo
    document: dict[str, object]
    sample_id: int


def quality_from_percentile(percentile: float) -> str:
    if not math.isfinite(percentile) or not 0.0 <= percentile <= 1.0:
        raise QualityPipelineError(f"invalid aesthetic percentile: {percentile!r}")
    if percentile >= 0.95:
        return "masterpiece"
    if percentile >= 0.85:
        return "best"
    if percentile >= 0.75:
        return "great"
    if percentile >= 0.50:
        return "good"
    if percentile >= 0.25:
        return "normal"
    if percentile >= 0.10:
        return "low"
    return "worst"


def _load_labels(spec: ModelSpec, directory: Path) -> tuple[str, ...]:
    meta_path = directory / "meta.json"
    try:
        document = json.loads(meta_path.read_text(encoding="utf-8"))
        labels = tuple(document["labels"])
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise QualityPipelineError(f"invalid model metadata: {meta_path}") from error
    if labels != spec.labels:
        raise QualityPipelineError(
            f"model labels changed for {spec.repo}/{spec.model}: {labels!r}"
        )
    return labels


def _new_session(model_path: Path, threads: int) -> ort.InferenceSession:
    if threads <= 0:
        raise QualityPipelineError("ONNX thread count must be positive")
    if not model_path.is_file():
        raise QualityPipelineError(f"model is missing: {model_path}")
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    return ort.InferenceSession(
        str(model_path), options, providers=["CPUExecutionProvider"]
    )


def _validate_probability_output(
    output: np.ndarray, *, rows: int, labels: Sequence[str], model_name: str
) -> np.ndarray:
    expected = (rows, len(labels))
    if output.shape != expected:
        raise QualityPipelineError(
            f"{model_name} output shape {output.shape!r} differs from {expected!r}"
        )
    if output.dtype not in (np.float32, np.float64):
        raise QualityPipelineError(f"{model_name} output dtype is {output.dtype!r}")
    if not np.isfinite(output).all():
        raise QualityPipelineError(f"{model_name} emitted a non-finite score")
    if np.any(output < -1e-6) or np.any(output > 1.0 + 1e-6):
        raise QualityPipelineError(f"{model_name} scores are not probabilities")
    totals = output.sum(axis=1)
    if not np.allclose(totals, 1.0, rtol=1e-4, atol=1e-4):
        raise QualityPipelineError(f"{model_name} probability rows do not sum to one")
    return output


def _has_alpha(image: Image.Image) -> bool:
    return image.mode in {"RGBA", "LA", "PA"} or "transparency" in image.info


def _decode_rgb(payload: bytes, *, sample_id: int) -> Image.Image:
    try:
        with Image.open(io.BytesIO(payload)) as source:
            source.load()
            if _has_alpha(source):
                rgba = source.convert("RGBA")
                background = Image.new("RGBA", rgba.size, "white")
                background.alpha_composite(rgba)
                return background.convert("RGB")
            return source.convert("RGB")
    except Exception as error:
        raise QualityPipelineError(f"image {sample_id} cannot be decoded") from error


def _encode_batch(images: Sequence[Image.Image], size: int) -> np.ndarray:
    encoded: list[np.ndarray] = []
    for image in images:
        resized = image.resize((size, size), Image.Resampling.BILINEAR)
        data = np.asarray(resized, dtype=np.float32)
        if data.shape != (size, size, 3):
            raise QualityPipelineError(f"preprocessed image has shape {data.shape!r}")
        data = np.transpose(data / 255.0, (2, 0, 1))
        encoded.append((data - 0.5) / 0.5)
    return np.stack(encoded).astype(np.float32, copy=False)


class ModelSuite:
    def __init__(self, model_root: Path, *, threads: int) -> None:
        self.specs = MODEL_SPECS
        self.sessions: dict[str, ort.InferenceSession] = {}
        for spec in self.specs:
            directory = spec.directory(model_root)
            _load_labels(spec, directory)
            session = _new_session(directory / "model.onnx", threads)
            model_input = session.get_inputs()
            model_output = session.get_outputs()
            if len(model_input) != 1 or model_input[0].name != "input":
                raise QualityPipelineError(
                    f"unexpected input contract for {spec.model}"
                )
            if len(model_output) != 1 or model_output[0].name != "output":
                raise QualityPipelineError(
                    f"unexpected output contract for {spec.model}"
                )
            self.sessions[spec.repo] = session

        sample_path = AESTHETIC.directory(model_root) / "samples.npz"
        try:
            stacked = np.load(sample_path)["arr_0"]
        except (OSError, KeyError, ValueError) as error:
            raise QualityPipelineError(
                f"invalid aesthetic calibration file: {sample_path}"
            ) from error
        if (
            stacked.ndim != 2
            or stacked.shape[0] != 2
            or stacked.shape[1] < 2
            or not np.isfinite(stacked).all()
            or np.any(np.diff(stacked[0]) < 0)
            or np.any(np.diff(stacked[1]) < 0)
        ):
            raise QualityPipelineError("aesthetic calibration table is invalid")
        self.calibration_scores = stacked[0]
        self.calibration_percentiles = stacked[1]

    def _run(self, spec: ModelSpec, batch: np.ndarray) -> np.ndarray:
        session = self.sessions[spec.repo]
        output = session.run(["output"], {"input": batch})[0]
        return _validate_probability_output(
            output, rows=batch.shape[0], labels=spec.labels, model_name=spec.model
        )

    def _run_all(
        self, inputs: dict[ModelSpec, np.ndarray]
    ) -> dict[ModelSpec, np.ndarray]:
        with ThreadPoolExecutor(
            max_workers=len(inputs), thread_name_prefix="deepghs-model"
        ) as executor:
            futures = {
                spec: executor.submit(self._run, spec, batch)
                for spec, batch in inputs.items()
            }
            return {spec: future.result() for spec, future in futures.items()}

    def _aesthetic_percentiles(self, probabilities: np.ndarray) -> np.ndarray:
        ordinal = np.asarray([6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0], dtype=np.float64)
        scores = probabilities.astype(np.float64) @ ordinal
        clipped = np.clip(
            scores, self.calibration_scores[0], self.calibration_scores[-1]
        )
        indices = np.searchsorted(self.calibration_scores, clipped)
        result = np.empty_like(clipped)
        for row, (score, index) in enumerate(zip(clipped, indices, strict=True)):
            if index >= self.calibration_scores.shape[0] - 1:
                result[row] = self.calibration_percentiles[index]
                continue
            x0 = self.calibration_scores[index]
            y0 = self.calibration_percentiles[index]
            x1 = self.calibration_scores[index + 1]
            y1 = self.calibration_percentiles[index + 1]
            if np.isclose(x1, x0):
                result[row] = y0
            else:
                result[row] = np.clip(
                    (score - x0) / (x1 - x0) * (y1 - y0) + y0,
                    self.calibration_percentiles[0],
                    self.calibration_percentiles[-1],
                )
        return result

    def classify(self, pairs: Sequence[_TarPair]) -> tuple[Tags, ...]:
        if not pairs:
            return ()
        images = [
            _decode_rgb(pair.image_bytes, sample_id=pair.sample_id) for pair in pairs
        ]
        try:
            batch_384 = _encode_batch(images, 384)
            batch_448 = _encode_batch(images, 448)
        finally:
            for image in images:
                image.close()

        inputs = {
            AESTHETIC: batch_448,
            COMPLETENESS: batch_384,
            CORRUPTED: batch_384,
            CLASSIFICATION: batch_384,
        }
        outputs = self._run_all(inputs)
        aesthetic = outputs[AESTHETIC]
        completeness = outputs[COMPLETENESS]
        corrupted = outputs[CORRUPTED]
        classification = outputs[CLASSIFICATION]
        percentiles = self._aesthetic_percentiles(aesthetic)

        results: list[Tags] = []
        for index, pair in enumerate(pairs):
            completeness_label = COMPLETENESS.labels[
                int(np.argmax(completeness[index]))
            ]
            corrupted_label = CORRUPTED.labels[int(np.argmax(corrupted[index]))]
            classification_label = CLASSIFICATION.labels[
                int(np.argmax(classification[index]))
            ]
            results.append(
                Tags(
                    sample_id=pair.sample_id,
                    quality=quality_from_percentile(float(percentiles[index])),
                    anime_completeness=completeness_label,
                    ai_image_corrupted=(
                        "corrupted" if corrupted_label == "corrupted" else None
                    ),
                    anime_classification=classification_label,
                )
            )
        return tuple(results)


class TorchModelSuite(ModelSuite):
    """Strict FP32 GPU implementation of the published deepghs checkpoints."""

    def __init__(self, model_root: Path, *, device: str) -> None:
        try:
            import timm
            import torch
        except ImportError as error:
            raise QualityPipelineError(
                "the torch backend dependencies are missing"
            ) from error
        if not device.startswith("cuda:") or not torch.cuda.is_available():
            raise QualityPipelineError(f"the torch device is unavailable: {device}")
        self.specs = MODEL_SPECS
        self._torch = torch
        self.device = torch.device(device)
        torch.cuda.set_device(self.device)
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        if free_bytes < 8 * 1024**3:
            raise QualityPipelineError(
                f"the torch device has only {free_bytes / 1024**3:.2f} GiB free"
            )

        self.models: dict[str, Any] = {}
        for spec in self.specs:
            directory = spec.directory(model_root)
            _load_labels(spec, directory)
            checkpoint_path = directory / "model.ckpt"
            if not checkpoint_path.is_file():
                raise QualityPipelineError(f"model is missing: {checkpoint_path}")
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False
            )
            if not isinstance(checkpoint, dict) or not isinstance(
                checkpoint.get("state_dict"), dict
            ):
                raise QualityPipelineError(f"invalid checkpoint: {checkpoint_path}")
            arguments = checkpoint.get("arguments")
            if (
                not isinstance(arguments, dict)
                or arguments.get("name") != TORCH_CHECKPOINT_NAMES[spec.repo]
                or tuple(arguments.get("labels", ())) != spec.labels
            ):
                raise QualityPipelineError(
                    f"checkpoint contract changed: {checkpoint_path}"
                )

            model_name, model_arguments = TORCH_MODEL_ARGUMENTS[spec.repo]
            model = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=len(spec.labels),
                **model_arguments,
            )
            state = {
                key: value
                for key, value in checkpoint["state_dict"].items()
                if not key.endswith(("total_ops", "total_params"))
            }
            try:
                model.load_state_dict(state, strict=True)
            except RuntimeError as error:
                raise QualityPipelineError(
                    f"checkpoint does not strictly match {model_name}"
                ) from error
            self.models[spec.repo] = model.eval().to(self.device)
            print(f"MODEL_READY backend=torch model={spec.model}", flush=True)

        sample_path = AESTHETIC.directory(model_root) / "samples.npz"
        try:
            stacked = np.load(sample_path)["arr_0"]
        except (OSError, KeyError, ValueError) as error:
            raise QualityPipelineError(
                f"invalid aesthetic calibration file: {sample_path}"
            ) from error
        if (
            stacked.ndim != 2
            or stacked.shape[0] != 2
            or stacked.shape[1] < 2
            or not np.isfinite(stacked).all()
            or np.any(np.diff(stacked[0]) < 0)
            or np.any(np.diff(stacked[1]) < 0)
        ):
            raise QualityPipelineError("aesthetic calibration table is invalid")
        self.calibration_scores = stacked[0]
        self.calibration_percentiles = stacked[1]

    def _run(self, spec: ModelSpec, batch: np.ndarray) -> np.ndarray:
        torch = self._torch
        with torch.inference_mode():
            tensor = torch.from_numpy(batch).to(self.device)
            output = (
                torch.softmax(self.models[spec.repo](tensor).float(), dim=1)
                .cpu()
                .numpy()
            )
        return _validate_probability_output(
            output,
            rows=batch.shape[0],
            labels=spec.labels,
            model_name=spec.model,
        )

    def _run_all(
        self, inputs: dict[ModelSpec, np.ndarray]
    ) -> dict[ModelSpec, np.ndarray]:
        return {spec: self._run(spec, batch) for spec, batch in inputs.items()}


def _read_member(tf: tarfile.TarFile, member: tarfile.TarInfo) -> bytes:
    if not member.isfile():
        raise QualityPipelineError(f"tar member is not a regular file: {member.name}")
    handle = tf.extractfile(member)
    if handle is None:
        raise QualityPipelineError(f"tar member cannot be read: {member.name}")
    payload = handle.read()
    if len(payload) != member.size:
        raise QualityPipelineError(f"tar member ended early: {member.name}")
    return payload


def _pair_iterator(tf: tarfile.TarFile) -> Iterator[_TarPair]:
    members = iter(tf)
    seen_ids: set[int] = set()
    while True:
        try:
            image_info = next(members)
        except StopIteration:
            return
        image_suffix = Path(image_info.name).suffix.casefold()
        if image_suffix not in IMAGE_SUFFIXES:
            raise QualityPipelineError(
                f"expected an image member, found {image_info.name!r}"
            )
        try:
            json_info = next(members)
        except StopIteration as error:
            raise QualityPipelineError(
                f"image has no JSON partner: {image_info.name}"
            ) from error
        image_stem = str(Path(image_info.name).with_suffix(""))
        json_path = Path(json_info.name)
        if (
            json_path.suffix.casefold() != ".json"
            or str(json_path.with_suffix("")) != image_stem
        ):
            raise QualityPipelineError(
                f"member pair mismatch: {image_info.name!r}, {json_info.name!r}"
            )
        try:
            sample_id = int(Path(image_stem).name)
        except ValueError as error:
            raise QualityPipelineError(
                f"sample filename is not a numeric ID: {image_info.name}"
            ) from error
        if sample_id in seen_ids:
            raise QualityPipelineError(f"duplicate sample ID in tar: {sample_id}")
        seen_ids.add(sample_id)
        image_bytes = _read_member(tf, image_info)
        json_bytes = _read_member(tf, json_info)
        try:
            document = json.loads(json_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise QualityPipelineError(
                f"invalid JSON for sample {sample_id}"
            ) from error
        if not isinstance(document, dict) or document.get("id") != sample_id:
            raise QualityPipelineError(
                f"JSON ID differs from member filename for sample {sample_id}"
            )
        yield _TarPair(
            image_info=image_info,
            image_bytes=image_bytes,
            json_info=json_info,
            document=document,
            sample_id=sample_id,
        )


def _write_pair(
    output: tarfile.TarFile,
    result_handle: Any,
    pair: _TarPair,
    tags: Tags,
) -> None:
    if pair.sample_id != tags.sample_id:
        raise QualityPipelineError("model result order differs from tar order")
    document = dict(pair.document)
    for key in DB_COLUMNS:
        document.pop(key, None)
    document.update(tags.json_fields())
    json_bytes = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )

    image_info = copy.copy(pair.image_info)
    output.addfile(image_info, io.BytesIO(pair.image_bytes))
    json_info = copy.copy(pair.json_info)
    json_info.size = len(json_bytes)
    output.addfile(json_info, io.BytesIO(json_bytes))
    result_handle.write(json.dumps(tags.result_record(), separators=(",", ":")))
    result_handle.write("\n")


def rewrite_tar(
    input_path: Path,
    output_path: Path,
    result_path: Path,
    suite: ModelSuite,
    *,
    batch_size: int,
    progress: Callable[[int], None] | None = None,
) -> int:
    if batch_size <= 0:
        raise QualityPipelineError("batch size must be positive")
    if not input_path.is_file():
        raise QualityPipelineError(f"input tar is missing: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise QualityPipelineError("input and output tar paths must differ")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    output_part = output_path.with_name(f"{output_path.name}.part")
    result_part = result_path.with_name(f"{result_path.name}.part")
    output_part.unlink(missing_ok=True)
    result_part.unlink(missing_ok=True)
    count = 0
    try:
        with (
            tarfile.open(input_path, "r:") as source,
            tarfile.open(output_part, "w:", format=tarfile.PAX_FORMAT) as output,
            result_part.open("w", encoding="utf-8", newline="\n") as result_handle,
        ):
            batch: list[_TarPair] = []
            for pair in _pair_iterator(source):
                batch.append(pair)
                if len(batch) == batch_size:
                    tags_batch = suite.classify(batch)
                    for current_pair, tags in zip(batch, tags_batch, strict=True):
                        _write_pair(output, result_handle, current_pair, tags)
                    count += len(batch)
                    print(f"HEARTBEAT classified_samples={count}", flush=True)
                    if progress is not None:
                        progress(count)
                    batch.clear()
            if batch:
                tags_batch = suite.classify(batch)
                for current_pair, tags in zip(batch, tags_batch, strict=True):
                    _write_pair(output, result_handle, current_pair, tags)
                count += len(batch)
                print(f"HEARTBEAT classified_samples={count}", flush=True)
                if progress is not None:
                    progress(count)
        if count <= 0:
            raise QualityPipelineError(f"tar contains no samples: {input_path}")
        os.replace(output_part, output_path)
        os.replace(result_part, result_path)
        return count
    except BaseException:
        output_part.unlink(missing_ok=True)
        result_part.unlink(missing_ok=True)
        raise


def verify_rewrite(input_path: Path, output_path: Path, result_path: Path) -> int:
    if (
        not input_path.is_file()
        or not output_path.is_file()
        or not result_path.is_file()
    ):
        raise QualityPipelineError("tar verification input is missing")
    try:
        records = [json.loads(line) for line in result_path.read_text().splitlines()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise QualityPipelineError("result file is invalid") from error
    expected_keys = {"id", *DB_COLUMNS}
    count = 0
    with (
        tarfile.open(input_path, "r:") as source,
        tarfile.open(output_path, "r:") as output,
    ):
        for count, (before, after, record) in enumerate(
            zip(_pair_iterator(source), _pair_iterator(output), records, strict=True),
            start=1,
        ):
            if set(record) != expected_keys or record.get("id") != before.sample_id:
                raise QualityPipelineError(
                    f"invalid result record for {before.sample_id}"
                )
            if (
                before.image_info.name != after.image_info.name
                or before.json_info.name != after.json_info.name
                or before.image_bytes != after.image_bytes
            ):
                raise QualityPipelineError(f"tar image changed for {before.sample_id}")
            tags = Tags(
                sample_id=before.sample_id,
                quality=record["quality"],
                anime_completeness=record["anime_completeness"],
                ai_image_corrupted=record["ai_image_corrupted"],
                anime_classification=record["anime_classification"],
            )
            expected_document = dict(before.document)
            for key in DB_COLUMNS:
                expected_document.pop(key, None)
            expected_document.update(tags.json_fields())
            if after.document != expected_document:
                raise QualityPipelineError(
                    f"rewritten JSON differs for {before.sample_id}"
                )
            if count % 1000 == 0:
                print(f"HEARTBEAT verified_samples={count}", flush=True)
    if count <= 0:
        raise QualityPipelineError("verified tar contains no samples")
    return count


def initialize_database(database: Path) -> None:
    import duckdb

    if not database.is_file():
        raise QualityPipelineError(f"database is missing: {database}")
    connection = duckdb.connect(str(database))
    try:
        connection.execute("BEGIN TRANSACTION")
        existing = {
            row[0] for row in connection.execute("DESCRIBE metadata").fetchall()
        }
        for column in DB_COLUMNS:
            if column not in existing:
                connection.execute(
                    f'ALTER TABLE metadata ADD COLUMN "{column}" VARCHAR'
                )
        connection.execute("COMMIT")
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def update_database(database: Path, result_path: Path) -> int:
    import duckdb

    if not database.is_file() or not result_path.is_file():
        raise QualityPipelineError("database update input is missing")
    connection = duckdb.connect(str(database))
    try:
        connection.execute("BEGIN TRANSACTION")
        existing = {
            row[0] for row in connection.execute("DESCRIBE metadata").fetchall()
        }
        missing_columns = set(DB_COLUMNS) - existing
        if missing_columns:
            raise QualityPipelineError(
                f"database columns were not initialized: {sorted(missing_columns)!r}"
            )
        escaped = str(result_path).replace("'", "''")
        connection.execute(
            f"CREATE TEMP TABLE quality_updates AS "
            f"SELECT * FROM read_ndjson_auto('{escaped}')"
        )
        rows, unique_rows = connection.execute(
            "SELECT count(*), count(DISTINCT id) FROM quality_updates"
        ).fetchone()
        if rows <= 0 or rows != unique_rows:
            raise QualityPipelineError("result file is empty or contains duplicate IDs")
        matched = connection.execute(
            "SELECT count(*) FROM metadata m INNER JOIN quality_updates u USING (id)"
        ).fetchone()[0]
        if matched <= 0:
            raise QualityPipelineError(
                f"database matched none of {rows} result rows"
            )
        skipped = int(rows - matched)
        if skipped:
            missing_ids = [
                int(row[0])
                for row in connection.execute(
                    """
                    SELECT u.id FROM quality_updates u
                    LEFT JOIN metadata m USING (id)
                    WHERE m.id IS NULL
                    ORDER BY u.id
                    LIMIT 20
                    """
                ).fetchall()
            ]
            print(
                f"DB_SKIPPED_MISSING count={skipped} ids={missing_ids!r}",
                flush=True,
            )
        connection.execute(
            """
            UPDATE metadata AS m SET
                quality = u.quality,
                anime_completeness = u.anime_completeness,
                ai_image_corrupted = u.ai_image_corrupted,
                anime_classification = u.anime_classification
            FROM quality_updates AS u
            WHERE m.id = u.id
            """
        )
        connection.execute("COMMIT")
        return int(matched)
    except BaseException:
        connection.execute("ROLLBACK")
        raise
    finally:
        connection.close()


def run_classify_worker(
    model_root: Path,
    *,
    batch_size: int,
    device: str,
) -> int:
    if batch_size <= 0:
        raise QualityPipelineError("batch size must be positive")
    suite = TorchModelSuite(model_root, device=device)
    print("WORKER_READY", flush=True)
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.endswith("\n") or not line.strip():
            raise QualityPipelineError(
                f"invalid worker request framing at line {line_number}"
            )
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            raise QualityPipelineError(
                f"invalid worker request JSON at line {line_number}"
            ) from error
        if (
            not isinstance(request, dict)
            or set(request) != {"request_id", "input", "output", "results"}
            or type(request["request_id"]) is not int
            or request["request_id"] < 0
            or not all(
                isinstance(request[key], str) and request[key]
                for key in ("input", "output", "results")
            )
        ):
            raise QualityPipelineError(
                f"invalid worker request contract at line {line_number}"
            )
        started = time.monotonic()
        count = rewrite_tar(
            Path(request["input"]),
            Path(request["output"]),
            Path(request["results"]),
            suite,
            batch_size=batch_size,
        )
        elapsed = time.monotonic() - started
        response = {
            "request_id": request["request_id"],
            "samples": count,
            "elapsed_seconds": elapsed,
        }
        print(
            "WORKER_DONE "
            + json.dumps(response, sort_keys=True, separators=(",", ":")),
            flush=True,
        )
    print("WORKER_EXIT_OK", flush=True)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    classify = subparsers.add_parser("classify-tar")
    classify.add_argument("--input", type=Path, required=True)
    classify.add_argument("--output", type=Path, required=True)
    classify.add_argument("--results", type=Path, required=True)
    classify.add_argument("--model-root", type=Path, required=True)
    classify.add_argument("--batch-size", type=int, default=16)
    classify.add_argument("--threads", type=int, default=32)
    classify.add_argument("--backend", choices=("onnx", "torch"), default="onnx")
    classify.add_argument("--device", default="cuda:0")

    worker = subparsers.add_parser("classify-worker")
    worker.add_argument("--model-root", type=Path, required=True)
    worker.add_argument("--batch-size", type=int, default=16)
    worker.add_argument("--device", default="cuda:0")

    verify = subparsers.add_parser("verify-tar")
    verify.add_argument("--input", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--results", type=Path, required=True)

    init_db = subparsers.add_parser("init-db")
    init_db.add_argument("--database", type=Path, required=True)

    update_db = subparsers.add_parser("update-db")
    update_db.add_argument("--database", type=Path, required=True)
    update_db.add_argument("--results", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    started = time.monotonic()
    if args.command == "classify-tar":
        suite = (
            TorchModelSuite(args.model_root, device=args.device)
            if args.backend == "torch"
            else ModelSuite(args.model_root, threads=args.threads)
        )
        count = rewrite_tar(
            args.input,
            args.output,
            args.results,
            suite,
            batch_size=args.batch_size,
        )
        elapsed = time.monotonic() - started
        print(
            f"CLASSIFY_OK samples={count} elapsed={elapsed:.3f}s "
            f"images_per_second={count / elapsed:.3f}",
            flush=True,
        )
    elif args.command == "classify-worker":
        return run_classify_worker(
            args.model_root,
            batch_size=args.batch_size,
            device=args.device,
        )
    elif args.command == "verify-tar":
        count = verify_rewrite(args.input, args.output, args.results)
        print(f"VERIFY_OK samples={count}", flush=True)
    elif args.command == "init-db":
        initialize_database(args.database)
        print("DB_INIT_OK", flush=True)
    elif args.command == "update-db":
        count = update_database(args.database, args.results)
        print(f"DB_UPDATE_OK updated={count}", flush=True)
    else:
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("FAST_FAIL: interrupted", file=sys.stderr, flush=True)
        raise SystemExit(130) from None
    except Exception as error:
        print(
            f"FAST_FAIL: {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
