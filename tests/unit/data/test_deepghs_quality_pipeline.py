from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
from pathlib import Path
from types import ModuleType

import pytest


def _load_pipeline() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "deepghs_quality_pipeline.py"
    spec = importlib.util.spec_from_file_location("deepghs_quality_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


pipeline = _load_pipeline()


@pytest.mark.parametrize(
    ("percentile", "expected"),
    [
        (1.0, "masterpiece"),
        (0.95, "masterpiece"),
        (0.949999, "best"),
        (0.85, "best"),
        (0.849999, "great"),
        (0.75, "great"),
        (0.749999, "good"),
        (0.50, "good"),
        (0.499999, "normal"),
        (0.25, "normal"),
        (0.249999, "low"),
        (0.10, "low"),
        (0.099999, "worst"),
        (0.0, "worst"),
    ],
)
def test_quality_threshold_boundaries(percentile: float, expected: str) -> None:
    assert pipeline.quality_from_percentile(percentile) == expected


@pytest.mark.parametrize("percentile", [-0.1, 1.1, float("nan"), float("inf")])
def test_quality_threshold_rejects_invalid_values(percentile: float) -> None:
    with pytest.raises(pipeline.QualityPipelineError):
        pipeline.quality_from_percentile(percentile)


def _add_member(tf: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    tf.addfile(info, io.BytesIO(payload))


def test_pair_iterator_requires_matching_numeric_id(tmp_path: Path) -> None:
    path = tmp_path / "input.tar"
    with tarfile.open(path, "w:") as tf:
        _add_member(tf, "data/12.webp", b"image")
        _add_member(tf, "data/12.json", json.dumps({"id": 13}).encode())

    with (
        tarfile.open(path, "r:") as tf,
        pytest.raises(pipeline.QualityPipelineError, match="JSON ID"),
    ):
        tuple(pipeline._pair_iterator(tf))


def test_tags_omit_normal_corruption() -> None:
    tags = pipeline.Tags(1, "good", "polished", None, "illustration")
    assert tags.json_fields() == {
        "quality": "good",
        "anime_completeness": "polished",
        "anime_classification": "illustration",
    }
    assert tags.result_record()["ai_image_corrupted"] is None


def _write_verified_fixture(
    path: Path, *, image: bytes, include_corrupted: bool = False
) -> None:
    document = {
        "id": 12,
        "quality": "good",
        "anime_completeness": "polished",
        "anime_classification": "illustration",
    }
    if include_corrupted:
        document["ai_image_corrupted"] = "corrupted"
    with tarfile.open(path, "w:") as tf:
        _add_member(tf, "data/12.webp", image)
        _add_member(tf, "data/12.json", json.dumps(document).encode())


def _write_result(path: Path, *, corrupted: str | None = None) -> None:
    path.write_text(
        json.dumps(
            {
                "id": 12,
                "quality": "good",
                "anime_completeness": "polished",
                "ai_image_corrupted": corrupted,
                "anime_classification": "illustration",
            }
        )
        + "\n"
    )


def test_verify_rewrite_accepts_exact_image_and_fields(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tar"
    output_path = tmp_path / "output.tar"
    result_path = tmp_path / "result.ndjson"
    with tarfile.open(input_path, "w:") as tf:
        _add_member(tf, "data/12.webp", b"original")
        _add_member(tf, "data/12.json", json.dumps({"id": 12}).encode())
    _write_verified_fixture(output_path, image=b"original")
    _write_result(result_path)

    assert pipeline.verify_rewrite(input_path, output_path, result_path) == 1


def test_verify_rewrite_rejects_changed_image(tmp_path: Path) -> None:
    input_path = tmp_path / "input.tar"
    output_path = tmp_path / "output.tar"
    result_path = tmp_path / "result.ndjson"
    with tarfile.open(input_path, "w:") as tf:
        _add_member(tf, "data/12.webp", b"original")
        _add_member(tf, "data/12.json", json.dumps({"id": 12}).encode())
    _write_verified_fixture(output_path, image=b"changed")
    _write_result(result_path)

    with pytest.raises(pipeline.QualityPipelineError, match="tar image changed"):
        pipeline.verify_rewrite(input_path, output_path, result_path)

def test_classify_worker_rejects_invalid_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pipeline, "TorchModelSuite", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(sys, "stdin", io.StringIO("{}\n"))

    with pytest.raises(pipeline.QualityPipelineError, match="request contract"):
        pipeline.run_classify_worker(Path("models"), batch_size=16, device="cuda:0")


def test_classify_worker_emits_matching_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    input_path = tmp_path / "input.tar"
    output_path = tmp_path / "output.tar"
    result_path = tmp_path / "result.ndjson"
    request = {
        "request_id": 7,
        "input": str(input_path),
        "output": str(output_path),
        "results": str(result_path),
    }
    calls: list[tuple[Path, Path, Path, int]] = []

    monkeypatch.setattr(pipeline, "TorchModelSuite", lambda *_args, **_kwargs: object())

    def fake_rewrite(
        current_input: Path,
        current_output: Path,
        current_result: Path,
        _suite: object,
        *,
        batch_size: int,
    ) -> int:
        calls.append((current_input, current_output, current_result, batch_size))
        return 2

    monkeypatch.setattr(pipeline, "rewrite_tar", fake_rewrite)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(request) + "\n"))

    assert (
        pipeline.run_classify_worker(
            Path("models"), batch_size=16, device="cuda:0"
        )
        == 0
    )
    assert calls == [(input_path, output_path, result_path, 16)]
    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == "WORKER_READY"
    assert json.loads(lines[1].removeprefix("WORKER_DONE ")) == {
        "elapsed_seconds": pytest.approx(float(lines[1].split('"elapsed_seconds":')[1].split(",")[0])),
        "request_id": 7,
        "samples": 2,
    }
    assert lines[2] == "WORKER_EXIT_OK"
