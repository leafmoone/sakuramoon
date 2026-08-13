from __future__ import annotations

import importlib.util
import json
import sys
import threading
from argparse import Namespace
from pathlib import Path
from types import ModuleType

import pytest


def _load_orchestrator() -> ModuleType:
    path = Path(__file__).parents[3] / "scripts" / "run_deepghs_quality_pipeline.py"
    spec = importlib.util.spec_from_file_location(
        "run_deepghs_quality_pipeline", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


orchestrator = _load_orchestrator()


def _record(sample_id: int) -> dict[str, object]:
    return {
        "id": sample_id,
        "quality": "good",
        "anime_completeness": "polished",
        "ai_image_corrupted": None,
        "anime_classification": "illustration",
    }


def test_count_result_records_accepts_strict_ndjson(tmp_path: Path) -> None:
    path = tmp_path / "results.ndjson"
    path.write_text(
        "".join(json.dumps(_record(sample_id)) + "\n" for sample_id in (12, 13)),
        encoding="utf-8",
    )

    assert orchestrator._count_result_records(path) == 2


def test_count_result_records_rejects_duplicate_id(tmp_path: Path) -> None:
    path = tmp_path / "results.ndjson"
    path.write_text(
        json.dumps(_record(12)) + "\n" + json.dumps(_record(12)) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(orchestrator.OrchestrationError, match="invalid result record"):
        orchestrator._count_result_records(path)


def test_count_result_records_rejects_invalid_label(tmp_path: Path) -> None:
    path = tmp_path / "results.ndjson"
    record = _record(12)
    record["quality"] = "unknown"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(orchestrator.OrchestrationError, match="invalid result record"):
        orchestrator._count_result_records(path)


def test_count_result_records_requires_newline(tmp_path: Path) -> None:
    path = tmp_path / "results.ndjson"
    path.write_text(json.dumps(_record(12)), encoding="utf-8")

    with pytest.raises(orchestrator.OrchestrationError, match="NDJSON framing"):
        orchestrator._count_result_records(path)


def test_watchdog_terminates_silent_child() -> None:
    with pytest.raises(orchestrator.OrchestrationError, match="made no progress"):
        orchestrator._run_with_watchdog(
            (sys.executable, "-c", "import time; time.sleep(60)"),
            heartbeat=orchestrator.Heartbeat("test"),
            timeout_seconds=0.05,
        )


def _pipeline_args(tmp_path: Path) -> Namespace:
    return Namespace(
        batch_size=64,
        cache_root=tmp_path / "cache",
        database=tmp_path / "metadata.db",
        devices=("cuda:0", "cuda:1"),
        download_concurrency=8,
        limit=2,
        manifest=tmp_path / "manifest.json",
        model_root=tmp_path / "models",
        no_progress_timeout=300.0,
        start_index=0,
        verify_only=False,
        work_root=tmp_path / "work",
    )


def test_start_workers_binds_one_persistent_worker_per_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    devices: list[str] = []

    class FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            devices.append(str(kwargs["device"]))

        def abort(self) -> None:
            pass

    pipeline = object.__new__(orchestrator.Pipeline)
    pipeline.args = _pipeline_args(tmp_path)
    pipeline.classify_workers = []
    pipeline.classify_executors = []
    pipeline.stop_event = threading.Event()
    monkeypatch.setattr(orchestrator, "ClassifyWorker", FakeWorker)

    pipeline._start_workers()
    try:
        assert devices == ["cuda:0", "cuda:1"]
        assert len(pipeline.classify_workers) == 2
        assert len(pipeline.classify_executors) == 2
    finally:
        pipeline.abort()


def test_start_workers_aborts_first_worker_when_second_gpu_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    aborted: list[str] = []

    class FakeWorker:
        def __init__(self, **kwargs: object) -> None:
            self.device = str(kwargs["device"])
            if self.device == "cuda:1":
                raise orchestrator.OrchestrationError("gpu startup failed")

        def abort(self) -> None:
            aborted.append(self.device)

    pipeline = object.__new__(orchestrator.Pipeline)
    pipeline.args = _pipeline_args(tmp_path)
    pipeline.classify_workers = []
    pipeline.classify_executors = []
    pipeline.stop_event = threading.Event()
    monkeypatch.setattr(orchestrator, "ClassifyWorker", FakeWorker)

    with pytest.raises(orchestrator.OrchestrationError, match="gpu startup"):
        pipeline._start_workers()

    assert aborted == ["cuda:0"]
    assert pipeline.classify_workers == []
    assert pipeline.classify_executors == []


def test_classify_uses_worker_selected_by_scheduler(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    class FakeWorker:
        def __init__(self, device: str) -> None:
            self.device = device

        def classify(
            self,
            input_path: Path,
            output_path: Path,
            result_path: Path,
            *,
            heartbeat: object,
        ) -> int:
            del heartbeat
            calls.append((self.device, input_path))
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"tar")
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(json.dumps(_record(12)) + "\n")
            return 1

    shard = type("Shard", (), {"path": "data/shard-000005.tar", "bytes": 3})()
    input_path = tmp_path / "input.tar"
    input_path.write_bytes(b"tar")
    pipeline = object.__new__(orchestrator.Pipeline)
    pipeline.args = _pipeline_args(tmp_path)
    pipeline.classify_workers = [FakeWorker("cuda:0"), FakeWorker("cuda:1")]
    monkeypatch.setattr(orchestrator, "_count_result_records", lambda _path: 1)

    pipeline._classify(1, shard, input_path)

    assert calls == [("cuda:1", input_path)]
    state = orchestrator._read_state(
        orchestrator._paths(pipeline.args.work_root, shard.path).state,
        shard.path,
    )
    assert state["stage"] == "classified"
    assert state["samples"] == 1
