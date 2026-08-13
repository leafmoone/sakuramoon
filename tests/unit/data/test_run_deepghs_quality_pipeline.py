from __future__ import annotations

import importlib.util
import json
import sys
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
