"""Recoverable master-to-danbooru deepghs WebDataset enrichment."""

from __future__ import annotations

import argparse
import json
import os
import selectors
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import deepghs_quality_pipeline as quality

REPO_ID: Final = "leafmoone/webdataset_danbooru"
SOURCE_REVISION: Final = "master"
TARGET_REVISION: Final = "danbooru"
STAGES: Final = (
    "pending",
    "downloaded",
    "classified",
    "verified",
    "db_updated",
    "uploaded",
    "complete",
)
SECRET_KEYS: Final = (
    "MODELSCOPE_API_TOKEN",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "no_proxy",
)


class OrchestrationError(RuntimeError):
    """The enrichment pipeline cannot safely continue."""


@dataclass(frozen=True)
class Paths:
    input: Path
    output: Path
    results: Path
    state: Path


class Heartbeat:
    def __init__(self, shard: str) -> None:
        self.shard = shard
        self._last = time.monotonic()
        self._lock = threading.Lock()

    @property
    def age(self) -> float:
        with self._lock:
            return time.monotonic() - self._last

    def ping(self, event: str) -> None:
        with self._lock:
            self._last = time.monotonic()
        print(f"HEARTBEAT shard={self.shard} event={event}", flush=True)


def _atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(document, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _read_state(path: Path, shard: str) -> dict[str, object]:
    if not path.exists():
        return {"shard": shard, "stage": "pending"}
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError(f"state path is not a regular file: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrchestrationError(f"state file is invalid: {path}") from error
    if (
        not isinstance(document, dict)
        or document.get("shard") != shard
        or document.get("stage") not in STAGES
    ):
        raise OrchestrationError(f"state contract is invalid: {path}")
    return document


def _transition(
    path: Path,
    document: dict[str, object],
    stage: str,
    **facts: object,
) -> dict[str, object]:
    if stage not in STAGES:
        raise OrchestrationError(f"unknown stage: {stage}")
    old_stage = document.get("stage")
    if not isinstance(old_stage, str):
        raise OrchestrationError("state stage is not a string")
    if STAGES.index(stage) < STAGES.index(old_stage):
        raise OrchestrationError(f"state regression from {old_stage} to {stage}")
    updated = dict(document)
    updated.update(facts)
    updated["stage"] = stage
    updated["updated_at"] = time.time()
    _atomic_json(path, updated)
    print(f"STATE shard={updated['shard']} stage={stage}", flush=True)
    return updated


def _paths(work_root: Path, shard: str) -> Paths:
    relative = Path(*PurePosixPath(shard).parts)
    return Paths(
        input=work_root / "input" / relative,
        output=work_root / "output" / relative,
        results=work_root / "results" / relative.with_suffix(".ndjson"),
        state=work_root / "state" / relative.with_suffix(f"{relative.suffix}.json"),
    )


def _load_process_environment() -> str:
    token = os.environ.get("MODELSCOPE_API_TOKEN")
    if token:
        return token
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ")
            if b"sakuramoon.cli.data_service" not in command:
                continue
            values: dict[str, str] = {}
            for item in (entry / "environ").read_bytes().split(b"\0"):
                if b"=" not in item:
                    continue
                key, value = item.split(b"=", 1)
                decoded_key = key.decode(errors="ignore")
                if decoded_key in SECRET_KEYS:
                    values[decoded_key] = value.decode(errors="ignore")
            token = values.get("MODELSCOPE_API_TOKEN")
            if not token:
                continue
            for key, value in values.items():
                os.environ.setdefault(key, value)
            os.environ["MODELSCOPE_API_TOKEN"] = token
            return token
        except OSError:
            continue
    raise OrchestrationError("ModelScope credentials are unavailable")


def _load_manifest(path: Path) -> Any:
    from sakuramoon.data.manifest import parse_dataset_manifest_bytes

    try:
        manifest = parse_dataset_manifest_bytes(path.read_bytes())
    except (OSError, ValueError) as error:
        raise OrchestrationError(f"dataset manifest is invalid: {path}") from error
    if (
        manifest.source.repo_id != REPO_ID
        or manifest.source.revision != SOURCE_REVISION
        or len(manifest.shards) != 1877
    ):
        raise OrchestrationError("dataset manifest identity or shard count changed")
    return manifest


def _transport(token: str) -> Any:
    from sakuramoon.config.schema import DataTransportConfig
    from sakuramoon.data.modelscope import ModelScopeDatasetTransport

    return ModelScopeDatasetTransport(
        token,
        DataTransportConfig(
            connect_timeout_seconds=10.0,
            read_timeout_seconds=30.0,
            max_retries=3,
            retry_backoff_seconds=1.0,
            stream_chunk_bytes=4 * 1024 * 1024,
            streams_per_shard=4,
        ),
    )


def _require_file(path: Path, *, size: int | None = None) -> None:
    if path.is_symlink() or not path.is_file():
        raise OrchestrationError(f"required file is missing: {path}")
    if size is not None and path.stat().st_size != size:
        raise OrchestrationError(
            f"file size differs for {path}: {path.stat().st_size} != {size}"
        )

def _count_result_records(path: Path) -> int:
    _require_file(path)
    expected_keys = {"id", *quality.DB_COLUMNS}
    seen_ids: set[int] = set()
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.endswith("\n") or not line.strip():
                    raise OrchestrationError(
                        f"invalid NDJSON framing at {path}:{line_number}"
                    )
                record = json.loads(line)
                sample_id = record.get("id") if isinstance(record, dict) else None
                if (
                    not isinstance(record, dict)
                    or set(record) != expected_keys
                    or type(sample_id) is not int
                    or sample_id in seen_ids
                    or record["quality"] not in quality.AESTHETIC.labels
                    or record["anime_completeness"]
                    not in quality.COMPLETENESS.labels
                    or record["ai_image_corrupted"] not in (None, "corrupted")
                    or record["anime_classification"]
                    not in quality.CLASSIFICATION.labels
                ):
                    raise OrchestrationError(
                        f"invalid result record at {path}:{line_number}"
                    )
                seen_ids.add(sample_id)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise OrchestrationError(f"result file is invalid: {path}") from error
    if not seen_ids:
        raise OrchestrationError(f"result file is empty: {path}")
    return len(seen_ids)



def _download_one(
    transport: Any,
    manifest: Any,
    shard: Any,
    paths: Paths,
    cache_root: Path,
    input_root: Path,
    heartbeat: Heartbeat,
) -> Path:
    from sakuramoon.data.modelscope import fetch_dataset_shard

    if paths.input.is_file():
        _require_file(paths.input, size=shard.bytes)
        heartbeat.ping("download_reused_work")
        return paths.input
    cached = cache_root / Path(*PurePosixPath(shard.path).parts)
    if cached.exists():
        _require_file(cached, size=shard.bytes)
        paths.input.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(cached, paths.input)
        except FileExistsError:
            _require_file(paths.input, size=shard.bytes)
        heartbeat.ping("download_reused_training_cache")
        return paths.input

    def progress(done: int, total: int, elapsed: float, rate: float) -> None:
        heartbeat.ping(f"download_bytes={done}/{total} rate_mib={rate / 1024**2:.2f}")

    fetched = fetch_dataset_shard(
        transport,
        manifest,
        shard.path,
        input_root,
        progress=progress,
    )
    if fetched.path != paths.input:
        raise OrchestrationError("download transport returned an unexpected path")
    _require_file(paths.input, size=shard.bytes)
    heartbeat.ping("download_complete")
    return paths.input


def _run_with_watchdog(
    command: Sequence[str],
    *,
    heartbeat: Heartbeat,
    timeout_seconds: float,
    cancel_event: threading.Event | None = None,
) -> None:
    process = subprocess.Popen(
        tuple(command),
        cwd=PROJECT_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if process.stdout is None:
        process.kill()
        raise OrchestrationError("child process output pipe is unavailable")
    os.set_blocking(process.stdout.fileno(), False)
    selector = selectors.DefaultSelector()
    selector.register(process.stdout, selectors.EVENT_READ)
    try:
        while process.poll() is None:
            for key, _ in selector.select(timeout=1.0):
                chunk = os.read(key.fd, 65536)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    heartbeat.ping("child_output")
            if cancel_event is not None and cancel_event.is_set():
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise OrchestrationError("child process cancelled after peer failure")
            if heartbeat.age > timeout_seconds:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise OrchestrationError(
                    f"child made no progress for {timeout_seconds:.0f}s"
                )
        remainder = process.stdout.read()
        if remainder:
            sys.stdout.buffer.write(remainder)
            sys.stdout.buffer.flush()
        if process.returncode != 0:
            raise OrchestrationError(
                f"child process failed with exit code {process.returncode}"
            )
    finally:
        selector.close()
        if process.poll() is None:
            process.kill()
            process.wait()

class ClassifyWorker:
    def __init__(
        self,
        *,
        model_root: Path,
        batch_size: int,
        device: str,
        timeout_seconds: float,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.request_id = 0
        self.buffer = bytearray()
        self.process = subprocess.Popen(
            (
                sys.executable,
                str(PROJECT_ROOT / "scripts/deepghs_quality_pipeline.py"),
                "classify-worker",
                "--model-root",
                str(model_root),
                "--batch-size",
                str(batch_size),
                "--device",
                device,
            ),
            cwd=PROJECT_ROOT,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if self.process.stdin is None or self.process.stdout is None:
            self.abort()
            raise OrchestrationError("classify worker pipes are unavailable")
        os.set_blocking(self.process.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self._wait_for_line("WORKER_READY", Heartbeat("classify-worker-startup"))
        except BaseException:
            self.abort()
            raise

    def _read_line(self, heartbeat: Heartbeat) -> str:
        stdout = self.process.stdout
        if stdout is None:
            raise OrchestrationError("classify worker output pipe is unavailable")
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                payload = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                return payload.decode("utf-8", errors="replace").rstrip("\r")
            if self.process.poll() is not None:
                remainder = stdout.read()
                if remainder:
                    self.buffer.extend(remainder)
                    continue
                raise OrchestrationError(
                    f"classify worker exited early with code {self.process.returncode}"
                )
            for key, _ in self.selector.select(timeout=1.0):
                chunk = os.read(key.fd, 65536)
                if chunk:
                    sys.stdout.buffer.write(chunk)
                    sys.stdout.buffer.flush()
                    self.buffer.extend(chunk)
                    heartbeat.ping("classify_worker_output")
            if heartbeat.age > self.timeout_seconds:
                self.abort()
                raise OrchestrationError(
                    f"classify worker made no progress for "
                    f"{self.timeout_seconds:.0f}s"
                )

    def _wait_for_line(self, expected: str, heartbeat: Heartbeat) -> None:
        while self._read_line(heartbeat) != expected:
            pass

    def classify(
        self,
        input_path: Path,
        output_path: Path,
        result_path: Path,
        *,
        heartbeat: Heartbeat,
    ) -> int:
        if self.process.poll() is not None or self.process.stdin is None:
            raise OrchestrationError("classify worker is not running")
        request_id = self.request_id
        self.request_id += 1
        request = {
            "input": str(input_path),
            "output": str(output_path),
            "request_id": request_id,
            "results": str(result_path),
        }
        payload = (
            json.dumps(request, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        try:
            self.process.stdin.write(payload)
            self.process.stdin.flush()
        except (BrokenPipeError, OSError) as error:
            self.abort()
            raise OrchestrationError("classify worker request failed") from error

        while True:
            line = self._read_line(heartbeat)
            if not line.startswith("WORKER_DONE "):
                continue
            try:
                response = json.loads(line.removeprefix("WORKER_DONE "))
            except json.JSONDecodeError as error:
                self.abort()
                raise OrchestrationError(
                    "classify worker response is invalid JSON"
                ) from error
            if (
                not isinstance(response, dict)
                or set(response)
                != {"request_id", "samples", "elapsed_seconds"}
                or response.get("request_id") != request_id
                or type(response.get("samples")) is not int
                or response["samples"] <= 0
                or not isinstance(response.get("elapsed_seconds"), (int, float))
                or response["elapsed_seconds"] <= 0
            ):
                self.abort()
                raise OrchestrationError(
                    f"classify worker response contract changed: request={request_id}"
                )
            return int(response["samples"])

    def close(self) -> None:
        if self.process.poll() is not None:
            raise OrchestrationError(
                f"classify worker exited before close with code "
                f"{self.process.returncode}"
            )
        if self.process.stdin is None:
            raise OrchestrationError("classify worker input pipe is unavailable")
        self.process.stdin.close()
        self._wait_for_line("WORKER_EXIT_OK", Heartbeat("classify-worker-close"))
        try:
            return_code = self.process.wait(timeout=15)
        except subprocess.TimeoutExpired as error:
            self.abort()
            raise OrchestrationError("classify worker did not exit") from error
        self.selector.close()
        if return_code != 0:
            raise OrchestrationError(
                f"classify worker failed during close with code {return_code}"
            )

    def abort(self) -> None:
        selector = getattr(self, "selector", None)
        if selector is not None:
            selector.close()
        process = self.process
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()

def _remote_size(token: str, remote_path: str) -> int | None:
    from modelscope.hub.api import HubApi

    parent = str(PurePosixPath(remote_path).parent)
    if parent == ".":
        parent = "/"
    matches: list[dict[str, object]] = []
    api = HubApi()
    for page in range(1, 1001):
        entries = api.get_dataset_files(
            REPO_ID,
            revision=TARGET_REVISION,
            root_path=parent,
            recursive=False,
            page_number=page,
            page_size=1000,
            token=token,
        )
        if not entries:
            break
        matches.extend(
            entry
            for entry in entries
            if isinstance(entry, dict) and entry.get("Path") == remote_path
        )
        if len(entries) < 1000:
            break
    if not matches:
        return None
    if len(matches) != 1 or type(matches[0].get("Size")) is not int:
        raise OrchestrationError(f"remote listing is ambiguous: {remote_path}")
    return int(matches[0]["Size"])


def _upload_child(path: Path, remote_path: str, expected_size: int) -> int:
    token = _load_process_environment()
    _require_file(path, size=expected_size)
    existing = _remote_size(token, remote_path)
    if existing is not None:
        if existing != expected_size:
            raise OrchestrationError(
                f"remote size differs for {remote_path}: {existing} != {expected_size}"
            )
        print(f"UPLOAD_REUSED path={remote_path} bytes={expected_size}", flush=True)
        return 0

    from modelscope_hub.api import HubApi

    print(f"UPLOAD_START path={remote_path} bytes={expected_size}", flush=True)
    response = HubApi(token=token).upload_file(
        repo_id=REPO_ID,
        repo_type="dataset",
        path_or_fileobj=path,
        path_in_repo=remote_path,
        revision=TARGET_REVISION,
        commit_message=f"Enrich {remote_path} with deepghs metadata",
        buffer_size_mb=64,
        disable_tqdm=False,
    )
    if not isinstance(response, dict):
        raise OrchestrationError("ModelScope upload returned an invalid response")
    print(f"UPLOAD_RETURNED path={remote_path}", flush=True)
    actual = _remote_size(token, remote_path)
    if actual != expected_size:
        raise OrchestrationError(
            f"uploaded size differs for {remote_path}: {actual} != {expected_size}"
        )
    print(f"UPLOAD_VERIFIED path={remote_path} bytes={expected_size}", flush=True)
    return 0


class Pipeline:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.token = _load_process_environment()
        self.manifest = _load_manifest(args.manifest)
        self.transport = _transport(self.token)
        self.classify_workers: list[ClassifyWorker] = []
        self.classify_executors: list[ThreadPoolExecutor] = []
        self.stop_event = threading.Event()

    def _start_workers(self) -> None:
        if self.classify_workers or self.classify_executors:
            raise OrchestrationError("classify workers were already started")
        try:
            for index, device in enumerate(self.args.devices):
                worker = ClassifyWorker(
                    model_root=self.args.model_root,
                    batch_size=self.args.batch_size,
                    device=device,
                    timeout_seconds=self.args.no_progress_timeout,
                )
                self.classify_workers.append(worker)
                self.classify_executors.append(
                    ThreadPoolExecutor(
                        max_workers=1,
                        thread_name_prefix=f"deepghs-classify-{index}",
                    )
                )
                print(
                    f"GPU_WORKER_READY worker={index} device={device}", flush=True
                )
        except BaseException:
            self.abort()
            raise

    def _close_workers(self) -> None:
        errors: list[BaseException] = []
        for worker in self.classify_workers:
            try:
                worker.close()
            except OrchestrationError as error:
                errors.append(error)
        for executor in self.classify_executors:
            executor.shutdown(wait=True, cancel_futures=False)
        self.classify_workers.clear()
        self.classify_executors.clear()
        if errors:
            raise OrchestrationError(
                f"{len(errors)} classify worker(s) failed during close"
            ) from errors[0]

    def abort(self) -> None:
        self.stop_event.set()
        for worker in self.classify_workers:
            worker.abort()
        for executor in self.classify_executors:
            executor.shutdown(wait=False, cancel_futures=True)
        self.classify_workers.clear()
        self.classify_executors.clear()

    def _classify(
        self, worker_index: int, shard: Any, input_path: Path
    ) -> None:
        paths = _paths(self.args.work_root, shard.path)
        state = _read_state(paths.state, shard.path)
        stage = str(state["stage"])
        heartbeat = Heartbeat(shard.path)
        if STAGES.index(stage) < STAGES.index("downloaded"):
            _require_file(input_path, size=shard.bytes)
            state = _transition(
                paths.state, state, "downloaded", input_bytes=shard.bytes
            )
            stage = "downloaded"

        if STAGES.index(stage) < STAGES.index("classified"):
            paths.output.parent.mkdir(parents=True, exist_ok=True)
            paths.results.parent.mkdir(parents=True, exist_ok=True)
            worker_count = self.classify_workers[worker_index].classify(
                input_path,
                paths.output,
                paths.results,
                heartbeat=heartbeat,
            )
            _require_file(paths.output)
            count = _count_result_records(paths.results)
            if count != worker_count:
                raise OrchestrationError(
                    "worker and result record counts differ"
                )
            state = _transition(
                paths.state,
                state,
                "classified",
                samples=count,
                output_bytes=paths.output.stat().st_size,
            )
            stage = "classified"

        if stage != "classified":
            raise OrchestrationError(
                f"classification task entered an invalid stage: {stage}"
            )

    def _finalize(self, shard: Any, input_path: Path) -> None:
        paths = _paths(self.args.work_root, shard.path)
        state = _read_state(paths.state, shard.path)
        stage = str(state["stage"])
        heartbeat = Heartbeat(shard.path)
        if STAGES.index(stage) < STAGES.index("classified"):
            raise OrchestrationError(
                f"cannot finalize unclassified shard {shard.path}: {stage}"
            )

        _require_file(paths.output, size=int(state["output_bytes"]))
        _require_file(paths.results)
        if STAGES.index(stage) < STAGES.index("verified"):
            _run_with_watchdog(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/deepghs_quality_pipeline.py"),
                    "verify-tar",
                    "--input",
                    str(input_path),
                    "--output",
                    str(paths.output),
                    "--results",
                    str(paths.results),
                ),
                heartbeat=heartbeat,
                timeout_seconds=self.args.no_progress_timeout,
                cancel_event=self.stop_event,
            )
            count = _count_result_records(paths.results)
            if count != state.get("samples"):
                raise OrchestrationError(
                    "classification and verification counts differ"
                )
            state = _transition(paths.state, state, "verified")
            stage = "verified"

        if self.args.verify_only:
            print(f"VERIFY_ONLY_OK shard={shard.path}", flush=True)
            return

        if STAGES.index(stage) < STAGES.index("db_updated"):
            _run_with_watchdog(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/deepghs_quality_pipeline.py"),
                    "update-db",
                    "--database",
                    str(self.args.database),
                    "--results",
                    str(paths.results),
                ),
                heartbeat=heartbeat,
                timeout_seconds=self.args.no_progress_timeout,
                cancel_event=self.stop_event,
            )
            state = _transition(paths.state, state, "db_updated")
            stage = "db_updated"

        if STAGES.index(stage) < STAGES.index("uploaded"):
            _run_with_watchdog(
                (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "upload-one",
                    "--path",
                    str(paths.output),
                    "--remote-path",
                    shard.path,
                    "--expected-size",
                    str(state["output_bytes"]),
                ),
                heartbeat=heartbeat,
                timeout_seconds=self.args.no_progress_timeout,
                cancel_event=self.stop_event,
            )
            state = _transition(paths.state, state, "uploaded")
            stage = "uploaded"

        if stage == "uploaded":
            _transition(paths.state, state, "complete")
            for artifact in (paths.input, paths.output, paths.results):
                artifact.unlink()
            print(f"CLEANUP_OK shard={shard.path}", flush=True)

    def _publish_manifest(self) -> None:
        records: list[dict[str, object]] = []
        total = 0
        for shard in self.manifest.shards:
            state_path = _paths(self.args.work_root, shard.path).state
            state = _read_state(state_path, shard.path)
            if (
                state["stage"] != "complete"
                or type(state.get("output_bytes")) is not int
            ):
                raise OrchestrationError(
                    f"cannot publish incomplete target manifest: {shard.path}"
                )
            size = int(state["output_bytes"])
            records.append({"bytes": size, "path": shard.path})
            total += size
        document: dict[str, object] = {
            "aggregates": {"bytes": total, "shards": len(records)},
            "dataset_id": f"{REPO_ID}@{TARGET_REVISION}",
            "schema_version": 3,
            "shards": records,
            "source": {"repo_id": REPO_ID, "revision": TARGET_REVISION},
        }
        target = self.args.work_root / "target-dataset-manifest.json"
        _atomic_json(target, document)
        _upload_child(target, "dataset-manifest.json", target.stat().st_size)
        print(
            f"TARGET_MANIFEST_OK shards={len(records)} bytes={total}",
            flush=True,
        )

    def run(self) -> int:
        selected = self.manifest.shards[self.args.start_index :]
        if self.args.limit is not None:
            selected = selected[: self.args.limit]
        if not selected:
            raise OrchestrationError("no shards were selected")

        if not self.args.verify_only:
            _run_with_watchdog(
                (
                    sys.executable,
                    str(PROJECT_ROOT / "scripts/deepghs_quality_pipeline.py"),
                    "init-db",
                    "--database",
                    str(self.args.database),
                ),
                heartbeat=Heartbeat("database"),
                timeout_seconds=self.args.no_progress_timeout,
            )

        download_futures: dict[Future[Path], tuple[int, Any]] = {}
        ready_downloads: dict[int, tuple[Any, Path]] = {}
        classify_futures: dict[Future[None], tuple[int, int, Any, Path]] = {}
        finalize_ready: dict[int, tuple[Any, Path]] = {}
        finalize_future: Future[None] | None = None
        finalize_index: int | None = None
        free_workers: deque[int] = deque()
        next_index = 0
        completed = 0
        input_root = self.args.work_root / "input"
        download_executor = ThreadPoolExecutor(
            max_workers=self.args.download_concurrency,
            thread_name_prefix="deepghs-download",
        )
        finalize_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="deepghs-finalize",
        )
        succeeded = False
        try:
            def outstanding() -> int:
                return (
                    len(download_futures)
                    + len(ready_downloads)
                    + len(classify_futures)
                    + len(finalize_ready)
                    + (finalize_future is not None)
                )

            def fill_downloads() -> None:
                nonlocal next_index, completed
                while (
                    next_index < len(selected)
                    and outstanding() < self.args.download_concurrency
                ):
                    index = next_index
                    shard = selected[index]
                    next_index += 1
                    state = _read_state(
                        _paths(self.args.work_root, shard.path).state,
                        shard.path,
                    )
                    if state["stage"] == "complete":
                        print(f"SKIP_COMPLETE shard={shard.path}", flush=True)
                        completed += 1
                        continue
                    future = download_executor.submit(
                        _download_one,
                        self.transport,
                        self.manifest,
                        shard,
                        _paths(self.args.work_root, shard.path),
                        self.args.cache_root,
                        input_root,
                        Heartbeat(shard.path),
                    )
                    download_futures[future] = (index, shard)

            fill_downloads()
            while completed < len(selected):
                made_progress = False

                for future in tuple(download_futures):
                    if not future.done():
                        continue
                    index, shard = download_futures.pop(future)
                    ready_downloads[index] = (shard, future.result())
                    made_progress = True

                for future in tuple(classify_futures):
                    if not future.done():
                        continue
                    worker_index, _index, shard, input_path = (
                        classify_futures.pop(future)
                    )
                    future.result()
                    free_workers.append(worker_index)
                    finalize_ready[_index] = (shard, input_path)
                    made_progress = True

                if finalize_future is not None and finalize_future.done():
                    finalize_future.result()
                    if finalize_index is None:
                        raise OrchestrationError(
                            "finalize future lost its shard index"
                        )
                    completed += 1
                    finalize_future = None
                    finalize_index = None
                    made_progress = True

                for index in sorted(ready_downloads):
                    shard, input_path = ready_downloads[index]
                    state = _read_state(
                        _paths(self.args.work_root, shard.path).state,
                        shard.path,
                    )
                    if STAGES.index(str(state["stage"])) >= STAGES.index(
                        "classified"
                    ):
                        del ready_downloads[index]
                        finalize_ready[index] = (shard, input_path)
                        made_progress = True
                        continue
                    if not self.classify_workers:
                        self._start_workers()
                        free_workers.extend(range(len(self.classify_workers)))
                    if not free_workers:
                        break
                    worker_index = free_workers.popleft()
                    del ready_downloads[index]
                    future = self.classify_executors[worker_index].submit(
                        self._classify, worker_index, shard, input_path
                    )
                    classify_futures[future] = (
                        worker_index,
                        index,
                        shard,
                        input_path,
                    )
                    print(
                        f"CLASSIFY_DISPATCH shard={shard.path} "
                        f"worker={worker_index} "
                        f"device={self.args.devices[worker_index]}",
                        flush=True,
                    )
                    made_progress = True

                if finalize_future is None and finalize_ready:
                    finalize_index = min(finalize_ready)
                    shard, input_path = finalize_ready.pop(finalize_index)
                    finalize_future = finalize_executor.submit(
                        self._finalize, shard, input_path
                    )
                    print(
                        f"FINALIZE_DISPATCH shard={shard.path}", flush=True
                    )
                    made_progress = True

                fill_downloads()
                if completed >= len(selected):
                    break
                if made_progress:
                    continue
                pending = [*download_futures, *classify_futures]
                if finalize_future is not None:
                    pending.append(finalize_future)
                if not pending:
                    raise OrchestrationError(
                        "pipeline stalled without pending work"
                    )
                wait(pending, return_when=FIRST_COMPLETED)

            if (
                download_futures
                or ready_downloads
                or classify_futures
                or finalize_ready
                or finalize_future is not None
            ):
                raise OrchestrationError("pipeline finished with orphaned work")
            if self.classify_workers:
                self._close_workers()
            succeeded = True
        finally:
            if not succeeded:
                self.stop_event.set()
                for future in download_futures:
                    future.cancel()
                for future in classify_futures:
                    future.cancel()
            download_executor.shutdown(
                wait=succeeded, cancel_futures=not succeeded
            )
            finalize_executor.shutdown(
                wait=True, cancel_futures=not succeeded
            )

        if (
            not self.args.verify_only
            and self.args.start_index == 0
            and self.args.limit is None
        ):
            self._publish_manifest()
        print(
            f"RUN_OK selected={len(selected)} verify_only={self.args.verify_only}",
            flush=True,
        )
        return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument(
        "--manifest",
        type=Path,
        default=Path("/sakuramoon-runtime/data/dataset-manifest.json"),
    )
    run.add_argument(
        "--database",
        type=Path,
        default=PROJECT_ROOT / "db/dan_8_13.db",
    )
    run.add_argument(
        "--model-root",
        type=Path,
        default=Path("/sakuramoon-runtime/quality-pipeline/models"),
    )
    run.add_argument(
        "--work-root",
        type=Path,
        default=Path("/sakuramoon-runtime/quality-pipeline/work"),
    )
    run.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/sakuramoon-runtime/cache/data"),
    )
    run.add_argument(
        "--devices", nargs="+", default=("cuda:0", "cuda:1")
    )
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--download-concurrency", type=int, default=8)
    run.add_argument("--no-progress-timeout", type=float, default=300.0)
    run.add_argument("--start-index", type=int, default=0)
    run.add_argument("--limit", type=int)
    run.add_argument("--verify-only", action="store_true")

    upload = commands.add_parser("upload-one")
    upload.add_argument("--path", type=Path, required=True)
    upload.add_argument("--remote-path", required=True)
    upload.add_argument("--expected-size", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "upload-one":
        return _upload_child(args.path, args.remote_path, args.expected_size)
    if (
        args.batch_size != 64
        or tuple(args.devices) != ("cuda:0", "cuda:1")
        or args.download_concurrency != 8
        or args.no_progress_timeout != 300.0
        or args.start_index < 0
        or (args.limit is not None and args.limit <= 0)
    ):
        raise OrchestrationError("pipeline limits differ from the required contract")
    usage = shutil.disk_usage(args.work_root.parent)
    if usage.free < 32 * 1024**3:
        raise OrchestrationError("work filesystem has less than 32 GiB free")
    pipeline = Pipeline(args)
    try:
        return pipeline.run()
    except BaseException:
        pipeline.abort()
        raise


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
