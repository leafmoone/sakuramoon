"""CPU-only offline merge contract, including executable aggregation mutations."""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

import pytest

from sakuramoon.cli import guard_window_merge as merge
from sakuramoon.optim.guarded_canonical import stable_owner
from sakuramoon.telemetry import guard_publish

R = ("observations", "skip_total", "skip_by_role", "skip_by_fqn")
O = (
    "bf16_attempts",
    "bf16_safety_failures",
    "fp32_attempts",
    "fp32_rescues",
    "fp32_rescue_failures",
    "rescue_by_role",
)
SEMANTIC = [
    "attention_scale",
    "attention_shift",
    "attention_gate",
    "mlp_scale",
    "mlp_shift",
    "mlp_gate",
]
ROUTING: list[dict[str, Any]] = [
    {
        "fqn": "probe0",
        "shape": [4, 4],
        "chunk_count": 1,
        "chunk_dim": 0,
        "roles": ["attention_q"],
        "role": "attention_q",
    },
    {
        "fqn": "probe1",
        "shape": [8, 4],
        "chunk_count": 2,
        "chunk_dim": 0,
        "roles": ["ffn_gate", "ffn_value"],
        "role": "ffn_in",
    },
    {
        "fqn": "probe2",
        "shape": [24, 4],
        "chunk_count": 6,
        "chunk_dim": 0,
        "roles": SEMANTIC,
        "role": "adaln_shared",
    },
]


def _partition(world: int) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {str(rank): [] for rank in range(world)}
    for spec in ROUTING:
        for chunk in range(spec["chunk_count"]):
            result[str(stable_owner(spec["fqn"], chunk, world))].append(
                f"{spec['fqn']}#{chunk}"
            )
    return {rank: sorted(chunks) for rank, chunks in result.items()}


def _counts(update: int, owner: int, *, rescue: bool = True) -> dict[str, Any]:
    # Replicated decision maps exercise canonical buckets, not per-chunk roles.
    counts: dict[str, Any] = {
        "observations": update,
        "skip_total": 9 * update,
        "skip_by_role": {
            "attention_q": update,
            "ffn_in": 2 * update,
            "adaln_shared": 6 * update,
        }
        if update
        else {},
        "skip_by_fqn": {"probe0": update, "probe1": 2 * update, "probe2": 6 * update}
        if update
        else {},
    }
    if rescue:
        counts.update(
            {
                field: 0 if field == "fp32_rescue_failures" else owner
                for field in O
                if field != "rescue_by_role"
            }
        )
        counts["rescue_by_role"] = {"adaln_shared": owner} if owner else {}
    return counts


def _diff(end: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    return {
        field: (
            {
                key: cast("dict[str, int]", value).get(key, 0)
                - cast("dict[str, int]", base[field]).get(key, 0)
                for key in set(cast("dict[str, int]", value))
                | set(cast("dict[str, int]", base[field]))
            }
            if isinstance(value, dict)
            else value - base[field]
        )
        for field, value in end.items()
    }


def _records(
    world: int = 2,
    start: int = 1,
    end: int = 2,
    *,
    complete: bool = True,
    outcome: str | None = None,
    session: str = "session",
    restored: bool = False,
    rescue: bool = True,
) -> list[dict[str, Any]]:
    partition = _partition(world)
    result: list[dict[str, Any]] = []
    resume = "step-10" if restored else "fresh-start"
    resume_update = 10 if restored else 0
    for rank in range(world):
        local = len(partition[str(rank)])
        base = _counts(start - 1, 2 if restored else local * (start - 1), rescue=rescue)
        absolute = _counts(
            end,
            (2 if restored else local * (start - 1)) + local * (end - start + 1),
            rescue=rescue,
        )
        fields = (*R, *O) if rescue else R
        result.append(
            {
                "schema_version": 1,
                "run_id": "run",
                "session_id": session,
                "rank": rank,
                "world_size": world,
                "window_id": f"{session}:u{start}-{end}:c{resume}",
                "start_update": start,
                "end_update": end,
                "complete": complete,
                "run_outcome": outcome,
                "routing": deepcopy(ROUTING),
                "owner_partition": deepcopy(partition),
                "launch": {
                    "optimizer": (
                        "hybrid_cmuon_canonical_ns4_fp32_rescue"
                        if rescue
                        else "hybrid_cmuon_guarded_canonical"
                    ),
                    "resume_source_checkpoint": resume,
                    "resume_successful_updates": resume_update,
                    "launch_kind": "user_resume" if restored else "fresh_bootstrap",
                    "owner_mapping_version": "fnv1a64-v1",
                    "last_published_checkpoint": None,
                },
                "baseline": {
                    "source": "restored_rank0" if restored else "fresh",
                    "counts": base,
                },
                "absolute": absolute,
                "delta": _diff(absolute, base),
                "delta_status": {field: "ok" for field in fields},
                "delta_coverage": "complete",
                "absolute_history": {
                    field: (
                        "rank_provenance"
                        if field in R
                        else "rank0_owner_only"
                        if restored
                        else "local"
                    )
                    for field in fields
                },
            }
        )
    return result


def _merged(
    records: list[dict[str, Any]], start: int = 1, end: int = 2, **kwargs: Any
) -> dict[str, Any]:
    return merge.merge_records(records, start_update=start, end_update=end, **kwargs)


def _assert_positive(world: int) -> None:
    result = _merged(list(reversed(_records(world))))
    assert result["schema"]["version"] == 1
    assert result["integrity"]["coverage"] == "complete"
    assert result["integrity"]["owner_partition_verified"] is True
    assert result["integrity"]["routing_anchor"] == "embedded_writer_manifest"
    delta = result["windows"][0]["delta"]
    expected = _counts(2, 18)
    expected["rescue_by_role"].update({"attention_q": 0, "ffn_in": 0})
    assert delta == expected
    assert delta["skip_by_role"]["adaln_shared"] == 12
    assert result["windows"][0]["complete"] is True
    assert "absolute" not in result["windows"][0]
    assert len(result["windows"][0]["per_rank"]) == world


@pytest.mark.parametrize("world", [1, 2, 4])
def test_owner_sum_replicated_retained_once(world: int) -> None:
    _assert_positive(world)


def test_two_intervals_and_clean_short_tail_checkpoint_metadata_not_identity() -> None:
    rows = (
        _records(4)
        + _records(4, 3, 4)
        + _records(4, 5, 5, complete=False, outcome="clean")
    )
    for row in rows[4:]:
        row["launch"]["last_published_checkpoint"] = "step-4"
    result = _merged(rows, end=5)
    assert result["integrity"]["coverage"] == "complete"
    assert [window["delta"]["bf16_attempts"] for window in result["windows"]] == [
        18,
        18,
        9,
    ]
    assert [window["delta"]["observations"] for window in result["windows"]] == [
        2,
        2,
        1,
    ]
    tail = result["windows"][-1]
    assert tail["complete"] is False
    assert tail["run_outcome"] == "clean"
    assert tail["per_rank"][0]["launch"]["last_published_checkpoint"] == "step-4"
    assert (
        tail["per_rank"][0]["baseline"]["counts"]
        == result["windows"][1]["per_rank"][0]["absolute"]
    )


def test_rank0_restored_history_delta_not_absolute_sum() -> None:
    rows = _records(4, 11, 11, restored=True)
    # Exact copied-history example: baseline 2, new local events 1/3/3/2.
    for row, absolute in zip(rows, [3, 5, 5, 4], strict=True):
        for field in O:
            if field == "rescue_by_role":
                row["absolute"][field] = {"adaln_shared": absolute}
            else:
                row["absolute"][field] = (
                    0 if field == "fp32_rescue_failures" else absolute
                )
        row["delta"] = _diff(row["absolute"], row["baseline"]["counts"])
    result = _merged(rows, 11, 11)
    window = result["windows"][0]
    assert window["delta"]["bf16_attempts"] == 9
    assert window["delta"]["rescue_by_role"]["adaln_shared"] == 9
    assert [rank["absolute"]["bf16_attempts"] for rank in window["per_rank"]] == [
        3,
        5,
        5,
        4,
    ]
    assert all(
        rank["absolute_history"]["bf16_attempts"] == "rank0_owner_only"
        for rank in window["per_rank"]
    )
    assert all(
        rank["baseline"]["counts"]["bf16_attempts"] == 2 for rank in window["per_rank"]
    )
    assert window["delta_coverage"] == "complete"


def test_base_variant_omits_owner_fields_and_normalizes_empty_sparse_maps() -> None:
    result = _merged(_records(2, rescue=False))
    assert set(result["windows"][0]["delta"]) == set(R)
    base = result["windows"][0]["per_rank"][0]["baseline"]["counts"]
    assert base["skip_by_role"] == {"attention_q": 0, "ffn_in": 0, "adaln_shared": 0}
    assert base["skip_by_fqn"] == {"probe0": 0, "probe1": 0, "probe2": 0}


def test_external_routing_anchor() -> None:
    result = _merged(_records(), routing_manifest=ROUTING)
    assert result["integrity"]["routing_anchor"] == "external_manifest"
    forged = deepcopy(ROUTING)
    forged[0]["shape"] = [8, 4]
    with pytest.raises(ValueError, match="expected manifest"):
        _merged(_records(), routing_manifest=forged)


@pytest.mark.parametrize("conflict", [False, True])
def test_even_identical_duplicate_is_rejected(conflict: bool) -> None:
    rows = _records()
    duplicate = deepcopy(rows[0])
    if conflict:
        duplicate["launch"]["last_published_checkpoint"] = "other"
    with pytest.raises(ValueError, match="duplicate rank/window"):
        _merged([*rows, duplicate])


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("rank", True),
        ("rank", -1),
        ("rank", 2),
        ("world_size", True),
        ("world_size", 0),
        ("world_size", 4),
        ("run_id", "other"),
        ("session_id", "other"),
        ("complete", 1),
        ("window_id", "forged"),
        ("start_update", True),
        ("end_update", 0),
    ],
)
def test_invalid_identity_schema_and_bounds(field: str, value: object) -> None:
    rows = _records()
    rows[1][field] = value
    with pytest.raises(ValueError):
        _merged(rows)


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner_mapping_version", "round-robin"),
        ("resume_source_checkpoint", "other"),
        ("resume_successful_updates", True),
        ("resume_successful_updates", 2),
    ],
)
def test_invalid_launch_identity(field: str, value: object) -> None:
    rows = _records()
    rows[1]["launch"][field] = value
    with pytest.raises(ValueError):
        _merged(rows)


def test_cross_session_contiguous_updates_fail() -> None:
    with pytest.raises(ValueError, match="identity"):
        _merged(_records() + _records(2, 3, 4, session="another"), end=4)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "extra",
        "duplicate_chunk",
        "missing_chunk",
        "wrong_owner",
        "partition_world",
        "routing",
        "routing_duplicate",
    ],
)
def test_rank_and_owner_partition_failures(kind: str) -> None:
    rows = _records()
    partition = rows[0]["owner_partition"]
    if kind == "missing":
        rows.pop()
    elif kind == "extra":
        rows.append(deepcopy(rows[0]))
        rows[-1]["rank"] = 2
    elif kind == "duplicate_chunk":
        partition["0"].append(partition["0"][0])
    elif kind == "missing_chunk":
        partition["0"].pop()
    elif kind == "wrong_owner":
        partition["1"].append(partition["0"].pop())
    elif kind == "partition_world":
        partition["2"] = []
    elif kind == "routing":
        rows[1]["routing"][0]["shape"] = [8, 4]
    else:
        rows[0]["routing"].append(deepcopy(ROUTING[0]))
    with pytest.raises(ValueError):
        _merged(rows)


@pytest.mark.parametrize("where", ["baseline", "absolute", "delta"])
@pytest.mark.parametrize(
    "field", ["observations", "skip_by_role", "bf16_attempts", "rescue_by_role"]
)
def test_whole_missing_counter_never_zero(where: str, field: str) -> None:
    rows = _records()
    counts = rows[0][where]["counts"] if where == "baseline" else rows[0][where]
    del counts[field]
    with pytest.raises(ValueError, match="counter field"):
        _merged(rows)


@pytest.mark.parametrize(
    "kind",
    [
        "negative_endpoint",
        "bool_counter",
        "negative_delta",
        "forged_delta",
        "status",
        "coverage",
        "role_confusion",
        "fqn",
        "replicated_delta",
        "replicated_endpoints",
        "history",
    ],
)
def test_counter_validation(kind: str) -> None:
    rows = _records()
    row = rows[1]
    if kind == "negative_endpoint":
        row["absolute"]["bf16_attempts"] = -1
    elif kind == "bool_counter":
        row["absolute"]["observations"] = True
    elif kind == "negative_delta":
        row["baseline"]["counts"]["bf16_attempts"] = 999
    elif kind == "forged_delta":
        row["delta"]["bf16_attempts"] += 1
    elif kind == "status":
        row["delta_status"]["observations"] = "rollback"
    elif kind == "coverage":
        row["delta_coverage"] = "unknown"
    elif kind == "role_confusion":
        row["absolute"]["rescue_by_role"] = {"attention_scale": 12}
    elif kind == "fqn":
        row["absolute"]["skip_by_fqn"]["unknown"] = 0
    elif kind == "replicated_delta":
        row["absolute"]["skip_total"] += 1
        row["delta"]["skip_total"] += 1
    elif kind == "replicated_endpoints":
        row["absolute"]["skip_total"] += 1
        row["baseline"]["counts"]["skip_total"] += 1
    else:
        row["absolute_history"]["bf16_attempts"] = "global"
    with pytest.raises(ValueError):
        _merged(rows)


@pytest.mark.parametrize(
    "kind",
    ["chain", "gap", "overlap", "head", "tail", "rank_bounds", "terminal", "cadence"],
)
def test_interval_and_endpoint_chain_validation(kind: str) -> None:
    rows = _records() + _records(2, 3, 4)
    start, end = 1, 4
    if kind == "chain":
        row = rows[-1]
        row["baseline"]["counts"]["bf16_attempts"] += 1
        row["absolute"]["bf16_attempts"] += 1
    elif kind == "gap":
        rows = _records() + _records(2, 4, 5)
        end = 5
    elif kind == "overlap":
        rows = _records() + _records(2, 2, 3)
    elif kind == "head":
        start = 0
    elif kind == "tail":
        end = 5
    elif kind == "rank_bounds":
        rows[-1] = _records(2, 3, 3)[1]
    elif kind == "terminal":
        for row in rows[:2]:
            row["run_outcome"] = "clean"
    else:
        rows = _records() + _records(2, 3, 5)
        end = 5
    with pytest.raises(ValueError):
        _merged(rows, start, end)


@pytest.mark.parametrize("outcome", [None, "raised", "unknown"])
@pytest.mark.parametrize("complete", [False, True])
def test_abort_unknown_tail_never_complete(outcome: str | None, complete: bool) -> None:
    if complete and outcome is None:
        assert _merged(_records(complete=True))["integrity"]["coverage"] == "complete"
        return
    with pytest.raises(ValueError):
        _merged(_records(complete=complete, outcome=outcome))


def test_explicit_interval_can_end_on_periodic_record() -> None:
    assert _merged(_records())["integrity"]["coverage"] == "complete"
    assert _merged(_records(2, 3, 4), 3, 4)["integrity"]["coverage"] == "complete"
    tail = _merged(_records(2, 1, 1, complete=False, outcome="clean"), 1, 1)
    assert tail["integrity"]["coverage"] == "complete"
    assert tail["windows"][0]["complete"] is False


def test_explicit_legacy_partial_never_upgrades_modern() -> None:
    legacy = {"bf16_attempts": 7, "observations": 2}
    result: Any = merge.merge_records(
        [], start_update=1, end_update=2, legacy_records=[legacy]
    )
    assert result["integrity"]["coverage"] == "partial"
    assert result["integrity"]["modern_coverage"] == "unknown"
    assert result["integrity"]["identity"] is None
    assert result["windows"] == []
    assert result["legacy"][0]["record"] == legacy
    assert result["legacy"][0]["identity"] is None
    assert result["legacy"][0]["delta"] is None
    assert result["legacy"][0]["owner_coverage"] == "rank0_owner_only"
    assert (
        _merged(_records(), legacy_records=[legacy])["integrity"]["coverage"]
        == "partial"
    )
    with pytest.raises(ValueError, match="missing/extra rank"):
        _merged(_records()[:1], legacy_records=[legacy])
    with pytest.raises(ValueError):
        _merged([legacy])
    with pytest.raises(ValueError, match="modern identity"):
        _merged([], legacy_records=[_records()[0]])


def _jsonl(path: Path, rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def _args(source: Path, output: Path) -> list[str]:
    return [
        "--input",
        str(source),
        "--output",
        str(output),
        "--start-update",
        "1",
        "--end-update",
        "2",
    ]


def test_cli_directory_jsonl_expansion_and_no_clobber(tmp_path: Path) -> None:
    directory = tmp_path / "input"
    _jsonl(directory / "rank-0" / "rows.jsonl", _records()[:1])
    _jsonl(directory / "rank-1" / "rows.jsonl", _records()[1:])
    for name in (
        ".claimed",
        ".claimed.jsonl",
        ".tmp/broken.jsonl",
        "aborted.tmp.jsonl",
    ):
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("not JSON", encoding="utf-8")
    output = tmp_path / "out" / "merged.json"
    assert merge.main(_args(directory, output)) == 0
    original = output.read_bytes()
    assert json.loads(original)["integrity"]["coverage"] == "complete"
    assert not (output.parent / "schema.json").exists()
    assert list((output.parent / ".tmp").iterdir()) == []
    with pytest.raises(FileExistsError):
        merge.main(_args(directory, output))
    assert output.read_bytes() == original
    assert list((output.parent / ".tmp").iterdir()) == []


def test_cli_multiple_rows_and_explicit_legacy_file(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "modern.jsonl", _records())
    manifest = tmp_path / "routing.json"
    manifest.write_text(json.dumps(ROUTING), encoding="utf-8")
    output = tmp_path / "modern.json"
    assert (
        merge.main([*_args(source, output), "--routing-manifest", str(manifest)]) == 0
    )
    assert (
        json.loads(output.read_text(encoding="utf-8"))["integrity"]["routing_anchor"]
        == "external_manifest"
    )
    legacy = _jsonl(tmp_path / "legacy.jsonl", [{"bf16_attempts": 2}])
    output = tmp_path / "legacy.json"
    assert (
        merge.main(
            [
                "--legacy",
                str(legacy),
                "--output",
                str(output),
                "--start-update",
                "1",
                "--end-update",
                "2",
            ]
        )
        == 0
    )
    assert (
        json.loads(output.read_text(encoding="utf-8"))["integrity"]["coverage"]
        == "partial"
    )


@pytest.mark.parametrize(
    "text",
    [
        "{broken",
        '{"schema_version":1,"schema_version":1}',
        '{"bf16_attempts":NaN}',
        "{}",
    ],
)
def test_cli_malformed_modern_no_output(tmp_path: Path, text: str) -> None:
    source = tmp_path / "bad.jsonl"
    source.write_text(text, encoding="utf-8")
    output = tmp_path / "not-published.json"
    with pytest.raises(ValueError):
        merge.main(_args(source, output))
    assert not output.exists()


def test_cli_missing_rank_traceback_nonzero(tmp_path: Path) -> None:
    source = _jsonl(tmp_path / "missing.jsonl", _records()[:1])
    output = tmp_path / "not-published.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "sakuramoon.cli.guard_window_merge",
            *_args(source, output),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert completed.returncode != 0
    assert "Traceback (most recent call last)" in completed.stderr
    assert "missing/extra rank" in completed.stderr
    assert not output.exists()


def test_import_and_help_are_stdlib_only() -> None:
    code = "import sys; from sakuramoon.cli.guard_window_merge import main; assert 'torch' not in sys.modules; main(['--help'])"
    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stderr
    assert "--start-update" in completed.stdout
    assert "--legacy" in completed.stdout


def test_output_fsync_failure_is_precommit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "merged.json"

    def fail(_fd: int) -> None:
        raise OSError("fsync failed")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="fsync failed"):
        merge.write_output(output, {"ready": True})
    assert not output.exists()
    assert list((tmp_path / ".tmp").iterdir()) == []


def test_postcommit_cleanup_warning_is_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "merged.json"
    unlink = Path.unlink

    def fail_temp(path: Path, missing_ok: bool = False) -> None:
        if path.parent.name == ".tmp":
            raise OSError("temp cleanup fault")
        unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", fail_temp)
    merge.write_output(output, {"ready": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ready": True}
    assert "temp cleanup fault" in capsys.readouterr().err


def test_prepare_fsync_close_before_link_no_postlink_fsync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "merged.json"
    events: list[str] = []
    fsync, link = os.fsync, os.link

    def observe_fsync(fd: int) -> None:
        assert not output.exists()
        events.append("fsync")
        fsync(fd)

    def observe_link(source: Path, target: Path) -> None:
        assert events == ["fsync"]
        # On Windows a normal open handle prevents rename: confirms close.
        renamed = source.with_suffix(".closed")
        source.rename(renamed)
        renamed.rename(source)
        assert json.loads(source.read_text(encoding="utf-8")) == {"ready": True}
        events.append("link")
        link(source, target)

    monkeypatch.setattr(os, "fsync", observe_fsync)
    monkeypatch.setattr(guard_publish.os, "link", observe_link)
    merge.write_output(output, {"ready": True})
    assert events == ["fsync", "link"]


@pytest.mark.parametrize("mutation", ["owner_max", "replicated_sum"])
@pytest.mark.parametrize("world", [2, 4])
def test_aggregation_mutations_are_killed_in_memory(
    mutation: str, world: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = merge._merge_delta  # pyright: ignore[reportPrivateUsage]
    source = inspect.getsource(original)
    if mutation == "owner_max":
        old = 'result[field] = sum(cast("int", record.delta[field]) for record in records)'
        new = 'result[field] = max(cast("int", record.delta[field]) for record in records)'
    else:
        old = "result: Counts = {field: first.delta[field] for field in _R_FIELDS}"
        new = 'result: Counts = {field: (sum(cast("int", record.delta[field]) for record in records) if isinstance(first.delta[field], int) else first.delta[field]) for field in _R_FIELDS}'
    assert source.count(old) == 1
    namespace = dict(vars(merge))
    exec(compile(source.replace(old, new), "<guard-merge-mutation>", "exec"), namespace)  # noqa: S102 - deliberate in-memory mutation test
    monkeypatch.setattr(merge, "_merge_delta", namespace["_merge_delta"])
    with pytest.raises(AssertionError):
        _assert_positive(world)


@pytest.mark.parametrize("rescue", [False, True])
def test_empty_routing_cannot_back_complete_counter_evidence(rescue: bool) -> None:
    rows = _records(2, rescue=rescue)
    for row in rows:
        row["routing"] = []
        row["owner_partition"] = {"0": [], "1": []}
        for surface in (row["baseline"]["counts"], row["absolute"], row["delta"]):
            for field in ("skip_by_role", "skip_by_fqn", "rescue_by_role"):
                if field in surface:
                    surface[field] = {}
    with pytest.raises(ValueError, match="routing"):
        _merged(rows)


def test_rescue_variant_cannot_downgrade_when_all_owner_fields_are_missing() -> None:
    rows = _records(2)
    for row in rows:
        for surface in (
            row["baseline"]["counts"],
            row["absolute"],
            row["delta"],
            row["delta_status"],
            row["absolute_history"],
        ):
            for field in O:
                surface.pop(field)
    with pytest.raises(ValueError, match="counter field"):
        _merged(rows)


@pytest.mark.parametrize("variant", [None, "other_optimizer", True])
def test_missing_or_unknown_optimizer_variant_is_rejected(variant: object) -> None:
    rows = _records(2)
    rows[0]["launch"]["optimizer"] = variant
    with pytest.raises(ValueError, match="optimizer"):
        _merged(rows)


def _single_owned_restored_rows() -> list[dict[str, Any]]:
    rows = _records(4, 11, 11, restored=True)
    routing = deepcopy(ROUTING[:1])
    owner = stable_owner("probe0", 0, 4)
    partition = {str(rank): ["probe0#0"] if rank == owner else [] for rank in range(4)}
    for row in rows:
        row["routing"] = deepcopy(routing)
        row["owner_partition"] = deepcopy(partition)
        base = row["baseline"]["counts"]
        end = row["absolute"]
        for counts in (base, end):
            counts["skip_total"] = 0
            counts["skip_by_role"] = {"attention_q": 0}
            counts["skip_by_fqn"] = {"probe0": 0}
            counts["rescue_by_role"] = {"attention_q": 2}
        for field in O:
            if field != "rescue_by_role":
                base[field] = 0 if field == "fp32_rescue_failures" else 2
                end[field] = base[field] + (
                    int(row["rank"] == owner) if field != "fp32_rescue_failures" else 0
                )
        end["rescue_by_role"]["attention_q"] += int(row["rank"] == owner)
        row["delta"] = _diff(end, base)
    return rows


def test_rank_without_owner_may_keep_positive_restored_history_but_not_delta() -> None:
    rows = _single_owned_restored_rows()
    result = _merged(rows, 11, 11)
    assert result["windows"][0]["delta"]["bf16_attempts"] == 1
    assert result["windows"][0]["per_rank"][0]["absolute"]["bf16_attempts"] == 2
    rows[0]["absolute"]["bf16_attempts"] += 1
    rows[0]["delta"] = _diff(rows[0]["absolute"], rows[0]["baseline"]["counts"])
    with pytest.raises(ValueError, match="no owned chunk"):
        _merged(rows, 11, 11)


def test_rescue_role_delta_requires_that_rank_to_own_that_canonical_role() -> None:
    rows = _records(4)
    # probe0/attention_q belongs only to rank3; rank0 owns other roles.
    assert stable_owner("probe0", 0, 4) != 0
    rows[0]["absolute"]["rescue_by_role"]["attention_q"] = 1
    for field in O:
        if field not in ("rescue_by_role", "fp32_rescue_failures"):
            rows[0]["absolute"][field] += 1
    rows[0]["delta"] = _diff(rows[0]["absolute"], rows[0]["baseline"]["counts"])
    with pytest.raises(ValueError, match="role increment"):
        _merged(rows)


@pytest.mark.parametrize("mode", ["stretched_interval", "zero_observations"])
def test_observations_delta_must_cover_exact_successful_update_span(mode: str) -> None:
    rows = _records(2)
    end = 2
    for row in rows:
        if mode == "stretched_interval":
            end = 100
            row["end_update"] = end
            row["window_id"] = "session:u1-100:cfresh-start"
        else:
            row["absolute"]["observations"] = row["baseline"]["counts"]["observations"]
            row["delta"] = _diff(row["absolute"], row["baseline"]["counts"])
    with pytest.raises(ValueError, match="observations delta"):
        _merged(rows, end=end)


@pytest.mark.parametrize(
    "fault",
    [
        "skip_role_total",
        "skip_fqn_total",
        "rescue_role_total",
        "too_many_verdicts",
        "bad_attempts",
    ],
)
def test_counter_deltas_are_internally_consistent(fault: str) -> None:
    rows = _records(2)
    for row in rows:
        end = row["absolute"]
        if fault == "skip_role_total":
            end["skip_by_role"]["attention_q"] += 1
        elif fault == "skip_fqn_total":
            end["skip_by_fqn"]["probe0"] += 1
        elif fault == "rescue_role_total":
            end["rescue_by_role"]["adaln_shared"] += 1
        elif fault == "too_many_verdicts":
            end["fp32_rescue_failures"] += 1
        else:
            end["fp32_attempts"] += 1
        row["delta"] = _diff(end, row["baseline"]["counts"])
    with pytest.raises(ValueError, match="differs|inconsistent|exceed"):
        _merged(rows)


@pytest.mark.parametrize("kind", [None, "unknown_launch", "user_resume"])
def test_launch_kind_is_required_and_part_of_rank_identity(kind: object) -> None:
    rows = _records(2)
    rows[1]["launch"]["launch_kind"] = kind
    with pytest.raises(ValueError, match="launch_kind|identity"):
        _merged(rows)
