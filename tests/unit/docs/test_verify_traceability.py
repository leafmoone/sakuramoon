from __future__ import annotations

import copy
import hashlib
import importlib.util
import shutil
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import tomli_w

ROOT = Path(__file__).parents[3]
REGISTRY = Path("docs/model-architecture/progress/traceability.toml")
SPEC = importlib.util.spec_from_file_location(
    "verify_traceability", ROOT / "tools/verify_traceability.py"
)
assert SPEC is not None and SPEC.loader is not None
vt = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = vt
SPEC.loader.exec_module(vt)


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    model_docs = root / "docs/model-architecture"
    model_docs.mkdir(parents=True)
    for directory in ("current", "archive"):
        shutil.copytree(
            ROOT / "docs/model-architecture" / directory,
            model_docs / directory,
        )
    shutil.copytree(
        ROOT / "docs/model-architecture/reviews/ROADMAP",
        model_docs / "reviews/ROADMAP",
    )
    shutil.copytree(
        ROOT / "src/sakuramoon",
        root / "src/sakuramoon",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(
        ROOT / "tests",
        root / "tests",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copytree(ROOT / "config", root / "config")
    for relative in (
        "SHA256SUMS",
        "progress/IMPLEMENTATION_ROADMAP.md",
        "progress/traceability.toml",
    ):
        destination = model_docs / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / "docs/model-architecture" / relative, destination)
    for relative in (
        ".gitignore",
        "README.md",
        "pyproject.toml",
        "uv.lock",
        ".python-version",
        "AGENTS.md",
        "docs/model-architecture/progress/environment-lock.md",
        "docs/model-architecture/progress/asset-policy.md",
        "docs/model-architecture/progress/tasks/R001.md",
        "docs/model-architecture/progress/tasks/R002.md",
        "docs/model-architecture/progress/tasks/D001.md",
        "docs/model-architecture/progress/tasks/C001.md",
        "docs/model-architecture/progress/tasks/A001.md",
        "docs/model-architecture/progress/tasks/A002.md",
        "docs/model-architecture/progress/tasks/D010.md",
        "docs/model-architecture/progress/tasks/D011.md",
        "docs/model-architecture/progress/tasks/D012.md",
        "docs/model-architecture/progress/tasks/D013.md",
        "docs/model-architecture/progress/tasks/D014.md",
        "docs/model-architecture/progress/tasks/D015.md",
        "docs/model-architecture/progress/tasks/T020.md",
        "docs/model-architecture/progress/tasks/T050.md",
        "docs/model-architecture/progress/tasks/T051.md",
        "docs/model-architecture/progress/tasks/T052.md",
        "docs/model-architecture/progress/time-log.jsonl",
        "docs/model-architecture/reviews/R001/implementation_report.md",
        "docs/model-architecture/reviews/R001/reference_manifest.json",
        "docs/model-architecture/reviews/R001/review_remediation.md",
        "docs/model-architecture/reviews/R001/test_report.json",
        "docs/model-architecture/reviews/R001/timing.json",
        "docs/model-architecture/reviews/R001/artifacts.json",
        "docs/model-architecture/reviews/R001/tracked-files.txt",
        "docs/model-architecture/reviews/R002/implementation_report.md",
        "docs/model-architecture/reviews/R002/ai_review.md",
        "docs/model-architecture/reviews/R002/infra_review.md",
        "docs/model-architecture/reviews/R002/test_report.json",
        "docs/model-architecture/reviews/R002/artifacts.json",
        "docs/model-architecture/reviews/R002/dependency-licenses.json",
        "docs/model-architecture/reviews/R002/fresh-env-report.json",
        "docs/model-architecture/reviews/R002/cold-rebuild-report.json",
        "docs/model-architecture/reviews/R002/timing.json",
        "docs/model-architecture/reviews/D001/implementation_report.md",
        "docs/model-architecture/reviews/D001/test_report.json",
        "docs/model-architecture/reviews/D001/timing.json",
        "docs/model-architecture/reviews/D001/artifacts.json",
        "docs/model-architecture/reviews/D001/traceability-report.json",
        "docs/model-architecture/reviews/D001/ai_review.md",
        "docs/model-architecture/reviews/D001/infra_review.md",
        "docs/model-architecture/reviews/C001/implementation_report.md",
        "docs/model-architecture/reviews/C001/test_report.json",
        "docs/model-architecture/reviews/C001/timing.json",
        "docs/model-architecture/reviews/C001/artifacts.json",
        "docs/model-architecture/reviews/C001/ai_review.md",
        "docs/model-architecture/reviews/C001/infra_review.md",
        "docs/model-architecture/reviews/A001/implementation_report.md",
        "docs/model-architecture/reviews/A001/test_report.json",
        "docs/model-architecture/reviews/A001/timing.json",
        "docs/model-architecture/reviews/A001/artifacts.json",
        "docs/model-architecture/reviews/A001/ai_review.md",
        "docs/model-architecture/reviews/A001/infra_review.md",
        "docs/model-architecture/reviews/D010/implementation_report.md",
        "docs/model-architecture/reviews/D010/task.md",
        "docs/model-architecture/reviews/D010/test_report.json",
        "docs/model-architecture/reviews/D010/timing.json",
        "docs/model-architecture/reviews/D010/ai_review.md",
        "docs/model-architecture/reviews/D010/infra_review.md",
        "docs/model-architecture/reviews/D011/implementation_report.md",
        "docs/model-architecture/reviews/D011/task.md",
        "docs/model-architecture/reviews/D011/test_report.json",
        "docs/model-architecture/reviews/D011/timing.json",
        "docs/model-architecture/reviews/D011/ai_review.md",
        "docs/model-architecture/reviews/D011/infra_review.md",
        "docs/model-architecture/reviews/D012/implementation_report.md",
        "docs/model-architecture/reviews/D012/test_report.json",
        "docs/model-architecture/reviews/D012/timing.json",
        "docs/model-architecture/reviews/D012/ai_review.md",
        "docs/model-architecture/reviews/D012/infra_review.md",
        "docs/model-architecture/reviews/D013/implementation_report.md",
        "docs/model-architecture/reviews/D013/test_report.json",
        "docs/model-architecture/reviews/D013/timing.json",
        "docs/model-architecture/reviews/D013/ai_review.md",
        "docs/model-architecture/reviews/D013/infra_review.md",
        "docs/model-architecture/reviews/D014/implementation_report.md",
        "docs/model-architecture/reviews/D014/test_report.json",
        "docs/model-architecture/reviews/D014/timing.json",
        "docs/model-architecture/reviews/D014/ai_review.md",
        "docs/model-architecture/reviews/D014/infra_review.md",
        "docs/model-architecture/reviews/D015/implementation_report.md",
        "docs/model-architecture/reviews/D015/test_report.json",
        "docs/model-architecture/reviews/D015/timing.json",
        "docs/model-architecture/reviews/D015/ai_review.md",
        "docs/model-architecture/reviews/D015/infra_review.md",
        "docs/model-architecture/reviews/T020/implementation_report.md",
        "docs/model-architecture/reviews/T020/test_report.json",
        "docs/model-architecture/reviews/T020/timing.json",
        "docs/model-architecture/reviews/T020/ai_review.md",
        "docs/model-architecture/reviews/T020/infra_review.md",
        "docs/model-architecture/reviews/T050/task.md",
        "docs/model-architecture/reviews/T050/implementation_report.md",
        "docs/model-architecture/reviews/T050/test_report.json",
        "docs/model-architecture/reviews/T050/timing.json",
        "docs/model-architecture/reviews/T050/ai_review.md",
        "docs/model-architecture/reviews/T050/infra_review.md",
        "docs/model-architecture/reviews/T051/task.md",
        "docs/model-architecture/reviews/T051/implementation_report.md",
        "docs/model-architecture/reviews/T051/test_report.json",
        "docs/model-architecture/reviews/T051/timing.json",
        "docs/model-architecture/reviews/T051/ai_review.md",
        "docs/model-architecture/reviews/T051/infra_review.md",
        "docs/model-architecture/reviews/T051/perf_baseline.json",
        "docs/model-architecture/reviews/T051/perf_after.json",
        "docs/model-architecture/reviews/T052/task.md",
        "docs/model-architecture/reviews/T052/implementation_report.md",
        "docs/model-architecture/reviews/T052/test_report.json",
        "docs/model-architecture/reviews/T052/timing.json",
        "docs/model-architecture/reviews/T052/ai_review.md",
        "docs/model-architecture/reviews/T052/infra_review.md",
        "docs/model-architecture/reviews/T052/determinism_report.json",
        "docs/model-architecture/progress/tasks/T053.md",
        "docs/model-architecture/reviews/T053/test_report.json",
        "docs/model-architecture/reviews/T053/ai_review.md",
        "docs/model-architecture/reviews/T053/infra_review.md",
        "tests/unit/assets/test_local_models.py",
        "tests/unit/config/conftest.py",
        "tests/unit/config/test_resolve_redact.py",
        "tests/unit/config/test_schema.py",
        "tests/unit/docs/test_verify_traceability.py",
        "tools/__init__.py",
    ):
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    tool = root / "tools/verify_traceability.py"
    tool.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "tools/verify_traceability.py", tool)
    return root


def load_registry(root: Path) -> dict[str, Any]:
    return tomllib.loads((root / REGISTRY).read_text(encoding="utf-8"))


def rewrite_registry(root: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    data = load_registry(root)
    mutate(data)
    (root / REGISTRY).write_text(tomli_w.dumps(data), encoding="utf-8")


def error_text(root: Path) -> str:
    report = vt.verify(root)
    assert not report.ok
    return "\n".join(report.errors)


def source_by_kind(data: dict[str, Any], kind: str) -> dict[str, Any]:
    return next(source for source in data["sources"] if source["kind"] == kind)


def test_live_registry_and_source_coverage() -> None:
    report = vt.verify(ROOT)
    assert report.ok, report.errors
    assert report.requirement_count == report.source_node_count == 221
    assert report.archive_file_count == 16

    data = load_registry(ROOT)
    expected = {
        "confirmed": (110, {f"{index}." for index in range(15)}),
        "open_items": (99, {f"{index}." for index in range(11)}),
        "observability": (12, {"可观测性与评估补充决定"}),
    }
    for kind, (count, heading_prefixes) in expected.items():
        nodes = vt.extract_source_nodes(ROOT, source_by_kind(data, kind))
        assert len(nodes) == count
        assert {
            next(prefix for prefix in heading_prefixes if node.heading_path[0].startswith(prefix))
            for node in nodes
        } == heading_prefixes

    confirmed = vt.extract_source_nodes(ROOT, source_by_kind(data, "confirmed"))
    assert any(node.kind == "code_block" and "flowchart TB" in node.text for node in confirmed)
    assert any("clean endpoint" in node.text for node in confirmed)
    assert any("Anima 只作为效果方向参照" in node.text for node in confirmed)
    assert all("来源：" not in node.text for node in confirmed)


def test_bootstrap_refuses_to_replace_existing_registry(repo_copy: Path) -> None:
    with pytest.raises(FileExistsError, match="refusing to replace"):
        vt._write_bootstrap(repo_copy, REGISTRY)


def test_new_clause_preserves_existing_ids(repo_copy: Path) -> None:
    before = load_registry(repo_copy)
    existing_ids = [item["id"] for item in before["requirements"]]
    source = repo_copy / "docs/model-architecture/current/confirmed-decisions.md"
    text = source.read_text(encoding="utf-8")
    text = text.replace(
        "# 0. 判定规则与当前边界\n",
        "# 0. 判定规则与当前边界\n- 新增的本地治理条款。\n",
        1,
    )
    source.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def mutate(data: dict[str, Any]) -> None:
        source_entry = source_by_kind(data, "confirmed")
        previous = source_entry["sha256"]
        source_entry["sha256"] = digest
        source_entry["revision"] = 4
        data["changes"].append(
            {
                "source_path": source_entry["path"],
                "revision": 4,
                "previous_sha256": previous,
                "new_sha256": digest,
                "changed_at": "2026-07-29",
                "summary": "Add one governance clause.",
            }
        )
        nodes = vt.extract_source_nodes(repo_copy, source_entry)
        node = next(item for item in nodes if item.text == "新增的本地治理条款。")
        template = copy.deepcopy(
            next(item for item in data["requirements"] if item["id"] == "DOC-001")
        )
        template.update(
            {
                "id": "DOC-008",
                "heading_path": list(node.heading_path),
                "node_kind": node.kind,
                "source_fingerprint": node.fingerprint,
                "source_occurrence": node.occurrence,
            }
        )
        data["requirements"].append(template)

    rewrite_registry(repo_copy, mutate)
    after = load_registry(repo_copy)
    assert [item["id"] for item in after["requirements"][:-1]] == existing_ids
    report = vt.verify(repo_copy)
    assert report.ok, report.errors


def test_deleting_a_requirement_mapping_is_rejected(repo_copy: Path) -> None:
    rewrite_registry(repo_copy, lambda data: data["requirements"].pop())
    assert "unregistered normative node" in error_text(repo_copy)


def test_deleting_one_requirement_mapping_dimension_is_rejected(repo_copy: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        requirement = next(item for item in data["requirements"] if item["config_keys"])
        requirement["config_keys"] = []

    rewrite_registry(repo_copy, mutate)
    assert "config_keys is empty without explicit not-applicable" in error_text(repo_copy)


def test_fingerprint_drift_is_rejected(repo_copy: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["requirements"][0]["source_fingerprint"] = "0" * 64

    rewrite_registry(repo_copy, mutate)
    errors = error_text(repo_copy)
    assert "source node missing or fingerprint drifted" in errors
    assert "unregistered normative node" in errors


def test_unknown_registry_key_is_rejected(repo_copy: Path) -> None:
    rewrite_registry(repo_copy, lambda data: data.update({"fallback": True}))
    assert "unknown keys: fallback" in error_text(repo_copy)


def test_empty_profile_mapping_requires_explicit_not_applicable(repo_copy: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        profile = next(item for item in data["profiles"] if item["name"] == "governance")
        profile["tests"] = []

    rewrite_registry(repo_copy, mutate)
    assert "tests is empty without explicit not-applicable" in error_text(repo_copy)


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("sources", "canonical source set"),
        ("scope", "canonical include_top_headings scope was changed"),
        ("profiles", "profiles must not be empty"),
        ("requirements", "requirements must not be empty"),
    ],
)
def test_canonical_coverage_cannot_be_shrunk(
    repo_copy: Path,
    case: str,
    expected: str,
) -> None:
    def mutation(data: dict[str, Any]) -> None:
        if case == "scope":
            data["sources"][0]["include_top_headings"].pop()
        else:
            data[case] = []

    rewrite_registry(repo_copy, mutation)
    assert expected in error_text(repo_copy)


@pytest.mark.parametrize("status", ["alias", "superseded", "blocked"])
def test_status_edges_are_strict(repo_copy: Path, status: str) -> None:
    def mutate(data: dict[str, Any]) -> None:
        requirement = next(item for item in data["requirements"] if item["status"] == status)
        if status == "alias":
            requirement["alias_of"] = "MISSING-999"
        elif status == "superseded":
            requirement["superseded_by"] = requirement["id"]
        else:
            requirement["blocked_by"] = []

    rewrite_registry(repo_copy, mutate)
    errors = error_text(repo_copy)
    expected = {
        "alias": "dangling alias target",
        "superseded": "supersession target cannot self-reference",
        "blocked": "blocked requirement has no blocker",
    }
    assert expected[status] in errors


def test_archive_checksum_change_is_rejected(repo_copy: Path) -> None:
    archive_file = next((repo_copy / "docs/model-architecture/archive").rglob("*.md"))
    archive_file.write_text(
        archive_file.read_text(encoding="utf-8") + "\nchanged\n",
        encoding="utf-8",
    )
    assert "archive checksum mismatch" in error_text(repo_copy)


def test_archive_and_checksum_manifest_cannot_be_changed_together(
    repo_copy: Path,
) -> None:
    model_docs = repo_copy / "docs/model-architecture"
    archive_file = next((model_docs / "archive").rglob("*.md"))
    relative = archive_file.relative_to(model_docs).as_posix()
    old_digest = hashlib.sha256(archive_file.read_bytes()).hexdigest()
    archive_file.write_text(
        archive_file.read_text(encoding="utf-8") + "\nchanged together\n",
        encoding="utf-8",
    )
    new_digest = hashlib.sha256(archive_file.read_bytes()).hexdigest()
    manifest = model_docs / "SHA256SUMS"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace(
            f"{old_digest}  {relative}", f"{new_digest}  {relative}"
        ),
        encoding="utf-8",
    )
    assert "archive manifest does not match the trusted bootstrap anchor" in error_text(
        repo_copy
    )


def test_requirement_id_history_rejects_exchange_and_reuse() -> None:
    baseline = load_registry(ROOT)
    exchanged = copy.deepcopy(baseline)
    exchanged["registry_revision"] += 1
    first, second = exchanged["requirements"][:2]
    first["id"], second["id"] = second["id"], first["id"]

    errors: list[str] = []
    vt._validate_registry_history([baseline, exchanged], errors)
    assert any("source locator was historically bound" in error for error in errors)

    removed = copy.deepcopy(baseline)
    removed["registry_revision"] += 1
    retired = removed["requirements"].pop()
    reused = copy.deepcopy(removed)
    reused["registry_revision"] += 1
    retired["source_fingerprint"] = "f" * 64
    reused["requirements"].append(retired)

    errors = []
    vt._validate_registry_history([baseline, removed, reused], errors)
    assert any("stable requirement IDs were removed" in error for error in errors)

    unchanged_revision = copy.deepcopy(baseline)
    unchanged_revision["requirements"][0]["status"] = "planned"
    errors = []
    vt._validate_registry_history([baseline, unchanged_revision], errors)
    assert any("registry_revision must increment by exactly one" in error for error in errors)


def test_bootstrap_binding_anchor_rejects_exchange_without_history() -> None:
    exchanged = copy.deepcopy(load_registry(ROOT))
    first, second = exchanged["requirements"][:2]
    first["id"], second["id"] = second["id"], first["id"]

    errors: list[str] = []
    vt._validate_registry_history([exchanged], errors)

    assert any("trusted locator anchor" in error for error in errors)

@pytest.mark.parametrize("target_is_directory", [False, True])
def test_archive_symlink_is_rejected(
    repo_copy: Path, tmp_path: Path, target_is_directory: bool
) -> None:
    target = tmp_path / ("outside-dir" if target_is_directory else "outside.md")
    if target_is_directory:
        target.mkdir()
    else:
        target.write_text("outside\n", encoding="utf-8")
    link = repo_copy / "docs/model-architecture/archive/forbidden-link"
    link.symlink_to(target, target_is_directory=target_is_directory)
    assert "archive symlink is forbidden" in error_text(repo_copy)


def test_reverse_module_and_config_inventory_is_required(repo_copy: Path) -> None:
    module = repo_copy / "src/sakuramoon/unmapped.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")
    config = repo_copy / "config/unmapped.toml"
    config.parent.mkdir(exist_ok=True)
    config.write_text("unexpected = 1\n", encoding="utf-8")

    errors = error_text(repo_copy)
    assert "production module has no reverse requirement mapping" in errors
    assert "runtime config key has no reverse requirement mapping" in errors


def test_inventory_ignore_cannot_hide_production_files(repo_copy: Path) -> None:
    module = repo_copy / "src/sakuramoon/hidden.py"
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text("VALUE = 1\n", encoding="utf-8")

    def mutate(data: dict[str, Any]) -> None:
        data["inventory"]["ignored_module_paths"] = ["**"]

    rewrite_registry(repo_copy, mutate)
    assert "ignored_module_paths must remain empty" in error_text(repo_copy)


def test_one_gpu_evidence_cannot_close_four_gpu_requirement(repo_copy: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        hardware = {profile["name"]: profile["hardware"] for profile in data["profiles"]}
        requirement = next(
            item
            for item in data["requirements"]
            if item["status"] == "planned" and hardware[item["profile"]] == "4GPU"
        )
        requirement.update(
            {
                "status": "verified",
                "implementation_commit_ref": "task:TEST",
                "implementation_paths": [REGISTRY.as_posix()],
                "evidence_hardware": "1GPU",
                "ai_review": "review:ai",
                "infra_review": "review:infra",
            }
        )

    rewrite_registry(repo_copy, mutate)
    errors = error_text(repo_copy)
    assert "1GPU evidence cannot close 4GPU requirement" in errors
    assert "implementation commit reference is invalid" in errors
    assert "review evidence does not exist" in errors
    assert "verified requirement lacks evidence artifacts" in errors


def test_verified_requirement_requires_real_independent_evidence(repo_copy: Path) -> None:
    review_dir = repo_copy / "docs/model-architecture/reviews/D001"
    review_dir.mkdir(parents=True, exist_ok=True)
    ai_review = review_dir / "ai_review.md"
    infra_review = review_dir / "infra_review.md"
    ai_review.write_text("AI review passed.\n", encoding="utf-8")
    infra_review.write_text("Infra review passed.\n", encoding="utf-8")

    def mutate(data: dict[str, Any]) -> None:
        requirement = next(item for item in data["requirements"] if item["id"] == "DOC-001")
        requirement.update(
            {
                "status": "verified",
                "implementation_commit_ref": "task:D001",
                "implementation_paths": [REGISTRY.as_posix()],
                "evidence_hardware": "CPU",
                "evidence_artifacts": [
                    "docs/model-architecture/reviews/D001/ai_review.md"
                ],
                "ai_review": "docs/model-architecture/reviews/D001/ai_review.md",
                "infra_review": "docs/model-architecture/reviews/D001/infra_review.md",
            }
        )

    rewrite_registry(repo_copy, mutate)
    report = vt.verify(repo_copy)
    assert report.ok, report.errors


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("source_path", "expected str, got list"),
        ("profile_modules", "expected list, got int"),
        ("requirement_id", "expected str, got int"),
        ("requirement_config_keys", "expected list[str]"),
    ],
)
def test_malformed_nested_types_return_errors_without_traceback(
    repo_copy: Path,
    case: str,
    expected: str,
) -> None:
    def mutation(data: dict[str, Any]) -> None:
        if case == "source_path":
            data["sources"][0]["path"] = []
        elif case == "profile_modules":
            data["profiles"][0]["modules"] = 1
        elif case == "requirement_id":
            data["requirements"][0]["id"] = 1
        else:
            data["requirements"][0]["config_keys"] = [1]

    rewrite_registry(repo_copy, mutation)
    assert expected in error_text(repo_copy)


def test_node_kind_must_match_source_ast(repo_copy: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        data["requirements"][0]["node_kind"] = "code_block"

    rewrite_registry(repo_copy, mutate)
    assert "node_kind code_block does not match" in error_text(repo_copy)


def test_alias_must_resolve_to_terminal_requirement(repo_copy: Path) -> None:
    def mutate(data: dict[str, Any]) -> None:
        aliases = [item for item in data["requirements"] if item["status"] == "alias"]
        aliases[0]["alias_of"] = aliases[1]["id"]

    rewrite_registry(repo_copy, mutate)
    assert "alias target is not a terminal requirement" in error_text(repo_copy)


def test_registry_path_escape_is_rejected_before_read(repo_copy: Path) -> None:
    report = vt.verify(repo_copy, Path("../outside.toml"))
    assert not report.ok
    assert "repository-relative without traversal" in "\n".join(report.errors)


def test_source_symlink_is_rejected(repo_copy: Path, tmp_path: Path) -> None:
    source = repo_copy / "docs/model-architecture/current/confirmed-decisions.md"
    outside = tmp_path / "outside.md"
    outside.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    source.unlink()
    source.symlink_to(outside)
    assert "symlink path component is forbidden" in error_text(repo_copy)


def test_current_change_requires_revision_and_changelog(repo_copy: Path) -> None:
    source = repo_copy / "docs/model-architecture/current/confirmed-decisions.md"
    source.write_text(source.read_text(encoding="utf-8") + "\n<!-- local change -->\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def mutate(data: dict[str, Any]) -> None:
        source_entry = source_by_kind(data, "confirmed")
        source_entry["sha256"] = digest

    rewrite_registry(repo_copy, mutate)
    assert "latest changelog hash does not match source" in error_text(repo_copy)


def test_current_change_cannot_rewrite_canonical_initial_sha(repo_copy: Path) -> None:
    source = repo_copy / "docs/model-architecture/current/confirmed-decisions.md"
    source.write_text(source.read_text(encoding="utf-8") + "\n<!-- local change -->\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def mutate(data: dict[str, Any]) -> None:
        source_entry = source_by_kind(data, "confirmed")
        source_entry["sha256"] = digest
        source_entry["initial_sha256"] = digest

    rewrite_registry(repo_copy, mutate)
    assert "canonical initial_sha256 scope was changed" in error_text(repo_copy)


def test_valid_current_changelog_chain_passes(repo_copy: Path) -> None:
    source = repo_copy / "docs/model-architecture/current/confirmed-decisions.md"
    source.write_text(source.read_text(encoding="utf-8") + "\n<!-- local change -->\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()

    def mutate(data: dict[str, Any]) -> None:
        source_entry = source_by_kind(data, "confirmed")
        previous = source_entry["sha256"]
        source_entry["sha256"] = digest
        source_entry["revision"] = 4
        data["changes"].append(
            {
                "source_path": source_entry["path"],
                "revision": 4,
                "previous_sha256": previous,
                "new_sha256": digest,
                "changed_at": "2026-07-29",
                "summary": "Test-only non-normative change.",
            }
        )

    rewrite_registry(repo_copy, mutate)
    report = vt.verify(repo_copy)
    assert report.ok, report.errors
