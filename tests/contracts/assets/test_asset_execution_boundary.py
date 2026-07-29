from __future__ import annotations

import ast
from collections.abc import Iterator
from pathlib import Path

from sakuramoon.assets import load_manifest

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "assets/manifest.toml"
MODEL_LOCAL_PATHS = {
    "qwen": "model/qwen_3.5_2B",
    "vae": "model/vae",
}
MODEL_DOWNLOAD_CALLS = {
    "hf_hub_download",
    "model_file_download",
    "snapshot_download",
}
DYNAMIC_CODE_CALLS = {
    "SourceFileLoader",
    "check_call",
    "check_output",
    "compile",
    "exec",
    "execfile",
    "Popen",
    "run",
    "run_path",
    "spec_from_file_location",
    "system",
}
MODULE_LOADER_CALLS = {"import_module", "run_module"}
CODE_SUFFIXES = {".py", ".pyc", ".pyo", ".sh", ".bash", ".zsh"}


def _python_sources() -> Iterator[Path]:
    this_file = Path(__file__).resolve()
    for base in (ROOT / "src", ROOT / "tests"):
        for path in base.rglob("*.py"):
            if path.resolve() != this_file and "__pycache__" not in path.parts:
                yield path


def _call_name(node: ast.Call) -> str:
    function = node.func
    if isinstance(function, ast.Name):
        return function.id
    if isinstance(function, ast.Attribute):
        return function.attr
    return ""


def _string_literals(node: ast.AST) -> Iterator[str]:
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            yield child.value


def _is_reference_module(value: str | None) -> bool:
    if value is None:
        return False
    return "reference" in value.casefold().split(".")


def _looks_like_reference_code(value: str) -> bool:
    normalized = value.casefold().replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    return (
        "reference" in parts
        and any(normalized.endswith(suffix) for suffix in CODE_SUFFIXES)
    )


def _literals_reference_code(values: tuple[str, ...]) -> bool:
    normalized = tuple(value.casefold().replace("\\", "/") for value in values)
    return any(_looks_like_reference_code(value) for value in normalized) or (
        any("reference" in value.split("/") for value in normalized)
        and any(
            value.endswith(suffix)
            for value in normalized
            for suffix in CODE_SUFFIXES
        )
    )


def _is_dataset_download_scope(path: Path) -> bool:
    relative = path.relative_to(ROOT).as_posix()
    return relative.startswith("src/sakuramoon/data/") or relative == (
        "src/sakuramoon/cli/manifest.py"
    )


def test_manifest_locks_the_two_prepared_model_directories() -> None:
    manifest = load_manifest(MANIFEST)

    assert {asset.kind: asset.local_path for asset in manifest.models} == (
        MODEL_LOCAL_PATHS
    )
    assert all(asset.lock_state == "ready" for asset in manifest.models)


def test_production_model_loading_is_explicitly_local_only() -> None:
    violations: list[str] = []
    for path in (ROOT / "src/sakuramoon").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        download_call_names = set(MODEL_DOWNLOAD_CALLS)
        for imported in ast.walk(tree):
            if isinstance(imported, ast.ImportFrom):
                for alias in imported.names:
                    if alias.name in MODEL_DOWNLOAD_CALLS:
                        download_call_names.add(alias.asname or alias.name)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            location = f"{path.relative_to(ROOT)}:{node.lineno}"
            if name == "from_pretrained":
                local_only = next(
                    (
                        keyword.value
                        for keyword in node.keywords
                        if keyword.arg == "local_files_only"
                    ),
                    None,
                )
                if not (
                    isinstance(local_only, ast.Constant)
                    and local_only.value is True
                ):
                    violations.append(f"{location}: from_pretrained is not local-only")
            if (
                name in download_call_names
                and not _is_dataset_download_scope(path)
            ):
                violations.append(f"{location}: forbidden model download API {name}")

    assert violations == []


def test_reference_execution_detector_covers_direct_and_composed_paths() -> None:
    assert _is_reference_module("reference.hdm.model")
    assert _looks_like_reference_code("reference/JLT/train.py")
    assert _literals_reference_code(("python", "reference", "entry.py"))
    assert not _literals_reference_code(("git", "status", "reference"))


def test_no_engineering_path_imports_or_executes_reference_code() -> None:
    violations: list[str] = []
    for path in _python_sources():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            location = f"{path.relative_to(ROOT)}:{getattr(node, 'lineno', 1)}"
            if isinstance(node, ast.Import):
                if any(_is_reference_module(alias.name) for alias in node.names):
                    violations.append(f"{location}: imports reference code")
            elif isinstance(node, ast.ImportFrom):
                if _is_reference_module(node.module):
                    violations.append(f"{location}: imports reference code")
            elif isinstance(node, ast.Call):
                name = _call_name(node)
                literals = tuple(_string_literals(node))
                if name in DYNAMIC_CODE_CALLS and _literals_reference_code(literals):
                    violations.append(f"{location}: dynamically executes reference code")
                if name in MODULE_LOADER_CALLS and any(
                    _is_reference_module(value) for value in literals
                ):
                    violations.append(f"{location}: dynamically imports reference code")
                if name in {"append", "insert"} and any(
                    "reference" in value.casefold().replace("\\", "/").split("/")
                    for value in literals
                ):
                    violations.append(f"{location}: injects reference into a search path")

    assert violations == []
