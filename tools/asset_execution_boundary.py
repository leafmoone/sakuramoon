"""Static, fail-closed checks for local model and reference execution boundaries."""

from __future__ import annotations

import ast
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path

MODEL_ASSET_IDS = frozenset({"qwen_text_encoder", "mage_vae"})
DATASET_REPO_ID = "leafmoone/webdataset_danbooru"
DATASET_TRANSPORT_PATH = "src/sakuramoon/data/modelscope.py"
DATASET_TRANSPORT_FUNCTION = "fetch_dataset_shard"
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")

_DOWNLOAD_CALLS = frozenset(
    {
        "huggingface_hub.hf_hub_download",
        "huggingface_hub.snapshot_download",
        "modelscope.hub.file_download.model_file_download",
        "modelscope.hub.snapshot_download",
        "modelscope.hub.snapshot_download.snapshot_download",
        "modelscope.model_file_download",
        "modelscope.snapshot_download",
    }
)
_PRETRAINED_MODULES = ("transformers.", "diffusers.")
_VERIFIED_SELECTION_TYPES = frozenset(
    {
        "sakuramoon.assets.VerifiedAssetSelection",
        "sakuramoon.assets.inspect.VerifiedAssetSelection",
    }
)
_VERIFIED_SELECTION_FACTORIES = frozenset(
    {
        "sakuramoon.assets.require_runtime_assets_ready",
        "sakuramoon.assets.inspect.require_runtime_assets_ready",
    }
)
_DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "importlib.import_module",
        "runpy.run_module",
        "builtins.__import__",
        "__import__",
    }
)
_DYNAMIC_CODE_CALLS = frozenset(
    {
        "builtins.compile",
        "builtins.exec",
        "builtins.execfile",
        "compile",
        "exec",
        "execfile",
        "importlib.machinery.SourceFileLoader",
        "importlib.util.spec_from_file_location",
        "runpy.run_path",
    }
)
_PROCESS_PREFIXES = (
    "asyncio.create_subprocess_",
    "asyncio.subprocess.",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "subprocess.",
)
_PROCESS_EXACT = frozenset({"os.popen", "os.system"})
@dataclass(frozen=True)
class BoundaryViolation:
    path: str
    line: int
    code: str
    detail: str

    def render(self) -> str:
        return f"{self.path}:{self.line}: {self.code}: {self.detail}"


class SourceBoundaryError(RuntimeError):
    """Raised before a repository scanner follows an unsafe source path."""


def _terminal(value: str) -> str:
    return value.rsplit(".", 1)[-1]


def _literal_string(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _keyword(call: ast.Call, name: str) -> ast.AST | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _call_argument(call: ast.Call, keyword: str, position: int) -> ast.AST | None:
    value = _keyword(call, keyword)
    if value is not None:
        return value
    return call.args[position] if len(call.args) > position else None


def _annotation_name(node: ast.AST | None) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _annotation_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _annotation_name(node.value)
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ""


def _resolve_dotted_name(value: str, resolver: _Resolver) -> str:
    first, separator, remainder = value.partition(".")
    resolved = resolver.aliases.get(first, first)
    return f"{resolved}.{remainder}" if separator else resolved


class _Resolver:
    def __init__(self, tree: ast.AST) -> None:
        self.aliases: dict[str, str] = {
            "compile": "builtins.compile",
            "exec": "builtins.exec",
            "__import__": "builtins.__import__",
        }
        self._collect_imports(tree)
        self._collect_callable_aliases(tree)

    def _collect_imports(self, tree: ast.AST) -> None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for item in node.names:
                    self.aliases[item.asname or item.name.split(".", 1)[0]] = item.name
            elif isinstance(node, ast.ImportFrom) and node.module:
                for item in node.names:
                    if item.name != "*":
                        self.aliases[item.asname or item.name] = f"{node.module}.{item.name}"

    def _collect_callable_aliases(self, tree: ast.AST) -> None:
        assignments: list[tuple[str, ast.AST]] = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                value = node.value
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                if value is None:
                    continue
                for target in targets:
                    if isinstance(target, ast.Name):
                        assignments.append((target.id, value))
        for _ in range(len(assignments) + 1):
            changed = False
            for name, value in assignments:
                resolved = self.resolve_callable(value)
                if resolved and self.aliases.get(name) != resolved:
                    self.aliases[name] = resolved
                    changed = True
            if not changed:
                break

    def resolve_callable(self, node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return self.aliases.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            prefix = self.resolve_callable(node.value)
            return f"{prefix}.{node.attr}" if prefix else node.attr
        if isinstance(node, ast.Call):
            called = self.resolve_callable(node.func)
            if called in {"builtins.getattr", "getattr"} and len(node.args) >= 2:
                attr = _literal_string(node.args[1])
                base = self.resolve_callable(node.args[0])
                return f"{base}.{attr}" if base and attr else ""
            if called in {"functools.partial", "partial"} and node.args:
                return self.resolve_callable(node.args[0])
        return ""


def _is_reference_text(value: str) -> bool:
    normalized = value.casefold().replace("\\", "/")
    dotted = normalized.replace("/", ".")
    parts = tuple(part for part in dotted.split(".") if part)
    return "reference" in parts


def _expr_references(node: ast.AST | None, tainted: set[str]) -> bool:
    if node is None:
        return False
    if isinstance(node, ast.Name):
        return node.id in tainted
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return _is_reference_text(node.value)
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return any(_expr_references(item, tainted) for item in node.elts)
    if isinstance(node, ast.Dict):
        return any(_expr_references(item, tainted) for item in (*node.keys, *node.values))
    if isinstance(node, ast.JoinedStr):
        return any(_expr_references(item, tainted) for item in node.values)
    if isinstance(node, ast.FormattedValue):
        return _expr_references(node.value, tainted)
    if isinstance(node, ast.BinOp):
        return _expr_references(node.left, tainted) or _expr_references(node.right, tainted)
    if isinstance(node, ast.UnaryOp):
        return _expr_references(node.operand, tainted)
    if isinstance(node, ast.Attribute):
        return _expr_references(node.value, tainted)
    if isinstance(node, ast.Subscript):
        return _expr_references(node.value, tainted) or _expr_references(node.slice, tainted)
    if isinstance(node, ast.Call):
        return _expr_references(node.func, tainted) or any(
            _expr_references(item, tainted)
            for item in (*node.args, *(item.value for item in node.keywords))
        )
    return any(_expr_references(child, tainted) for child in ast.iter_child_nodes(node))


def _tainted_names(tree: ast.AST) -> set[str]:
    assignments: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append((target.id, node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.value:
            assignments.append((node.target.id, node.value))
    tainted: set[str] = set()
    for _ in range(len(assignments) + 1):
        changed = False
        for name, value in assignments:
            if name not in tainted and _expr_references(value, tainted):
                tainted.add(name)
                changed = True
        if not changed:
            break
    return tainted


def _verified_selection_names(tree: ast.AST, resolver: _Resolver) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            arguments = (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)
            for argument in arguments:
                annotation = _resolve_dotted_name(
                    _annotation_name(argument.annotation), resolver
                )
                if annotation in _VERIFIED_SELECTION_TYPES:
                    names.add(argument.arg)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            annotation = _resolve_dotted_name(_annotation_name(node.annotation), resolver)
            if annotation in _VERIFIED_SELECTION_TYPES:
                names.add(node.target.id)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value:
            if (
                isinstance(node.value, ast.Call)
                and resolver.resolve_callable(node.value.func)
                in _VERIFIED_SELECTION_FACTORIES
            ):
                targets = node.targets if isinstance(node, ast.Assign) else [node.target]
                names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


def _verified_root_vars(
    tree: ast.AST,
    resolver: _Resolver,
    selections: set[str],
) -> dict[str, str]:
    roots: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)) or node.value is None:
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _terminal(resolver.resolve_callable(value.func)) != "verified_root":
            continue
        receiver = value.func.value if isinstance(value.func, ast.Attribute) else None
        asset_id = _literal_string(value.args[0]) if value.args else None
        if not isinstance(receiver, ast.Name) or receiver.id not in selections or asset_id not in MODEL_ASSET_IDS:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                roots[target.id] = asset_id
    return roots


def _verified_model_source(
    node: ast.AST | None,
    selections: set[str],
    root_vars: dict[str, str],
) -> bool:
    if isinstance(node, ast.Name):
        return node.id in root_vars
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
        return False
    if node.func.attr != "verified_root" or not isinstance(node.func.value, ast.Name):
        return False
    asset_id = _literal_string(node.args[0]) if node.args else None
    return node.func.value.id in selections and asset_id in MODEL_ASSET_IDS


def _is_pretrained_call(name: str) -> bool:
    return name.endswith(".from_pretrained") and name.startswith(_PRETRAINED_MODULES)


def _dataset_transport_allowed(path: str, function: str, call: ast.Call) -> bool:
    if path != DATASET_TRANSPORT_PATH or function != DATASET_TRANSPORT_FUNCTION:
        return False
    repo_id = _literal_string(_call_argument(call, "repo_id", 0))
    revision = _literal_string(_keyword(call, "revision"))
    repo_type = _literal_string(_keyword(call, "repo_type"))
    return (
        repo_id == DATASET_REPO_ID
        and revision is not None
        and _COMMIT_RE.fullmatch(revision) is not None
        and repo_type == "dataset"
    )


def _enclosing_functions(tree: ast.AST) -> dict[int, str]:
    result: dict[int, str] = {}

    class Visitor(ast.NodeVisitor):
        def __init__(self) -> None:
            self.stack: list[str] = []

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            self.stack.append(node.name)
            self.generic_visit(node)
            self.stack.pop()

        def visit_Call(self, node: ast.Call) -> None:
            result[id(node)] = self.stack[-1] if self.stack else ""
            self.generic_visit(node)

    Visitor().visit(tree)
    return result


def _safe_git_metadata_call(node: ast.Call, path: str) -> bool:
    if not path.startswith("tests/"):
        return False
    if not node.args:
        return False
    command = node.args[0]
    if not isinstance(command, (ast.Tuple, ast.List)):
        return False
    literals = tuple(_literal_string(item) for item in command.elts)
    if not literals or literals[0] != "git":
        return False
    values = tuple(item for item in literals if item is not None)
    return any(
        marker in values
        for marker in (
            "add",
            "check-ignore",
            "commit",
            "config",
            "get-url",
            "init",
            "ls-files",
            "rev-parse",
            "set-url",
            "status",
        )
    )


def scan_source(source: str, path: str) -> tuple[BoundaryViolation, ...]:
    """Return deterministic policy violations for one Python source string."""

    tree = ast.parse(source, filename=path)
    resolver = _Resolver(tree)
    tainted = _tainted_names(tree)
    selections = _verified_selection_names(tree, resolver)
    root_vars = _verified_root_vars(tree, resolver, selections)
    functions = _enclosing_functions(tree)
    violations: list[BoundaryViolation] = []

    def add(node: ast.AST, code: str, detail: str) -> None:
        violations.append(BoundaryViolation(path, getattr(node, "lineno", 1), code, detail))

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(_is_reference_text(item.name) for item in node.names):
                add(node, "reference_import", "engineering code imports reference code")
        elif isinstance(node, ast.ImportFrom):
            if node.module and _is_reference_text(node.module):
                add(node, "reference_import", "engineering code imports reference code")
        elif isinstance(node, ast.Call):
            name = resolver.resolve_callable(node.func)
            if _is_pretrained_call(name):
                source_node = _call_argument(node, "pretrained_model_name_or_path", 0)
                local_only = _keyword(node, "local_files_only")
                remote_code = _keyword(node, "trust_remote_code")
                if not _verified_model_source(source_node, selections, root_vars):
                    add(node, "unverified_model_source", "from_pretrained source is not an A001 verified model root")
                if not isinstance(local_only, ast.Constant) or local_only.value is not True:
                    add(node, "model_network_enabled", "from_pretrained must set local_files_only=True")
                if remote_code is not None and (
                    not isinstance(remote_code, ast.Constant) or remote_code.value is not False
                ):
                    add(node, "remote_code_enabled", "trust_remote_code must be absent or False")
                if _keyword(node, "cache_dir") is not None:
                    add(node, "model_cache_enabled", "verified local models must not use a cache directory")
            elif name in _DOWNLOAD_CALLS and not _dataset_transport_allowed(
                path, functions.get(id(node), ""), node
            ):
                add(node, "forbidden_download", f"download API is not the locked dataset transport: {name}")

            references = _expr_references(node, tainted)
            if name in _DYNAMIC_IMPORT_CALLS and references:
                add(node, "reference_dynamic_import", "dynamically imports reference code")
            if name in _DYNAMIC_CODE_CALLS and references:
                add(node, "reference_dynamic_exec", "dynamically executes reference code")
            is_process = name in _PROCESS_EXACT or name.startswith(_PROCESS_PREFIXES)
            if is_process and references and not _safe_git_metadata_call(node, path):
                add(node, "reference_process_exec", "passes a reference path or code to a process")
            if name in {"sys.path.append", "sys.path.extend", "sys.path.insert", "site.addsitedir"} and references:
                add(node, "reference_search_path", "injects reference into Python search paths")
        elif isinstance(node, ast.AugAssign):
            target = resolver.resolve_callable(node.target)
            if target == "sys.path" and _expr_references(node.value, tainted):
                add(node, "reference_search_path", "injects reference into Python search paths")
        elif isinstance(node, ast.Assign) and _expr_references(node.value, tainted):
            for target_node in node.targets:
                target = resolver.resolve_callable(target_node)
                if target == "sys.path" or (
                    isinstance(target_node, ast.Subscript)
                    and resolver.resolve_callable(target_node.value) == "sys.path"
                ):
                    add(node, "reference_search_path", "injects reference into Python search paths")

    return tuple(sorted(violations, key=lambda item: (item.path, item.line, item.code, item.detail)))


def _assert_safe_source_path(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SourceBoundaryError("source path escapes repository root") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise SourceBoundaryError(f"source path contains symlink: {relative.as_posix()}")
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SourceBoundaryError(f"source path is missing or escapes root: {relative.as_posix()}") from exc


def _read_source(root: Path, path: Path) -> str:
    _assert_safe_source_path(root, path)
    try:
        before = path.lstat()
    except OSError as exc:
        raise SourceBoundaryError("source path cannot be inspected") from exc
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise SourceBoundaryError("source path is a symlink or is not a regular file")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise SourceBoundaryError("source path cannot be opened without following links") from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SourceBoundaryError("source identity changed before read")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            source = handle.read()
        after = path.lstat()
    except (OSError, UnicodeError) as exc:
        raise SourceBoundaryError("source path changed or could not be decoded") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    ):
        raise SourceBoundaryError("source identity changed during read")
    return source


def python_sources(root: Path) -> tuple[Path, ...]:
    """List scanned sources without following any symlink or out-of-root path."""

    resolved_root = root.resolve(strict=True)
    paths: list[Path] = []
    for relative_base in (Path("src"), Path("tests"), Path("tools")):
        base = resolved_root / relative_base
        if not base.exists():
            continue
        _assert_safe_source_path(resolved_root, base)
        pending = [base]
        while pending:
            directory = pending.pop()
            try:
                entries = tuple(os.scandir(directory))
            except OSError as exc:
                raise SourceBoundaryError("source directory cannot be enumerated") from exc
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise SourceBoundaryError(
                        f"source tree contains symlink: {path.relative_to(resolved_root)}"
                    )
                if entry.is_dir(follow_symlinks=False):
                    if entry.name != "__pycache__":
                        pending.append(path)
                elif entry.name.endswith(".py"):
                    _assert_safe_source_path(resolved_root, path)
                    paths.append(path)
    return tuple(sorted(paths))


def scan_file(root: Path, path: Path) -> tuple[BoundaryViolation, ...]:
    """Scan one root-confined regular Python file without following symlinks."""

    resolved_root = root.resolve(strict=True)
    candidate = path if path.is_absolute() else resolved_root / path
    _assert_safe_source_path(resolved_root, candidate)
    relative = candidate.relative_to(resolved_root).as_posix()
    return scan_source(_read_source(resolved_root, candidate), relative)


def scan_repository(root: Path) -> tuple[BoundaryViolation, ...]:
    """Scan tracked engineering source roots without accessing ignored assets."""

    resolved_root = root.resolve(strict=True)
    violations: list[BoundaryViolation] = []
    for path in python_sources(resolved_root):
        violations.extend(scan_file(resolved_root, path))
    return tuple(sorted(violations, key=lambda item: (item.path, item.line, item.code, item.detail)))
