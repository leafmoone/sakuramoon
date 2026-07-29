"""Static, fail-closed checks for local model and reference execution boundaries."""

from __future__ import annotations

import ast
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

MODEL_ASSET_IDS = frozenset({"qwen_text_encoder", "mage_vae"})
DATASET_REPO_ID = "leafmoone/webdataset_danbooru"
DATASET_TRANSPORT_PATH = "src/sakuramoon/data/modelscope.py"
DATASET_TRANSPORT_FUNCTION = "fetch_dataset_shard"
DATASET_DOWNLOAD_CALL = "modelscope.hub.snapshot_download.snapshot_download"

_MODEL_LOADER_PREFIXES = ("diffusers.", "modelscope.", "transformers.")
_DOWNLOAD_CALLS = frozenset(
    {
        "diffusers.utils.hub._get_model_file",
        "huggingface_hub._snapshot_download.snapshot_download",
        "huggingface_hub.file_download.hf_hub_download",
        "huggingface_hub.hf_hub_download",
        "huggingface_hub.snapshot_download",
        "modelscope.hub.file_download.model_file_download",
        DATASET_DOWNLOAD_CALL,
        "modelscope.model_file_download",
        "modelscope.snapshot_download",
        "transformers.utils.hub.cached_file",
        "transformers.utils.hub.cached_files",
        "transformers.utils.hub.get_file_from_repo",
    }
)
_SELECTION_FACTORIES = frozenset(
    {
        "sakuramoon.assets.inspect.require_runtime_assets_ready",
        "sakuramoon.assets.require_runtime_assets_ready",
    }
)
_SELECTION_GATES = frozenset(
    {
        "sakuramoon.assets.inspect.require_verified_selection",
        "sakuramoon.assets.require_verified_selection",
    }
)
_CONFIG_FACTORIES = frozenset(
    {
        "sakuramoon.config.load.load_config",
        "sakuramoon.config.load_config",
    }
)
_DYNAMIC_IMPORT_CALLS = frozenset(
    {
        "builtins.__import__",
        "importlib.import_module",
        "runpy.run_module",
    }
)
_DYNAMIC_CODE_CALLS = frozenset(
    {
        "builtins.compile",
        "builtins.eval",
        "builtins.exec",
        "builtins.execfile",
        "importlib.machinery.SourceFileLoader",
        "importlib.util.spec_from_file_location",
        "runpy.run_path",
    }
)
_PROCESS_PREFIXES = (
    "asyncio.create_subprocess_",
    "os.exec",
    "os.posix_spawn",
    "os.spawn",
    "subprocess.",
)
_PROCESS_EXACT = frozenset({"os.popen", "os.system"})
_SEARCH_PATH_CALLS = frozenset(
    {
        "site.addpackage",
        "site.addsitedir",
        "sys.path.append",
        "sys.path.extend",
        "sys.path.insert",
    }
)
_REF = "reference"
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])reference(?![A-Za-z0-9_])", re.IGNORECASE
)
_SKIPPED_SOURCE_DIRECTORIES = frozenset({".ipynb_checkpoints", "__pycache__"})


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


@dataclass(frozen=True)
class _Callable:
    name: str
    bound_args: tuple[_Fact, ...] = ()
    bound_keywords: tuple[tuple[str, _Fact], ...] = ()
    unknown_keywords: bool = False


@dataclass(frozen=True)
class _Fact:
    taints: frozenset[str] = frozenset()
    callable: _Callable | None = None
    selection: bool = False
    model_root: str | None = None
    config_path: tuple[str, ...] | None = None
    object_place: str | None = None
    literal: str | bool | None = None
    literal_known: bool = False


@dataclass(frozen=True)
class _FunctionSummary:
    parameters: tuple[str, ...]
    return_taints: frozenset[str]
    sink_parameters: tuple[tuple[str, tuple[int, ...]], ...]


def _empty_string_set() -> set[str]:
    return set()


def _empty_sink_map() -> dict[str, set[int]]:
    return {}


@dataclass
class _FunctionContext:
    name: str
    parameters: tuple[str, ...]
    return_taints: set[str] = field(default_factory=_empty_string_set)
    sink_parameters: dict[str, set[int]] = field(default_factory=_empty_sink_map)


class _Environment:
    def __init__(self, values: dict[str, _Fact] | None = None) -> None:
        self.values = {} if values is None else dict(values)

    def clone(self) -> _Environment:
        return _Environment(self.values)

    def get(self, place: str) -> _Fact:
        direct = self.values.get(place)
        if direct is not None:
            taints = set(direct.taints)
            origin = direct.object_place or place
            for key, child in self.values.items():
                if key.startswith((f"{origin}.", f"{origin}[")):
                    taints.update(child.taints)
            return replace(direct, taints=frozenset(taints))
        base = place.split(".", 1)[0].split("[", 1)[0]
        return self.values.get(base, _Fact())

    def assign(self, place: str, fact: _Fact) -> None:
        prefix_dot = f"{place}."
        prefix_item = f"{place}["
        for key in tuple(self.values):
            if key == place or key.startswith((prefix_dot, prefix_item)):
                del self.values[key]
        self.values[place] = fact


def _is_reference_text(value: str) -> bool:
    return _REFERENCE_RE.search(value.replace("\\", "/")) is not None


def _literal(node: ast.AST | None) -> str | bool | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, bool)):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _literal(node.left)
        right = _literal(node.right)
        if isinstance(left, str) and isinstance(right, str):
            return left + right
    if isinstance(node, ast.JoinedStr):
        values: list[str] = []
        for item in node.values:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, str):
                return None
            values.append(item.value)
        return "".join(values)
    return None


def _fact_literal(value: str | bool | None, *, known: bool = True) -> _Fact:
    taints: frozenset[str] = (
        frozenset({_REF})
        if isinstance(value, str) and _is_reference_text(value)
        else frozenset()
    )
    return _Fact(taints=taints, literal=value, literal_known=known)


def _union_taints(values: Iterable[frozenset[str]]) -> frozenset[str]:
    result: set[str] = set()
    for value in values:
        result.update(value)
    return frozenset(result)


def _merge_facts(left: _Fact, right: _Fact) -> _Fact:
    same_callable = left.callable if left.callable == right.callable else None
    same_root = left.model_root if left.model_root == right.model_root else None
    same_config = left.config_path if left.config_path == right.config_path else None
    literal_known = left.literal_known and right.literal_known and left.literal == right.literal
    return _Fact(
        taints=left.taints | right.taints,
        callable=same_callable,
        selection=left.selection and right.selection,
        model_root=same_root,
        config_path=same_config,
        object_place=left.object_place if left.object_place == right.object_place else None,
        literal=left.literal if literal_known else None,
        literal_known=literal_known,
    )


def _merge_environments(*environments: _Environment) -> _Environment:
    keys: set[str] = set()
    for environment in environments:
        keys.update(environment.values)
    merged = _Environment()
    for key in keys:
        facts = [environment.values.get(key, _Fact()) for environment in environments]
        value = facts[0]
        for fact in facts[1:]:
            value = _merge_facts(value, fact)
        merged.values[key] = value
    return merged


def _parameter_index(marker: str) -> int | None:
    if not marker.startswith("parameter:"):
        return None
    try:
        return int(marker.removeprefix("parameter:"))
    except ValueError:
        return None


class _Analyzer:
    def __init__(self, path: str) -> None:
        self.path = path
        self.violations: list[BoundaryViolation] = []
        self.summaries: dict[str, _FunctionSummary] = {}
        self._scope_serial = 0
        self.context = _FunctionContext("<module>", ())

    def add(self, node: ast.AST, code: str, detail: str) -> None:
        self.violations.append(
            BoundaryViolation(self.path, getattr(node, "lineno", 1), code, detail)
        )

    def _place(self, node: ast.AST, env: _Environment) -> str | None:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            base_fact = self._eval(node.value, env)
            base = (
                base_fact.callable.name
                if base_fact.callable is not None
                else base_fact.object_place or self._place(node.value, env)
            )
            return f"{base}.{node.attr}" if base else None
        if isinstance(node, ast.Subscript):
            base_fact = self._eval(node.value, env)
            base = (
                base_fact.callable.name
                if base_fact.callable is not None
                else base_fact.object_place or self._place(node.value, env)
            )
            if base is None:
                return None
            if isinstance(node.slice, ast.Slice):
                return f"{base}[:]"
            key = _literal(node.slice)
            return f"{base}[{key!r}]" if isinstance(key, (str, bool)) else f"{base}[*]"
        return None

    def _attribute_chain(self, node: ast.Attribute) -> tuple[str, tuple[str, ...]] | None:
        parts: list[str] = []
        current: ast.AST = node
        while isinstance(current, ast.Attribute):
            parts.append(current.attr)
            current = current.value
        if not isinstance(current, ast.Name):
            return None
        return current.id, tuple(reversed(parts))

    def _eval(self, node: ast.AST | None, env: _Environment) -> _Fact:
        if node is None:
            return _Fact()
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (str, bool)) or node.value is None:
                return _fact_literal(node.value)
            return _Fact(literal_known=True)
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.Attribute):
            place = self._place(node, env)
            if place is not None and place in env.values:
                return env.values[place]
            chain = self._attribute_chain(node)
            if chain is not None:
                root, attributes = chain
                root_fact = env.get(root)
                if root_fact.config_path is not None:
                    return _Fact(
                        taints=root_fact.taints,
                        config_path=(*root_fact.config_path, *attributes),
                    )
            base = self._eval(node.value, env)
            if base.callable is not None:
                return _Fact(
                    taints=base.taints,
                    callable=_Callable(f"{base.callable.name}.{node.attr}"),
                )
            return _Fact(taints=base.taints)
        if isinstance(node, ast.Subscript):
            place = self._place(node, env)
            if place is not None and place in env.values:
                return env.values[place]
            if (
                isinstance(node.value, ast.Attribute)
                and node.value.attr == "__dict__"
            ):
                base = self._eval(node.value.value, env)
                key = _literal(node.slice)
                if base.callable is not None and isinstance(key, str):
                    return _Fact(callable=_Callable(f"{base.callable.name}.{key}"))
            base = self._eval(node.value, env)
            key = self._eval(node.slice, env)
            return _Fact(taints=base.taints | key.taints)
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            taints = _union_taints(self._eval(item, env).taints for item in node.elts)
            return _Fact(taints=taints)
        if isinstance(node, ast.Dict):
            items = (*node.keys, *node.values)
            taints = _union_taints(
                self._eval(item, env).taints for item in items if item is not None
            )
            return _Fact(taints=taints)
        if isinstance(node, ast.JoinedStr):
            literal = _literal(node)
            if isinstance(literal, str):
                return _fact_literal(literal)
            return _Fact(
                taints=_union_taints(self._eval(item, env).taints for item in node.values)
            )
        if isinstance(node, ast.FormattedValue):
            return self._eval(node.value, env)
        if isinstance(node, ast.BinOp):
            literal = _literal(node)
            if isinstance(literal, str):
                return _fact_literal(literal)
            return _Fact(taints=self._eval(node.left, env).taints | self._eval(node.right, env).taints)
        if isinstance(node, ast.UnaryOp):
            return self._eval(node.operand, env)
        if isinstance(node, ast.IfExp):
            return _merge_facts(self._eval(node.body, env), self._eval(node.orelse, env))
        if isinstance(node, ast.NamedExpr):
            fact = self._eval(node.value, env)
            self._assign(node.target, fact, env)
            return fact
        if isinstance(node, ast.Call):
            return self._eval_call(node, env)
        taints = _union_taints(
            self._eval(child, env).taints for child in ast.iter_child_nodes(node)
        )
        return _Fact(taints=taints)

    def _callable(self, node: ast.AST, env: _Environment) -> _Callable | None:
        fact = self._eval(node, env)
        return fact.callable

    def _keywords(
        self, call: ast.Call, env: _Environment
    ) -> tuple[dict[str, _Fact], bool, frozenset[str]]:
        values: dict[str, _Fact] = {}
        unknown = False
        expansion_taints: set[str] = set()
        for keyword in call.keywords:
            if keyword.arg is not None:
                values[keyword.arg] = self._eval(keyword.value, env)
                continue
            expansion = self._eval(keyword.value, env)
            expansion_taints.update(expansion.taints)
            if not isinstance(keyword.value, ast.Dict):
                unknown = True
                continue
            for key_node, value_node in zip(keyword.value.keys, keyword.value.values, strict=True):
                key = _literal(key_node)
                if not isinstance(key, str):
                    unknown = True
                    continue
                values[key] = self._eval(value_node, env)
        return values, unknown, frozenset(expansion_taints)

    def _effective_arguments(
        self, call: ast.Call, callable_value: _Callable, env: _Environment
    ) -> tuple[tuple[_Fact, ...], dict[str, _Fact], bool, frozenset[str]]:
        positional = (*callable_value.bound_args, *(self._eval(item, env) for item in call.args))
        keywords = dict(callable_value.bound_keywords)
        current, unknown, expansion_taints = self._keywords(call, env)
        keywords.update(current)
        return positional, keywords, callable_value.unknown_keywords or unknown, expansion_taints

    def _record_sink(self, node: ast.AST, code: str, taints: frozenset[str], detail: str) -> None:
        if _REF in taints:
            self.add(node, code, detail)
        for marker in taints:
            index = _parameter_index(marker)
            if index is not None:
                self.context.sink_parameters.setdefault(code, set()).add(index)

    def _actual_parameter_facts(
        self,
        summary: _FunctionSummary,
        positional: tuple[_Fact, ...],
        keywords: dict[str, _Fact],
    ) -> tuple[_Fact, ...]:
        facts: list[_Fact] = []
        for index, name in enumerate(summary.parameters):
            if index < len(positional):
                facts.append(positional[index])
            else:
                facts.append(keywords.get(name, _Fact()))
        return tuple(facts)

    def _local_function_call(
        self,
        call: ast.Call,
        callable_value: _Callable,
        positional: tuple[_Fact, ...],
        keywords: dict[str, _Fact],
    ) -> _Fact | None:
        summary = self.summaries.get(callable_value.name)
        if summary is None:
            return None
        actual = self._actual_parameter_facts(summary, positional, keywords)
        result_taints: set[str] = set()
        for marker in summary.return_taints:
            index = _parameter_index(marker)
            if index is None:
                result_taints.add(marker)
            elif index < len(actual):
                result_taints.update(actual[index].taints)
        for code, indices in summary.sink_parameters:
            sink_taints = _union_taints(
                actual[index].taints for index in indices if index < len(actual)
            )
            self._record_sink(call, code, sink_taints, "reference reaches a sink through a helper call")
        return _Fact(taints=frozenset(result_taints))

    def _model_loader(self, name: str) -> bool:
        while name.endswith(".__call__"):
            name = name.removesuffix(".__call__")
        return name.endswith(".from_pretrained") and name.startswith(_MODEL_LOADER_PREFIXES)

    def _dataset_transport_allowed(
        self,
        call: ast.Call,
        name: str,
        positional: tuple[_Fact, ...],
        keywords: dict[str, _Fact],
        unknown_keywords: bool,
    ) -> bool:
        if (
            name != DATASET_DOWNLOAD_CALL
            or self.path != DATASET_TRANSPORT_PATH
            or self.context.name != DATASET_TRANSPORT_FUNCTION
            or unknown_keywords
        ):
            return False
        repo = keywords.get("repo_id", positional[0] if positional else _Fact())
        revision = keywords.get("revision", _Fact())
        repo_type = keywords.get("repo_type", _Fact())
        return (
            repo.literal_known
            and repo.literal == DATASET_REPO_ID
            and revision.config_path == ("config", "data", "source", "revision")
            and repo_type.literal_known
            and repo_type.literal == "dataset"
        )

    def _git_shape(self, call: ast.Call) -> tuple[str | None, ...] | None:
        if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
            return None
        shape: list[str | None] = []
        for item in call.args[0].elts:
            value = _literal(item)
            shape.append(value if isinstance(value, str) else None)
        return tuple(shape)

    def _safe_test_git(self, call: ast.Call, name: str) -> bool:
        if name != "subprocess.run":
            return False
        shape = self._git_shape(call)
        if shape is None or not shape or shape[0] != "git" or "-c" in shape:
            return False
        location = (self.path, self.context.name)
        if location == (
            "tests/contracts/assets/test_asset_boundary.py",
            "test_all_asset_roots_are_git_ignored_and_payloads_are_not_tracked",
        ):
            return shape == ("git", "ls-files", "model", "db", "reference")
        if location == ("tests/unit/assets/conftest.py", "make_reference"):
            allowed = {
                ("git", "init", "-q", None),
                ("git", "-C", None, "remote", "add", "origin", None),
                ("git", "-C", None, "add", "."),
                ("git", "-C", None, "config", "user.name", None),
                ("git", "-C", None, "config", "user.email", None),
                ("git", "-C", None, "commit", "-qm", "fixture"),
                ("git", "-C", None, "rev-parse", "HEAD"),
            }
            return shape in allowed
        if location == (
            "tests/unit/assets/test_inspect.py",
            "test_reference_origin_diagnostic_redacts_credentials",
        ):
            return shape == ("git", "-C", None, "remote", "set-url", "origin", None)
        if location == (
            "tests/unit/assets/test_inspect.py",
            "test_reference_git_audit_disables_hostile_local_configuration",
        ):
            return len(shape) == 6 and shape[:4] == ("git", "-C", None, "config") and shape[4] in {
                "core.fsmonitor",
                "core.hooksPath",
                "core.pager",
                "diff.external",
                "interactive.diffFilter",
                "pager.status",
            } and shape[5] is None
        return False

    def _eval_call(self, call: ast.Call, env: _Environment) -> _Fact:
        if isinstance(call.func, ast.Attribute) and call.func.attr == "verified_root":
            receiver = self._eval(call.func.value, env)
            asset_id = _literal(call.args[0]) if call.args else None
            if (
                receiver.selection
                and isinstance(asset_id, str)
                and asset_id in MODEL_ASSET_IDS
                and not call.keywords
                and len(call.args) == 1
            ):
                return _Fact(model_root=asset_id)

        callable_value = self._callable(call.func, env)
        if callable_value is None:
            taints = self._eval(call.func, env).taints | _union_taints(
                (
                    *(self._eval(item, env).taints for item in call.args),
                    *(self._eval(item.value, env).taints for item in call.keywords),
                )
            )
            if isinstance(call.func, ast.Attribute) and call.func.attr in {
                "append",
                "extend",
                "insert",
                "update",
            }:
                receiver = self._place(call.func.value, env)
                if receiver is not None:
                    current = env.get(receiver)
                    env.assign(receiver, _Fact(taints=current.taints | taints))
            return _Fact(taints=taints)
        name = callable_value.name

        if name in {"builtins.getattr", "getattr"} and len(call.args) >= 2:
            base = self._callable(call.args[0], env)
            attribute_fact = self._eval(call.args[1], env)
            attribute = (
                attribute_fact.literal
                if attribute_fact.literal_known
                and isinstance(attribute_fact.literal, str)
                else None
            )
            if base is not None and isinstance(attribute, str):
                return _Fact(callable=_Callable(f"{base.name}.{attribute}"))
            if base is not None and base.name.startswith(
                (
                    *_MODEL_LOADER_PREFIXES,
                    "asyncio.",
                    "huggingface_hub.",
                    "os.",
                    "runpy.",
                    "site.",
                    "subprocess.",
                )
            ):
                return _Fact(callable=_Callable(f"{base.name}.*"))
            return _Fact()

        if name in {"functools.partial", "partial"} and call.args:
            target = self._callable(call.args[0], env)
            if target is None:
                return _Fact()
            keywords, unknown, _ = self._keywords(call, env)
            return _Fact(
                callable=_Callable(
                    target.name,
                    (*target.bound_args, *(self._eval(item, env) for item in call.args[1:])),
                    (*target.bound_keywords, *keywords.items()),
                    target.unknown_keywords or unknown,
                )
            )

        positional, keywords, unknown_keywords, expansion_taints = self._effective_arguments(
            call, callable_value, env
        )
        all_taints = _union_taints(
            (
                *(item.taints for item in positional),
                *(item.taints for item in keywords.values()),
                expansion_taints,
            )
        )

        if name in _SELECTION_FACTORIES:
            return _Fact(selection=True)
        if name in _SELECTION_GATES:
            if len(positional) == 1 and not keywords and not unknown_keywords:
                return _Fact(selection=True)
            return _Fact()
        if name in _CONFIG_FACTORIES:
            return _Fact(config_path=())

        local = self._local_function_call(call, callable_value, positional, keywords)
        if local is not None:
            return local

        canonical = name
        while canonical.endswith(".__call__"):
            canonical = canonical.removesuffix(".__call__")
        if self._model_loader(name):
            source = keywords.get(
                "pretrained_model_name_or_path",
                keywords.get(
                    "model_name_or_path",
                    keywords.get("model_id", positional[0] if positional else _Fact()),
                ),
            )
            if source.model_root not in MODEL_ASSET_IDS:
                self.add(call, "unverified_model_source", "model source is not a live A001 verified root")
            local_only = keywords.get("local_files_only", _Fact())
            if not local_only.literal_known or local_only.literal is not True:
                self.add(call, "model_network_enabled", "model loader must set local_files_only=True")
            if "trust_remote_code" in keywords:
                self.add(call, "remote_code_option_forbidden", "trust_remote_code is forbidden")
            if "cache_dir" in keywords:
                self.add(call, "model_cache_option_forbidden", "cache_dir is forbidden")
            if unknown_keywords:
                self.add(call, "unknown_model_loader_kwargs", "model loader contains unprovable **kwargs")
        elif canonical.endswith(".*") and canonical.startswith(
            (*_MODEL_LOADER_PREFIXES, "huggingface_hub.")
        ):
            self.add(call, "ambiguous_sensitive_callable", "computed callable cannot be proven safe")
        elif canonical in _DOWNLOAD_CALLS and not self._dataset_transport_allowed(
            call, canonical, positional, keywords, unknown_keywords
        ):
            self.add(call, "forbidden_download", f"download API is not the locked dataset transport: {canonical}")

        if canonical in _DYNAMIC_IMPORT_CALLS:
            self._record_sink(call, "reference_dynamic_import", all_taints, "dynamically imports reference code")
        if canonical in _DYNAMIC_CODE_CALLS:
            self._record_sink(call, "reference_dynamic_exec", all_taints, "dynamically executes reference code")
        is_process = canonical in _PROCESS_EXACT or canonical.startswith(_PROCESS_PREFIXES)
        if canonical.endswith(".*") and canonical.startswith(
            ("asyncio.", "os.", "subprocess.")
        ):
            is_process = True
        if is_process and not self._safe_test_git(call, canonical):
            self._record_sink(call, "reference_process_exec", all_taints, "passes reference data to a process")
        if canonical in _SEARCH_PATH_CALLS:
            self._record_sink(call, "reference_search_path", all_taints, "injects reference into Python search paths")

        if isinstance(call.func, ast.Attribute) and call.func.attr in {"append", "extend", "insert", "update"}:
            receiver = self._place(call.func.value, env)
            if receiver is not None:
                current = env.get(receiver)
                env.assign(receiver, _Fact(taints=current.taints | all_taints))
        if canonical in {"builtins.setattr", "setattr"} and len(call.args) >= 3:
            receiver = self._place(call.args[0], env)
            attribute = _literal(call.args[1])
            if receiver is not None and isinstance(attribute, str):
                env.assign(f"{receiver}.{attribute}", self._eval(call.args[2], env))

        return _Fact(taints=all_taints)

    def _assign(self, target: ast.AST, fact: _Fact, env: _Environment) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._assign(item, fact, env)
            return
        place = self._place(target, env)
        if place is None:
            return
        if place == "sys.path" or place.startswith("sys.path["):
            self._record_sink(target, "reference_search_path", fact.taints, "assigns reference into sys.path")
        if (
            isinstance(target, ast.Name)
            and fact.object_place is None
            and fact.callable is None
            and fact.model_root is None
            and fact.config_path is None
            and not fact.literal_known
        ):
            fact = replace(fact, object_place=place)
        env.assign(place, fact)

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, outer: _Environment) -> _Fact:
        self._scope_serial += 1
        identifier = f"local:{self.path}:{node.lineno}:{node.name}:{self._scope_serial}"
        parameters = (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        parameter_names = tuple(item.arg for item in parameters)
        env = outer.clone()
        for index, parameter in enumerate(parameter_names):
            env.assign(parameter, _Fact(taints=frozenset({f"parameter:{index}"})))
        if node.args.vararg is not None:
            index = len(parameter_names)
            parameter_names = (*parameter_names, node.args.vararg.arg)
            env.assign(node.args.vararg.arg, _Fact(taints=frozenset({f"parameter:{index}"})))
        if node.args.kwarg is not None:
            index = len(parameter_names)
            parameter_names = (*parameter_names, node.args.kwarg.arg)
            env.assign(node.args.kwarg.arg, _Fact(taints=frozenset({f"parameter:{index}"})))
        previous = self.context
        self.context = _FunctionContext(node.name, parameter_names)
        self._block(node.body, env)
        summary = _FunctionSummary(
            parameters=parameter_names,
            return_taints=frozenset(self.context.return_taints),
            sink_parameters=tuple(
                sorted((code, tuple(sorted(indices))) for code, indices in self.context.sink_parameters.items())
            ),
        )
        self.context = previous
        self.summaries[identifier] = summary
        return _Fact(callable=_Callable(identifier))

    def _statement(self, node: ast.stmt, env: _Environment) -> _Environment:
        if isinstance(node, ast.Import):
            for item in node.names:
                local = item.asname or item.name.split(".", 1)[0]
                module = item.name if item.asname else item.name.split(".", 1)[0]
                env.assign(local, _Fact(callable=_Callable(module)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            for item in node.names:
                if item.name != "*":
                    env.assign(item.asname or item.name, _Fact(callable=_Callable(f"{node.module}.{item.name}")))
            if _is_reference_text(node.module):
                self.add(node, "reference_import", "engineering code imports reference code")
        elif isinstance(node, ast.Assign):
            fact = self._eval(node.value, env)
            for target in node.targets:
                self._assign(target, fact, env)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._assign(node.target, self._eval(node.value, env), env)
        elif isinstance(node, ast.AugAssign):
            fact = _merge_facts(self._eval(node.target, env), self._eval(node.value, env))
            self._assign(node.target, fact, env)
        elif isinstance(node, ast.Expr):
            self._eval(node.value, env)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            env.assign(node.name, self._function(node, env))
        elif isinstance(node, ast.ClassDef):
            class_env = env.clone()
            self._block(node.body, class_env)
            env.assign(node.name, _Fact(callable=_Callable(f"local-class:{node.name}:{node.lineno}")))
        elif isinstance(node, ast.Return):
            self.context.return_taints.update(self._eval(node.value, env).taints)
        elif isinstance(node, ast.If):
            self._eval(node.test, env)
            body = env.clone()
            other = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, other)
            env = _merge_environments(body, other)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            self._eval(node.iter, env)
            body = env.clone()
            self._assign(node.target, _Fact(), body)
            self._block(node.body, body)
            self._block(node.orelse, body)
            env = _merge_environments(env, body)
        elif isinstance(node, ast.While):
            self._eval(node.test, env)
            body = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, body)
            env = _merge_environments(env, body)
        elif isinstance(node, ast.Try):
            branches: list[_Environment] = []
            body = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, body)
            branches.append(body)
            for handler in node.handlers:
                branch = env.clone()
                if handler.name:
                    branch.assign(handler.name, _Fact())
                self._block(handler.body, branch)
                branches.append(branch)
            env = _merge_environments(*branches)
            self._block(node.finalbody, env)
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            for item in node.items:
                fact = self._eval(item.context_expr, env)
                if item.optional_vars is not None:
                    self._assign(item.optional_vars, fact, env)
            self._block(node.body, env)
        return env

    def _block(self, statements: list[ast.stmt], env: _Environment) -> _Environment:
        current = env
        for statement in statements:
            if isinstance(statement, ast.Import):
                for item in statement.names:
                    if _is_reference_text(item.name):
                        self.add(statement, "reference_import", "engineering code imports reference code")
            current = self._statement(statement, current)
        if current is not env:
            env.values = current.values
        return env

    def analyze(self, tree: ast.Module) -> tuple[BoundaryViolation, ...]:
        builtins = _Environment(
            {
                "__import__": _Fact(callable=_Callable("builtins.__import__")),
                "compile": _Fact(callable=_Callable("builtins.compile")),
                "eval": _Fact(callable=_Callable("builtins.eval")),
                "exec": _Fact(callable=_Callable("builtins.exec")),
                "getattr": _Fact(callable=_Callable("builtins.getattr")),
                "setattr": _Fact(callable=_Callable("builtins.setattr")),
            }
        )
        self._block(tree.body, builtins)
        return tuple(
            sorted(self.violations, key=lambda item: (item.path, item.line, item.code, item.detail))
        )


def scan_source(source: str, path: str) -> tuple[BoundaryViolation, ...]:
    """Return deterministic policy violations for one Python source string."""

    tree = ast.parse(source, filename=path)
    return _Analyzer(path).analyze(tree)


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
    if (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) != (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ):
        raise SourceBoundaryError("source identity changed during read")
    return source


def python_sources(root: Path) -> tuple[Path, ...]:
    """List scanned sources without following any symlink or ignored notebook state."""

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
                    if entry.name not in _SKIPPED_SOURCE_DIRECTORIES:
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
