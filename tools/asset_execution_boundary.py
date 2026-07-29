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
DATASET_TRANSPORT_PATH = "src/sakuramoon/data/modelscope.py"
DATASET_TRANSPORT_CLASS = "ModelScopeDatasetTransport"

_MODEL_LOADER_PREFIXES = ("diffusers.", "modelscope.", "transformers.")
_DOWNLOAD_CALLS = frozenset(
    {
        "diffusers.utils.hub._get_model_file",
        "huggingface_hub._snapshot_download.snapshot_download",
        "huggingface_hub.file_download.hf_hub_download",
        "huggingface_hub.hf_hub_download",
        "huggingface_hub.snapshot_download",
        "modelscope.hub.file_download.model_file_download",
        "modelscope.hub.snapshot_download.snapshot_download",
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
_DATASET_SELECTION_GATES = frozenset(
    {
        "sakuramoon.data.manifest.require_verified_dataset_manifest",
        "sakuramoon.data.require_verified_dataset_manifest",
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
_NETWORK_CALL_PREFIXES = (
    "aiohttp.",
    "http.client.",
    "httpx.",
    "requests.",
    "socket.",
    "urllib.request.",
)
_NETWORK_INSTANCE_CLASSES = frozenset(
    {"http.client.HTTPResponse", "http.client.HTTPSConnection"}
)
_NETWORK_MEMBER_NAMES = frozenset(
    {
        "close",
        "connect",
        "getheader",
        "getresponse",
        "putheader",
        "putrequest",
        "read",
        "request",
        "send",
        "set_debuglevel",
        "settimeout",
    }
)
_NETWORK_EXECUTION_MEMBER_NAMES = frozenset(
    {
        "connect",
        "getresponse",
        "putheader",
        "putrequest",
        "request",
        "send",
        "set_debuglevel",
    }
)
_DATASET_HTTP_HELPERS = frozenset(
    {
        "_close_response",
        "_follow_redirects",
        "_open_get",
        "_read_listing_once",
        "_read_response",
        "_request_headers",
    }
)
_DATASET_TARGET_FACTORIES = frozenset(
    {"_listing_target", "_redirect_target", "_shard_target"}
)
_DATASET_FORBIDDEN_IMPORT_PREFIXES = (
    "aiohttp",
    "httpx",
    "logging",
    "modelscope",
    "modelscope_hub",
    "requests",
    "socket",
    "traceback",
    "urllib.request",
)
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
_UNKNOWN_EXTERNAL = "unknown-external-provenance"
_REFERENCE_RE = re.compile(
    r"(?<![A-Za-z0-9_])reference(?![A-Za-z0-9_])", re.IGNORECASE
)
_SKIPPED_SOURCE_DIRECTORIES = frozenset({".ipynb_checkpoints", "__pycache__"})
_AUDITED_PARAMETER_SINK_FUNCTIONS = frozenset(
    {
        ("src/sakuramoon/assets/inspect.py", "_git"),
    }
)
_AUDITED_PARAMETER_CALLS = frozenset(
    {
        (
            "src/sakuramoon/assets/inspect.py",
            "_IdentityWeakRegistry",
            "contains",
            "reference",
        ),
    }
)


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
    capability_class: bool = False
    selection: bool = False
    model_root: str | None = None
    config_path: tuple[str, ...] | None = None
    dataset_manifest: bool = False
    dataset_selection: bool = False
    dataset_shard: bool = False
    object_place: str | None = None
    instance_class: str | None = None
    network_headers: bool = False
    network_query: str | None = None
    network_target: bool = False
    items: tuple[_Fact, ...] = ()
    mapping: tuple[tuple[str | bool | int | None, _Fact], ...] = ()
    literal: str | bool | None = None
    literal_known: bool = False


@dataclass(frozen=True)
class _FunctionSummary:
    parameters: tuple[str, ...]
    return_fact: _Fact
    sink_parameters: tuple[tuple[str, tuple[int, ...]], ...]


def _empty_sink_map() -> dict[str, set[int]]:
    return {}


@dataclass
class _FunctionContext:
    name: str
    parameters: tuple[str, ...]
    return_fact: _Fact | None = None
    sink_parameters: dict[str, set[int]] = field(default_factory=_empty_sink_map)
    direct_sink_parameters: dict[str, set[int]] = field(default_factory=_empty_sink_map)


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


def _sensitive_callable(value: _Callable | None) -> bool:
    if value is None:
        return False
    name = value.name
    while name.endswith(".__call__"):
        name = name.removesuffix(".__call__")
    return (
        (name.endswith(".from_pretrained") and name.startswith(_MODEL_LOADER_PREFIXES))
        or name in _DOWNLOAD_CALLS
        or name in _DYNAMIC_IMPORT_CALLS
        or name in _DYNAMIC_CODE_CALLS
        or name in _PROCESS_EXACT
        or name.startswith(_NETWORK_CALL_PREFIXES)
        or name.startswith(("asyncio.create_subprocess_", "os.exec", "os.posix_spawn", "os.spawn"))
        or name
        in {
            "subprocess.Popen",
            "subprocess.call",
            "subprocess.check_call",
            "subprocess.check_output",
            "subprocess.getoutput",
            "subprocess.getstatusoutput",
            "subprocess.run",
        }
        or name in _SEARCH_PATH_CALLS
    )


def _merge_facts(left: _Fact, right: _Fact) -> _Fact:
    same_callable = left.callable if left.callable == right.callable else None
    if same_callable is None and (
        _sensitive_callable(left.callable) or _sensitive_callable(right.callable)
    ):
        if left.callable is None:
            same_callable = right.callable
        elif right.callable is None:
            same_callable = left.callable
        else:
            same_callable = _Callable("ambiguous-sensitive.*")
    same_root = left.model_root if left.model_root == right.model_root else None
    same_config = left.config_path if left.config_path == right.config_path else None
    literal_known = left.literal_known and right.literal_known and left.literal == right.literal
    return _Fact(
        taints=left.taints | right.taints,
        callable=same_callable,
        capability_class=left.capability_class or right.capability_class,
        selection=left.selection and right.selection,
        model_root=same_root,
        config_path=same_config,
        dataset_manifest=left.dataset_manifest and right.dataset_manifest,
        dataset_selection=left.dataset_selection and right.dataset_selection,
        dataset_shard=left.dataset_shard and right.dataset_shard,
        object_place=left.object_place if left.object_place == right.object_place else None,
        instance_class=(
            left.instance_class
            if left.instance_class == right.instance_class
            else None
        ),
        network_headers=left.network_headers and right.network_headers,
        network_query=(
            left.network_query
            if left.network_query == right.network_query
            else None
        ),
        network_target=left.network_target and right.network_target,
        items=left.items if left.items == right.items else (),
        mapping=left.mapping if left.mapping == right.mapping else (),
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
        self.classes: dict[str, dict[str, _Fact]] = {}
        self.context = _FunctionContext("<module>", ())
        self.class_context: str | None = None

    def _function_identifier(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ) -> str:
        name = node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) else "<lambda>"
        return f"local:{self.path}:{node.lineno}:{node.col_offset}:{name}"

    def _class_identifier(self, node: ast.ClassDef) -> str:
        return f"local-class:{self.path}:{node.lineno}:{node.col_offset}:{node.name}"

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
            if base.dataset_selection and node.attr == "manifest":
                return _Fact(
                    taints=base.taints,
                    dataset_manifest=True,
                )
            if base.dataset_manifest and node.attr == "shard":
                return _Fact(
                    taints=base.taints,
                    callable=_Callable(
                        "audited-dataset-manifest.shard",
                        bound_args=(base,),
                    ),
                )
            if base.instance_class is not None:
                member = self.classes.get(base.instance_class, {}).get(node.attr)
                if member is not None:
                    if member.callable is not None:
                        return replace(
                            member,
                            callable=replace(
                                member.callable,
                                bound_args=(base, *member.callable.bound_args),
                            ),
                        )
                    return member
                if base.instance_class in _NETWORK_INSTANCE_CLASSES:
                    return _Fact(
                        taints=base.taints,
                        callable=_Callable(
                            f"{base.instance_class}.{node.attr}",
                            bound_args=(base,),
                        ),
                    )
            if base.callable is not None:
                member = self.classes.get(base.callable.name, {}).get(node.attr)
                if member is not None:
                    return member
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
            raw_key = node.slice.value if isinstance(node.slice, ast.Constant) else None
            if isinstance(raw_key, int) and not isinstance(raw_key, bool):
                index = raw_key if raw_key >= 0 else len(base.items) + raw_key
                if 0 <= index < len(base.items):
                    return base.items[index]
            if isinstance(raw_key, (str, bool, int)) or raw_key is None:
                for item_key, item_fact in base.mapping:
                    if item_key == raw_key:
                        return item_fact
            return _Fact(taints=base.taints | key.taints)
        if isinstance(node, (ast.List, ast.Set, ast.Tuple)):
            items = tuple(self._eval(item, env) for item in node.elts)
            taints = _union_taints(item.taints for item in items)
            return _Fact(taints=taints, items=items)
        if isinstance(node, ast.Dict):
            key_facts = tuple(
                self._eval(item, env) if item is not None else _Fact()
                for item in node.keys
            )
            value_facts = tuple(self._eval(item, env) for item in node.values)
            taints = _union_taints(
                item.taints for item in (*key_facts, *value_facts)
            )
            mapping: list[tuple[str | bool | int | None, _Fact]] = []
            for key_node, value_fact in zip(node.keys, value_facts, strict=True):
                if isinstance(key_node, ast.Constant) and (
                    isinstance(key_node.value, (str, bool, int))
                    or key_node.value is None
                ):
                    mapping.append((key_node.value, value_fact))
            return _Fact(taints=taints, mapping=tuple(mapping))
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)):
            comprehension_env = env.clone()
            aggregate = _Fact()
            for generator in node.generators:
                iterable = self._eval(generator.iter, comprehension_env)
                if iterable.items:
                    item_fact = iterable.items[0]
                    for candidate in iterable.items[1:]:
                        item_fact = _merge_facts(item_fact, candidate)
                else:
                    item_fact = iterable
                self._assign(generator.target, item_fact, comprehension_env)
                aggregate = _merge_facts(aggregate, iterable)
                for condition in generator.ifs:
                    aggregate = _merge_facts(
                        aggregate,
                        self._eval(condition, comprehension_env),
                    )
            if isinstance(node, ast.DictComp):
                result = _merge_facts(
                    self._eval(node.key, comprehension_env),
                    self._eval(node.value, comprehension_env),
                )
            else:
                result = self._eval(node.elt, comprehension_env)
            return replace(result, taints=result.taints | aggregate.taints)
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
        if isinstance(node, ast.Lambda):
            return self._lambda(node, env)
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

    def _record_sink(
        self,
        node: ast.AST,
        code: str,
        taints: frozenset[str],
        detail: str,
        *,
        direct: bool = True,
    ) -> None:
        if _REF in taints or _UNKNOWN_EXTERNAL in taints:
            self.add(node, code, detail)
        for marker in taints:
            index = _parameter_index(marker)
            if index is not None:
                self.context.sink_parameters.setdefault(code, set()).add(index)
                if direct:
                    self.context.direct_sink_parameters.setdefault(code, set()).add(index)

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
        parameter_markers: list[int] = []
        for marker in summary.return_fact.taints:
            index = _parameter_index(marker)
            if index is None:
                result_taints.add(marker)
            elif index < len(actual):
                parameter_markers.append(index)
                result_taints.update(actual[index].taints)
        for code, indices in summary.sink_parameters:
            sink_taints = _union_taints(
                actual[index].taints for index in indices if index < len(actual)
            )
            self._record_sink(
                call,
                code,
                sink_taints,
                "reference reaches a sink through a helper call",
                direct=False,
            )
        result = replace(summary.return_fact, taints=frozenset(result_taints))
        if len(parameter_markers) == 1 and summary.return_fact == _Fact(
            taints=frozenset({f"parameter:{parameter_markers[0]}"})
        ):
            return actual[parameter_markers[0]]
        return replace(
            result,
            items=tuple(self._substitute_fact(item, actual) for item in result.items),
            mapping=tuple(
                (key, self._substitute_fact(item, actual))
                for key, item in result.mapping
            ),
            callable=(
                replace(
                    result.callable,
                    bound_args=tuple(
                        self._substitute_fact(item, actual)
                        for item in result.callable.bound_args
                    ),
                    bound_keywords=tuple(
                        (key, self._substitute_fact(item, actual))
                        for key, item in result.callable.bound_keywords
                    ),
                )
                if result.callable is not None
                else None
            ),
        )

    def _substitute_fact(self, fact: _Fact, actual: tuple[_Fact, ...]) -> _Fact:
        parameter_markers = [
            index
            for marker in fact.taints
            if (index := _parameter_index(marker)) is not None and index < len(actual)
        ]
        if len(parameter_markers) == 1 and fact == _Fact(
            taints=frozenset({f"parameter:{parameter_markers[0]}"})
        ):
            return actual[parameter_markers[0]]
        taints = {
            marker for marker in fact.taints if _parameter_index(marker) is None
        }
        for index in parameter_markers:
            taints.update(actual[index].taints)
        return replace(fact, taints=frozenset(taints))

    def _model_loader(self, name: str) -> bool:
        while name.endswith(".__call__"):
            name = name.removesuffix(".__call__")
        return name.endswith(".from_pretrained") and name.startswith(_MODEL_LOADER_PREFIXES)

    def _capability_class_fact(self, fact: _Fact) -> bool:
        return fact.capability_class or (
            fact.callable is not None
            and fact.callable.name.endswith(
                (".VerifiedAssetFile", ".VerifiedAssetSelection")
            )
        )

    @staticmethod
    def _network_target_class_fact(fact: _Fact) -> bool:
        return fact.callable is not None and fact.callable.name.endswith(
            (":_ValidatedHttpTarget", "._ValidatedHttpTarget")
        )

    @staticmethod
    def _same_expression(node: ast.AST, expression: str) -> bool:
        expected = ast.parse(expression, mode="eval").body
        return ast.dump(node, include_attributes=False) == ast.dump(
            expected, include_attributes=False
        )

    @staticmethod
    def _keyword_nodes(call: ast.Call) -> dict[str, ast.expr] | None:
        result: dict[str, ast.expr] = {}
        for keyword in call.keywords:
            if keyword.arg is None or keyword.arg in result:
                return None
            result[keyword.arg] = keyword.value
        return result

    def _tls_policy_target(self, node: ast.AST) -> bool:
        place = self._place(node, _Environment())
        return place is not None and place.endswith(
            (".check_hostname", ".verify_mode")
        )

    @staticmethod
    def _same_statements(
        statements: list[ast.stmt], expected_source: str
    ) -> bool:
        expected = ast.parse(expected_source).body
        return ast.dump(ast.Module(body=statements, type_ignores=[])) == ast.dump(
            ast.Module(body=expected, type_ignores=[])
        )

    def _dataset_request_headers_shape_allowed(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        return self._same_statements(
            node.body,
            '''
headers = {
    "Accept": "application/json, application/octet-stream",
    "Accept-Encoding": "identity",
    "User-Agent": "SakuraMoon-D010/1",
}
if target.send_authorization:
    token = self._token.get_secret_value()
    headers["Authorization"] = f"Bearer {token}"
    headers["Cookie"] = f"m_session_id={token}"
return headers
''',
        )

    def _dataset_open_get_range_shape_allowed(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        header_writes: list[ast.stmt] = []
        for candidate in ast.walk(node):
            if not isinstance(candidate, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                continue
            targets: tuple[ast.expr, ...]
            if isinstance(candidate, ast.Assign):
                targets = tuple(candidate.targets)
            else:
                targets = (candidate.target,)
            if any(
                isinstance(target, ast.Subscript)
                and self._place(target.value, _Environment()) == "headers"
                for target in targets
            ):
                header_writes.append(candidate)
        expected = ast.parse(
            '''
if range_start is not None:
    headers["Range"] = f"bytes={range_start}-"
'''
        ).body[0]
        return (
            len(header_writes) == 1
            and any(
                ast.dump(candidate, include_attributes=False)
                == ast.dump(expected, include_attributes=False)
                for candidate in ast.walk(node)
            )
        )

    def _dataset_urlencode_kind(self, call: ast.Call) -> str | None:
        if (
            self.path != DATASET_TRANSPORT_PATH
            or self.class_context is not None
            or len(call.args) != 1
            or call.keywords
            or not isinstance(call.args[0], ast.Dict)
        ):
            return None
        mapping = call.args[0]
        keys = tuple(_literal(item) for item in mapping.keys)
        if self.context.name == "_listing_target" and keys == (
            "Revision",
            "Recursive",
            "PageNumber",
            "PageSize",
        ):
            return (
                "listing"
                if tuple(
                    self._place(item, _Environment())
                    for item in mapping.values
                )
                == (
                    "manifest.source.revision",
                    None,
                    "page_number",
                    "page_size",
                )
                and _literal(mapping.values[1]) == "True"
                else None
            )
        if self.context.name == "_shard_target" and keys == (
            "Revision",
            "FilePath",
        ):
            return (
                "shard"
                if tuple(
                    self._place(item, _Environment())
                    for item in mapping.values
                )
                == ("manifest.source.revision", "shard.path")
                else None
            )
        return None

    def _dataset_target_constructor_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        keywords = self._keyword_nodes(call)
        if (
            self.path != DATASET_TRANSPORT_PATH
            or self.class_context is not None
            or call.args
            or keywords is None
            or set(keywords)
            != {"host", "port", "request_target", "send_authorization"}
            or self._place(keywords["port"], _Environment()) != "_HTTPS_PORT"
        ):
            return False
        if self.context.name in {"_listing_target", "_shard_target"}:
            kind = self.context.name.removeprefix("_").removesuffix("_target")
            request_expression = (
                'f"/api/v1/datasets/{_source_path(manifest.source)}/repo/tree?{query}"'
                if kind == "listing"
                else 'f"/api/v1/datasets/{_source_path(manifest.source)}/repo?{query}"'
            )
            return (
                self._place(keywords["host"], _Environment())
                == "MODELSCOPE_DATASET_HOST"
                and _literal(keywords["send_authorization"]) is True
                and self._same_expression(
                    keywords["request_target"], request_expression
                )
                and env.get("query").network_query == kind
            )
        if self.context.name == "_redirect_target":
            return (
                self._place(keywords["host"], _Environment()) == "host"
                and self._place(keywords["request_target"], _Environment())
                == "request_target"
                and self._same_expression(
                    keywords["send_authorization"],
                    "current.send_authorization and host == MODELSCOPE_DATASET_HOST",
                )
            )
        return False

    def _dataset_https_connection_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        keywords = self._keyword_nodes(call)
        if (
            self.path != DATASET_TRANSPORT_PATH
            or self.class_context != DATASET_TRANSPORT_CLASS
            or self.context.name != "_open_get"
            or call.args
            or keywords is None
            or set(keywords) != {"context", "host", "port", "timeout"}
            or not env.get("target").network_target
        ):
            return False
        context = keywords["context"]
        return (
            self._place(keywords["host"], _Environment()) == "target.host"
            and self._place(keywords["port"], _Environment()) == "target.port"
            and self._place(keywords["timeout"], _Environment())
            == "self._policy.connect_timeout_seconds"
            and isinstance(context, ast.Call)
            and not context.args
            and not context.keywords
            and self._place(context.func, _Environment())
            == "ssl.create_default_context"
        )

    def _dataset_http_member_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        if self.path != DATASET_TRANSPORT_PATH or not isinstance(
            call.func, ast.Attribute
        ):
            return False
        method = call.func.attr
        receiver = self._place(call.func.value, _Environment())
        keywords = self._keyword_nodes(call)
        if keywords is None:
            return False
        if self.class_context == DATASET_TRANSPORT_CLASS:
            if self.context.name == "_open_get":
                if method == "request" and receiver == "connection":
                    return (
                        len(call.args) == 2
                        and env.get("target").network_target
                        and _literal(call.args[0]) == "GET"
                        and self._place(call.args[1], _Environment())
                        == "target.request_target"
                        and set(keywords)
                        == {"body", "encode_chunked", "headers"}
                        and isinstance(keywords["body"], ast.Constant)
                        and keywords["body"].value is None
                        and self._place(keywords["headers"], _Environment())
                        == "headers"
                        and env.get("headers").network_headers
                        and _literal(keywords["encode_chunked"]) is False
                    )
                if method == "getresponse" and receiver == "connection":
                    return not call.args and not keywords
                if method == "settimeout" and receiver == "connection.sock":
                    return (
                        len(call.args) == 1
                        and not keywords
                        and self._place(call.args[0], _Environment())
                        == "self._policy.read_timeout_seconds"
                    )
                if method == "close" and receiver == "connection":
                    return not call.args and not keywords
            if self.context.name == "_follow_redirects":
                if method == "getheader" and receiver == "response":
                    return (
                        len(call.args) == 1
                        and not keywords
                        and _literal(call.args[0]) == "Location"
                    )
                if method == "close" and receiver in {"connection", "response"}:
                    return not call.args and not keywords
            if (
                self.context.name == "_close_response"
                and method == "close"
                and receiver in {"connection", "response"}
            ):
                return not call.args and not keywords
            if (
                self.context.name == "_read_response"
                and method == "read"
                and receiver == "response"
            ):
                return (
                    len(call.args) == 1
                    and not keywords
                    and self._place(call.args[0], _Environment()) == "length"
                )
        if self.class_context is None and method == "getheader" and receiver == "response":
            allowed_headers = {
                "_parse_content_length": {"Content-Length"},
                "_validate_download_headers": {
                    "Content-Encoding",
                    "Content-Range",
                },
            }
            return (
                len(call.args) == 1
                and not keywords
                and _literal(call.args[0])
                in allowed_headers.get(self.context.name, set())
            )
        return False

    def _dataset_http_helper_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        if (
            self.path != DATASET_TRANSPORT_PATH
            or self.class_context != DATASET_TRANSPORT_CLASS
            or not isinstance(call.func, ast.Attribute)
            or not isinstance(call.func.value, ast.Name)
            or call.func.value.id != "self"
        ):
            return False
        method = call.func.attr
        places = tuple(self._place(item, _Environment()) for item in call.args)
        keywords = self._keyword_nodes(call)
        if keywords is None:
            return False
        if method == "_request_headers":
            return (
                self.context.name == "_open_get"
                and places == ("target",)
                and env.get("target").network_target
                and not keywords
            )
        if method == "_open_get":
            return (
                self.context.name == "_follow_redirects"
                and places == ("current",)
                and env.get("current").network_target
                and set(keywords) == {"range_start"}
                and self._place(keywords["range_start"], _Environment())
                == "range_start"
            )
        if method == "_follow_redirects":
            if self.context.name == "_read_listing_once":
                return (
                    places == ("target",)
                    and env.get("target").network_target
                    and set(keywords) == {"range_start"}
                    and isinstance(keywords["range_start"], ast.Constant)
                    and keywords["range_start"].value is None
                )
            return (
                self.context.name == "download_locked_shard_to_staging"
                and places == ("target",)
                and env.get("target").network_target
                and set(keywords) == {"range_start"}
                and self._place(keywords["range_start"], _Environment())
                == "range_start"
            )
        if method == "_read_listing_once":
            return (
                self.context.name == "list_locked_files"
                and places == ("target",)
                and env.get("target").network_target
                and not keywords
            )
        if method == "_read_response":
            return (
                self.context.name
                in {"_read_listing_once", "download_locked_shard_to_staging"}
                and len(call.args) == 2
                and places[0] == "response"
                and not keywords
            )
        if method == "_close_response":
            return (
                self.context.name
                in {
                    "_read_listing_once",
                    "download_locked_shard_to_staging",
                }
                and places == ("response", "connection")
                and not keywords
            )
        return False

    def _dataset_target_factory_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        if self.path != DATASET_TRANSPORT_PATH or self.class_context != DATASET_TRANSPORT_CLASS:
            return False
        places = tuple(self._place(item, _Environment()) for item in call.args)
        if call.keywords:
            return False
        if isinstance(call.func, ast.Name) and call.func.id == "_listing_target":
            return (
                self.context.name == "list_locked_files"
                and env.get("manifest").dataset_manifest
                and places
                == (
                    "manifest",
                    "page_number",
                    "self._policy.listing_page_size",
                )
            )
        if isinstance(call.func, ast.Name) and call.func.id == "_shard_target":
            return (
                self.context.name == "download_locked_shard_to_staging"
                and env.get("manifest").dataset_manifest
                and env.get("shard").dataset_shard
                and places == ("manifest", "shard")
            )
        if isinstance(call.func, ast.Name) and call.func.id == "_redirect_target":
            return (
                self.context.name == "_follow_redirects"
                and env.get("current").network_target
                and places
                == (
                    "current",
                    "location",
                    "self._policy.redirect_hosts",
                )
            )
        return False

    def _git_shape(self, call: ast.Call) -> tuple[str | None, ...] | None:
        if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
            return None
        shape: list[str | None] = []
        for item in call.args[0].elts:
            value = _literal(item)
            shape.append(value if isinstance(value, str) else None)
        return tuple(shape)

    def _safe_test_git(self, call: ast.Call, name: str, env: _Environment) -> bool:
        if name != "subprocess.run":
            return False
        shape = self._git_shape(call)
        if shape is None or not shape or shape[0] != "git" or "-c" in shape:
            return False
        location = (self.path, self.context.name)

        def has_trusted_root_provenance(index: int) -> bool:
            if not call.args or not isinstance(call.args[0], (ast.List, ast.Tuple)):
                return False
            if index >= len(call.args[0].elts):
                return False
            fact = self._eval(call.args[0].elts[index], env)
            return "parameter:0" in fact.taints

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
                (
                    "git",
                    "-C",
                    None,
                    "config",
                    "user.name",
                    "Synthetic Test",
                ),
                (
                    "git",
                    "-C",
                    None,
                    "config",
                    "user.email",
                    "synthetic@example.invalid",
                ),
                ("git", "-C", None, "commit", "-qm", "fixture"),
                ("git", "-C", None, "rev-parse", "HEAD"),
            }
            if shape not in allowed:
                return False
            repo_index = 3 if shape[:3] == ("git", "init", "-q") else 2
            return has_trusted_root_provenance(repo_index)
        if location == (
            "tests/unit/assets/test_inspect.py",
            "test_reference_origin_diagnostic_redacts_credentials",
        ):
            return (
                shape == ("git", "-C", None, "remote", "set-url", "origin", None)
                and has_trusted_root_provenance(2)
            )
        if location == (
            "tests/unit/assets/test_inspect.py",
            "test_reference_git_audit_disables_hostile_local_configuration",
        ):
            return (
                len(shape) == 6
                and shape[:4] == ("git", "-C", None, "config")
                and shape[4]
                in {
                    "core.fsmonitor",
                    "core.hooksPath",
                    "core.pager",
                    "diff.external",
                    "interactive.diffFilter",
                    "pager.status",
                }
                and shape[5] is None
                and has_trusted_root_provenance(2)
            )
        return False

    def _mutate_container(
        self,
        call: ast.Call,
        env: _Environment,
        taints: frozenset[str],
    ) -> None:
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
            "append",
            "extend",
            "insert",
            "update",
        }:
            return
        receiver = self._place(call.func.value, env)
        if receiver is None:
            return
        current = env.get(receiver)
        items = current.items
        mapping = current.mapping
        arguments = tuple(self._eval(item, env) for item in call.args)
        if call.func.attr == "append" and arguments:
            items = (*items, arguments[0])
        elif call.func.attr == "extend" and arguments:
            items = (*items, *(arguments[0].items or (arguments[0],)))
        elif call.func.attr == "insert" and len(arguments) >= 2:
            items = (*items, arguments[1])
        elif call.func.attr == "update" and arguments:
            mapping = (*mapping, *arguments[0].mapping)
        env.assign(
            receiver,
            replace(
                current,
                taints=current.taints | taints,
                items=items,
                mapping=mapping,
            ),
        )

    def _capability_constructor_allowed(
        self,
        call: ast.Call,
        constructor: str,
        env: _Environment,
    ) -> bool:
        del env
        if self.path != "src/sakuramoon/assets/inspect.py" or call.args:
            return False
        if any(keyword.arg is None for keyword in call.keywords):
            return False
        values = {
            keyword.arg: self._place(keyword.value, _Environment())
            for keyword in call.keywords
            if keyword.arg is not None
        }
        if constructor == "VerifiedAssetFile" and self.context.name == "_inspect_file":
            return values == {
                "asset_id": "asset_id",
                "relative_path": "lock.path",
                "kind": "lock.kind",
                "bytes": "lock.bytes",
                "sha256": "lock.sha256",
                "_base": "base",
                "_path": "path",
                "_identity": "after",
            }
        if constructor == "VerifiedAssetSelection" and self.context.name == "_selection":
            return values == {
                "manifest_revision": "snapshot.manifest.manifest_revision",
                "manifest_sha256": "snapshot.sha256",
                "files": "files",
                "_root": "snapshot.root",
                "_manifest_relative_path": "snapshot.relative_path",
                "_manifest_path": "snapshot.path",
                "_manifest_identity": "snapshot.identity",
            }
        return False

    def _eval_call(self, call: ast.Call, env: _Environment) -> _Fact:
        if (
            self.path == DATASET_TRANSPORT_PATH
            and isinstance(call.func, ast.Attribute)
            and self._place(call.func.value, _Environment()) == "headers"
            and call.func.attr
            in {
                "clear",
                "pop",
                "popitem",
                "setdefault",
                "update",
            }
        ):
            self.add(
                call,
                "dataset_headers_mutation_forbidden",
                "dataset request headers may only be built by the exact audited assignments",
            )
        if isinstance(call.func, ast.Attribute) and call.func.attr in _DATASET_HTTP_HELPERS:
            allowed_helper = self._dataset_http_helper_allowed(call, env)
            if self.path.startswith("src/sakuramoon/") and not allowed_helper:
                self.add(
                    call,
                    "network_helper_call_forbidden",
                    "dataset HTTP helpers are private to their exact audited call graph",
                )
            if allowed_helper and call.func.attr == "_request_headers":
                return _Fact(network_headers=True)
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in _NETWORK_MEMBER_NAMES
            and self.path.startswith("src/sakuramoon/")
        ):
            allowed_member = self._dataset_http_member_allowed(call, env)
            receiver = self._place(call.func.value, _Environment())
            is_dataset_file_descriptor_call = (
                receiver == "os" and call.func.attr in {"close", "read"}
            )
            if (
                (
                    (
                        self.path == DATASET_TRANSPORT_PATH
                        and not is_dataset_file_descriptor_call
                    )
                    or call.func.attr in _NETWORK_EXECUTION_MEMBER_NAMES
                )
                and not allowed_member
            ):
                self.add(
                    call,
                    "network_call_forbidden",
                    "network instance methods require an exact audited transport shape",
                )
        if (
            isinstance(call.func, ast.Name)
            and call.func.id in _DATASET_TARGET_FACTORIES
            and self.path.startswith("src/sakuramoon/")
            and not self._dataset_target_factory_allowed(call, env)
        ):
            self.add(
                call,
                "network_target_factory_call_forbidden",
                "validated HTTP target factories are private to the audited transport",
            )

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
            if self._dataset_http_member_allowed(call, env):
                return _Fact()
            function_fact = self._eval(call.func, env)
            argument_facts = tuple(self._eval(item, env) for item in call.args)
            keyword_facts = tuple(
                self._eval(item.value, env) for item in call.keywords
            )
            taints = function_fact.taints | _union_taints(
                (
                    *(item.taints for item in argument_facts),
                    *(item.taints for item in keyword_facts),
                )
            )
            if any(
                _sensitive_callable(item.callable)
                for item in (*argument_facts, *keyword_facts)
            ):
                self.add(
                    call,
                    "sensitive_callable_escape",
                    "sensitive callable passed to an unprovable higher-order target",
                )
            if (
                self.path.startswith("src/sakuramoon/")
                and isinstance(call.func, ast.Name)
                and call.func.id not in {"cls", "self"}
                and (
                    self.path,
                    self.class_context,
                    self.context.name,
                    call.func.id,
                )
                not in _AUDITED_PARAMETER_CALLS
                and any(
                    _parameter_index(marker) is not None
                    for marker in function_fact.taints
                )
            ):
                self.add(
                    call,
                    "callable_parameter_forbidden",
                    "calling a function parameter is not auditable for sensitive callable provenance",
                )
            self._mutate_container(call, env, taints)
            return _Fact(taints=taints)
        name = callable_value.name
        if self.path == DATASET_TRANSPORT_PATH and (
            name in {"builtins.delattr", "builtins.print", "builtins.setattr"}
            or name.startswith(
                (
                    "logging.",
                    "types.MethodType",
                    "unittest.mock.",
                )
            )
        ):
            self.add(
                call,
                "dataset_transport_mutation_forbidden",
                "dataset transport may not patch callables or emit HTTP diagnostics",
            )
        if name == "urllib.parse.urlencode":
            query_kind = self._dataset_urlencode_kind(call)
            if query_kind is not None:
                return _Fact(network_query=query_kind)
        if name.startswith(_NETWORK_CALL_PREFIXES):
            allowed_network = False
            if name == "http.client.HTTPSConnection":
                allowed_network = self._dataset_https_connection_allowed(call, env)
            elif name.startswith(tuple(f"{item}." for item in _NETWORK_INSTANCE_CLASSES)):
                allowed_network = self._dataset_http_member_allowed(call, env)
            if self.path.startswith("src/sakuramoon/") and not allowed_network:
                self.add(
                    call,
                    "network_call_forbidden",
                    "production network calls require an exact audited transport shape",
                )
            if name == "http.client.HTTPSConnection":
                return _Fact(instance_class="http.client.HTTPSConnection")
            if name == "http.client.HTTPSConnection.getresponse":
                return _Fact(instance_class="http.client.HTTPResponse")
            if allowed_network:
                return _Fact()
        if (
            self.path.startswith("src/sakuramoon/")
            and name.startswith("modelscope_hub.HubApi.")
        ):
            self.add(
                call,
                "forbidden_download",
                "direct HubApi method calls are outside the exact D010 transport",
            )
        reflection_name = name
        while reflection_name.endswith(".__call__"):
            reflection_name = reflection_name.removesuffix(".__call__")
        if self.path.startswith("src/sakuramoon/") and reflection_name == "builtins.type":
            type_arguments = tuple(self._eval(item, env) for item in call.args)
            if len(type_arguments) >= 2 and any(
                self._capability_class_fact(item)
                or self._network_target_class_fact(item)
                for item in type_arguments[1].items
            ):
                self.add(
                    call,
                    "sensitive_subclass_forbidden",
                    "dynamic capability subclasses are forbidden",
                )
                return _Fact(capability_class=True)
        if self.path.startswith("src/sakuramoon/") and (
            reflection_name == "builtins.object.__new__"
            or name == "builtins.type.__call__"
            or reflection_name.endswith(
                (".VerifiedAssetFile.__new__", ".VerifiedAssetSelection.__new__")
            )
        ):
            constructor_arguments = tuple(self._eval(item, env) for item in call.args)
            if any(
                self._capability_class_fact(item)
                or self._network_target_class_fact(item)
                for item in constructor_arguments
            ):
                self.add(
                    call,
                    "capability_reflection_forbidden",
                    "reflective capability construction is forbidden",
                )
                return _Fact(selection=True)
        if (
            self.path.startswith("src/sakuramoon/")
            and reflection_name == "builtins.object.__setattr__"
            and call.args
            and self._eval(call.args[0], env).selection
        ):
            self.add(
                call,
                "capability_mutation_forbidden",
                "reflective mutation of a verified capability is forbidden",
            )

        constructor_syntax = call.func.id if isinstance(call.func, ast.Name) else ""
        target_constructor = constructor_syntax == "_ValidatedHttpTarget"
        target_constructor_allowed = target_constructor and self._dataset_target_constructor_allowed(
            call, env
        )
        if (
            self.path.startswith("src/sakuramoon/")
            and target_constructor
            and not target_constructor_allowed
        ):
            self.add(
                call,
                "network_target_construction_forbidden",
                "HTTP targets may only be constructed by exact audited validators",
            )
        if (
            self.path.startswith("src/sakuramoon/")
            and constructor_syntax == DATASET_TRANSPORT_CLASS
        ):
            self.add(
                call,
                "dataset_transport_construction_forbidden",
                "dataset transport must use the fixed credential/config factory",
            )
        capability_constructor = constructor_syntax in {
            "VerifiedAssetFile",
            "VerifiedAssetSelection",
        } or name.endswith((".VerifiedAssetFile", ".VerifiedAssetSelection"))
        if capability_constructor:
            constructor = (
                constructor_syntax
                if constructor_syntax in {"VerifiedAssetFile", "VerifiedAssetSelection"}
                else name.rsplit(".", 1)[-1]
            )
            if not self._capability_constructor_allowed(
                call,
                constructor,
                env,
            ):
                self.add(
                    call,
                    "capability_construction_forbidden",
                    "verified asset capabilities may only be issued by exact audited factories",
                )

        if name in self.classes:
            positional, keywords, unknown_keywords, expansion_taints = self._effective_arguments(
                call, callable_value, env
            )
            if any(
                _sensitive_callable(item.callable)
                for item in (*positional, *keywords.values())
            ):
                self.add(
                    call,
                    "sensitive_callable_escape",
                    "sensitive callable passed into a local class constructor",
                )
            return _Fact(
                taints=_union_taints(
                    (
                        *(item.taints for item in positional),
                        *(item.taints for item in keywords.values()),
                        expansion_taints,
                    )
                ),
                instance_class=name,
                network_target=target_constructor_allowed,
            )

        if name in {"builtins.getattr", "getattr"} and len(call.args) >= 2:
            base = self._callable(call.args[0], env)
            base_fact = self._eval(call.args[0], env)
            base_place = self._place(call.args[0], env)
            attribute_fact = self._eval(call.args[1], env)
            attribute = (
                attribute_fact.literal
                if attribute_fact.literal_known
                and isinstance(attribute_fact.literal, str)
                else None
            )
            if base is not None and isinstance(attribute, str):
                return _Fact(callable=_Callable(f"{base.name}.{attribute}"))
            if (
                base_fact.instance_class in _NETWORK_INSTANCE_CLASSES
                and isinstance(attribute, str)
            ):
                return _Fact(
                    callable=_Callable(
                        f"{base_fact.instance_class}.{attribute}",
                        bound_args=(base_fact,),
                    )
                )
            if (
                self.path.startswith("src/sakuramoon/")
                and base_fact.instance_class in _NETWORK_INSTANCE_CLASSES
                and attribute is None
            ):
                self.add(
                    call,
                    "network_dynamic_callable_forbidden",
                    "dynamic reflection over a network client is forbidden",
                )
                return _Fact(callable=_Callable("http.client.*"))
            if base_place is not None and isinstance(attribute, str):
                member_place = f"{base_place}.{attribute}"
                if member_place in env.values:
                    return env.get(member_place)
            if (
                self.path.startswith("src/sakuramoon/")
                and base is not None
                and (
                    base.name.startswith("sakuramoon.assets")
                    or base.name in {"builtins.object", "builtins.type"}
                )
                and attribute is None
            ):
                self.add(
                    call,
                    "capability_reflection_forbidden",
                    "dynamic reflection over asset capability exports is forbidden",
                )
                return _Fact(capability_class=True)
            if base is not None and base.name.startswith(
                (
                    *_MODEL_LOADER_PREFIXES,
                    "asyncio.",
                    "aiohttp.",
                    "huggingface_hub.",
                    "http.client.",
                    "httpx.",
                    "os.",
                    "requests.",
                    "runpy.",
                    "site.",
                    "socket.",
                    "subprocess.",
                    "urllib.request.",
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
        if any(
            _sensitive_callable(item.callable)
            for item in (*positional, *keywords.values())
        ):
            self.add(
                call,
                "sensitive_callable_escape",
                "sensitive callable passed through a higher-order call",
            )
        unknown_cross_module = (
            self.path.startswith("src/sakuramoon/")
            and name.startswith("sakuramoon.")
            and name
            not in (
                *_SELECTION_FACTORIES,
                *_SELECTION_GATES,
                *_CONFIG_FACTORIES,
                *_DATASET_SELECTION_GATES,
            )
        )
        if unknown_cross_module:
            if _REF in all_taints:
                self.add(
                    call,
                    "reference_cross_module_call_forbidden",
                    "reference provenance may not cross an unaudited module call",
                )
            if any(item.model_root is not None for item in (*positional, *keywords.values())):
                self.add(
                    call,
                    "model_root_cross_module_call_forbidden",
                    "verified model roots may only enter an exact audited loader wrapper",
                )
        if (
            self.path.startswith("src/sakuramoon/")
            and name in {"copy.copy", "copy.deepcopy", "dataclasses.replace"}
            and any(item.selection for item in (*positional, *keywords.values()))
        ):
            self.add(
                call,
                "capability_reflection_forbidden",
                "copying or replacing a verified capability is forbidden",
            )

        if name in _SELECTION_FACTORIES:
            return _Fact(selection=True)
        if name in _SELECTION_GATES:
            if len(positional) == 1 and not keywords and not unknown_keywords:
                return _Fact(selection=True)
            return _Fact()
        if name in _CONFIG_FACTORIES:
            return _Fact(config_path=())
        if name in _DATASET_SELECTION_GATES:
            if len(positional) == 1 and not keywords and not unknown_keywords:
                return _Fact(dataset_selection=True)
            return _Fact()
        if name == "audited-dataset-manifest.shard":
            if len(positional) == 2 and not keywords and not unknown_keywords:
                return _Fact(dataset_shard=True)
            return _Fact()

        local = self._local_function_call(call, callable_value, positional, keywords)
        if local is not None:
            return local

        canonical = name
        while canonical.endswith(".__call__"):
            canonical = canonical.removesuffix(".__call__")
        if canonical == "ambiguous-sensitive.*":
            self.add(
                call,
                "ambiguous_sensitive_callable",
                "branch-dependent sensitive callable provenance is forbidden",
            )
        elif self._model_loader(name):
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
        elif canonical in _DOWNLOAD_CALLS:
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
        if is_process and not self._safe_test_git(call, canonical, env):
            self._record_sink(call, "reference_process_exec", all_taints, "passes reference data to a process")
        if canonical in _SEARCH_PATH_CALLS:
            self._record_sink(call, "reference_search_path", all_taints, "injects reference into Python search paths")

        self._mutate_container(call, env, all_taints)
        if canonical in {"builtins.setattr", "setattr"} and len(call.args) >= 3:
            receiver = self._place(call.args[0], env)
            attribute = _literal(call.args[1])
            if receiver is not None and isinstance(attribute, str):
                env.assign(f"{receiver}.{attribute}", self._eval(call.args[2], env))
            elif receiver is not None:
                attribute_fact = self._eval(call.args[1], env)
                value_fact = self._eval(call.args[2], env)
                current = env.get(receiver)
                env.assign(
                    receiver,
                    replace(
                        current,
                        taints=current.taints
                        | attribute_fact.taints
                        | value_fact.taints,
                    ),
                )
                self._record_sink(
                    call,
                    "reference_dynamic_attribute",
                    attribute_fact.taints | value_fact.taints,
                    "dynamic attribute assignment carries reference provenance",
                )
                if _sensitive_callable(value_fact.callable):
                    self.add(
                        call,
                        "ambiguous_sensitive_callable",
                        "sensitive callable stored under a dynamic attribute is forbidden",
                    )

        return _Fact(
            taints=(
                all_taints | frozenset({_UNKNOWN_EXTERNAL})
                if unknown_cross_module
                else all_taints
            )
        )

    def _assign(self, target: ast.AST, fact: _Fact, env: _Environment) -> None:
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                item_fact = fact.items[index] if index < len(fact.items) else fact
                self._assign(item, item_fact, env)
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

    def _parameter_environment(
        self,
        arguments: ast.arguments,
        outer: _Environment,
        function_name: str,
    ) -> tuple[_Environment, tuple[str, ...]]:
        parameters = (
            *arguments.posonlyargs,
            *arguments.args,
            *arguments.kwonlyargs,
        )
        parameter_names = tuple(item.arg for item in parameters)
        env = outer.clone()
        for index, parameter in enumerate(parameter_names):
            env.assign(parameter, _Fact(taints=frozenset({f"parameter:{index}"})))
        if (
            self.path == DATASET_TRANSPORT_PATH
            and self.class_context == DATASET_TRANSPORT_CLASS
            and function_name
            in {
                "_follow_redirects",
                "_open_get",
                "_read_listing_once",
                "_request_headers",
            }
            and "target" in parameter_names
        ):
            target = env.get("target")
            env.assign("target", replace(target, network_target=True))
        if arguments.vararg is not None:
            index = len(parameter_names)
            parameter_names = (*parameter_names, arguments.vararg.arg)
            env.assign(arguments.vararg.arg, _Fact(taints=frozenset({f"parameter:{index}"})))
        if arguments.kwarg is not None:
            index = len(parameter_names)
            parameter_names = (*parameter_names, arguments.kwarg.arg)
            env.assign(arguments.kwarg.arg, _Fact(taints=frozenset({f"parameter:{index}"})))
        return env, parameter_names

    def _function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, outer: _Environment) -> _Fact:
        identifier = self._function_identifier(node)
        if (
            self.path == DATASET_TRANSPORT_PATH
            and self.class_context == DATASET_TRANSPORT_CLASS
        ):
            if (
                node.name == "_request_headers"
                and not self._dataset_request_headers_shape_allowed(node)
            ):
                self.add(
                    node,
                    "dataset_headers_factory_forbidden",
                    "request headers must use the exact audited auth and identity-only shape",
                )
            if (
                node.name == "_open_get"
                and not self._dataset_open_get_range_shape_allowed(node)
            ):
                self.add(
                    node,
                    "dataset_range_header_forbidden",
                    "range headers must use the exact audited manifest-offset shape",
                )
        env, parameter_names = self._parameter_environment(
            node.args, outer, node.name
        )
        previous = self.context
        self.context = _FunctionContext(node.name, parameter_names)
        self._block(node.body, env)
        summary = _FunctionSummary(
            parameters=parameter_names,
            return_fact=self.context.return_fact or _Fact(),
            sink_parameters=tuple(
                sorted((code, tuple(sorted(indices))) for code, indices in self.context.sink_parameters.items())
            ),
        )
        if (
            self.path.startswith("src/sakuramoon/")
            and self.context.direct_sink_parameters
            and (self.path, node.name) not in _AUDITED_PARAMETER_SINK_FUNCTIONS
        ):
            self.add(
                node,
                "parameterized_execution_wrapper_forbidden",
                "execution sinks may not be exposed through an unaudited parameterized wrapper",
            )
        self.context = previous
        self.summaries[identifier] = summary
        return _Fact(callable=_Callable(identifier))

    def _lambda(self, node: ast.Lambda, outer: _Environment) -> _Fact:
        identifier = self._function_identifier(node)
        env, parameter_names = self._parameter_environment(
            node.args, outer, "<lambda>"
        )
        previous = self.context
        self.context = _FunctionContext("<lambda>", parameter_names)
        return_fact = self._eval(node.body, env)
        summary = _FunctionSummary(
            parameters=parameter_names,
            return_fact=return_fact,
            sink_parameters=tuple(
                sorted(
                    (code, tuple(sorted(indices)))
                    for code, indices in self.context.sink_parameters.items()
                )
            ),
        )
        if self.path.startswith("src/sakuramoon/") and self.context.direct_sink_parameters:
            self.add(
                node,
                "parameterized_execution_wrapper_forbidden",
                "execution sinks may not be exposed through a lambda parameter",
            )
        self.context = previous
        self.summaries[identifier] = summary
        return _Fact(callable=_Callable(identifier))

    def _definition_expressions(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        env: _Environment,
    ) -> None:
        for decorator in node.decorator_list:
            self._eval(decorator, env)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            self._eval(default, env)
        annotations = [
            *(item.annotation for item in node.args.posonlyargs),
            *(item.annotation for item in node.args.args),
            *(item.annotation for item in node.args.kwonlyargs),
            node.args.vararg.annotation if node.args.vararg is not None else None,
            node.args.kwarg.annotation if node.args.kwarg is not None else None,
            node.returns,
        ]
        for annotation in annotations:
            self._eval(annotation, env)

    def _declared_class_names(self, statements: list[ast.stmt]) -> set[str]:
        names: set[str] = set()

        def collect(target: ast.AST) -> None:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    collect(item)

        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                names.add(statement.name)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    collect(target)
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                collect(statement.target)
        return names

    def _class(self, node: ast.ClassDef, env: _Environment) -> _Fact:
        identifier = self._class_identifier(node)
        for decorator in node.decorator_list:
            self._eval(decorator, env)
        for base_node in node.bases:
            base = self._eval(base_node, env)
            if base.callable is not None and (
                base.callable.name.startswith(_MODEL_LOADER_PREFIXES)
                or base.callable.name.endswith(
                    (".VerifiedAssetFile", ".VerifiedAssetSelection")
                )
                or base.callable.name.endswith(
                    (":_ValidatedHttpTarget", "._ValidatedHttpTarget")
                )
            ):
                self.add(
                    base_node,
                    "sensitive_subclass_forbidden",
                    "sensitive model/capability classes may not be subclassed",
                )
        for keyword in node.keywords:
            self._eval(keyword.value, env)
        class_env = env.clone()
        previous_class = self.class_context
        self.class_context = node.name
        self._block(node.body, class_env)
        self.class_context = previous_class
        self.classes[identifier] = {
            name: class_env.get(name)
            for name in self._declared_class_names(node.body)
        }
        return _Fact(callable=_Callable(identifier))

    def _predeclare(self, statements: list[ast.stmt], env: _Environment) -> None:
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                env.assign(
                    statement.name,
                    _Fact(callable=_Callable(self._function_identifier(statement))),
                )
            elif isinstance(statement, ast.ClassDef):
                env.assign(
                    statement.name,
                    _Fact(callable=_Callable(self._class_identifier(statement))),
                )

    def _bind_pattern(self, pattern: ast.pattern, fact: _Fact, env: _Environment) -> None:
        if isinstance(pattern, ast.MatchAs):
            if pattern.pattern is not None:
                self._bind_pattern(pattern.pattern, fact, env)
            if pattern.name is not None:
                env.assign(pattern.name, fact)
        elif isinstance(pattern, ast.MatchStar):
            if pattern.name is not None:
                env.assign(pattern.name, fact)
        elif isinstance(pattern, ast.MatchSequence):
            for index, item in enumerate(pattern.patterns):
                item_fact = fact.items[index] if index < len(fact.items) else fact
                self._bind_pattern(item, item_fact, env)
        elif isinstance(pattern, ast.MatchMapping):
            for key_node, item in zip(pattern.keys, pattern.patterns, strict=True):
                key = key_node.value if isinstance(key_node, ast.Constant) else None
                item_fact = next(
                    (value for item_key, value in fact.mapping if item_key == key),
                    fact,
                )
                self._bind_pattern(item, item_fact, env)
            if pattern.rest is not None:
                env.assign(pattern.rest, fact)
        elif isinstance(pattern, ast.MatchClass):
            self._eval(pattern.cls, env)
            for item in (*pattern.patterns, *pattern.kwd_patterns):
                self._bind_pattern(item, fact, env)
        elif isinstance(pattern, ast.MatchOr):
            branches: list[_Environment] = []
            for item in pattern.patterns:
                branch = env.clone()
                self._bind_pattern(item, fact, branch)
                branches.append(branch)
            env.values = _merge_environments(*branches).values

    def _eval_direct_expressions(self, node: ast.stmt, env: _Environment) -> None:
        for item in ast.iter_child_nodes(node):
            if isinstance(item, ast.expr):
                self._eval(item, env)

    def _statement(self, node: ast.stmt, env: _Environment) -> _Environment:
        if isinstance(node, ast.Import):
            for item in node.names:
                if self.path == DATASET_TRANSPORT_PATH and item.name.startswith(
                    _DATASET_FORBIDDEN_IMPORT_PREFIXES
                ):
                    self.add(
                        node,
                        "dataset_transport_import_forbidden",
                        "dataset transport may not import SDK, alternate network, or logging modules",
                    )
                local = item.asname or item.name.split(".", 1)[0]
                module = item.name if item.asname else item.name.split(".", 1)[0]
                env.assign(local, _Fact(callable=_Callable(module)))
        elif isinstance(node, ast.ImportFrom) and node.module:
            if self.path == DATASET_TRANSPORT_PATH and node.module.startswith(
                _DATASET_FORBIDDEN_IMPORT_PREFIXES
            ):
                self.add(
                    node,
                    "dataset_transport_import_forbidden",
                    "dataset transport may not import SDK, alternate network, or logging modules",
                )
            for item in node.names:
                if item.name == "*":
                    if node.module.startswith(
                        (
                            "asyncio",
                            "aiohttp",
                            "diffusers",
                            "huggingface_hub",
                            "http.client",
                            "httpx",
                            "importlib",
                            "modelscope",
                            "os",
                            "requests",
                            "runpy",
                            "site",
                            "socket",
                            "subprocess",
                            "transformers",
                            "urllib.request",
                        )
                    ):
                        self.add(
                            node,
                            "sensitive_star_import_forbidden",
                            "star import from a model/download package is not auditable",
                        )
                else:
                    env.assign(item.asname or item.name, _Fact(callable=_Callable(f"{node.module}.{item.name}")))
            if _is_reference_text(node.module):
                self.add(node, "reference_import", "engineering code imports reference code")
        elif isinstance(node, ast.Assign):
            if (
                self.path == DATASET_TRANSPORT_PATH
                and isinstance(node.value, ast.Attribute)
                and node.value.attr in _NETWORK_MEMBER_NAMES
            ):
                self.add(
                    node,
                    "network_call_forbidden",
                    "dataset network methods may not be detached from exact audited receivers",
                )
            fact = self._eval(node.value, env)
            for target in node.targets:
                if self.path.startswith("src/sakuramoon/") and self._tls_policy_target(
                    target
                ):
                    self.add(
                        node,
                        "tls_policy_mutation_forbidden",
                        "TLS hostname and certificate verification may not be changed",
                    )
                self._assign(target, fact, env)
        elif isinstance(node, ast.AnnAssign):
            if node.value is not None:
                if self.path.startswith("src/sakuramoon/") and self._tls_policy_target(
                    node.target
                ):
                    self.add(
                        node,
                        "tls_policy_mutation_forbidden",
                        "TLS hostname and certificate verification may not be changed",
                    )
                self._assign(node.target, self._eval(node.value, env), env)
        elif isinstance(node, ast.AugAssign):
            if self.path.startswith("src/sakuramoon/") and self._tls_policy_target(
                node.target
            ):
                self.add(
                    node,
                    "tls_policy_mutation_forbidden",
                    "TLS hostname and certificate verification may not be changed",
                )
            fact = _merge_facts(self._eval(node.target, env), self._eval(node.value, env))
            self._assign(node.target, fact, env)
        elif isinstance(node, ast.Expr):
            self._eval(node.value, env)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            self._definition_expressions(node, env)
            env.assign(node.name, self._function(node, env))
        elif isinstance(node, ast.ClassDef):
            env.assign(node.name, self._class(node, env))
        elif isinstance(node, ast.Return):
            fact = self._eval(node.value, env)
            self.context.return_fact = (
                fact
                if self.context.return_fact is None
                else _merge_facts(self.context.return_fact, fact)
            )
        elif isinstance(node, ast.Assert):
            self._eval(node.test, env)
            self._eval(node.msg, env)
        elif isinstance(node, ast.Raise):
            self._eval(node.exc, env)
            self._eval(node.cause, env)
        elif isinstance(node, ast.Delete):
            for target in node.targets:
                place = self._place(target, env)
                if place is not None:
                    env.assign(place, _Fact())
        elif isinstance(node, ast.If):
            self._eval(node.test, env)
            body = env.clone()
            other = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, other)
            env = _merge_environments(body, other)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            iterable = self._eval(node.iter, env)
            body = env.clone()
            if iterable.items:
                item_fact = iterable.items[0]
                for candidate in iterable.items[1:]:
                    item_fact = _merge_facts(item_fact, candidate)
            else:
                item_fact = iterable
            self._assign(node.target, item_fact, body)
            self._block(node.body, body)
            self._block(node.orelse, body)
            env = _merge_environments(env, body)
        elif isinstance(node, ast.While):
            self._eval(node.test, env)
            body = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, body)
            env = _merge_environments(env, body)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            branches: list[_Environment] = []
            body = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, body)
            branches.append(body)
            for handler in node.handlers:
                branch = env.clone()
                self._eval(handler.type, branch)
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
        elif isinstance(node, ast.Match):
            subject = self._eval(node.subject, env)
            branches = [env.clone()]
            for case in node.cases:
                branch = env.clone()
                self._bind_pattern(case.pattern, subject, branch)
                self._eval(case.guard, branch)
                self._block(case.body, branch)
                branches.append(branch)
            env = _merge_environments(*branches)
        else:
            self._eval_direct_expressions(node, env)
        return env

    def _block(self, statements: list[ast.stmt], env: _Environment) -> _Environment:
        self._predeclare(statements, env)
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
        max_passes = 3 + sum(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef))
            for node in ast.walk(tree)
        )
        converged = False
        for _ in range(max_passes):
            before_summaries = dict(self.summaries)
            before_classes = {key: dict(value) for key, value in self.classes.items()}
            self.violations = []
            self.context = _FunctionContext("<module>", ())
            builtins = _Environment(
                {
                    "__import__": _Fact(callable=_Callable("builtins.__import__")),
                    "compile": _Fact(callable=_Callable("builtins.compile")),
                    "delattr": _Fact(callable=_Callable("builtins.delattr")),
                    "eval": _Fact(callable=_Callable("builtins.eval")),
                    "exec": _Fact(callable=_Callable("builtins.exec")),
                    "getattr": _Fact(callable=_Callable("builtins.getattr")),
                    "object": _Fact(callable=_Callable("builtins.object")),
                    "print": _Fact(callable=_Callable("builtins.print")),
                    "setattr": _Fact(callable=_Callable("builtins.setattr")),
                    "type": _Fact(callable=_Callable("builtins.type")),
                }
            )
            self._block(tree.body, builtins)
            if self.summaries == before_summaries and self.classes == before_classes:
                converged = True
                break
        if not converged:
            self.add(
                tree,
                "analysis_did_not_converge",
                "function/class provenance analysis did not reach a fixed point",
            )
        return tuple(
            sorted(
                set(self.violations),
                key=lambda item: (item.path, item.line, item.code, item.detail),
            )
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
