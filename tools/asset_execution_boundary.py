"""Static, fail-closed checks for local model and reference execution boundaries."""

from __future__ import annotations

import ast
import hashlib
import os
import re
import stat
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from pathlib import Path

MODEL_ASSET_IDS = frozenset({"qwen_text_encoder", "mage_vae"})
DATASET_TRANSPORT_PATH = "src/sakuramoon/data/modelscope.py"
DATASET_TRANSPORT_CLASS = "ModelScopeDatasetTransport"
_DATASET_TRANSPORT_CLASS_AST_SHA256 = (
    "226d16422aa57d5fcd8c7b1a05ef4cc07f52296f1d21c29e78c40ef23b1567f9"
)

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
_NAMESPACE_REFLECTION_CALLS = frozenset(
    {
        "builtins.globals",
        "builtins.locals",
        "builtins.vars",
        "inspect.currentframe",
        "inspect.getclosurevars",
        "inspect.getcoroutinelocals",
        "inspect.getargvalues",
        "inspect.getgeneratorlocals",
        "inspect.getmembers",
        "inspect.getmembers_static",
        "inspect.signature",
        "inspect.stack",
        "inspect.trace",
        "sys._current_frames",
        "sys._getframe",
        "typing.get_type_hints",
    }
)
_REFLECTION_CALLABLES = _NAMESPACE_REFLECTION_CALLS | frozenset(
    {
        "builtins.getattr",
        "builtins.object.__getattribute__",
        "builtins.type.__getattribute__",
        "builtins.vars",
        "getattr",
        "inspect.getattr_static",
        "operator.attrgetter",
        "operator.methodcaller",
        "vars",
    }
)
_FRAME_NAMESPACE_ATTRIBUTES = frozenset(
    {
        "ag_frame",
        "builtins",
        "cell_contents",
        "cr_frame",
        "globals",
        "nonlocals",
        "unbound",
        "__builtins__",
        "__annotations__",
        "__closure__",
        "__code__",
        "__dict__",
        "__defaults__",
        "__globals__",
        "__kwdefaults__",
        "default",
        "annotation",
        "parameters",
        "return_annotation",
        "f_back",
        "f_builtins",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_frame",
        "tb_frame",
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
_REFERENCE_RE = re.compile(r"(?<![A-Za-z0-9_])reference(?![A-Za-z0-9_])", re.IGNORECASE)
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
        (
            "src/sakuramoon/data/manifest.py",
            "_IdentityWeakRegistry",
            "contains",
            "reference",
        ),
    }
)
_AUDITED_VARS_CALLS = frozenset(
    {
        (
            "src/sakuramoon/assets/inspect.py",
            "_stat_fingerprint",
            "value",
        ),
        (
            "src/sakuramoon/data/manifest.py",
            "_verified_manifest_fingerprint",
            "identity",
        ),
    }
)
_AUDITED_DYNAMIC_GETATTR_CALLS = frozenset(
    {
        (
            "src/sakuramoon/assets/bindings.py",
            "require_runtime_assets_match_snapshot",
            base,
            "name",
        )
        for base in ("config.qwen", "config.vae")
    }
)
_AUDITED_OBJECT_GETATTRIBUTE_CALLS = frozenset(
    {
        (
            "src/sakuramoon/assets/inspect.py",
            "_verified_file_fingerprint",
            "value",
            attribute,
        )
        for attribute in (
            "_base",
            "_identity",
            "_path",
            "asset_id",
            "bytes",
            "kind",
            "relative_path",
            "sha256",
        )
    }
    | {
        (
            "src/sakuramoon/assets/inspect.py",
            "_verified_selection_fingerprint",
            "value",
            attribute,
        )
        for attribute in (
            "_identity",
            "_manifest_identity",
            "_manifest_path",
            "_manifest_relative_path",
            "_root",
            "files",
            "manifest_revision",
            "manifest_sha256",
        )
    }
    | {
        (
            "src/sakuramoon/data/manifest.py",
            "_revalidate_issued_manifest",
            "selection",
            "_identity",
        )
    }
    | {
        (
            "src/sakuramoon/data/manifest.py",
            "_verified_manifest_fingerprint",
            "value",
            attribute,
        )
        for attribute in ("_identity", "manifest", "path", "sha256")
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
    asset_capability_maybe: bool = False
    selection: bool = False
    model_root: str | None = None
    model_root_maybe: bool = False
    config_path: tuple[str, ...] | None = None
    dataset_capability_maybe: bool = False
    dataset_manifest: bool = False
    dataset_selection: bool = False
    dataset_shard: bool = False
    object_place: str | None = None
    instance_class: str | None = None
    network_instance_maybe: str | None = None
    network_capability_maybe: bool = False
    network_headers: bool = False
    network_query: str | None = None
    network_target: bool = False
    sensitive_callable_maybe: bool = False
    bounded_nonnegative: bool = False
    listing_payload_bounded: bool = False
    synthetic_git_helper: bool = False
    test_safe_relative: bool = False
    test_root_holder: bool = False
    test_root_path: bool = False
    container_items_unknown: bool = False
    container_mapping_unknown: bool = False
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


def _safe_test_relative_literal(value: str) -> bool:
    path = Path(value)
    return (
        bool(value)
        and not path.is_absolute()
        and "\\" not in value
        and all(part not in {"", ".", ".."} for part in path.parts)
    )


def _canonical_callable_name(name: str) -> str:
    while name.endswith(".__call__"):
        name = name.removesuffix(".__call__")
    return name


def _reflection_callable_name(name: str) -> bool:
    return _canonical_callable_name(name) in _REFLECTION_CALLABLES


def _union_taints(values: Iterable[frozenset[str]]) -> frozenset[str]:
    result: set[str] = set()
    for value in values:
        result.update(value)
    return frozenset(result)


def _sensitive_callable(value: _Callable | None) -> bool:
    if value is None:
        return False
    name = _canonical_callable_name(value.name)
    return (
        _reflection_callable_name(name)
        or (
            name.endswith(".from_pretrained")
            and name.startswith(_MODEL_LOADER_PREFIXES)
        )
        or name in _DOWNLOAD_CALLS
        or name in _DYNAMIC_IMPORT_CALLS
        or name in _DYNAMIC_CODE_CALLS
        or name in _PROCESS_EXACT
        or name.startswith(_NETWORK_CALL_PREFIXES)
        or name.startswith(
            ("asyncio.create_subprocess_", "os.exec", "os.posix_spawn", "os.spawn")
        )
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


def _fact_children(fact: _Fact) -> tuple[_Fact, ...]:
    callable_children: tuple[_Fact, ...] = ()
    if fact.callable is not None:
        callable_children = (
            *fact.callable.bound_args,
            *(value for _, value in fact.callable.bound_keywords),
        )
    return (
        *callable_children,
        *fact.items,
        *(value for _, value in fact.mapping),
    )


def _fact_contains_sensitive_callable(fact: _Fact) -> bool:
    return (
        fact.sensitive_callable_maybe
        or _sensitive_callable(fact.callable)
        or any(
            _fact_contains_sensitive_callable(child) for child in _fact_children(fact)
        )
    )


def _fact_contains_network_capability(fact: _Fact) -> bool:
    direct_callable = fact.callable
    direct = (
        fact.network_capability_maybe
        or fact.network_headers
        or fact.network_target
        or fact.instance_class in _NETWORK_INSTANCE_CLASSES
        or fact.network_instance_maybe in _NETWORK_INSTANCE_CLASSES
        or (
            direct_callable is not None
            and (
                direct_callable.name.startswith(_NETWORK_CALL_PREFIXES)
                or direct_callable.name.endswith(
                    (":_ValidatedHttpTarget", "._ValidatedHttpTarget")
                )
            )
        )
    )
    return direct or any(
        _fact_contains_network_capability(child) for child in _fact_children(fact)
    )


def _fact_contains_security_capability(fact: _Fact) -> bool:
    return (
        _fact_contains_sensitive_callable(fact)
        or _fact_contains_network_capability(fact)
        or fact.capability_class
        or fact.selection
        or fact.model_root is not None
        or fact.synthetic_git_helper
        or _fact_contains_dataset_capability(fact)
        or any(
            _fact_contains_security_capability(child) for child in _fact_children(fact)
        )
    )


def _fact_contains_synthetic_git_helper(fact: _Fact) -> bool:
    return fact.synthetic_git_helper or any(
        _fact_contains_synthetic_git_helper(child) for child in _fact_children(fact)
    )


def _fact_contains_model_root(fact: _Fact) -> bool:
    return (
        fact.model_root_maybe
        or fact.model_root is not None
        or any(_fact_contains_model_root(child) for child in _fact_children(fact))
    )


def _fact_contains_asset_capability(fact: _Fact) -> bool:
    return (
        fact.asset_capability_maybe
        or fact.capability_class
        or fact.selection
        or any(_fact_contains_asset_capability(child) for child in _fact_children(fact))
    )


def _fact_contains_dataset_capability(fact: _Fact) -> bool:
    return (
        fact.dataset_capability_maybe
        or fact.dataset_manifest
        or fact.dataset_selection
        or fact.dataset_shard
        or any(
            _fact_contains_dataset_capability(child) for child in _fact_children(fact)
        )
    )


def _without_network_capabilities(fact: _Fact) -> _Fact:
    callable_value = fact.callable
    if callable_value is not None:
        callable_value = replace(
            callable_value,
            bound_args=tuple(
                _without_network_capabilities(item)
                for item in callable_value.bound_args
            ),
            bound_keywords=tuple(
                (key, _without_network_capabilities(value))
                for key, value in callable_value.bound_keywords
            ),
        )
        if callable_value.name.startswith(
            _NETWORK_CALL_PREFIXES
        ) or callable_value.name.endswith(
            (":_ValidatedHttpTarget", "._ValidatedHttpTarget")
        ):
            callable_value = None
    return replace(
        fact,
        callable=callable_value,
        instance_class=(
            None
            if fact.instance_class in _NETWORK_INSTANCE_CLASSES
            else fact.instance_class
        ),
        network_instance_maybe=None,
        network_capability_maybe=False,
        network_headers=False,
        network_target=False,
        items=tuple(_without_network_capabilities(item) for item in fact.items),
        mapping=tuple(
            (key, _without_network_capabilities(value)) for key, value in fact.mapping
        ),
    )


def _merge_facts(left: _Fact, right: _Fact) -> _Fact:
    same_callable = left.callable if left.callable == right.callable else None
    left_contains_sensitive = _fact_contains_sensitive_callable(left)
    right_contains_sensitive = _fact_contains_sensitive_callable(right)
    if same_callable is None and (left_contains_sensitive or right_contains_sensitive):
        if left.callable is None:
            same_callable = right.callable
        elif right.callable is None:
            same_callable = left.callable
        else:
            same_callable = _Callable("ambiguous-sensitive.*")
    same_root = left.model_root if left.model_root == right.model_root else None
    same_config = left.config_path if left.config_path == right.config_path else None
    literal_known = (
        left.literal_known and right.literal_known and left.literal == right.literal
    )
    instance_class = (
        left.instance_class if left.instance_class == right.instance_class else None
    )
    network_instance_maybe: str | None = None
    network_classes = {
        value
        for value in (
            left.instance_class,
            left.network_instance_maybe,
            right.instance_class,
            right.network_instance_maybe,
        )
        if value in _NETWORK_INSTANCE_CLASSES
    }
    if instance_class is None and len(network_classes) == 1:
        candidate = next(iter(network_classes))
        left_compatible = (
            left.instance_class == candidate
            or left.network_instance_maybe == candidate
            or (left.literal_known and left.literal is None)
        )
        right_compatible = (
            right.instance_class == candidate
            or right.network_instance_maybe == candidate
            or (right.literal_known and right.literal is None)
        )
        if left_compatible and right_compatible:
            network_instance_maybe = candidate
    container_items_unknown = (
        left.container_items_unknown
        or right.container_items_unknown
        or len(left.items) != len(right.items)
    )
    items = ()
    if not container_items_unknown and left.items:
        items = tuple(
            _merge_facts(left_item, right_item)
            for left_item, right_item in zip(left.items, right.items, strict=True)
        )
    left_mapping_keys = tuple(key for key, _ in left.mapping)
    right_mapping_keys = tuple(key for key, _ in right.mapping)
    container_mapping_unknown = (
        left.container_mapping_unknown
        or right.container_mapping_unknown
        or left_mapping_keys != right_mapping_keys
    )
    mapping: tuple[tuple[str | bool | int | None, _Fact], ...] = ()
    if not container_mapping_unknown and left.mapping:
        mapping = tuple(
            (left_key, _merge_facts(left_value, right_value))
            for (left_key, left_value), (_, right_value) in zip(
                left.mapping, right.mapping, strict=True
            )
        )
    return _Fact(
        taints=left.taints | right.taints,
        callable=same_callable,
        capability_class=left.capability_class or right.capability_class,
        asset_capability_maybe=(
            _fact_contains_asset_capability(left)
            or _fact_contains_asset_capability(right)
        ),
        selection=left.selection and right.selection,
        model_root=same_root,
        model_root_maybe=(
            _fact_contains_model_root(left) or _fact_contains_model_root(right)
        ),
        config_path=same_config,
        dataset_capability_maybe=(
            _fact_contains_dataset_capability(left)
            or _fact_contains_dataset_capability(right)
        ),
        dataset_manifest=left.dataset_manifest and right.dataset_manifest,
        dataset_selection=left.dataset_selection and right.dataset_selection,
        dataset_shard=left.dataset_shard and right.dataset_shard,
        object_place=left.object_place
        if left.object_place == right.object_place
        else None,
        instance_class=instance_class,
        network_instance_maybe=network_instance_maybe,
        network_capability_maybe=(
            _fact_contains_network_capability(left)
            or _fact_contains_network_capability(right)
        ),
        network_headers=left.network_headers and right.network_headers,
        network_query=(
            left.network_query if left.network_query == right.network_query else None
        ),
        network_target=left.network_target and right.network_target,
        sensitive_callable_maybe=(left_contains_sensitive or right_contains_sensitive),
        bounded_nonnegative=(left.bounded_nonnegative and right.bounded_nonnegative),
        listing_payload_bounded=(
            left.listing_payload_bounded and right.listing_payload_bounded
        ),
        synthetic_git_helper=(left.synthetic_git_helper or right.synthetic_git_helper),
        test_safe_relative=(left.test_safe_relative and right.test_safe_relative),
        test_root_holder=left.test_root_holder and right.test_root_holder,
        test_root_path=left.test_root_path and right.test_root_path,
        container_items_unknown=container_items_unknown,
        container_mapping_unknown=container_mapping_unknown,
        items=items,
        mapping=mapping,
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
        self.loop_break_environments: list[list[_Environment]] = []
        self.loop_continue_environments: list[list[_Environment]] = []

    def _function_identifier(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef | ast.Lambda
    ) -> str:
        name = (
            node.name
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            else "<lambda>"
        )
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

    def _attribute_chain(
        self, node: ast.Attribute
    ) -> tuple[str, tuple[str, ...]] | None:
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
                fact = _fact_literal(node.value)
                if isinstance(node.value, str):
                    fact = replace(
                        fact,
                        test_safe_relative=_safe_test_relative_literal(node.value),
                    )
                return fact
            return _Fact(literal_known=True)
        if isinstance(node, ast.Name):
            if (
                self.path.startswith("src/sakuramoon/")
                and node.id == "__builtins__"
            ):
                self.add(
                    node,
                    "namespace_reflection_forbidden",
                    "production code may not inspect the dynamic builtins namespace",
                )
                return _Fact(
                    network_capability_maybe=True,
                    sensitive_callable_maybe=True,
                    capability_class=True,
                )
            return env.get(node.id)
        if isinstance(node, ast.Attribute):
            if (
                (
                    self.path.startswith("src/sakuramoon/")
                    or self.path == "tests/unit/assets/conftest.py"
                )
                and node.attr in _FRAME_NAMESPACE_ATTRIBUTES
            ):
                self.add(
                    node,
                    "namespace_reflection_forbidden",
                    "production code may not inspect frame or execution namespaces",
                )
                return _Fact(
                    network_capability_maybe=True,
                    sensitive_callable_maybe=True,
                    capability_class=True,
                )
            place = self._place(node, env)
            if (
                self.path.startswith("src/sakuramoon/")
                and place == "sys.modules"
            ):
                self.add(
                    node,
                    "namespace_reflection_forbidden",
                    "production code may not inspect the dynamic module namespace",
                )
                return _Fact(
                    network_capability_maybe=True,
                    sensitive_callable_maybe=True,
                    capability_class=True,
                )
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
            if (
                self.path.startswith("src/sakuramoon/")
                and node.attr == "from_pretrained"
                and any(_parameter_index(marker) is not None for marker in base.taints)
            ):
                self.add(
                    node,
                    "parameterized_sensitive_member_forbidden",
                    "a parameter may not select a model loader member",
                )
                return _Fact(
                    taints=base.taints,
                    callable=_Callable("ambiguous-sensitive.*"),
                    sensitive_callable_maybe=True,
                )
            if base.test_root_holder and node.attr == "root":
                return _Fact(
                    taints=base.taints,
                    test_root_path=True,
                )
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
                if (
                    base.instance_class in _NETWORK_INSTANCE_CLASSES
                    and node.attr in _NETWORK_MEMBER_NAMES
                ):
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
            if isinstance(node.value, ast.Attribute) and node.value.attr == "__dict__":
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
            if self.path.startswith(
                "src/sakuramoon/"
            ) and _fact_contains_security_capability(base):
                self.add(
                    node,
                    "ambiguous_sensitive_container_access",
                    "dynamic container access may select an execution capability",
                )
            return _Fact(
                taints=base.taints | key.taints,
                callable=(
                    _Callable("ambiguous-sensitive.*")
                    if _fact_contains_sensitive_callable(base)
                    else None
                ),
            )
        if isinstance(node, ast.Starred):
            return self._eval(node.value, env)
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
            taints = _union_taints(item.taints for item in (*key_facts, *value_facts))
            mapping: list[tuple[str | bool | int | None, _Fact]] = []
            for key_node, value_fact in zip(node.keys, value_facts, strict=True):
                if isinstance(key_node, ast.Constant) and (
                    isinstance(key_node.value, (str, bool, int))
                    or key_node.value is None
                ):
                    mapping.append((key_node.value, value_fact))
            return _Fact(taints=taints, mapping=tuple(mapping))
        if isinstance(
            node, (ast.ListComp, ast.SetComp, ast.GeneratorExp, ast.DictComp)
        ):
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
                taints=_union_taints(
                    self._eval(item, env).taints for item in node.values
                )
            )
        if isinstance(node, ast.FormattedValue):
            return self._eval(node.value, env)
        if isinstance(node, ast.BinOp):
            literal = _literal(node)
            if isinstance(literal, str):
                return _fact_literal(literal)
            left = self._eval(node.left, env)
            right = self._eval(node.right, env)
            test_root_path = False
            if isinstance(node.op, ast.Div) and left.test_root_path:
                test_root_path = (
                    right.literal_known
                    and isinstance(right.literal, str)
                    and _safe_test_relative_literal(right.literal)
                ) or right.taints == frozenset({"parameter:1"})
            return _Fact(
                taints=left.taints | right.taints,
                test_root_path=test_root_path,
            )
        if isinstance(node, ast.UnaryOp):
            return self._eval(node.operand, env)
        if isinstance(node, ast.IfExp):
            return _merge_facts(
                self._eval(node.body, env), self._eval(node.orelse, env)
            )
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
    ) -> tuple[dict[str, _Fact], bool, tuple[_Fact, ...]]:
        values: dict[str, _Fact] = {}
        unknown = False
        expansions: list[_Fact] = []
        for keyword in call.keywords:
            if keyword.arg is not None:
                values[keyword.arg] = self._eval(keyword.value, env)
                continue
            expansion = self._eval(keyword.value, env)
            expansions.append(expansion)
            if not isinstance(keyword.value, ast.Dict):
                unknown = True
                continue
            for key_node, value_node in zip(
                keyword.value.keys, keyword.value.values, strict=True
            ):
                key = _literal(key_node)
                if not isinstance(key, str):
                    unknown = True
                    continue
                values[key] = self._eval(value_node, env)
        return values, unknown, tuple(expansions)

    def _effective_arguments(
        self, call: ast.Call, callable_value: _Callable, env: _Environment
    ) -> tuple[tuple[_Fact, ...], dict[str, _Fact], bool, tuple[_Fact, ...]]:
        positional = (
            *callable_value.bound_args,
            *(self._eval(item, env) for item in call.args),
        )
        keywords = dict(callable_value.bound_keywords)
        current, unknown, expansions = self._keywords(call, env)
        keywords.update(current)
        return (
            positional,
            keywords,
            callable_value.unknown_keywords or unknown,
            expansions,
        )

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
                    self.context.direct_sink_parameters.setdefault(code, set()).add(
                        index
                    )

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
        taints = {marker for marker in fact.taints if _parameter_index(marker) is None}
        for index in parameter_markers:
            taints.update(actual[index].taints)
        return replace(fact, taints=frozenset(taints))

    def _model_loader(self, name: str) -> bool:
        name = _canonical_callable_name(name)
        return name.endswith(".from_pretrained") and name.startswith(
            _MODEL_LOADER_PREFIXES
        )

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

    def _object_getattribute_allowed(
        self,
        call: ast.Call,
    ) -> bool:
        if len(call.args) != 2 or call.keywords:
            return False
        receiver = self._place(call.args[0], _Environment())
        attribute = _literal(call.args[1])
        return (
            isinstance(receiver, str)
            and isinstance(attribute, str)
            and (
                self.path,
                self.context.name,
                receiver,
                attribute,
            )
            in _AUDITED_OBJECT_GETATTRIBUTE_CALLS
        )

    def _vars_call_allowed(self, call: ast.Call) -> bool:
        if len(call.args) != 1 or call.keywords:
            return False
        receiver = self._place(call.args[0], _Environment())
        return (
            self.path,
            self.context.name,
            receiver,
        ) in _AUDITED_VARS_CALLS

    def _dynamic_getattr_call_allowed(self, call: ast.Call) -> bool:
        if len(call.args) != 2 or call.keywords:
            return False
        receiver = self._place(call.args[0], _Environment())
        attribute = self._place(call.args[1], _Environment())
        return (
            self.path,
            self.context.name,
            receiver,
            attribute,
        ) in _AUDITED_DYNAMIC_GETATTR_CALLS

    def _reject_opaque_security_capability(
        self,
        call: ast.Call,
        facts: tuple[_Fact, ...],
    ) -> None:
        if not self.path.startswith("src/sakuramoon/") or not any(
            _fact_contains_security_capability(fact) for fact in facts
        ):
            return
        self.add(
            call,
            "security_capability_escape_forbidden",
            "security capabilities may not enter an opaque or higher-order call",
        )
        if any(_fact_contains_model_root(fact) for fact in facts):
            self.add(
                call,
                "model_root_cross_module_call_forbidden",
                "verified model roots may only enter an exact audited loader wrapper",
            )
        if any(_fact_contains_asset_capability(fact) for fact in facts):
            self.add(
                call,
                "capability_reflection_forbidden",
                "verified capabilities may not enter reflective or opaque calls",
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
        return place is not None and place.endswith((".check_hostname", ".verify_mode"))

    @staticmethod
    def _same_statements(statements: list[ast.stmt], expected_source: str) -> bool:
        expected = ast.parse(expected_source).body
        return ast.dump(ast.Module(body=statements, type_ignores=[])) == ast.dump(
            ast.Module(body=expected, type_ignores=[])
        )

    def _dataset_request_headers_shape_allowed(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        return self._same_statements(
            node.body,
            """
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
""",
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
            """
if range_start is not None:
    headers["Range"] = f"bytes={range_start}-"
"""
        ).body[0]
        return len(header_writes) == 1 and any(
            ast.dump(candidate, include_attributes=False)
            == ast.dump(expected, include_attributes=False)
            for candidate in ast.walk(node)
        )

    def _dataset_open_get_constructor_shape_allowed(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> bool:
        expected = ast.parse(
            """
try:
    connection = http.client.HTTPSConnection(
        host=target.host,
        port=target.port,
        timeout=self._policy.connect_timeout_seconds,
        context=ssl.create_default_context(),
    )
except (OSError, ValueError):
    raise DatasetTransportError(
        "ModelScope HTTPS client could not be initialized"
    ) from None
"""
        ).body[0]
        return bool(node.body) and ast.dump(
            node.body[0], include_attributes=False
        ) == ast.dump(expected, include_attributes=False)

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
                if tuple(self._place(item, _Environment()) for item in mapping.values)
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
                if tuple(self._place(item, _Environment()) for item in mapping.values)
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
            or set(keywords) != {"host", "port", "request_target", "send_authorization"}
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

    def _dataset_http_member_allowed(self, call: ast.Call, env: _Environment) -> bool:
        if self.path != DATASET_TRANSPORT_PATH or not isinstance(
            call.func, ast.Attribute
        ):
            return False
        method = call.func.attr
        receiver = self._place(call.func.value, _Environment())
        keywords = self._keyword_nodes(call)
        if keywords is None:
            return False
        connection_verified = (
            env.get("connection").instance_class == "http.client.HTTPSConnection"
        )
        response_verified = (
            env.get("response").instance_class == "http.client.HTTPResponse"
        )
        if self.class_context == DATASET_TRANSPORT_CLASS:
            if self.context.name == "_open_get":
                if method == "request" and receiver == "connection":
                    return (
                        connection_verified
                        and len(call.args) == 2
                        and env.get("target").network_target
                        and _literal(call.args[0]) == "GET"
                        and self._place(call.args[1], _Environment())
                        == "target.request_target"
                        and set(keywords) == {"body", "encode_chunked", "headers"}
                        and isinstance(keywords["body"], ast.Constant)
                        and keywords["body"].value is None
                        and self._place(keywords["headers"], _Environment())
                        == "headers"
                        and env.get("headers").network_headers
                        and _literal(keywords["encode_chunked"]) is False
                    )
                if method == "getresponse" and receiver == "connection":
                    return connection_verified and not call.args and not keywords
                if method == "settimeout" and receiver == "connection.sock":
                    return (
                        connection_verified
                        and len(call.args) == 1
                        and not keywords
                        and self._place(call.args[0], _Environment())
                        == "self._policy.read_timeout_seconds"
                    )
                if method == "close" and receiver == "connection":
                    return connection_verified and not call.args and not keywords
            if (
                self.context.name == "_follow_redirects"
                and method == "getheader"
                and receiver == "response"
            ):
                return (
                    response_verified
                    and len(call.args) == 1
                    and not keywords
                    and _literal(call.args[0]) == "Location"
                )
            if (
                self.context.name == "_close_response"
                and method == "close"
                and receiver in {"connection", "response"}
            ):
                receiver_verified = (
                    connection_verified
                    if receiver == "connection"
                    else response_verified
                )
                return receiver_verified and not call.args and not keywords
            if (
                self.context.name == "_read_response"
                and method == "read"
                and receiver == "response"
            ):
                return (
                    response_verified
                    and len(call.args) == 1
                    and not keywords
                    and self._place(call.args[0], _Environment()) == "length"
                )
        if (
            self.class_context is None
            and method == "getheader"
            and receiver == "response"
        ):
            allowed_headers = {
                "_parse_content_length": {"Content-Length"},
                "_validate_download_headers": {
                    "Content-Encoding",
                    "Content-Range",
                },
            }
            return (
                response_verified
                and len(call.args) == 1
                and not keywords
                and _literal(call.args[0])
                in allowed_headers.get(self.context.name, set())
            )
        return False

    def _dataset_http_helper_allowed(self, call: ast.Call, env: _Environment) -> bool:
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
            if (
                len(call.args) != 2
                or places[0] != "response"
                or env.get("response").instance_class != "http.client.HTTPResponse"
                or keywords
            ):
                return False
            length = call.args[1]
            if self.context.name == "_read_listing_once":
                return (
                    self._same_expression(
                        length,
                        "min(self._policy.stream_chunk_bytes, remaining)",
                    )
                    and env.get("remaining").bounded_nonnegative
                )
            if self.context.name == "download_locked_shard_to_staging":
                return (
                    isinstance(length, ast.Constant)
                    and type(length.value) is int
                    and length.value == 1
                ) or self._same_expression(
                    length,
                    "min(self._policy.stream_chunk_bytes, expected_response_bytes - received)",
                )
            return False
        if method == "_close_response":
            return (
                self.context.name
                in {
                    "_follow_redirects",
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
        if (
            self.path != DATASET_TRANSPORT_PATH
            or self.class_context != DATASET_TRANSPORT_CLASS
        ):
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

    def _dataset_response_helper_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        if (
            self.path != DATASET_TRANSPORT_PATH
            or not isinstance(call.func, ast.Name)
            or call.keywords
        ):
            return False
        places = tuple(self._place(item, _Environment()) for item in call.args)
        if call.func.id == "_parse_content_length":
            return (
                self.class_context is None
                and self.context.name == "_validate_download_headers"
                and places == ("response",)
                and env.get("response").instance_class == "http.client.HTTPResponse"
            )
        if call.func.id == "_validate_download_headers":
            return (
                self.class_context == DATASET_TRANSPORT_CLASS
                and self.context.name == "download_locked_shard_to_staging"
                and places == ("response", "downloaded", "shard.bytes")
                and env.get("response").instance_class == "http.client.HTTPResponse"
            )
        return False

    def _dataset_network_capability_call_allowed(
        self, call: ast.Call, env: _Environment
    ) -> bool:
        return (
            self._dataset_http_member_allowed(call, env)
            or self._dataset_http_helper_allowed(call, env)
            or self._dataset_target_factory_allowed(call, env)
            or self._dataset_response_helper_allowed(call, env)
        )

    @staticmethod
    def _invalidate_network_capabilities(env: _Environment) -> None:
        env.values = {
            key: _without_network_capabilities(value)
            for key, value in env.values.items()
        }

    def _reject_network_capability_escape(
        self,
        call: ast.Call,
        env: _Environment,
        facts: tuple[_Fact, ...],
    ) -> bool:
        if (
            not self.path.startswith("src/sakuramoon/")
            or not any(_fact_contains_network_capability(fact) for fact in facts)
            or self._dataset_network_capability_call_allowed(call, env)
        ):
            return False
        if any(fact.network_headers for fact in facts):
            self.add(
                call,
                "dataset_headers_mutation_forbidden",
                "audited request headers may not enter a non-exact call",
            )
        self.add(
            call,
            "network_capability_escape_forbidden",
            "network capabilities may only enter the exact audited transport graph",
        )
        self._invalidate_network_capabilities(env)
        return True

    def _dataset_header_write_allowed(
        self,
        node: ast.Assign,
        target: ast.AST,
        env: _Environment,
    ) -> bool:
        return (
            self.path == DATASET_TRANSPORT_PATH
            and self.class_context == DATASET_TRANSPORT_CLASS
            and self.context.name == "_open_get"
            and len(node.targets) == 1
            and isinstance(target, ast.Subscript)
            and self._eval(target.value, env).network_headers
            and isinstance(target.slice, ast.Constant)
            and target.slice.value == "Range"
            and self._same_expression(node.value, 'f"bytes={range_start}-"')
        )

    def _dataset_bounded_remaining_assignment(
        self,
        node: ast.Assign,
        target: ast.AST,
        env: _Environment,
    ) -> bool:
        return (
            self.path == DATASET_TRANSPORT_PATH
            and self.class_context == DATASET_TRANSPORT_CLASS
            and self.context.name == "_read_listing_once"
            and len(node.targets) == 1
            and isinstance(target, ast.Name)
            and target.id == "remaining"
            and env.get("payload").listing_payload_bounded
            and self._same_expression(
                node.value,
                "_LISTING_RESPONSE_LIMIT_BYTES + 1 - len(payload)",
            )
        )

    def _dataset_listing_payload_initialization(
        self,
        node: ast.Assign,
        target: ast.AST,
    ) -> bool:
        return (
            self.path == DATASET_TRANSPORT_PATH
            and self.class_context == DATASET_TRANSPORT_CLASS
            and self.context.name == "_read_listing_once"
            and len(node.targets) == 1
            and isinstance(target, ast.Name)
            and target.id == "payload"
            and self._same_expression(node.value, "bytearray()")
        )

    def _dataset_listing_payload_upper_bound_guard(
        self,
        node: ast.If,
    ) -> bool:
        return (
            self.path == DATASET_TRANSPORT_PATH
            and self.class_context == DATASET_TRANSPORT_CLASS
            and self.context.name == "_read_listing_once"
            and self._same_expression(
                node.test,
                "len(payload) > _LISTING_RESPONSE_LIMIT_BYTES",
            )
            and self._statements_definitely_terminate(node.body)
        )

    def _reject_dataset_header_write(
        self,
        node: ast.Assign | ast.AnnAssign | ast.AugAssign,
        target: ast.AST,
        env: _Environment,
    ) -> bool:
        if self.path != DATASET_TRANSPORT_PATH:
            return False
        if isinstance(target, (ast.Subscript, ast.Attribute)):
            target_fact = self._eval(target.value, env)
        else:
            target_fact = self._eval(target, env)
        if not target_fact.network_headers:
            return False
        if isinstance(node, ast.Assign) and self._dataset_header_write_allowed(
            node, target, env
        ):
            return False
        self.add(
            node,
            "dataset_headers_mutation_forbidden",
            "audited request headers only allow the exact Range assignment",
        )
        self._invalidate_network_capabilities(env)
        return True

    def _reject_dataset_network_binding_assignment(
        self,
        node: ast.Assign | ast.AnnAssign,
        target: ast.AST,
        fact: _Fact,
    ) -> None:
        if (
            self.path != DATASET_TRANSPORT_PATH
            or self.class_context != DATASET_TRANSPORT_CLASS
        ):
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for index, item in enumerate(target.elts):
                item_fact = fact.items[index] if index < len(fact.items) else fact
                self._reject_dataset_network_binding_assignment(node, item, item_fact)
            return
        if not isinstance(target, ast.Name) or target.id not in {
            "connection",
            "response",
        }:
            return
        expected = (
            "http.client.HTTPSConnection"
            if target.id == "connection"
            else "http.client.HTTPResponse"
        )
        if fact.instance_class == expected or (
            fact.literal_known and fact.literal is None
        ):
            return
        self.add(
            node,
            "network_binding_assignment_forbidden",
            "network response and connection bindings require exact audited provenance",
        )

    def _reject_namespace_assignment(
        self,
        node: ast.Assign | ast.AnnAssign | ast.AugAssign,
        target: ast.AST,
    ) -> None:
        if not self.path.startswith("src/sakuramoon/"):
            return
        if isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._reject_namespace_assignment(node, item)
            return
        if (
            isinstance(target, ast.Attribute)
            and target.attr in _FRAME_NAMESPACE_ATTRIBUTES
        ):
            self.add(
                node,
                "namespace_reflection_forbidden",
                "production code may not mutate callable or closure namespaces",
            )

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
            argument = call.args[0].elts[index]
            if (
                isinstance(argument, ast.Call)
                and isinstance(argument.func, ast.Name)
                and argument.func.id == "str"
                and len(argument.args) == 1
                and not argument.keywords
            ):
                argument = argument.args[0]
            return self._eval(argument, env).test_root_path

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
            return shape == (
                "git",
                "-C",
                None,
                "remote",
                "set-url",
                "origin",
                None,
            ) and has_trusted_root_provenance(2)
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
        if current.container_items_unknown:
            items = ()
        elif call.func.attr == "append" and arguments:
            items = (*items, arguments[0])
        elif call.func.attr == "extend" and arguments:
            items = (*items, *(arguments[0].items or (arguments[0],)))
        elif call.func.attr == "insert" and len(arguments) >= 2:
            items = (*items, arguments[1])
        elif (
            call.func.attr == "update"
            and arguments
            and not current.container_mapping_unknown
        ):
            mapping = (*mapping, *arguments[0].mapping)
        added_facts = arguments if arguments else ()
        env.assign(
            receiver,
            replace(
                current,
                taints=current.taints | taints,
                items=items,
                mapping=mapping,
                network_capability_maybe=(
                    current.network_capability_maybe
                    or any(
                        _fact_contains_network_capability(item) for item in added_facts
                    )
                ),
                sensitive_callable_maybe=(
                    current.sensitive_callable_maybe
                    or any(
                        _fact_contains_sensitive_callable(item) for item in added_facts
                    )
                ),
                listing_payload_bounded=False,
                model_root_maybe=(
                    current.model_root_maybe
                    or any(_fact_contains_model_root(item) for item in added_facts)
                ),
                asset_capability_maybe=(
                    current.asset_capability_maybe
                    or any(
                        _fact_contains_asset_capability(item) for item in added_facts
                    )
                ),
                dataset_capability_maybe=(
                    current.dataset_capability_maybe
                    or any(
                        _fact_contains_dataset_capability(item) for item in added_facts
                    )
                ),
            ),
        )

    def _container_extraction(
        self,
        call: ast.Call,
        env: _Environment,
    ) -> _Fact | None:
        if not isinstance(call.func, ast.Attribute) or call.func.attr not in {
            "pop",
            "popitem",
        }:
            return None
        receiver = self._place(call.func.value, env)
        container = self._eval(call.func.value, env)
        if (
            not container.items
            and not container.mapping
            and not _fact_contains_security_capability(container)
        ):
            return None
        if call.func.attr == "popitem" and not call.args and not call.keywords:
            if container.mapping:
                key, value = container.mapping[-1]
                if receiver is not None:
                    env.assign(
                        receiver,
                        replace(container, mapping=container.mapping[:-1]),
                    )
                key_fact = (
                    _fact_literal(key)
                    if isinstance(key, (str, bool)) or key is None
                    else _Fact(literal_known=True)
                )
                return _Fact(items=(key_fact, value))
            return container
        if call.func.attr != "pop" or call.keywords:
            return container
        if container.mapping and call.args:
            key = _literal(call.args[0])
            matches = tuple(
                value for item_key, value in container.mapping if item_key == key
            )
            if matches:
                if receiver is not None:
                    env.assign(
                        receiver,
                        replace(
                            container,
                            mapping=tuple(
                                (item_key, value)
                                for item_key, value in container.mapping
                                if item_key != key
                            ),
                        ),
                    )
                return matches[-1]
            if len(call.args) >= 2:
                return self._eval(call.args[1], env)
            return container
        if container.items:
            if call.args and isinstance(call.args[0], ast.Constant):
                index = call.args[0].value
                if type(index) is int:
                    normalized = index if index >= 0 else len(container.items) + index
                    if 0 <= normalized < len(container.items):
                        if receiver is not None:
                            env.assign(
                                receiver,
                                replace(
                                    container,
                                    items=tuple(
                                        item
                                        for item_index, item in enumerate(
                                            container.items
                                        )
                                        if item_index != normalized
                                    ),
                                ),
                            )
                        return container.items[normalized]
            result = container.items[0]
            for candidate in container.items[1:]:
                result = _merge_facts(result, candidate)
            if receiver is not None:
                env.assign(
                    receiver,
                    replace(
                        container,
                        items=(),
                        container_items_unknown=True,
                        network_capability_maybe=(
                            container.network_capability_maybe
                            or _fact_contains_network_capability(container)
                        ),
                        sensitive_callable_maybe=(
                            container.sensitive_callable_maybe
                            or _fact_contains_sensitive_callable(container)
                        ),
                    ),
                )
            return result
        return container

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
        if (
            constructor == "VerifiedAssetSelection"
            and self.context.name == "_selection"
        ):
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
            and not self._dataset_network_capability_call_allowed(call, env)
        ):
            self.add(
                call,
                "dataset_headers_mutation_forbidden",
                "the audited headers binding may not receive method calls",
            )
        extracted = self._container_extraction(call, env)
        if extracted is not None:
            self._reject_network_capability_escape(
                call,
                env,
                (extracted,),
            )
            return extracted
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in _DATASET_HTTP_HELPERS
        ):
            allowed_helper = self._dataset_http_helper_allowed(call, env)
            if self.path.startswith("src/sakuramoon/") and not allowed_helper:
                self.add(
                    call,
                    "network_helper_call_forbidden",
                    "dataset HTTP helpers are private to their exact audited call graph",
                )
            if allowed_helper and call.func.attr == "_request_headers":
                return _Fact(network_headers=True)
            if allowed_helper and call.func.attr in {
                "_follow_redirects",
                "_open_get",
            }:
                return _Fact(
                    items=(
                        _Fact(instance_class="http.client.HTTPResponse"),
                        _Fact(instance_class="http.client.HTTPSConnection"),
                    )
                )
            if allowed_helper:
                return _Fact()
        if (
            isinstance(call.func, ast.Attribute)
            and call.func.attr in _NETWORK_MEMBER_NAMES
            and self.path.startswith("src/sakuramoon/")
        ):
            allowed_member = self._dataset_http_member_allowed(call, env)
            receiver = self._place(call.func.value, _Environment())
            is_dataset_file_descriptor_call = receiver == "os" and call.func.attr in {
                "close",
                "read",
            }
            if (
                (
                    self.path == DATASET_TRANSPORT_PATH
                    and not is_dataset_file_descriptor_call
                )
                or call.func.attr in _NETWORK_EXECUTION_MEMBER_NAMES
            ) and not allowed_member:
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
            receiver_fact = (
                self._eval(call.func.value, env)
                if isinstance(call.func, ast.Attribute)
                else _Fact()
            )
            argument_facts = tuple(self._eval(item, env) for item in call.args)
            keyword_facts = tuple(self._eval(item.value, env) for item in call.keywords)
            taints = function_fact.taints | _union_taints(
                (
                    *(item.taints for item in argument_facts),
                    *(item.taints for item in keyword_facts),
                )
            )
            opaque_facts = (
                function_fact,
                receiver_fact,
                *argument_facts,
                *keyword_facts,
            )
            self._reject_network_capability_escape(
                call,
                env,
                opaque_facts,
            )
            if not self._dataset_network_capability_call_allowed(call, env):
                self._reject_opaque_security_capability(call, opaque_facts)
            if any(
                _fact_contains_sensitive_callable(item)
                for item in (
                    function_fact,
                    *argument_facts,
                    *keyword_facts,
                )
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
        for factory_name in _DATASET_TARGET_FACTORIES:
            if (
                self.path.startswith("src/sakuramoon/")
                and name.endswith(f":{factory_name}")
                and not (
                    isinstance(call.func, ast.Name) and call.func.id == factory_name
                )
            ):
                self.add(
                    call,
                    "network_target_factory_call_forbidden",
                    "validated HTTP target factories may not be aliased",
                )
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
            elif name.startswith(
                tuple(f"{item}." for item in _NETWORK_INSTANCE_CLASSES)
            ):
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
        if self.path.startswith("src/sakuramoon/") and name.startswith(
            "modelscope_hub.HubApi."
        ):
            self.add(
                call,
                "forbidden_download",
                "direct HubApi method calls are outside the exact D010 transport",
            )
        reflection_name = _canonical_callable_name(name)
        if (
            (
                self.path.startswith("src/sakuramoon/")
                or self.path == "tests/unit/assets/conftest.py"
            )
            and reflection_name in _NAMESPACE_REFLECTION_CALLS
            and not (
                reflection_name == "builtins.vars" and self._vars_call_allowed(call)
            )
        ):
            if (
                reflection_name == "builtins.vars"
                and call.args
                and (
                    _fact_contains_security_capability(self._eval(call.args[0], env))
                    or self._capability_class_fact(self._eval(call.args[0], env))
                )
            ):
                self.add(
                    call,
                    "capability_reflection_forbidden",
                    "capability exports may not be accessed through reflection mappings",
                )
            self.add(
                call,
                "namespace_reflection_forbidden",
                "production code may not inspect dynamic execution namespaces",
            )
            return _Fact(
                network_capability_maybe=True,
                sensitive_callable_maybe=True,
                capability_class=True,
            )
        if self.path.startswith("src/sakuramoon/") and reflection_name in {
            "functools.reduce",
            "operator.attrgetter",
            "operator.methodcaller",
        }:
            self.add(
                call,
                "callable_reflection_forbidden",
                "production callable adapters are not auditable",
            )
            return _Fact(callable=_Callable("ambiguous-sensitive.*"))
        if (
            self.path.startswith("src/sakuramoon/")
            and reflection_name
            in {
                "builtins.object.__getattribute__",
                "builtins.type.__getattribute__",
            }
            and (
                reflection_name != "builtins.object.__getattribute__"
                or not self._object_getattribute_allowed(call)
            )
        ):
            if reflection_name == "builtins.type.__getattribute__":
                self.add(
                    call,
                    "callable_reflection_forbidden",
                    "type.__getattribute__ may not resolve production callables",
                )
            self.add(
                call,
                "capability_reflection_forbidden",
                "object.__getattribute__ is restricted to exact capability fingerprint readers",
            )
            return _Fact(
                network_capability_maybe=True,
                sensitive_callable_maybe=True,
                capability_class=True,
            )
        if (
            self.path.startswith("src/sakuramoon/")
            and reflection_name == "builtins.object.__setattr__"
            and len(call.args) >= 2
            and _literal(call.args[1]) in _FRAME_NAMESPACE_ATTRIBUTES
        ):
            self.add(
                call,
                "namespace_reflection_forbidden",
                "production code may not mutate callable or closure namespaces",
            )
            return _Fact(
                network_capability_maybe=True,
                sensitive_callable_maybe=True,
                capability_class=True,
            )
        if (
            self.path.startswith("src/sakuramoon/")
            and reflection_name == "inspect.getattr_static"
        ):
            if call.args and (
                _fact_contains_security_capability(self._eval(call.args[0], env))
                or self._capability_class_fact(self._eval(call.args[0], env))
            ):
                self.add(
                    call,
                    "capability_reflection_forbidden",
                    "capability exports may not be accessed through static reflection",
                )
            self.add(
                call,
                "callable_reflection_forbidden",
                "static attribute reflection is not auditable in production",
            )
            return _Fact(
                callable=_Callable("ambiguous-sensitive.*"),
                sensitive_callable_maybe=True,
            )
        if (
            self.path.startswith("src/sakuramoon/")
            and reflection_name == "builtins.vars"
            and call.args
        ):
            reflected = self._eval(call.args[0], env)
            reflected_name = (
                reflected.callable.name if reflected.callable is not None else ""
            )
            if _fact_contains_security_capability(
                reflected
            ) or reflected_name.startswith("sakuramoon.assets"):
                self.add(
                    call,
                    "capability_reflection_forbidden",
                    "capability exports may not be accessed through reflection mappings",
                )
                return _Fact(capability_class=True)
        if (
            self.path.startswith("src/sakuramoon/")
            and reflection_name == "builtins.type"
        ):
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
        target_constructor = (
            constructor_syntax == "_ValidatedHttpTarget"
            or name.endswith((":_ValidatedHttpTarget", "._ValidatedHttpTarget"))
        )
        target_constructor_allowed = (
            constructor_syntax == "_ValidatedHttpTarget"
            and self._dataset_target_constructor_allowed(call, env)
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
            positional, keywords, unknown_keywords, expansions = (
                self._effective_arguments(call, callable_value, env)
            )
            argument_facts = (*positional, *keywords.values(), *expansions)
            self._reject_network_capability_escape(
                call,
                env,
                argument_facts,
            )
            self._reject_opaque_security_capability(call, argument_facts)
            if any(_fact_contains_sensitive_callable(item) for item in argument_facts):
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
                        *(item.taints for item in expansions),
                    )
                ),
                instance_class=name,
                network_target=target_constructor_allowed,
            )

        if reflection_name in {"builtins.getattr", "getattr"} and len(call.args) >= 2:
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
            if (
                (
                    self.path.startswith("src/sakuramoon/")
                    or self.path == "tests/unit/assets/conftest.py"
                )
                and attribute in _FRAME_NAMESPACE_ATTRIBUTES
            ):
                self.add(
                    call,
                    "namespace_reflection_forbidden",
                    "callable execution namespaces may not be resolved through getattr",
                )
                return _Fact(
                    network_capability_maybe=True,
                    sensitive_callable_maybe=True,
                    capability_class=True,
                )
            if (
                self.path.startswith("src/sakuramoon/")
                and attribute is None
                and not self._dynamic_getattr_call_allowed(call)
            ):
                reflected_name = (
                    base_fact.callable.name if base_fact.callable is not None else ""
                )
                if (
                    _fact_contains_security_capability(base_fact)
                    or reflected_name.startswith("sakuramoon.assets")
                    or reflected_name in {"builtins.object", "builtins.type"}
                ):
                    self.add(
                        call,
                        "capability_reflection_forbidden",
                        "dynamic reflection over capability exports is forbidden",
                    )
                self.add(
                    call,
                    "dynamic_getattr_forbidden",
                    "nonliteral attribute selection is not auditable in production",
                )
                return _Fact(
                    callable=_Callable("ambiguous-sensitive.*"),
                    sensitive_callable_maybe=True,
                )
            if (
                self.path.startswith("src/sakuramoon/")
                and attribute == "from_pretrained"
                and any(
                    _parameter_index(marker) is not None for marker in base_fact.taints
                )
            ):
                self.add(
                    call,
                    "parameterized_sensitive_member_forbidden",
                    "a parameter may not select a model loader member",
                )
                return _Fact(
                    taints=base_fact.taints,
                    callable=_Callable("ambiguous-sensitive.*"),
                    sensitive_callable_maybe=True,
                )
            class_identifier = base_fact.instance_class
            if (
                class_identifier is None
                and base_fact.callable is not None
                and base_fact.callable.name in self.classes
            ):
                class_identifier = base_fact.callable.name
            class_member = (
                self.classes.get(class_identifier, {}).get(attribute)
                if class_identifier is not None and isinstance(attribute, str)
                else None
            )
            if self.path.startswith("src/sakuramoon/") and (
                _fact_contains_security_capability(base_fact)
                or (
                    class_member is not None
                    and _fact_contains_security_capability(class_member)
                )
                or (
                    self.path == DATASET_TRANSPORT_PATH
                    and self.class_context == DATASET_TRANSPORT_CLASS
                    and base_place == "self"
                    and attribute
                    in (
                        _DATASET_HTTP_HELPERS
                        | _DATASET_TARGET_FACTORIES
                        | _NETWORK_MEMBER_NAMES
                    )
                )
            ):
                self.add(
                    call,
                    "callable_reflection_forbidden",
                    "sensitive class members may not be resolved through getattr",
                )
                return _Fact(callable=_Callable("ambiguous-sensitive.*"))
            if base is not None and isinstance(attribute, str):
                return _Fact(callable=_Callable(f"{base.name}.{attribute}"))
            if base_fact.instance_class in _NETWORK_INSTANCE_CLASSES and isinstance(
                attribute, str
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

        if name in {
            "functools.partial",
            "functools.partialmethod",
            "partial",
            "partialmethod",
        } and call.args:
            target = self._callable(call.args[0], env)
            if target is None:
                return _Fact()
            target_fact = self._eval(call.args[0], env)
            if (
                (
                    self.path.startswith("src/sakuramoon/")
                    or self.path == "tests/unit/assets/conftest.py"
                )
                and _reflection_callable_name(target.name)
            ):
                self.add(
                    call,
                    "callable_reflection_forbidden",
                    "reflection callables may not be bound through partial adapters",
                )
                return _Fact(
                    callable=_Callable("ambiguous-sensitive.*"),
                    sensitive_callable_maybe=True,
                )
            if _fact_contains_synthetic_git_helper(target_fact):
                self.add(
                    call,
                    "synthetic_git_helper_escape_forbidden",
                    "the synthetic Git helper may not be bound through partial adapters",
                )
                return _Fact(
                    callable=_Callable("ambiguous-sensitive.*"),
                    sensitive_callable_maybe=True,
                )
            bound_args = tuple(self._eval(item, env) for item in call.args[1:])
            keywords, unknown, expansions = self._keywords(call, env)
            bound_facts = (*bound_args, *keywords.values(), *expansions)
            self._reject_network_capability_escape(call, env, bound_facts)
            self._reject_opaque_security_capability(call, bound_facts)
            if any(_fact_contains_sensitive_callable(item) for item in bound_facts):
                self.add(
                    call,
                    "sensitive_callable_escape",
                    "sensitive callable bound through a higher-order adapter",
                )
            return _Fact(
                callable=_Callable(
                    target.name,
                    (*target.bound_args, *bound_args),
                    (*target.bound_keywords, *keywords.items()),
                    target.unknown_keywords or unknown,
                )
            )

        positional, keywords, unknown_keywords, expansions = self._effective_arguments(
            call, callable_value, env
        )
        argument_facts = (*positional, *keywords.values(), *expansions)
        if any(_fact_contains_synthetic_git_helper(item) for item in argument_facts):
            self.add(
                call,
                "synthetic_git_helper_escape_forbidden",
                "the synthetic Git helper may not enter a higher-order call",
            )
        if (
            self.path == "tests/unit/assets/conftest.py"
            and name.startswith("local:tests/unit/assets/conftest.py:")
            and name.endswith(":make_reference")
            and (len(positional) < 2 or not positional[1].test_safe_relative)
        ):
            self.add(
                call,
                "synthetic_git_path_argument_forbidden",
                "synthetic Git helper paths require a proven safe relative call argument",
            )
        all_taints = _union_taints(
            (
                *(item.taints for item in positional),
                *(item.taints for item in keywords.values()),
                *(item.taints for item in expansions),
            )
        )
        self._reject_network_capability_escape(
            call,
            env,
            argument_facts,
        )
        if any(_fact_contains_sensitive_callable(item) for item in argument_facts):
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
            if any(_fact_contains_model_root(item) for item in argument_facts):
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

        if not self._model_loader(name):
            self._reject_opaque_security_capability(call, argument_facts)

        canonical = _canonical_callable_name(name)
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
                self.add(
                    call,
                    "unverified_model_source",
                    "model source is not a live A001 verified root",
                )
            local_only = keywords.get("local_files_only", _Fact())
            if not local_only.literal_known or local_only.literal is not True:
                self.add(
                    call,
                    "model_network_enabled",
                    "model loader must set local_files_only=True",
                )
            if "trust_remote_code" in keywords:
                self.add(
                    call,
                    "remote_code_option_forbidden",
                    "trust_remote_code is forbidden",
                )
            if "cache_dir" in keywords:
                self.add(call, "model_cache_option_forbidden", "cache_dir is forbidden")
            if unknown_keywords:
                self.add(
                    call,
                    "unknown_model_loader_kwargs",
                    "model loader contains unprovable **kwargs",
                )
        elif canonical.endswith(".*") and canonical.startswith(
            (*_MODEL_LOADER_PREFIXES, "huggingface_hub.")
        ):
            self.add(
                call,
                "ambiguous_sensitive_callable",
                "computed callable cannot be proven safe",
            )
        elif canonical in _DOWNLOAD_CALLS:
            self.add(
                call,
                "forbidden_download",
                f"download API is not the locked dataset transport: {canonical}",
            )

        if canonical in _DYNAMIC_IMPORT_CALLS:
            self._record_sink(
                call,
                "reference_dynamic_import",
                all_taints,
                "dynamically imports reference code",
            )
            if self.path.startswith("src/sakuramoon/"):
                self.add(
                    call,
                    "dynamic_import_forbidden",
                    "production dynamic imports cannot prove execution provenance",
                )
                return _Fact(callable=_Callable("ambiguous-sensitive.*"))
        if canonical in _DYNAMIC_CODE_CALLS:
            self._record_sink(
                call,
                "reference_dynamic_exec",
                all_taints,
                "dynamically executes reference code",
            )
            if self.path.startswith("src/sakuramoon/"):
                self.add(
                    call,
                    "dynamic_code_forbidden",
                    "production dynamic code evaluation is forbidden",
                )
                return _Fact(
                    network_capability_maybe=True,
                    sensitive_callable_maybe=True,
                )
        is_process = canonical in _PROCESS_EXACT or canonical.startswith(
            _PROCESS_PREFIXES
        )
        if canonical.endswith(".*") and canonical.startswith(
            ("asyncio.", "os.", "subprocess.")
        ):
            is_process = True
        if is_process and not self._safe_test_git(call, canonical, env):
            self._record_sink(
                call,
                "reference_process_exec",
                all_taints,
                "passes reference data to a process",
            )
        if canonical in _SEARCH_PATH_CALLS:
            self._record_sink(
                call,
                "reference_search_path",
                all_taints,
                "injects reference into Python search paths",
            )

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
            self._record_sink(
                target,
                "reference_search_path",
                fact.taints,
                "assigns reference into sys.path",
            )
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
            fact = _Fact(taints=frozenset({f"parameter:{index}"}))
            if (
                index == 0
                and self.path == "tests/unit/assets/conftest.py"
                and self.class_context is None
                and function_name == "make_reference"
            ):
                fact = replace(fact, test_root_path=True)
            if (
                index == 0
                and self.path == "tests/unit/assets/test_inspect.py"
                and self.class_context is None
                and function_name
                in {
                    "test_reference_git_audit_disables_hostile_local_configuration",
                    "test_reference_origin_diagnostic_redacts_credentials",
                }
            ):
                fact = replace(fact, test_root_holder=True)
            env.assign(parameter, fact)
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
        if self.path == DATASET_TRANSPORT_PATH:
            if (
                self.class_context == DATASET_TRANSPORT_CLASS
                and function_name in {"_close_response", "_read_response"}
                and "response" in parameter_names
            ):
                response = env.get("response")
                env.assign(
                    "response",
                    replace(
                        response,
                        instance_class="http.client.HTTPResponse",
                    ),
                )
            if (
                self.class_context == DATASET_TRANSPORT_CLASS
                and function_name == "_close_response"
                and "connection" in parameter_names
            ):
                connection = env.get("connection")
                env.assign(
                    "connection",
                    replace(
                        connection,
                        instance_class="http.client.HTTPSConnection",
                    ),
                )
            if (
                self.class_context is None
                and function_name
                in {"_parse_content_length", "_validate_download_headers"}
                and "response" in parameter_names
            ):
                response = env.get("response")
                env.assign(
                    "response",
                    replace(
                        response,
                        instance_class="http.client.HTTPResponse",
                    ),
                )
        if arguments.vararg is not None:
            index = len(parameter_names)
            parameter_names = (*parameter_names, arguments.vararg.arg)
            env.assign(
                arguments.vararg.arg, _Fact(taints=frozenset({f"parameter:{index}"}))
            )
        if arguments.kwarg is not None:
            index = len(parameter_names)
            parameter_names = (*parameter_names, arguments.kwarg.arg)
            env.assign(
                arguments.kwarg.arg, _Fact(taints=frozenset({f"parameter:{index}"}))
            )
        return env, parameter_names

    def _function(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef, outer: _Environment
    ) -> _Fact:
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
            if (
                node.name == "_open_get"
                and not self._dataset_open_get_constructor_shape_allowed(node)
            ):
                self.add(
                    node,
                    "dataset_connection_factory_forbidden",
                    "HTTPS construction must use the exact redacted initialization guard",
                )
        env, parameter_names = self._parameter_environment(node.args, outer, node.name)
        previous = self.context
        self.context = _FunctionContext(node.name, parameter_names)
        self._block(node.body, env)
        summary = _FunctionSummary(
            parameters=parameter_names,
            return_fact=self.context.return_fact or _Fact(),
            sink_parameters=tuple(
                sorted(
                    (code, tuple(sorted(indices)))
                    for code, indices in self.context.sink_parameters.items()
                )
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
        return _Fact(
            callable=_Callable(identifier),
            synthetic_git_helper=(
                self.path == "tests/unit/assets/conftest.py"
                and self.class_context is None
                and node.name == "make_reference"
            ),
        )

    def _lambda(self, node: ast.Lambda, outer: _Environment) -> _Fact:
        identifier = self._function_identifier(node)
        env, parameter_names = self._parameter_environment(node.args, outer, "<lambda>")
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
        if (
            self.path.startswith("src/sakuramoon/")
            and self.context.direct_sink_parameters
        ):
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
            fact = self._eval(default, env)
            if (
                default is not None
                and _fact_contains_security_capability(fact)
                and (
                    self.path.startswith("src/sakuramoon/")
                    or _fact_contains_synthetic_git_helper(fact)
                )
            ):
                self.add(
                    default,
                    "security_capability_default_forbidden",
                    "function defaults may not capture execution security capabilities",
                )
        annotations = [
            *(item.annotation for item in node.args.posonlyargs),
            *(item.annotation for item in node.args.args),
            *(item.annotation for item in node.args.kwonlyargs),
            node.args.vararg.annotation if node.args.vararg is not None else None,
            node.args.kwarg.annotation if node.args.kwarg is not None else None,
            node.returns,
        ]
        for annotation in annotations:
            fact = self._eval(annotation, env)
            if (
                annotation is not None
                and self.path.startswith("src/sakuramoon/")
                and _fact_contains_security_capability(fact)
                and not self._audited_network_type_annotation(annotation)
            ):
                self.add(
                    annotation,
                    "security_capability_annotation_forbidden",
                    "function annotations may not capture execution security capabilities",
                )

    def _audited_network_type_annotation(self, node: ast.AST) -> bool:
        if isinstance(node, ast.Constant) and node.value is None:
            return True
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
            return self._audited_network_type_annotation(
                node.left
            ) and self._audited_network_type_annotation(node.right)
        return self._place(node, _Environment()) in {
            "_ValidatedHttpTarget",
            "http.client.HTTPResponse",
            "http.client.HTTPSConnection",
        }

    def _declared_class_names(self, statements: list[ast.stmt]) -> set[str]:
        names: set[str] = set()

        def collect(target: ast.AST) -> None:
            if isinstance(target, ast.Name):
                names.add(target.id)
            elif isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    collect(item)

        for statement in statements:
            if isinstance(
                statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            ):
                names.add(statement.name)
            elif isinstance(statement, ast.Assign):
                for target in statement.targets:
                    collect(target)
            elif isinstance(statement, (ast.AnnAssign, ast.AugAssign)):
                collect(statement.target)
        return names

    def _class(self, node: ast.ClassDef, env: _Environment) -> _Fact:
        identifier = self._class_identifier(node)
        if (
            self.path == DATASET_TRANSPORT_PATH
            and node.name == DATASET_TRANSPORT_CLASS
            and hashlib.sha256(
                ast.dump(node, include_attributes=False).encode("utf-8")
            ).hexdigest()
            != _DATASET_TRANSPORT_CLASS_AST_SHA256
        ):
            self.add(
                node,
                "dataset_transport_structure_forbidden",
                "the frozen dataset transport class normalized AST changed",
            )
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
            name: class_env.get(name) for name in self._declared_class_names(node.body)
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

    def _bind_pattern(
        self, pattern: ast.pattern, fact: _Fact, env: _Environment
    ) -> None:
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

    def _loop_fixed_point(
        self,
        node: ast.For | ast.AsyncFor | ast.While,
        env: _Environment,
    ) -> _Environment:
        state = env.clone()
        max_passes = 3 + len(node.body) + len(env.values)
        converged = False
        break_exits: list[_Environment] = []
        for _ in range(max_passes):
            body = state.clone()
            if isinstance(node, (ast.For, ast.AsyncFor)):
                iterable = self._eval(node.iter, body)
                if iterable.items:
                    item_fact = iterable.items[0]
                    for candidate in iterable.items[1:]:
                        item_fact = _merge_facts(item_fact, candidate)
                else:
                    item_fact = iterable
                self._assign(node.target, item_fact, body)
            else:
                self._eval(node.test, body)
            self.loop_break_environments.append([])
            self.loop_continue_environments.append([])
            self._block(node.body, body)
            current_breaks = self.loop_break_environments.pop()
            current_continues = self.loop_continue_environments.pop()
            break_exits.extend(current_breaks)
            iteration_states = [*current_continues]
            if not self._statements_definitely_terminate(node.body):
                iteration_states.append(body)
            next_state = _merge_environments(env, *iteration_states)
            if next_state.values == state.values:
                state = next_state
                converged = True
                break
            state = next_state
        if not converged:
            self.add(
                node,
                "loop_analysis_did_not_converge",
                "loop-carried execution provenance did not reach a fixed point",
            )
        normal_exit = state.clone()
        self._block(node.orelse, normal_exit)
        return _merge_environments(normal_exit, *break_exits)

    @classmethod
    def _statements_definitely_terminate(cls, statements: list[ast.stmt]) -> bool:
        if not statements:
            return False
        final = statements[-1]
        if isinstance(final, (ast.Break, ast.Continue, ast.Raise, ast.Return)):
            return True
        if isinstance(final, ast.If):
            return (
                bool(final.orelse)
                and cls._statements_definitely_terminate(final.body)
                and cls._statements_definitely_terminate(final.orelse)
            )
        return False

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
                            "sakuramoon.assets",
                            "site",
                            "socket",
                            "subprocess",
                            "transformers",
                            "urllib.request",
                        )
                    ) or (node.level > 0 and node.module.split(".")[-1] == "assets"):
                        self.add(
                            node,
                            "sensitive_star_import_forbidden",
                            "star import from a model/download package is not auditable",
                        )
                else:
                    env.assign(
                        item.asname or item.name,
                        _Fact(callable=_Callable(f"{node.module}.{item.name}")),
                    )
            if _is_reference_text(node.module):
                self.add(
                    node, "reference_import", "engineering code imports reference code"
                )
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
                self._reject_namespace_assignment(node, target)
                if self._dataset_bounded_remaining_assignment(node, target, env):
                    fact = replace(fact, bounded_nonnegative=True)
                if self._dataset_listing_payload_initialization(node, target):
                    fact = replace(fact, listing_payload_bounded=True)
                self._reject_dataset_network_binding_assignment(node, target, fact)
                if self._reject_dataset_header_write(node, target, env):
                    fact = _without_network_capabilities(fact)
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
                fact = self._eval(node.value, env)
                self._reject_namespace_assignment(node, node.target)
                self._reject_dataset_network_binding_assignment(node, node.target, fact)
                if self._reject_dataset_header_write(node, node.target, env):
                    fact = _without_network_capabilities(fact)
                if self.path.startswith("src/sakuramoon/") and self._tls_policy_target(
                    node.target
                ):
                    self.add(
                        node,
                        "tls_policy_mutation_forbidden",
                        "TLS hostname and certificate verification may not be changed",
                    )
                self._assign(node.target, fact, env)
        elif isinstance(node, ast.AugAssign):
            target_fact = self._eval(node.target, env)
            value_fact = self._eval(node.value, env)
            self._reject_namespace_assignment(node, node.target)
            rejected_header_write = self._reject_dataset_header_write(
                node, node.target, env
            )
            if self.path.startswith("src/sakuramoon/") and self._tls_policy_target(
                node.target
            ):
                self.add(
                    node,
                    "tls_policy_mutation_forbidden",
                    "TLS hostname and certificate verification may not be changed",
                )
            fact = _merge_facts(target_fact, value_fact)
            if rejected_header_write:
                fact = _without_network_capabilities(fact)
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
        elif isinstance(node, ast.Break):
            if self.loop_break_environments:
                self.loop_break_environments[-1].append(env.clone())
        elif isinstance(node, ast.Continue):
            if self.loop_continue_environments:
                self.loop_continue_environments[-1].append(env.clone())
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
            if self._dataset_listing_payload_upper_bound_guard(node):
                payload = other.get("payload")
                other.assign(
                    "payload",
                    replace(payload, listing_payload_bounded=True),
                )
            self._block(node.body, body)
            self._block(node.orelse, other)
            continuing: list[_Environment] = []
            if not self._statements_definitely_terminate(node.body):
                continuing.append(body)
            if not self._statements_definitely_terminate(node.orelse):
                continuing.append(other)
            if continuing:
                env = _merge_environments(*continuing)
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            env = self._loop_fixed_point(node, env)
        elif isinstance(node, (ast.Try, ast.TryStar)):
            branches: list[_Environment] = []
            body = env.clone()
            self._block(node.body, body)
            self._block(node.orelse, body)
            if not self._statements_definitely_terminate([*node.body, *node.orelse]):
                branches.append(body)
            for handler in node.handlers:
                branch = env.clone()
                self._eval(handler.type, branch)
                if handler.name:
                    branch.assign(handler.name, _Fact())
                self._block(handler.body, branch)
                if not self._statements_definitely_terminate(handler.body):
                    branches.append(branch)
            if branches:
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
                        self.add(
                            statement,
                            "reference_import",
                            "engineering code imports reference code",
                        )
            current = self._statement(statement, current)
            if self._statements_definitely_terminate([statement]):
                break
        if current is not env:
            env.values = current.values
        return env

    def analyze(self, tree: ast.Module) -> tuple[BoundaryViolation, ...]:
        max_passes = 3 + sum(
            isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
            )
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
                    "globals": _Fact(callable=_Callable("builtins.globals")),
                    "locals": _Fact(callable=_Callable("builtins.locals")),
                    "object": _Fact(callable=_Callable("builtins.object")),
                    "print": _Fact(callable=_Callable("builtins.print")),
                    "setattr": _Fact(callable=_Callable("builtins.setattr")),
                    "type": _Fact(callable=_Callable("builtins.type")),
                    "vars": _Fact(callable=_Callable("builtins.vars")),
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
            raise SourceBoundaryError(
                f"source path contains symlink: {relative.as_posix()}"
            )
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise SourceBoundaryError(
            f"source path is missing or escapes root: {relative.as_posix()}"
        ) from exc


def _read_source(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise SourceBoundaryError("source path escapes repository root") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise SourceBoundaryError("source path contains an unsafe component")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    file_flags = (
        os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    )
    directory_descriptor = -1
    descriptor = -1
    try:
        directory_descriptor = os.open(root, directory_flags)
        for part in relative.parts[:-1]:
            child_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=directory_descriptor,
            )
            os.close(directory_descriptor)
            directory_descriptor = child_descriptor
        leaf = relative.parts[-1]
        before = os.stat(
            leaf,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            os.close(directory_descriptor)
            directory_descriptor = -1
            raise SourceBoundaryError(
                "source path is a symlink or is not a regular file"
            )
        descriptor = os.open(
            leaf,
            file_flags,
            dir_fd=directory_descriptor,
        )
    except OSError as exc:
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
            directory_descriptor = -1
        raise SourceBoundaryError(
            "source path cannot be opened through anchored no-follow descriptors"
        ) from exc
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SourceBoundaryError("source identity changed before read")
        with os.fdopen(descriptor, "r", encoding="utf-8") as handle:
            descriptor = -1
            source = handle.read()
        after = os.stat(
            leaf,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except (OSError, UnicodeError) as exc:
        raise SourceBoundaryError(
            "source path changed or could not be decoded"
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if directory_descriptor >= 0:
            os.close(directory_descriptor)
    if (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ) != (
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
                raise SourceBoundaryError(
                    "source directory cannot be enumerated"
                ) from exc
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
    return tuple(
        sorted(
            violations, key=lambda item: (item.path, item.line, item.code, item.detail)
        )
    )
