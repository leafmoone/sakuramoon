from __future__ import annotations

from pathlib import Path

import pytest

import tools.asset_execution_boundary as boundary
from sakuramoon.assets import load_manifest
from tools.asset_execution_boundary import (
    SourceBoundaryError,
    python_sources,
    scan_file,
    scan_repository,
    scan_source,
)

ROOT = Path(__file__).parents[3]
MANIFEST = ROOT / "assets/manifest.toml"
MODEL_LOCAL_PATHS = {
    "qwen": "model/qwen_3.5_2B",
    "vae": "model/vae",
}


def _codes(source: str, path: str = "src/sakuramoon/encoders/qwen.py") -> set[str]:
    return {item.code for item in scan_source(source, path)}


def test_manifest_locks_the_two_prepared_model_directories() -> None:
    manifest = load_manifest(MANIFEST)

    assert {asset.kind: asset.local_path for asset in manifest.models} == MODEL_LOCAL_PATHS
    assert all(asset.lock_state == "ready" for asset in manifest.models)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """
from transformers import AutoModel
AutoModel.from_pretrained("other/repo", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from pathlib import Path
from transformers import AutoModel
AutoModel.from_pretrained(Path("/tmp/cache/model"), local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from transformers import AutoModel
loader = AutoModel.from_pretrained
loader("other/repo", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from transformers import AutoModel
loader = getattr(AutoModel, "from_pretrained")
loader("other/repo", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from functools import partial
from transformers import AutoModel
loader = partial(AutoModel.from_pretrained)
loader("other/repo", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from transformers import AutoModel
from sakuramoon.assets import VerifiedAssetSelection
def load(selection: VerifiedAssetSelection):
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(root, local_files_only=False)
""",
            "model_network_enabled",
        ),
        (
            """
from transformers import AutoModel
from sakuramoon.assets import VerifiedAssetSelection
def load(selection: VerifiedAssetSelection):
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(
        root, local_files_only=True, trust_remote_code=True
    )
""",
            "remote_code_option_forbidden",
        ),
        (
            """
from transformers import AutoModel
from sakuramoon.assets import VerifiedAssetSelection
def load(selection: VerifiedAssetSelection):
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(
        root, local_files_only=True, cache_dir="/tmp/model-cache"
    )
""",
            "model_cache_option_forbidden",
        ),
        (
            """
from transformers import AutoModel
class VerifiedAssetSelection:
    def verified_root(self, asset_id):
        return "model/qwen_3.5_2B"
def load(selection: VerifiedAssetSelection):
    return AutoModel.from_pretrained(
        selection.verified_root("qwen_text_encoder"), local_files_only=True
    )
""",
            "unverified_model_source",
        ),
    ],
)
def test_model_loader_bypasses_are_rejected(source: str, expected: str) -> None:
    assert expected in _codes(source)


def test_verified_model_roots_are_the_only_pretrained_source() -> None:
    source = """
from transformers import AutoModel
from sakuramoon.assets import require_verified_selection
def load(value):
    selection = require_verified_selection(value)
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(root, local_files_only=True)
"""

    assert scan_source(source, "src/sakuramoon/encoders/qwen.py") == ()


@pytest.mark.parametrize(
    "path",
    [
        "src/sakuramoon/data/model_fetch.py",
        "src/sakuramoon/data/modelscope.py",
        "src/sakuramoon/cli/manifest.py",
        "src/sakuramoon/encoders/qwen.py",
    ],
)
def test_model_download_cannot_hide_in_dataset_or_cli_paths(path: str) -> None:
    source = """
from modelscope.hub.snapshot_download import snapshot_download
def fetch_dataset_shard():
    return snapshot_download("other/model")
"""

    assert "forbidden_download" in _codes(source, path)


@pytest.mark.parametrize(
    "source",
    [
        """
from modelscope.hub.snapshot_download import snapshot_download
def fetch_dataset_shard():
    return snapshot_download(
        "leafmoone/webdataset_danbooru",
        revision="0123456789abcdef0123456789abcdef01234567",
        repo_type="model",
    )
""",
        """
from modelscope.hub.snapshot_download import snapshot_download
def fetch_dataset_shard():
    return snapshot_download(
        "other/dataset",
        revision="0123456789abcdef0123456789abcdef01234567",
        repo_type="dataset",
    )
""",
        """
from modelscope.hub.snapshot_download import snapshot_download
def fetch_dataset_shard():
    return snapshot_download(
        "leafmoone/webdataset_danbooru",
        revision="main",
        repo_type="dataset",
    )
""",
    ],
)
def test_dataset_transport_requires_locked_identity(source: str) -> None:
    assert "forbidden_download" in _codes(source, "src/sakuramoon/data/modelscope.py")


def test_legacy_snapshot_dataset_transport_is_forbidden() -> None:
    source = """
from modelscope.hub.snapshot_download import snapshot_download
from sakuramoon.config import load_config
def fetch_dataset_shard():
    loaded = load_config("stage.toml", config_root="config")
    return snapshot_download(
        "leafmoone/webdataset_danbooru",
        revision=loaded.config.data.source.revision,
        repo_type="dataset",
    )
"""

    assert "forbidden_download" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


def test_exact_d010_stdlib_https_open_shape_is_allowed() -> None:
    source = """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _request_headers(self, target):
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
    def _open_get(self, target, *, range_start):
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
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={range_start}-"
        connection.request(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
        response = connection.getresponse()
        connection.sock.settimeout(self._policy.read_timeout_seconds)
        return response, connection
"""

    assert scan_source(source, "src/sakuramoon/data/modelscope.py") == ()


def test_exact_d010_target_factories_and_verified_entrypoints_are_allowed() -> None:
    source = """
from dataclasses import dataclass
from urllib.parse import urlencode
from sakuramoon.data.manifest import require_verified_dataset_manifest
MODELSCOPE_DATASET_HOST = "modelscope.cn"
_HTTPS_PORT = 443
@dataclass(frozen=True)
class _ValidatedHttpTarget:
    host: str
    port: int
    request_target: str
    send_authorization: bool
def _source_path(source):
    return source.repo_id
def _listing_target(manifest, page_number, page_size):
    query = urlencode({
        "Revision": manifest.source.revision,
        "Recursive": "True",
        "PageNumber": page_number,
        "PageSize": page_size,
    })
    return _ValidatedHttpTarget(
        host=MODELSCOPE_DATASET_HOST,
        port=_HTTPS_PORT,
        request_target=f"/api/v1/datasets/{_source_path(manifest.source)}/repo/tree?{query}",
        send_authorization=True,
    )
def _shard_target(manifest, shard):
    query = urlencode({
        "Revision": manifest.source.revision,
        "FilePath": shard.path,
    })
    return _ValidatedHttpTarget(
        host=MODELSCOPE_DATASET_HOST,
        port=_HTTPS_PORT,
        request_target=f"/api/v1/datasets/{_source_path(manifest.source)}/repo?{query}",
        send_authorization=True,
    )
def _redirect_target(current, location, allowed_hosts):
    host = location
    request_target = location
    return _ValidatedHttpTarget(
        host=host,
        port=_HTTPS_PORT,
        request_target=request_target,
        send_authorization=current.send_authorization and host == MODELSCOPE_DATASET_HOST,
    )
class ModelScopeDatasetTransport:
    def list_locked_files(self, selection):
        manifest = require_verified_dataset_manifest(selection).manifest
        page_number = 1
        target = _listing_target(
            manifest,
            page_number,
            self._policy.listing_page_size,
        )
        return self._read_listing_once(target)
    def download_locked_shard_to_staging(self, selection, shard_path, range_start):
        manifest = require_verified_dataset_manifest(selection).manifest
        shard = manifest.shard(shard_path)
        target = _shard_target(manifest, shard)
        return self._follow_redirects(target, range_start=range_start)
"""

    assert scan_source(source, "src/sakuramoon/data/modelscope.py") == ()


def test_exact_d010_response_access_shape_is_allowed() -> None:
    source = """
class ModelScopeDatasetTransport:
    def _follow_redirects(self, target, *, range_start):
        current = target
        response, connection = self._open_get(current, range_start=range_start)
        location = response.getheader("Location")
        self._close_response(response, connection)
        current = _redirect_target(
            current,
            location,
            self._policy.redirect_hosts,
        )
        return response, connection
    @staticmethod
    def _close_response(response, connection):
        response.close()
        connection.close()
    @staticmethod
    def _read_response(response, length):
        return response.read(length)
def _parse_content_length(response):
    return response.getheader("Content-Length")
def _validate_download_headers(response):
    return response.getheader("Content-Encoding"), response.getheader("Content-Range")
"""

    assert scan_source(source, "src/sakuramoon/data/modelscope.py") == ()


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import modelscope_hub", "dataset_transport_import_forbidden"),
        ("import requests", "dataset_transport_import_forbidden"),
        (
            """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        return http.client.HTTPSConnection(
            host="modelscope.cn",
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
""",
            "network_call_forbidden",
        ),
        (
            """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        target = untrusted_target
        return http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
""",
            "network_call_forbidden",
        ),
        (
            """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        return http.client.HTTPConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
        )
""",
            "network_call_forbidden",
        ),
        (
            """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        return http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl._create_unverified_context(),
        )
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection.request(
            "POST",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        other.request(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection.request(
            "GET",
            "/changed",
            body=None,
            headers=headers,
            encode_chunked=False,
        )
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection.request(
            "GET",
            target.request_target,
            **request_options,
        )
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        request = connection.request
        request(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _request_headers(self, target):
        return injected_headers
""",
            "dataset_headers_factory_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        headers.update(extra_headers)
""",
            "dataset_headers_mutation_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def _follow_redirects(self, target, *, range_start):
        current = untrusted_target
        return self._open_get(current, range_start=range_start)
""",
            "network_helper_call_forbidden",
        ),
        (
            """
class _ValidatedHttpTarget: pass
def make_target(host, request_target):
    return _ValidatedHttpTarget(
        host=host,
        port=443,
        request_target=request_target,
        send_authorization=True,
    )
""",
            "network_target_construction_forbidden",
        ),
        (
            """
def helper(manifest, page_number, page_size):
    return _listing_target(manifest, page_number, page_size)
""",
            "network_target_factory_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def download_locked_shard_to_staging(self, response):
        return response.read(1024)
""",
            "network_call_forbidden",
        ),
        (
            """
def _parse_content_length(response, name):
    return response.getheader(name)
""",
            "network_call_forbidden",
        ),
        (
            """
class ModelScopeDatasetTransport:
    def patch(self, transport):
        setattr(transport, "_open_get", replacement)
""",
            "dataset_transport_mutation_forbidden",
        ),
        (
            """
import ssl
context = ssl.create_default_context()
context.check_hostname = False
""",
            "tls_policy_mutation_forbidden",
        ),
    ],
)
def test_d010_https_transport_rejects_unverified_or_changed_shapes(
    source: str,
    expected: str,
) -> None:
    assert expected in _codes(source, "src/sakuramoon/data/modelscope.py")


def test_exact_d010_https_constructor_is_rejected_outside_locked_location() -> None:
    source = """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        return http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
"""

    assert "network_call_forbidden" in _codes(
        source, "src/sakuramoon/data/other.py"
    )


def test_exact_d010_https_constructor_is_rejected_outside_locked_class() -> None:
    source = """
import http.client
import ssl
class OtherTransport:
    def _open_get(self, target, *, range_start):
        return http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
"""

    assert "network_call_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


def _d010_header_attack(statement: str) -> str:
    return f'''
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={{range_start}}-"
        {statement}
        connection.request(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
'''


@pytest.mark.parametrize(
    "statement",
    [
        'headers.__setitem__("Host", "attacker.invalid")',
        'dict.__setitem__(headers, "Host", "attacker.invalid")',
        'headers.__ior__({"Host": "attacker.invalid"})',
        'operator.setitem(headers, "Host", "attacker.invalid")',
        'mapping = headers\n        mapping.__setitem__("Host", "attacker.invalid")',
        'mutate_headers(headers)',
    ],
)
def test_d010_audited_headers_reject_every_non_exact_mutation_or_escape(
    statement: str,
) -> None:
    source = _d010_header_attack(statement)
    if "operator." in statement:
        source = f"import operator\n{source}"

    codes = _codes(source, "src/sakuramoon/data/modelscope.py")
    assert "network_capability_escape_forbidden" in codes
    assert "network_call_forbidden" in codes


def test_d010_audited_headers_reject_in_place_mapping_merge() -> None:
    source = _d010_header_attack('headers |= {"Host": "attacker.invalid"}')

    codes = _codes(source, "src/sakuramoon/data/modelscope.py")
    assert "dataset_headers_mutation_forbidden" in codes
    assert "network_call_forbidden" in codes


@pytest.mark.parametrize("capability", ["target", "headers", "connection"])
def test_d010_network_capability_cannot_enter_opaque_calls(
    capability: str,
) -> None:
    source = f'''
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection = http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={{range_start}}-"
        external({capability})
        connection.request(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
'''

    codes = _codes(source, "src/sakuramoon/data/modelscope.py")
    assert "network_capability_escape_forbidden" in codes
    assert "network_call_forbidden" in codes


@pytest.mark.parametrize(
    "statement",
    [
        "external(*[headers])",
        "external(*(target,))",
        "values = [connection]\n        external(*values)",
        "values = (headers,)\n        external(*values)",
        "external(*((target,),))",
        "external(*(item for item in (connection,)))",
        (
            "values = [headers] if range_start is None else unknown_values"
            "\n        external(*values)"
        ),
        "external(*unknown_values, headers)",
    ],
)
def test_d010_starred_positional_expansion_preserves_network_facts(
    statement: str,
) -> None:
    source = f'''
import http.client
import ssl
from sakuramoon.adapters import external
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection = http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={{range_start}}-"
        {statement}
        connection.request(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )
'''

    codes = _codes(source, "src/sakuramoon/data/modelscope.py")
    assert "network_capability_escape_forbidden" in codes
    assert "network_call_forbidden" in codes


@pytest.mark.parametrize(
    "statement",
    [
        'options = {"headers": headers}\n        external(**options)',
        (
            'options = {"nested": {"target": target}}'
            "\n        external(**options)"
        ),
        (
            'options = {"connection": connection} if range_start is None '
            "else unknown_options\n        external(**options)"
        ),
        "external(**unknown_options, headers=headers)",
    ],
)
def test_d010_keyword_expansion_preserves_nested_network_facts(
    statement: str,
) -> None:
    source = f'''
import http.client
import ssl
from sakuramoon.adapters import external
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection = http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={{range_start}}-"
        {statement}
'''

    assert "network_capability_escape_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


@pytest.mark.parametrize(
    "adapter",
    [
        '''operator.methodcaller(
            "request",
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )(connection)''',
        '''operator.attrgetter("request")(connection)(
            "GET",
            target.request_target,
            body=None,
            headers=headers,
            encode_chunked=False,
        )''',
        'operator.methodcaller("getresponse")(connection)',
        'operator.methodcaller("close")(connection)',
    ],
)
def test_d010_higher_order_network_adapters_are_rejected(adapter: str) -> None:
    source = f'''
import http.client
import operator
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection = http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={{range_start}}-"
        {adapter}
'''

    assert "network_capability_escape_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


@pytest.mark.parametrize(
    ("statement", "expected"),
    [
        (
            'locals()["headers"].__setitem__("Host", "attacker.invalid")',
            "namespace_reflection_forbidden",
        ),
        (
            'vars()["headers"].__setitem__("Host", "attacker.invalid")',
            "namespace_reflection_forbidden",
        ),
        ('eval("headers.clear()")', "dynamic_code_forbidden"),
        ('exec("headers.clear()")', "dynamic_code_forbidden"),
        (
            'inspect.currentframe().f_locals["headers"].clear()',
            "namespace_reflection_forbidden",
        ),
        (
            'getattr(self, "_open_get")(target, range_start=range_start)',
            "callable_reflection_forbidden",
        ),
        (
            'operator.attrgetter("_open_get")(self)('
            + "target, range_start=range_start)",
            "callable_reflection_forbidden",
        ),
    ],
)
def test_d010_dynamic_namespace_and_callable_reflection_is_forbidden(
    statement: str,
    expected: str,
) -> None:
    source = f'''
import http.client
import inspect
import operator
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection = http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        headers = self._request_headers(target)
        if range_start is not None:
            headers["Range"] = f"bytes={{range_start}}-"
        {statement}
'''

    assert expected in _codes(source, "src/sakuramoon/data/modelscope.py")


def test_d010_target_constructor_alias_is_rejected() -> None:
    source = """
class _ValidatedHttpTarget: pass
Target = _ValidatedHttpTarget
def helper(host, request_target):
    return Target(
        host=host,
        port=_HTTPS_PORT,
        request_target=request_target,
        send_authorization=True,
    )
"""

    assert "network_target_construction_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


@pytest.mark.parametrize(
    ("method", "length"),
    [
        ("_read_listing_once", "requested_length"),
        ("download_locked_shard_to_staging", "requested_length"),
        ("download_locked_shard_to_staging", "self._policy.stream_chunk_bytes"),
    ],
)
def test_d010_read_response_length_is_locked_to_exact_bounded_shapes(
    method: str,
    length: str,
) -> None:
    source = f'''
class ModelScopeDatasetTransport:
    def {method}(self, target):
        response, connection = self._follow_redirects(target, range_start=None)
        return self._read_response(response, {length})
'''

    assert "network_helper_call_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


def test_d010_listing_remaining_requires_live_nonnegative_provenance() -> None:
    source = """
class ModelScopeDatasetTransport:
    def _read_listing_once(self, target):
        response, connection = self._follow_redirects(target, range_start=None)
        payload = bytearray()
        remaining = _LISTING_RESPONSE_LIMIT_BYTES + 1 - len(payload)
        remaining = -1
        return self._read_response(
            response,
            min(self._policy.stream_chunk_bytes, remaining),
        )
"""

    assert "network_helper_call_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


def test_d010_unwrapped_connection_constructor_is_rejected() -> None:
    source = """
import http.client
import ssl
class ModelScopeDatasetTransport:
    def _open_get(self, target, *, range_start):
        connection = http.client.HTTPSConnection(
            host=target.host,
            port=target.port,
            timeout=self._policy.connect_timeout_seconds,
            context=ssl.create_default_context(),
        )
        if range_start is not None:
            headers["Range"] = f"bytes={range_start}-"
"""

    assert "dataset_connection_factory_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


def test_d010_redirect_cleanup_must_use_the_audited_close_helper() -> None:
    source = """
class ModelScopeDatasetTransport:
    def _follow_redirects(self, target, *, range_start):
        response, connection = self._open_get(target, range_start=range_start)
        response.close()
        connection.close()
"""

    assert "network_call_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


@pytest.mark.parametrize("binding", ["response", "connection"])
def test_d010_network_bindings_reject_unverified_overwrite(binding: str) -> None:
    source = f'''
class ModelScopeDatasetTransport:
    def _read_listing_once(self, target):
        response, connection = self._follow_redirects(target, range_start=None)
        {binding} = attacker_controlled
        self._close_response(response, connection)
'''

    assert "network_binding_assignment_forbidden" in _codes(
        source, "src/sakuramoon/data/modelscope.py"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """
from sakuramoon.assets import require_runtime_assets_ready
from transformers import AutoModel
def load(config, manifest, repository):
    selection = require_runtime_assets_ready(config, manifest, root=repository)
    root = selection.verified_root("qwen_text_encoder")
    root = "other-org/other-model"
    return AutoModel.from_pretrained(root, local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
from transformers import AutoModel
from typing import cast
def load(value):
    selection = cast(VerifiedAssetSelection, value)
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(root, local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
from transformers import AutoModel
def load(selection: VerifiedAssetSelection):
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(root, local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from transformers import AutoModel
loader = getattr(AutoModel, "from_" + "pretrained")
loader("other/model", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from transformers import AutoModel
AutoModel.from_pretrained.__call__("other/model", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from modelscope import Model
Model.from_pretrained("other/model", local_files_only=True)
""",
            "unverified_model_source",
        ),
        (
            """
from sakuramoon.assets import require_runtime_assets_ready
from transformers import AutoModel
selection = require_runtime_assets_ready(config, manifest, root=repository)
root = selection.verified_root("qwen_text_encoder")
AutoModel.from_pretrained(
    root,
    local_files_only=True,
    **{"trust_remote_code": True},
)
""",
            "remote_code_option_forbidden",
        ),
        (
            """
from sakuramoon.assets import require_runtime_assets_ready
from transformers import AutoModel
selection = require_runtime_assets_ready(config, manifest, root=repository)
root = selection.verified_root("qwen_text_encoder")
AutoModel.from_pretrained(root, local_files_only=True, **{"cache_dir": "/tmp/cache"})
""",
            "model_cache_option_forbidden",
        ),
        (
            """
from sakuramoon.assets import require_runtime_assets_ready
from transformers import AutoModel
selection = require_runtime_assets_ready(config, manifest, root=repository)
root = selection.verified_root("qwen_text_encoder")
AutoModel.from_pretrained(root, local_files_only=True, **options)
""",
            "unknown_model_loader_kwargs",
        ),
    ],
)
def test_scope_sensitive_model_provenance_rejects_bypasses(
    source: str, expected: str
) -> None:
    assert expected in _codes(source)


@pytest.mark.parametrize(
    "body",
    [
        "external(*[AutoModel.from_pretrained])",
        (
            "callbacks = (AutoModel.from_pretrained,)"
            "\nexternal(*callbacks)"
        ),
        "external(*((AutoModel.from_pretrained,),))",
        (
            "external(*(callback for callback in "
            "(AutoModel.from_pretrained,)))"
        ),
        (
            "callbacks = [AutoModel.from_pretrained] if enabled else unknown"
            "\nexternal(*callbacks)"
        ),
        (
            'options = {"callback": AutoModel.from_pretrained}'
            "\nexternal(**options)"
        ),
        (
            'options = {"nested": {"callback": AutoModel.from_pretrained}}'
            "\nexternal(**options)"
        ),
    ],
)
def test_argument_expansion_preserves_nested_sensitive_callables(
    body: str,
) -> None:
    source = f'''
from sakuramoon.adapters import external
from transformers import AutoModel
{body}
'''

    assert "sensitive_callable_escape" in _codes(source)


@pytest.mark.parametrize(
    "container",
    [
        "{AutoModel.from_pretrained}",
        '{"model": AutoModel.from_pretrained}',
    ],
)
def test_pop_extraction_preserves_sensitive_loader_fact(
    container: str,
) -> None:
    argument = '"model"' if container.startswith("{") and ":" in container else ""
    source = f'''
from transformers import AutoModel
loaders = {container}
loader = loaders.pop({argument})
loader("remote/model")
'''

    codes = _codes(source)
    assert "unverified_model_source" in codes
    assert "model_network_enabled" in codes


@pytest.mark.parametrize("terminator", ["break", "continue"])
def test_loop_exit_facts_preserve_sensitive_loader_provenance(
    terminator: str,
) -> None:
    source = f'''
from transformers import AutoModel
loader = safe_loader
for item in values:
    loader = AutoModel.from_pretrained
    {terminator}
loader("remote/model")
'''

    codes = _codes(source)
    assert "unverified_model_source" in codes
    assert "loop_analysis_did_not_converge" not in codes


@pytest.mark.parametrize("expansion", ["*[root]", '**{"root": root}', "**options"])
def test_model_root_cannot_escape_through_argument_expansion(
    expansion: str,
) -> None:
    setup = 'options = {"root": root}\n' if expansion == "**options" else ""
    source = f'''
from sakuramoon.assets import require_runtime_assets_ready
from sakuramoon.encoders.external import construct
selection = require_runtime_assets_ready(config, manifest, root=repository)
root = selection.verified_root("qwen_text_encoder")
{setup}construct({expansion})
'''

    assert "model_root_cross_module_call_forbidden" in _codes(source)


def test_model_provenance_does_not_leak_across_functions() -> None:
    source = """
from sakuramoon.assets import require_runtime_assets_ready
from transformers import AutoModel
def valid(config, manifest, repository):
    selection = require_runtime_assets_ready(config, manifest, root=repository)
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(root, local_files_only=True)
def invalid():
    root = "other/model"
    return AutoModel.from_pretrained(root, local_files_only=True)
"""

    assert "unverified_model_source" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
from transformers import AutoModel
def construct(value=AutoModel.from_pretrained("other/model", local_files_only=True)):
    return value
""",
        """
from transformers import AutoModel
def construct(*, value=AutoModel.from_pretrained("other/model", local_files_only=True)):
    return value
""",
        """
from transformers import AutoModel
@AutoModel.from_pretrained("other/model", local_files_only=True)
def construct():
    pass
""",
        """
from transformers import AutoModel
assert AutoModel.from_pretrained("other/model", local_files_only=True)
""",
    ],
)
def test_model_loaders_in_definition_and_assert_expressions_are_rejected(
    source: str,
) -> None:
    assert "unverified_model_source" in _codes(source)


def test_sensitive_model_and_capability_subclasses_are_rejected() -> None:
    source = """
from sakuramoon.assets import VerifiedAssetFile, VerifiedAssetSelection
from transformers import AutoModel
class ForgedSelection(VerifiedAssetSelection):
    pass
class ForgedFile(VerifiedAssetFile):
    pass
class DerivedModel(AutoModel):
    pass
"""

    violations = scan_source(source, "src/sakuramoon/encoders/qwen.py")
    assert [item.code for item in violations].count("sensitive_subclass_forbidden") == 3


@pytest.mark.parametrize("capability", ["VerifiedAssetFile", "VerifiedAssetSelection"])
def test_verified_capability_construction_is_factory_restricted(
    capability: str,
) -> None:
    source = f"""
from sakuramoon.assets import {capability}
candidate = {capability}(None)
"""

    assert "capability_construction_forbidden" in _codes(source)


def test_capability_factory_name_does_not_bypass_exact_constructor_shape() -> None:
    source = """
from sakuramoon.assets.inspect import VerifiedAssetSelection
def _selection(snapshot, files):
    return VerifiedAssetSelection(
        manifest_revision=1,
        manifest_sha256="0" * 64,
        files=files,
        _root="model-cache",
        _manifest_relative_path="assets/manifest.toml",
        _manifest_path="assets/manifest.toml",
        _manifest_identity=None,
    )
"""

    assert "capability_construction_forbidden" in _codes(
        source, "src/sakuramoon/assets/inspect.py"
    )


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
candidate = object.__new__(VerifiedAssetSelection)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
constructor = getattr(object, "__new__")
candidate = constructor(VerifiedAssetSelection)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
constructor = getattr(object, constructor_name)
candidate = constructor(VerifiedAssetSelection)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
candidate = VerifiedAssetSelection.__new__(VerifiedAssetSelection)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
forged_type = type("Forged", (VerifiedAssetSelection,), {})
""",
            "sensitive_subclass_forbidden",
        ),
        (
            """
from sakuramoon.assets import require_verified_selection
selection = require_verified_selection(value)
object.__setattr__(selection, "_root", cache_root)
""",
            "capability_mutation_forbidden",
        ),
        (
            """
from dataclasses import replace
from sakuramoon.assets import require_verified_selection
selection = require_verified_selection(value)
candidate = replace(selection, _root=cache_root)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
import sakuramoon.assets as assets
constructor = getattr(assets, capability_name)
candidate = constructor(payload)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
from sakuramoon.assets import VerifiedAssetSelection
constructor = vars(VerifiedAssetSelection)[capability_name]
candidate = constructor(payload)
""",
            "capability_reflection_forbidden",
        ),
        (
            """
import inspect
from sakuramoon.assets import VerifiedAssetSelection
constructor = inspect.getattr_static(VerifiedAssetSelection, capability_name)
candidate = constructor(payload)
""",
            "capability_reflection_forbidden",
        ),
    ],
)
def test_reflective_capability_construction_and_mutation_are_rejected(
    source: str,
    expected: str,
) -> None:
    assert expected in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
from transformers import AutoModel
class Holder:
    loader = AutoModel.from_pretrained
loader = getattr(Holder, "loader")
loader("remote/model")
""",
        """
import operator
from transformers import AutoModel
class Holder:
    loader = AutoModel.from_pretrained
loader = operator.attrgetter("loader")(Holder)
loader("remote/model")
""",
    ],
)
def test_reflective_class_member_resolution_is_forbidden(source: str) -> None:
    assert "callable_reflection_forbidden" in _codes(source)


def test_object_getattribute_cannot_extract_verified_capability_members() -> None:
    source = """
from sakuramoon.assets import require_runtime_assets_ready
selection = require_runtime_assets_ready(config, manifest, root=repository)
method = object.__getattribute__(selection, "verified_root")
method("qwen_text_encoder")
"""

    assert "capability_reflection_forbidden" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
from transformers import AutoModel
loaders = [AutoModel.from_pretrained]
loaders[0]("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
loaders = {"qwen": AutoModel.from_pretrained}
loaders["qwen"]("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
loaders = []
loaders.append(AutoModel.from_pretrained)
loaders[0]("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
loaders = {}
loaders.update({"qwen": AutoModel.from_pretrained})
loaders["qwen"]("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
loader, ignored = (AutoModel.from_pretrained, print)
loader("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
def loader_factory():
    return AutoModel.from_pretrained
loader_factory()("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
class LoaderTable:
    loader = AutoModel.from_pretrained
LoaderTable.loader("other/model", local_files_only=True)
""",
        """
from transformers import AutoModel
class LoaderTable:
    pass
table = LoaderTable()
setattr(table, "loader", AutoModel.from_pretrained)
getattr(table, "loader")("other/model", local_files_only=True)
""",
    ],
)
def test_model_callable_provenance_survives_containers_helpers_and_classes(
    source: str,
) -> None:
    assert "unverified_model_source" in _codes(source)


def test_sensitive_callable_under_dynamic_attribute_is_denied_at_assignment() -> None:
    source = """
from transformers import AutoModel
class LoaderTable:
    pass
table = LoaderTable()
attribute = input()
setattr(table, attribute, AutoModel.from_pretrained)
"""

    assert "ambiguous_sensitive_callable" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
from transformers import *
AutoModel.from_pretrained("other/model", local_files_only=True)
""",
        """
from subprocess import *
run(["python", "reference/JLT/train.py"])
""",
        """
from http.client import *
HTTPSConnection("modelscope.cn")
""",
        """
from sakuramoon.assets import *
candidate = VerifiedAssetSelection(payload)
""",
        """
from ..assets import *
candidate = VerifiedAssetSelection(payload)
""",
    ],
)
def test_sensitive_star_import_is_rejected_without_guessing_exported_names(
    source: str,
) -> None:

    assert "sensitive_star_import_forbidden" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
from http.client import HTTPSConnection
HTTPSConnection("modelscope.cn")
""",
        """
from urllib.request import urlopen
urlopen("https://modelscope.cn/api/v1/datasets")
""",
        """
import requests
requests.get("https://modelscope.cn/api/v1/datasets")
""",
        """
import httpx
httpx.Client()
""",
        """
import aiohttp
aiohttp.ClientSession()
""",
        """
import socket
socket.create_connection(("modelscope.cn", 443))
""",
    ],
)
def test_generic_production_network_calls_are_denied(source: str) -> None:
    assert "network_call_forbidden" in _codes(
        source, "src/sakuramoon/runtime/network.py"
    )


def test_network_callable_cannot_escape_to_an_unknown_higher_order_target() -> None:
    source = """
from http.client import HTTPSConnection
register(HTTPSConnection)
"""

    assert "sensitive_callable_escape" in _codes(
        source, "src/sakuramoon/runtime/network.py"
    )


@pytest.mark.parametrize(
    "module_name",
    [
        '"transformers"',
        '"requests"',
        '"modelscope_hub"',
        "input()",
    ],
)
def test_dynamic_import_cannot_manufacture_unknown_sensitive_callables(
    module_name: str,
) -> None:
    source = f'''
module = __import__({module_name})
loader = getattr(module, input())
loader("other/model")
'''

    assert "dynamic_import_forbidden" in _codes(
        source, "src/sakuramoon/runtime/imports.py"
    )


def test_dynamic_container_subscript_cannot_drop_sensitive_callable() -> None:
    source = """
from transformers import AutoModel
loaders = [AutoModel.from_pretrained]
loaders[index]("other/model", local_files_only=True)
"""

    assert "ambiguous_sensitive_container_access" in _codes(source)


def test_nested_sensitive_container_cannot_enter_higher_order_call() -> None:
    source = """
from transformers import AutoModel
register({"nested": [[AutoModel.from_pretrained]]})
"""

    assert "sensitive_callable_escape" in _codes(source)


def test_branch_dependent_sensitive_callable_is_denied_fail_closed() -> None:
    source = """
from transformers import AutoModel
from pathlib import Path
loader = AutoModel.from_pretrained if condition else Path
loader("other/model", local_files_only=True)
"""

    assert "ambiguous_sensitive_callable" in _codes(source)


def test_reassignment_kills_loader_alias_without_false_positive() -> None:
    source = """
from transformers import AutoModel
class LocalFactory:
    def load(self, value):
        return value
loader = AutoModel.from_pretrained
loader = LocalFactory().load
loader("ordinary local metadata")
"""

    assert scan_source(source, "src/sakuramoon/report.py") == ()


def test_huggingface_transport_cannot_use_dataset_exception() -> None:
    source = """
from huggingface_hub import snapshot_download
from sakuramoon.config import load_config
def fetch_dataset_shard():
    loaded = load_config("stage.toml", config_root="config")
    return snapshot_download(
        "leafmoone/webdataset_danbooru",
        revision=loaded.config.data.source.revision,
        repo_type="dataset",
    )
"""

    assert "forbidden_download" in _codes(source, "src/sakuramoon/data/modelscope.py")


def test_huggingface_qualified_file_download_is_forbidden() -> None:
    source = """
from huggingface_hub.file_download import hf_hub_download
hf_hub_download("other/model", "weights.bin")
"""

    assert "forbidden_download" in _codes(source)


@pytest.mark.parametrize(
    "source",
    [
        """
from huggingface_hub import snapshot_download
transports = [snapshot_download]
transports[0]("other/model")
""",
        """
from huggingface_hub import snapshot_download
transports = {"fetch": snapshot_download}
transports["fetch"]("other/model")
""",
        """
from huggingface_hub import snapshot_download
def transport_factory():
    return snapshot_download
transport_factory()("other/model")
""",
    ],
)
def test_download_callable_provenance_survives_containers_and_helpers(
    source: str,
) -> None:
    assert "forbidden_download" in _codes(source)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("import reference.JLT.train", "reference_import"),
        (
            """
from importlib import import_module as load_module
load_module("reference.JLT.train")
""",
            "reference_dynamic_import",
        ),
        ("__import__('reference.JLT.train')", "reference_dynamic_import"),
        (
            """
from pathlib import Path
from runpy import run_path
p = Path("reference") / "JLT" / "train.py"
run_path(p)
""",
            "reference_dynamic_exec",
        ),
        (
            """
from pathlib import Path
p = Path("reference") / "JLT" / "train.py"
code = p.read_text()
exec(code)
""",
            "reference_dynamic_exec",
        ),
        (
            """
import subprocess as sp
from pathlib import Path
p = Path("reference") / "JLT" / "train.py"
sp.call(["python", p])
""",
            "reference_process_exec",
        ),
        (
            """
import os
from pathlib import Path
p = Path("reference") / "JLT" / "train.py"
os.spawnv(os.P_WAIT, "/usr/bin/python", ["python", p])
""",
            "reference_process_exec",
        ),
        (
            """
import asyncio
from pathlib import Path
p = Path("reference") / "JLT" / "train.py"
asyncio.create_subprocess_exec("python", p)
""",
            "reference_process_exec",
        ),
        (
            """
import sys
from pathlib import Path
p = Path("reference") / "JLT"
sys.path.extend([p])
""",
            "reference_search_path",
        ),
        (
            """
import site
from pathlib import Path
p = Path("reference") / "JLT"
site.addsitedir(p)
""",
            "reference_search_path",
        ),
        (
            """
import sys
from pathlib import Path
p = Path("reference") / "JLT"
sys.path += [p]
""",
            "reference_search_path",
        ),
    ],
)
def test_reference_execution_bypasses_are_rejected(source: str, expected: str) -> None:
    assert expected in _codes(source, "src/sakuramoon/train/step.py")


@pytest.mark.parametrize(
    "source",
    [
        """
import subprocess
subprocess.run("python reference/JLT/train.py", shell=True)
""",
        """
import subprocess
from pathlib import Path
class Holder: pass
holder = Holder()
holder.path = Path("reference") / "JLT" / "train.py"
subprocess.run(["python", holder.path])
""",
        """
import subprocess
from pathlib import Path
class Holder: pass
holder = Holder()
holder.path = Path("reference") / "JLT" / "train.py"
alias = holder
subprocess.run(["python", alias.path])
""",
        """
import subprocess
from pathlib import Path
holder = {}
holder["path"] = Path("reference") / "JLT" / "train.py"
subprocess.run(["python", holder["path"]])
""",
        """
import subprocess
from pathlib import Path
command = ["python"]
command.append(Path("reference") / "JLT" / "train.py")
subprocess.run(command)
""",
        """
import subprocess
from pathlib import Path
def reference_entry():
    return Path("reference") / "JLT" / "train.py"
subprocess.run(["python", reference_entry()])
""",
        """
import subprocess
from pathlib import Path
def execute(path):
    subprocess.run(["python", path])
execute(Path("reference") / "JLT" / "train.py")
""",
        """
import subprocess
from pathlib import Path
method = "r" + "un"
getattr(subprocess, method)(["python", Path("reference") / "JLT" / "train.py"])
""",
    ],
)
def test_reference_taint_flows_through_shell_attributes_containers_and_helpers(
    source: str,
) -> None:
    assert "reference_process_exec" in _codes(source, "src/sakuramoon/train/step.py")


@pytest.mark.parametrize(
    "source",
    [
        """
import subprocess
from pathlib import Path
def entry():
    execute(Path("reference") / "JLT" / "train.py")
def execute(path):
    subprocess.run(["python", path])
entry()
""",
        """
import subprocess
from pathlib import Path
def entry():
    subprocess.run(["python", reference_entry()])
def reference_entry():
    return Path("reference") / "JLT" / "train.py"
entry()
""",
        """
import subprocess
from pathlib import Path
class Executor:
    def run(self, path):
        subprocess.run(["python", path])
Executor().run(Path("reference") / "JLT" / "train.py")
""",
        """
import subprocess
from pathlib import Path
execute = lambda path: subprocess.run(["python", path])
execute(Path("reference") / "JLT" / "train.py")
""",
        """
import subprocess
from pathlib import Path
path = Path("reference") / "JLT" / "train.py"
match path:
    case selected:
        subprocess.run(["python", selected])
""",
        """
import subprocess
from pathlib import Path
assert subprocess.run(["python", Path("reference") / "JLT" / "train.py"])
""",
        """
import subprocess
from pathlib import Path
try:
    raise ExceptionGroup("group", [ValueError("x")])
except* ValueError:
    subprocess.run(["python", Path("reference") / "JLT" / "train.py"])
""",
        """
import subprocess
from pathlib import Path
for path in [Path("reference") / "JLT" / "train.py"]:
    subprocess.run(["python", path])
""",
        """
import subprocess
from pathlib import Path
commands = [
    subprocess.run(["python", path])
    for path in [Path("reference") / "JLT" / "train.py"]
]
""",
    ],
)
def test_reference_execution_covers_forward_class_lambda_and_all_statement_forms(
    source: str,
) -> None:
    assert "reference_process_exec" in _codes(
        source, "src/sakuramoon/train/step.py"
    )


@pytest.mark.parametrize(
    "source",
    [
        """
import subprocess
from pathlib import Path
path = "ordinary.py"
for _ in range(2):
    subprocess.run(["python", path])
    path = Path("reference") / "JLT" / "train.py"
""",
        """
import subprocess
from pathlib import Path
path = "ordinary.py"
while condition:
    subprocess.run(["python", path])
    path = Path("reference") / "JLT" / "train.py"
""",
    ],
)
def test_loop_carried_reference_provenance_reaches_execution_sink(
    source: str,
) -> None:
    assert "reference_process_exec" in _codes(
        source, "src/sakuramoon/train/step.py"
    )


@pytest.mark.parametrize(
    "source",
    [
        """
import subprocess
from pathlib import Path
def configured(value=subprocess.run([Path("reference") / "JLT" / "train.py"])):
    return value
""",
        """
import subprocess
from pathlib import Path
def configured(*, value=subprocess.run([Path("reference") / "JLT" / "train.py"])):
    return value
""",
        """
import subprocess
from pathlib import Path
@subprocess.run([Path("reference") / "JLT" / "train.py"])
def configured():
    pass
""",
    ],
)
def test_reference_process_calls_in_function_definition_expressions_are_rejected(
    source: str,
) -> None:
    assert "reference_process_exec" in _codes(
        source, "src/sakuramoon/train/step.py"
    )


def test_dynamic_setattr_reference_taint_is_rejected() -> None:
    source = """
import subprocess
from pathlib import Path
class Holder:
    pass
holder = Holder()
attribute = input()
setattr(holder, attribute, Path("reference") / "JLT" / "train.py")
subprocess.run([holder])
"""

    codes = _codes(source, "src/sakuramoon/train/step.py")
    assert "reference_dynamic_attribute" in codes
    assert "reference_process_exec" in codes


def test_production_parameterized_process_wrapper_is_denied_for_cross_module_calls() -> None:
    source = """
import subprocess
def execute(path):
    subprocess.run(["python", path])
"""

    assert "parameterized_execution_wrapper_forbidden" in _codes(
        source, "src/sakuramoon/runtime/process.py"
    )


def test_reference_cannot_enter_an_unknown_imported_wrapper() -> None:
    source = """
from pathlib import Path
from sakuramoon.runtime.process import execute
execute(Path("reference") / "JLT" / "train.py")
"""

    assert "reference_cross_module_call_forbidden" in _codes(
        source, "src/sakuramoon/train/step.py"
    )


def test_unknown_imported_wrapper_return_cannot_enter_an_execution_sink() -> None:
    source = """
import subprocess
from sakuramoon.runtime.paths import entrypoint
subprocess.run(["python", entrypoint()])
"""

    assert "reference_process_exec" in _codes(
        source, "src/sakuramoon/train/step.py"
    )


def test_verified_model_root_cannot_enter_an_unknown_imported_wrapper() -> None:
    source = """
from sakuramoon.assets import require_runtime_assets_ready
from sakuramoon.encoders.external import construct
selection = require_runtime_assets_ready(config, manifest, root=repository)
root = selection.verified_root("qwen_text_encoder")
construct(root)
"""

    assert "model_root_cross_module_call_forbidden" in _codes(source)


def test_higher_order_callable_parameters_are_denied_in_production() -> None:
    source = """
def invoke(callback, value):
    return callback(value)
"""

    assert "callable_parameter_forbidden" in _codes(
        source, "src/sakuramoon/runtime/callback.py"
    )


@pytest.mark.parametrize(
    "path",
    [
        "src/sakuramoon/assets/inspect.py",
        "src/sakuramoon/data/manifest.py",
    ],
)
def test_identity_registry_weak_reference_dereference_is_exactly_audited(
    path: str,
) -> None:
    source = """
class _IdentityWeakRegistry:
    def contains(self, value):
        reference = self._references.get(id(value))
        return reference is not None and reference() is value
"""

    assert scan_source(source, path) == ()


@pytest.mark.parametrize(
    "path",
    [
        "src/sakuramoon/assets/inspect.py",
        "src/sakuramoon/data/manifest.py",
    ],
)
def test_identity_registry_exception_does_not_allow_other_callable_parameters(
    path: str,
) -> None:
    source = """
class OtherRegistry:
    def contains(self, value):
        reference = self._references.get(id(value))
        return reference is not None and reference() is value
"""

    assert "callable_parameter_forbidden" in _codes(
        source, path
    )


@pytest.mark.parametrize(
    "source",
    [
        """
from transformers import AutoModel
register(AutoModel.from_pretrained)
""",
        """
from transformers import AutoModel
class Holder:
    pass
Holder(AutoModel.from_pretrained)
""",
    ],
)
def test_sensitive_callable_cannot_escape_to_an_unknown_higher_order_target(
    source: str,
) -> None:

    assert "sensitive_callable_escape" in _codes(
        source, "src/sakuramoon/runtime/callback.py"
    )


def test_sys_path_slice_assignment_is_rejected() -> None:
    source = """
import sys
from pathlib import Path
path = Path("reference") / "JLT"
sys.path[:] += [path]
"""

    assert "reference_search_path" in _codes(source, "src/sakuramoon/train/step.py")


def test_reference_taint_reassignment_and_function_scope_do_not_false_positive() -> None:
    source = """
import subprocess
from pathlib import Path
def unrelated():
    path = Path("reference") / "JLT" / "train.py"
    return path.name
def safe():
    path = Path("reference") / "JLT" / "train.py"
    path = "/usr/bin/true"
    subprocess.run([path], check=True)
"""

    assert scan_source(source, "src/sakuramoon/report.py") == ()


def test_reference_attribute_reassignment_kills_taint() -> None:
    source = """
import subprocess
from pathlib import Path
class Holder: pass
holder = Holder()
holder.path = Path("reference") / "JLT" / "train.py"
holder.path = "/usr/bin/true"
alias = holder
subprocess.run([alias.path], check=True)
"""

    assert scan_source(source, "src/sakuramoon/report.py") == ()


@pytest.mark.parametrize(
    "command",
    [
        '("git", "-C", "reference/JLT", "clean", "-fd", "status")',
        '''(
            "git", "-C", "reference/JLT", "-c",
            "alias.safe=!python reference/JLT/train.py", "safe"
        )''',
        '("git", "-C", "reference/JLT", command)',
    ],
)
def test_git_test_exception_rejects_markers_aliases_and_unknown_argv(
    command: str,
) -> None:
    source = f"""
import subprocess
def test_reference_origin_diagnostic_redacts_credentials():
    command = "status"
    subprocess.run({command}, check=True)
"""

    assert "reference_process_exec" in _codes(
        source, "tests/unit/assets/test_inspect.py"
    )


def test_synthetic_git_exception_requires_temp_root_provenance() -> None:
    source = """
import subprocess
def make_reference(root, relative, origin, licenses):
    repo = "reference/JLT"
    subprocess.run(("git", "-C", repo, "add", "."), check=True)
"""

    assert "reference_process_exec" in _codes(
        source, "tests/unit/assets/conftest.py"
    )


def test_synthetic_git_exception_accepts_the_exact_temp_root_shape() -> None:
    source = """
import subprocess
def make_reference(root, relative, origin, licenses):
    repo = root / relative
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
"""

    assert scan_source(source, "tests/unit/assets/conftest.py") == ()


@pytest.mark.parametrize(
    "relative",
    ['"../../reference"', '"/tmp/reference"', '"safe/../reference"'],
)
def test_synthetic_git_exception_rejects_temp_root_escape(
    relative: str,
) -> None:
    source = f'''
import subprocess
def make_reference(root, relative, origin, licenses):
    repo = root / {relative}
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
'''

    assert "reference_process_exec" in _codes(
        source, "tests/unit/assets/conftest.py"
    )


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ('"reference/HDM"', None),
        ('"../../reference"', "synthetic_git_path_argument_forbidden"),
        ('"/absolute/reference"', "synthetic_git_path_argument_forbidden"),
    ],
)
def test_synthetic_git_helper_requires_safe_relative_call_argument(
    relative: str,
    expected: str | None,
) -> None:
    source = f'''
import subprocess
def make_reference(root, relative, origin, licenses):
    repo = root / relative
    subprocess.run(("git", "-C", str(repo), "add", "."), check=True)
make_reference(tmp_path, {relative}, origin, licenses)
'''

    codes = _codes(source, "tests/unit/assets/conftest.py")
    if expected is None:
        assert not codes
    else:
        assert expected in codes


def test_unrelated_method_names_and_read_only_metadata_are_not_false_positives() -> None:
    source = """
class Reporter:
    def run(self, value):
        return value
    def append(self, value):
        return value
reporter = Reporter()
reporter.run("reference status")
reporter.append("reference metadata")
"""

    assert scan_source(source, "src/sakuramoon/report.py") == ()


def test_restricted_language_keeps_provably_safe_forward_and_container_calls() -> None:
    source = """
import subprocess
def entry():
    safe_process()
def safe_process():
    subprocess.run(["/usr/bin/true"], check=True)
class Reporter:
    def render(self, value):
        return value
callbacks = {"render": Reporter().render}
assert callbacks["render"]("ordinary metadata") == "ordinary metadata"
match "ordinary metadata":
    case value:
        callbacks["render"](value)
entry()
"""

    assert scan_source(source, "src/sakuramoon/report.py") == ()


def test_repository_scanner_includes_its_own_tool_and_current_tree_is_clean() -> None:
    relative = {path.relative_to(ROOT).as_posix() for path in python_sources(ROOT)}

    assert "tools/asset_execution_boundary.py" in relative
    assert scan_repository(ROOT) == ()


def test_repository_scanner_rejects_source_symlinks_before_reading(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("SENTINEL = 'must not be read'\n", encoding="utf-8")
    (tmp_path / "src/linked.py").symlink_to(outside)

    with pytest.raises(SourceBoundaryError, match="symlink"):
        python_sources(tmp_path)


def test_repository_scanner_rejects_symlinked_source_directories(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-dir"
    outside.mkdir()
    (outside / "hidden.py").write_text("SENTINEL = True\n", encoding="utf-8")
    (tmp_path / "src/linked").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SourceBoundaryError, match="symlink"):
        python_sources(tmp_path)


def test_explicit_source_scan_rejects_out_of_root_paths(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    outside = tmp_path.parent / f"{tmp_path.name}-outside-explicit.py"
    outside.write_text("SENTINEL = True\n", encoding="utf-8")

    with pytest.raises(SourceBoundaryError, match="escapes"):
        scan_file(tmp_path, outside)


def test_source_read_rejects_parent_symlink_replacement_after_precheck(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "src/package"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("SAFE = True\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.py").write_text(
        "import requests\nrequests.get('https://attacker.invalid')\n",
        encoding="utf-8",
    )
    detached = tmp_path / "detached"
    original = vars(boundary)["_assert_safe_source_path"]
    replaced = False

    def replace_parent(root: Path, path: Path) -> None:
        nonlocal replaced
        original(root, path)
        if not replaced and path == source:
            package.rename(detached)
            package.symlink_to(outside, target_is_directory=True)
            replaced = True

    monkeypatch.setattr(boundary, "_assert_safe_source_path", replace_parent)

    with pytest.raises(SourceBoundaryError, match="anchored no-follow"):
        scan_file(tmp_path, source)


def test_source_read_keeps_open_parent_anchor_during_leaf_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / "src/package"
    package.mkdir(parents=True)
    source = package / "module.py"
    source.write_text("SAFE = True\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "module.py").write_text(
        "import requests\nrequests.get('https://attacker.invalid')\n",
        encoding="utf-8",
    )
    detached = tmp_path / "detached"
    original_open = boundary.os.open
    replaced = False

    def replace_after_parent_open(
        path: str | bytes | Path,
        flags: int,
        mode: int = 0o777,
        *,
        dir_fd: int | None = None,
    ) -> int:
        nonlocal replaced
        if not replaced and path == "module.py" and dir_fd is not None:
            package.rename(detached)
            package.symlink_to(outside, target_is_directory=True)
            replaced = True
        return original_open(path, flags, mode, dir_fd=dir_fd)

    monkeypatch.setattr(boundary.os, "open", replace_after_parent_open)

    assert scan_file(tmp_path, source) == ()
