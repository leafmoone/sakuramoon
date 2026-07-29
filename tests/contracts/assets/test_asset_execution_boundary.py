from __future__ import annotations

from pathlib import Path

import pytest

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
            "remote_code_enabled",
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
            "model_cache_enabled",
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
from sakuramoon.assets import VerifiedAssetSelection
def load(selection: VerifiedAssetSelection):
    root = selection.verified_root("qwen_text_encoder")
    return AutoModel.from_pretrained(
        root, local_files_only=True, trust_remote_code=False
    )
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


def test_only_the_exact_dataset_transport_shape_is_allowed() -> None:
    source = """
from modelscope.hub.snapshot_download import snapshot_download
def fetch_dataset_shard():
    return snapshot_download(
        "leafmoone/webdataset_danbooru",
        revision="0123456789abcdef0123456789abcdef01234567",
        repo_type="dataset",
    )
"""

    assert scan_source(source, "src/sakuramoon/data/modelscope.py") == ()


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
