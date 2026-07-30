from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[3]


def test_project_requires_exact_locked_uv_version() -> None:
    document = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())

    assert document["tool"]["uv"]["required-version"] == "==0.12.0"


def test_uv_rejects_an_incompatible_required_version(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        """[project]
name = "uv-version-negative-contract"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = []

[tool.uv]
required-version = "==0.0.1"
""",
        encoding="utf-8",
    )

    result = subprocess.run(
        ["uv", "lock", "--check"],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "0.0.1" in result.stderr
    assert "0.12.0" in result.stderr
