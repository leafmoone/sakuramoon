from __future__ import annotations

import json

import pytest

from sakuramoon.cli.manifest import main


def test_manifest_cli_has_no_credential_argv_surface_and_redacts_unknown_values(
    capsys: pytest.CaptureFixture[str],
) -> None:
    marker = "raw-command-line-secret-marker"

    assert main(("--token", marker)) == 2

    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.count("\n") == 1
    assert json.loads(captured.out) == {"error": "invalid_arguments", "ok": False}
    assert marker not in captured.out
