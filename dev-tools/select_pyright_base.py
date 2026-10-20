"""Select the baseline commit for the CI pyright no-new-diagnostics gate.

The gate must never turn a git failure into an empty "no changed files"
pass.  Base selection is fail-closed:

- ``pull_request``: merge-base against the PR's actual target branch
  (``GITHUB_BASE_REF``), not a hardcoded branch.
- ``push`` to a branch other than ``dev``: merge-base against ``origin/dev``.
- ``push`` to ``dev``: ``GITHUB_BEFORE``, the true pre-push SHA, which covers
  single-commit, merge-commit and multi-commit pushes alike.

Every selected SHA must exist in the repository.  The dev-push SHA must also
be an ancestor of HEAD and must not equal HEAD.  Zero, unknown, missing or
self SHAs, and unknown event names, raise instead of degrading to an empty
diff.

CI usage: ``base=$(uv run --no-sync python dev-tools/select_pyright_base.py)``

Prints the selected base SHA on stdout and the selection source on stderr.
Exits nonzero with a specific message on any violation.
"""

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import cast

ZERO_SHA = "0" * 40
_SHA_RE = re.compile(r"[0-9a-f]{40}")


class BaseSelectionError(RuntimeError):
    """Fail-closed base selection violation; never an empty diff pass."""


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise BaseSelectionError(
            f"git {' '.join(args)} failed: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    return result.stdout.strip()


def _require_sha(sha: str, label: str) -> None:
    if not _SHA_RE.fullmatch(sha):
        raise BaseSelectionError(f"{label} is not a full 40-hex SHA: {sha!r}")
    if sha == ZERO_SHA:
        raise BaseSelectionError(
            f"{label} is the zero SHA; refusing to use it as a baseline"
        )


def _require_commit(repo: Path, sha: str) -> None:
    _git(repo, "cat-file", "-e", sha + "^{commit}")


def _require_ancestor(repo: Path, ancestor: str, descendant: str, label: str) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "--is-ancestor", ancestor, descendant],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        raise BaseSelectionError(
            f"git merge-base --is-ancestor failed: {result.stderr.strip()}"
        )
    if result.returncode == 1:
        raise BaseSelectionError(f"{label} is not an ancestor of HEAD")


def _merge_base(repo: Path, left: str, right: str, label: str) -> str:
    sha = _git(repo, "merge-base", left, right)
    _require_commit(repo, sha)
    return sha


def select_base(
    repo: Path,
    event: str,
    ref_name: str,
    base_ref: str,
    before: str,
    dev_branch: str,
) -> tuple[str, str]:
    """Return ``(base_sha, source)`` or raise :class:`BaseSelectionError`.

    ``event`` is the GitHub event name, ``ref_name`` the pushed branch for
    push events, ``base_ref`` the PR target branch for pull_request events,
    and ``before`` the pre-push SHA (``GITHUB_BEFORE``) for push events.
    """
    head = _git(repo, "rev-parse", "HEAD")
    if event == "pull_request":
        if not base_ref:
            raise BaseSelectionError("pull_request event without GITHUB_BASE_REF")
        target = f"origin/{base_ref}"
        _git(repo, "rev-parse", target + "^{commit}")
        return (
            _merge_base(repo, target, "HEAD", f"merge-base({target}, HEAD)"),
            "pr-target-merge-base",
        )
    if event == "push":
        if not ref_name:
            raise BaseSelectionError("push event without GITHUB_REF_NAME")
        if ref_name == dev_branch:
            if not before:
                raise BaseSelectionError("dev push without GITHUB_BEFORE")
            _require_sha(before, "GITHUB_BEFORE")
            if before == head:
                raise BaseSelectionError(
                    "GITHUB_BEFORE equals HEAD (self); no push delta to validate"
                )
            _require_commit(repo, before)
            _require_ancestor(repo, before, head, "GITHUB_BEFORE")
            return before, "push-before"
        target = f"origin/{dev_branch}"
        _git(repo, "rev-parse", target + "^{commit}")
        return (
            _merge_base(repo, target, "HEAD", f"merge-base({target}, HEAD)"),
            "dev-merge-base",
        )
    raise BaseSelectionError(f"unknown GITHUB_EVENT_NAME: {event!r}")


def _event_before(event_path: Path) -> str:
    """Best-effort ``before`` from the push payload; empty on any problem.

    An empty result fails closed later (``dev push without GITHUB_BEFORE``)
    instead of silently weakening the check.
    """
    try:
        payload: object = json.loads(event_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(payload, dict):
        return ""
    before: object = cast("dict[str, object]", payload).get("before", "")
    return before if isinstance(before, str) else ""


def main() -> int:
    repo = Path.cwd()
    event = os.environ.get("GITHUB_EVENT_NAME", "")
    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    base_ref = os.environ.get("GITHUB_BASE_REF", "")
    before = os.environ.get("GITHUB_BEFORE", "")
    if event == "push" and not before:
        # GITHUB_BEFORE is not a default GitHub environment variable; CI maps
        # it from github.event.before, and the payload path is the fallback.
        event_path = os.environ.get("GITHUB_EVENT_PATH", "")
        if event_path:
            before = _event_before(Path(event_path))
    try:
        base, source = select_base(repo, event, ref_name, base_ref, before, "dev")
    except BaseSelectionError as error:
        print(f"pyright base selection failed: {error}", file=sys.stderr)
        return 1
    print(f"pyright base source={source}", file=sys.stderr)
    print(base)
    return 0


if __name__ == "__main__":
    sys.exit(main())
