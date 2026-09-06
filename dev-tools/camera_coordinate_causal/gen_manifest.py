"""Generate reports/camera-coordinate-causal-code-manifest.json in the review
worktree from the actual on-disk files (sha256 + byte sizes), never from
memory. Read-only everywhere except the single manifest write inside the
review worktree reports/ directory. Run before staging/commit.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

RT = Path("/sakuramoon-runtime/sakuramoon-camera-causal-review")
AUDIT_OUT = Path("/tmp/camera-coordinate-causal")
BASE_COMMIT = "b2443af436b268fafb2cb6c05d724e3ab0d6042c"
BOOTSTRAP_SEED = 20260906


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def script_entry(source: Path, tracked: Path) -> dict:
    st = sha(tracked)
    ss = sha(source)
    return {
        "source_path": str(source),
        "tracked_path": tracked.relative_to(RT).as_posix(),
        "sha256": st,
        "bytes": tracked.stat().st_size,
        "source_sha256": ss,
        "source_identity": st == ss,
    }


snap = RT / "dev-tools/camera_coordinate_causal/final_snapshot"
final_scripts = []
for name in ("cc_common.py", "cc_stage1.py", "cc_stage2.py", "cc_stage2b.py", "cc_stage3.py"):
    final_scripts.append(script_entry(AUDIT_OUT / name, snap / name))
assert all(e["source_identity"] for e in final_scripts), "snapshot drifted from the executed /tmp scripts"

rep = RT / "reports"


def rep_entry(name: str) -> dict:
    p = rep / name
    return {"name": name, "sha256": sha(p), "bytes": p.stat().st_size}


causal_reports = [rep_entry(n) for n in (
    "camera-coordinate-causal-audit.md",
    "camera-coordinate-causal-audit.json",
    "camera-coordinate-causal-metrics.json",
    "camera-coordinate-causal-units.csv",
    "camera-coordinate-causal-copy-report.md",
)]
expanded_reports = [rep_entry(n) for n in (
    "camera-p25-expanded-effectiveness-audit.md",
    "camera-p25-expanded-effectiveness-audit.json",
    "camera-p25-expanded-effectiveness-metrics.json",
    "camera-p25-expanded-effectiveness-points.csv",
    "camera-p25-expanded-effectiveness-prompts.csv",
    "camera-p25-expanded-effectiveness-copy-report.md",
)]

audit = json.loads((rep / "camera-coordinate-causal-audit.json").read_text(encoding="utf-8"))
units_section_sha = audit["provenance"]["unit_manifest_sha256"]

doc = {
    "schema_version": 1,
    "base_commit": BASE_COMMIT,
    "branch": "camera-v2-causal-audit-review",
    "bootstrap_seed": BOOTSTRAP_SEED,
    "final_numbers_generated_by_snapshot": True,
    "final_scripts": final_scripts,
    "input_manifest_sha256": sha(AUDIT_OUT / "stage1-manifest.json"),
    "input_manifest_units_section_sha256": units_section_sha,
    "provenance_note": (
        "audit.json provenance.script_sha256 records the shas at stage1-manifest time: ",
        "cc_stage2=3420f055... and cc_stage3=c9d5646e... are PRE-FIX versions and cc_stage2b is ",
        "absent (added after stage1). The authoritative final executed scripts are the ",
        "final_snapshot/ entries above (cc_stage2=e5b7ae91..., cc_stage2b=aea987d6..., ",
        "cc_stage3=4a855b31...); the full patch chain is documented in ",
        "dev-tools/camera_coordinate_causal/README.md (Known bug history) and in the protected ",
        "/tmp/camera-coordinate-causal/cc-audit-deployed-shas-rerun.txt (not in git)."
    ),
    "causal_reports": causal_reports,
    "expanded_effectiveness_reports": expanded_reports,
}
out = rep / "camera-coordinate-causal-code-manifest.json"
out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print("WROTE " + str(out) + " " + str(out.stat().st_size) + " bytes; source_identity all PASS")
