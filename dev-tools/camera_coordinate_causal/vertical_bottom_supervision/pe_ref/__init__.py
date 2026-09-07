"""pe_ref - self-contained frozen PE-Spatial-B16-512 vision tower.

Verbatim copy of the iprea-branch vendored package
(src/sakuramoon/pe_spatial in the come2 irepa-prod worktree at
/sakuramoon-runtime/sakuramoon-irepa-prod, HEAD recorded in the audit
provenance block), re-based to relative imports.  Used ONLY by
feature_audit.py for the frozen content feature audit; it is never
imported by production code.
"""
from .config import PE_VISION_CONFIG, PEConfig
from .pe_vision import VisionTransformer

__all__ = ["PE_VISION_CONFIG", "PEConfig", "VisionTransformer"]
