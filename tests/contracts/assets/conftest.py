"""Make repository verification tools importable under the src-only test path."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
