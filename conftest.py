from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SRC_PATHS = (
    ROOT / "serapeum-core" / "src",
    ROOT / "serapeum-integrations" / "llms" / "serapeum-ollama" / "src",
)

for path in SRC_PATHS:
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
