from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR_DIR = ROOT / "tools" / "dataset_collector"

if str(COLLECTOR_DIR) not in sys.path:
    sys.path.insert(0, str(COLLECTOR_DIR))

runpy.run_path(str(COLLECTOR_DIR / "app.py"), run_name="__main__")
