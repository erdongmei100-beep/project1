"""Quick import verifier for the emergency lane project."""

# --- bootstrap ---
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# -----------------

import importlib
import sys


def try_import(mod: str) -> bool:
    try:
        importlib.import_module(mod)
        print(f"[OK] import {mod}")
        return True
    except Exception as e:  # pragma: no cover - diagnostic tool
        print(f"[FAIL] import {mod}: {e.__class__.__name__}: {e}")
        return False


mods = [
    "src",
    "src.roi",
    "src.roi.laneatt",
    "src.lane_detection",
    "src.lane_detection.laneatt",
]

ok = all(try_import(m) for m in mods)
print("RESULT:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
