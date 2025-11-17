"""Quick LaneATT ROI visualiser.

Usage:
    python tools/test_laneatt_roi.py \
      --source data/videos/ambulance.mp4 \
      --config configs/default.yaml \
      --save-debug
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

# --- add project root to sys.path (bootstrap) ---
import sys
ROOT = Path(__file__).resolve().parents[1]  # <repo-root>/project1
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------

from src.roi import AutoCVParams, LaneATTParams, estimate_roi_laneatt
from src.utils.paths import OUTPUTS_DIR


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test LaneATT based ROI generation")
    parser.add_argument("--source", required=True, help="Video path for ROI estimation")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument(
        "--save-debug",
        action="store_true",
        help="Save debug overlays to data/outputs/laneatt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(Path(args.config))
    roi_cfg = (cfg.get("roi") or {}).get("laneatt") or {}
    auto_cv_cfg = (cfg.get("roi") or {}).get("auto_cv") or {}

    params = LaneATTParams.from_config(roi_cfg)
    params.save_debug = params.save_debug or args.save_debug
    auto_params = AutoCVParams.from_config(auto_cv_cfg)

    result = estimate_roi_laneatt(
        Path(args.source),
        params,
        auto_cv_params=auto_params,
        overlay=params.save_debug,
        overlay_dir=OUTPUTS_DIR / "laneatt",
    )

    print(f"LaneATT ROI success: {result.success}")
    if result.polygon:
        print(f"Polygon vertices: {result.polygon}")
    print(f"Used frames: {result.used_frames}")
    if params.save_debug:
        for key, value in result.metrics.items():
            if key.startswith("debug_"):
                print(f"Saved {key}: {value}")


if __name__ == "__main__":
    main()
