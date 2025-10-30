"""CLI helper for a single auto_cv ROI estimation run."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from src.roi.auto_cv import AutoCVParams, estimate_roi, save_roi_json
from src.utils.config import load_config, resolve_path
from src.utils.paths import ROOT


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate ROI for a single video using the auto_cv pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source", required=True, help="Video path relative to repository root")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Configuration file providing roi.auto_cv parameters",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Destination ROI JSON path. Defaults to data/rois/<video_stem>.json",
    )
    parser.add_argument(
        "--save-overlay",
        action="store_true",
        help="Persist overlay PNG alongside the video stem",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)
    roi_cfg = dict(config.get("roi", {}) or {})
    auto_cfg = roi_cfg.get("auto_cv", {})
    params = AutoCVParams.from_config(auto_cfg)

    source_path = resolve_path(ROOT, args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    out_path: Path
    if args.out:
        out_path = resolve_path(ROOT, args.out)
    else:
        out_dir = ROOT / "data" / "rois"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{source_path.stem}.json"

    print(f"Running auto_cv for {source_path}")
    result = estimate_roi(source_path, params, overlay=args.save_overlay)
    if not result.success or result.polygon is None:
        raise SystemExit(f"Auto ROI failed: {result.message}")

    meta = {
        "mode": "auto_cv",
        "metrics": result.metrics,
        "params": {
            "min_box_h_px": params.min_box_h_px,
            "min_rel_area": params.min_rel_area,
            "min_sharpness": params.min_sharpness,
            "bbox_aspect_min": params.bbox_aspect_min,
        },
    }
    save_roi_json(out_path, result.base_size, result.polygon, meta)
    print(f"ROI saved to {out_path}")
    overlay_path = result.metrics.get("overlay_path")
    if overlay_path:
        print(f"Overlay saved to {overlay_path}")


if __name__ == "__main__":
    main()
