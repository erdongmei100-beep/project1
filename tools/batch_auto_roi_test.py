from __future__ import annotations

import argparse
import importlib
import importlib.util
from pathlib import Path
from typing import Any, List, Sequence, Tuple

from src.roi.auto_cv import AutoCVParams, estimate_roi, save_roi_json
from src.utils.config import load_config, resolve_path
from src.utils.device import select_device
from src.utils.paths import ROOT


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch evaluate auto ROI generation across a directory of videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--videos-dir", required=True, help="Directory containing input videos")
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Configuration file providing roi.auto_cv parameters",
    )
    parser.add_argument(
        "--output-root",
        default=str(ROOT / "data" / "rois"),
        help="Directory to store ROI JSON and optional visualizations",
    )
    parser.add_argument(
        "--pattern",
        default="*.mp4",
        help="Glob pattern for videos within --videos-dir",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Computation device for auto ROI inference (e.g. cuda or cpu)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ROI JSON outputs",
    )
    parser.add_argument(
        "--debug-vis",
        action="store_true",
        help="Save debug visualizations with the ROI polygon overlaid",
    )
    return parser.parse_args(argv)


def _collect_videos(directory: Path, pattern: str) -> List[Path]:
    candidates = [path for path in directory.glob(pattern) if path.is_file()]
    return sorted(candidates, key=lambda path: path.name.lower())


def _require_cv2() -> Any:
    spec = importlib.util.find_spec("cv2")
    if spec is None:
        raise RuntimeError(
            "OpenCV (cv2) is required for --debug-vis but is not installed in the environment."
        )
    return importlib.import_module("cv2")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)

    videos_dir = resolve_path(ROOT, args.videos_dir)
    if not videos_dir.exists() or not videos_dir.is_dir():
        raise NotADirectoryError(f"Video directory not found: {videos_dir}")

    output_root = resolve_path(ROOT, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)
    roi_cfg = dict(config.get("roi", {}) or {})
    auto_cfg = roi_cfg.get("auto_cv", {})
    params = AutoCVParams.from_config(auto_cfg)
    if args.debug_vis and not params.save_debug:
        params = params.evolve(save_debug=True)

    device = select_device(args.device, feature="batch auto ROI")
    print(f"Using device for auto ROI: {device}")

    videos = _collect_videos(videos_dir, args.pattern)
    if not videos:
        print(f"No videos matched pattern '{args.pattern}' under {videos_dir}")
        return 0

    print(f"Found {len(videos)} video(s) for processing")

    stats = {"success": 0, "skipped": 0, "failed": 0}
    failures: List[Tuple[str, str]] = []

    cv2_module: Any | None = None
    if args.debug_vis:
        cv2_module = _require_cv2()

    for video in videos:
        run_name = video.stem
        roi_path = output_root / f"{run_name}.json"
        preview_path = output_root / f"{run_name}_roi_preview.jpg"

        if not args.overwrite and roi_path.exists():
            print(f"[SKIP] {run_name}: existing ROI at {roi_path}")
            stats["skipped"] += 1
            continue

        try:
            result = estimate_roi(
                video,
                params,
                overlay=args.debug_vis,
                overlay_dir=output_root if args.debug_vis else None,
                device=device,
            )
        except Exception as exc:
            stats["failed"] += 1
            failures.append((video.name, str(exc)))
            print(f"[FAIL] {run_name}: {exc}")
            continue

        if not result.polygon:
            stats["failed"] += 1
            reason = result.message or "ROI polygon not generated"
            failures.append((video.name, reason))
            print(f"[FAIL] {run_name}: {reason}")
            continue

        params_used = result.params_used
        meta = {
            "mode": "auto_cv" if result.success else "auto_cv_relaxed",
            "metrics": result.metrics,
            "device": device,
            "params": {
                "min_box_h_px": params_used.min_box_h_px,
                "min_rel_area": params_used.min_rel_area,
                "min_sharpness": params_used.min_sharpness,
                "bbox_aspect_min": params_used.bbox_aspect_min,
            },
        }

        save_roi_json(roi_path, result.base_size, result.polygon, meta)
        stats["success"] += 1
        print(f"[OK] {run_name}: ROI saved to {roi_path}")

        if args.debug_vis and cv2_module is not None:
            overlay_img = result.overlay
            overlay_metric = result.metrics.get("overlay_path")
            if overlay_img is None and overlay_metric:
                overlay_file = Path(overlay_metric)
                if overlay_file.exists():
                    overlay_img = cv2_module.imread(str(overlay_file))
            if overlay_img is not None:
                if cv2_module.imwrite(str(preview_path), overlay_img):
                    print(f"      Debug preview saved to {preview_path}")
            else:
                print("      Debug overlay unavailable; skipping preview save")

            if overlay_metric:
                overlay_file = Path(overlay_metric)
                if overlay_file.exists() and overlay_file.parent == output_root:
                    try:
                        overlay_file.unlink()
                    except OSError:
                        pass
        elif args.debug_vis:
            print("      Debug overlay unavailable because OpenCV could not be loaded")

    total = len(videos)
    print("\nBatch auto ROI summary:")
    print(f"  Total videos: {total}")
    print(f"  Success: {stats['success']}")
    print(f"  Skipped: {stats['skipped']}")
    print(f"  Failed: {stats['failed']}")
    if failures:
        print("  Failure details:")
        for name, reason in failures:
            print(f"    - {name}: {reason}")

    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

