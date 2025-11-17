"""Batch evaluator for the auto_cv ROI tuning pipeline."""
from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd
from tqdm import tqdm

# --- add project root to sys.path (bootstrap) ---
import sys
ROOT = Path(__file__).resolve().parents[1]  # <repo-root>/project1
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------

from src.roi.auto_cv import AutoCVParams, estimate_roi, save_roi_json
from src.utils.config import load_config, resolve_path, save_config
from src.utils.paths import ROOT


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate auto_cv ROI parameters across multiple videos",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        required=True,
        help="Video paths relative to the repository root (e.g. data/videos/ambulance.mp4)",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "configs" / "default.yaml"),
        help="Configuration file containing roi.auto_cv defaults",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "reports" / "auto_cv_tune.csv"),
        help="Output CSV path for the tuning report",
    )
    parser.add_argument(
        "--save-overlays",
        action="store_true",
        help="Persist overlay PNGs for the final parameter sweep",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing ROI JSON files instead of skipping",
    )
    parser.add_argument(
        "--no-update-config",
        action="store_true",
        help="Skip writing tuned parameters back to the config file",
    )
    parser.add_argument(
        "--coarse-step",
        type=float,
        default=1.0,
        help="Multiplier applied to coarse adjustments (1.0 keeps defaults)",
    )
    parser.add_argument(
        "--fine-step",
        type=float,
        default=0.5,
        help="Multiplier applied to fine adjustments relative to coarse deltas",
    )
    return parser.parse_args(argv)


def _resolve_videos(videos: Iterable[str]) -> List[Path]:
    resolved: List[Path] = []
    for video in videos:
        candidate = resolve_path(ROOT, video)
        resolved.append(candidate)
    return resolved


def _coarse_offsets(step_scale: float) -> Dict[str, Tuple[float, ...]]:
    base = {
        "min_box_h_px": (-8, 0, 8),
        "min_rel_area": (-0.0015, 0.0, 0.0015),
        "min_sharpness": (-25.0, 0.0, 25.0),
        "bbox_aspect_min": (-0.15, 0.0, 0.15),
    }
    scaled: Dict[str, Tuple[float, ...]] = {}
    for key, values in base.items():
        scaled[key] = tuple(v * step_scale for v in values)
    return scaled


def _fine_offsets(coarse_best: AutoCVParams, coarse_offsets: Dict[str, Tuple[float, ...]], step_scale: float) -> Dict[str, Tuple[float, ...]]:
    fine: Dict[str, Tuple[float, ...]] = {}
    for key, values in coarse_offsets.items():
        span = max(abs(v) for v in values if not math.isclose(v, 0.0))
        fine[key] = (-span * step_scale, 0.0, span * step_scale)
    return fine


def _apply_offsets(params: AutoCVParams, offsets: Dict[str, float]) -> AutoCVParams:
    data = asdict(params)
    for key, delta in offsets.items():
        if key not in data:
            continue
        value = data[key]
        updated = value + delta
        if key == "min_box_h_px":
            updated = max(10, int(round(updated)))
        elif key == "min_sharpness":
            updated = max(10.0, float(updated))
        elif key == "min_rel_area":
            updated = float(max(1e-4, updated))
        elif key == "bbox_aspect_min":
            updated = float(max(0.1, updated))
        data[key] = updated
    return AutoCVParams(**data)


def _score_result(result) -> float:
    if result is None:
        return -1.0
    base = 5.0 if result.success else -2.0
    area = float(result.metrics.get("rel_area", 0.0))
    sharp = float(result.metrics.get("sharpness", 0.0))
    aspect = float(result.metrics.get("roi_aspect", 0.0))
    base += min(1.5, area * 120.0)
    base += min(1.5, sharp / 80.0)
    base += min(1.0, aspect / 1.2)
    return base


def _evaluate(videos: Sequence[Path], params: AutoCVParams, *, overlay: bool = False) -> List[Tuple[Path, object]]:
    results: List[Tuple[Path, object]] = []
    for video in videos:
        try:
            result = estimate_roi(video, params, overlay=overlay)
        except Exception as exc:
            result = None
            print(f"[WARN] Failed to evaluate {video.name} with error: {exc}")
        results.append((video, result))
    return results


def _aggregate_score(results: Sequence[Tuple[Path, object]]) -> float:
    total = 0.0
    for _, result in results:
        total += _score_result(result)
    return total


def _best_from_grid(videos: Sequence[Path], params: AutoCVParams, offsets: Dict[str, Tuple[float, ...]]) -> AutoCVParams:
    keys = list(offsets.keys())
    best_params = params
    best_score = float("-inf")
    for deltas in itertools.product(*[offsets[key] for key in keys]):
        offset_map = {key: deltas[idx] for idx, key in enumerate(keys)}
        candidate = _apply_offsets(params, offset_map)
        results = _evaluate(videos, candidate)
        score = _aggregate_score(results)
        if score > best_score:
            best_score = score
            best_params = candidate
    return best_params


def _ensure_reports_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _run_final_pass(
    videos: Sequence[Path],
    params: AutoCVParams,
    *,
    save_overlays: bool,
    overwrite: bool,
    report_path: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for video in tqdm(videos, desc="Final auto_cv pass"):
        start = time.time()
        try:
            result = estimate_roi(video, params, overlay=save_overlays)
        except Exception as exc:
            duration = time.time() - start
            rows.append(
                {
                    "video": str(video.relative_to(ROOT)),
                    "success": False,
                    "message": str(exc),
                    "duration_s": duration,
                }
            )
            continue

        duration = time.time() - start
        roi_dir = ROOT / "data" / "rois"
        roi_dir.mkdir(parents=True, exist_ok=True)
        roi_path = roi_dir / f"{video.stem}.json"
        if result.success and result.polygon:
            if overwrite or not roi_path.exists():
                meta = {
                    "mode": "auto_cv",
                    "metrics": result.metrics,
                    "params": asdict(result.params_used),
                }
                save_roi_json(roi_path, result.base_size, result.polygon, meta)
        rows.append(
            {
                "video": str(video.relative_to(ROOT)),
                "success": bool(result.success),
                "duration_s": duration,
                "rel_area": result.metrics.get("rel_area"),
                "roi_height": result.metrics.get("roi_height"),
                "roi_aspect": result.metrics.get("roi_aspect"),
                "sharpness": result.metrics.get("sharpness"),
                "fallback_variant": result.metrics.get("fallback_variant"),
                "overlay_path": result.metrics.get("overlay_path"),
                "roi_path": str(roi_path.relative_to(ROOT)),
                "message": result.message,
            }
        )

    df = pd.DataFrame(rows)
    _ensure_reports_dir(report_path)
    df.to_csv(report_path, index=False)
    return df


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    videos = _resolve_videos(args.videos)
    config_path = resolve_path(ROOT, args.config)
    config = load_config(config_path)

    roi_cfg = dict(config.get("roi", {}) or {})
    auto_cfg = roi_cfg.get("auto_cv", {})
    base_params = AutoCVParams.from_config(auto_cfg)

    coarse_offsets = _coarse_offsets(args.coarse_step)
    coarse_best = _best_from_grid(videos, base_params, coarse_offsets)

    fine_offsets = _fine_offsets(coarse_best, coarse_offsets, args.fine_step)
    final_params = _best_from_grid(videos, coarse_best, fine_offsets)

    report_path = resolve_path(ROOT, args.out)
    df = _run_final_pass(
        videos,
        final_params,
        save_overlays=args.save_overlays,
        overwrite=args.overwrite,
        report_path=report_path,
    )

    print("\nAuto ROI tuning summary:")
    print(df[["video", "success", "duration_s", "rel_area", "sharpness"]])

    if not args.no_update_config:
        roi_cfg.setdefault("auto_cv", {})
        roi_cfg["auto_cv"].update({
            "min_box_h_px": final_params.min_box_h_px,
            "min_rel_area": final_params.min_rel_area,
            "min_sharpness": final_params.min_sharpness,
            "bbox_aspect_min": final_params.bbox_aspect_min,
        })
        config["roi"] = roi_cfg
        save_config(config_path, config)
        print(f"Updated config with tuned parameters: {config_path}")

    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
