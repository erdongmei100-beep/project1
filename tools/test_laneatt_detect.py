"""LaneATT detection visualiser with rightmost-lane ROI overlay.

Example:
    python tools/test_laneatt_detect.py \
        --source data/videos/2.mp4 \
        --frame 160 \
        --config configs/default.yaml \
        --save-dir data/outputs/laneatt_detect \
        --device cpu
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import statistics
import yaml

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

from lane_detection.laneatt import LaneATTConfig, LaneATTDetector, LaneDetection

from src.roi.laneatt import (
    LaneATTParams,
    _aggregate_lane,
    _build_polygon,
    _render_overlay,
    _sample_frames,
)
from src.utils.paths import OUTPUTS_DIR

Point = Tuple[int, int]


COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
]


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _read_frame(video_path: Path, frame_idx: int) -> Optional[Tuple[int, Any]]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for LaneATT debug visualisation.")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_idx))
    ret, frame = capture.read()
    capture.release()
    if not ret or frame is None:
        return None
    return frame_idx, frame


def _draw_lanes(frame, lanes: Sequence[LaneDetection]) -> Any:
    vis = frame.copy()
    for idx, lane in enumerate(lanes):
        color = COLORS[idx % len(COLORS)]
        cv2.line(vis, lane.points[0], lane.points[1], color, 2)
        bottom = lane.bottom_point
        label = f"{idx}:{int(bottom[0])}"
        cv2.circle(vis, bottom, 4, color, -1)
        cv2.putText(
            vis,
            label,
            (int(bottom[0]), int(bottom[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LaneATT lane visualiser")
    parser.add_argument("--source", required=True, help="Video path")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--frame", type=int, default=None, help="Optional frame index to debug")
    parser.add_argument(
        "--save-dir",
        default=str(OUTPUTS_DIR / "laneatt_detect"),
        help="Directory for final overlays",
    )
    parser.add_argument(
        "--debug-dir",
        default=str(OUTPUTS_DIR / "laneatt_debug"),
        help="Directory for per-frame lane visualisation",
    )
    parser.add_argument("--device", default=None, help="Override lane detector device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = _load_config(Path(args.config))
    roi_cfg = (cfg.get("roi") or {}).get("laneatt") or {}
    laneatt_cfg = dict(roi_cfg.get("laneatt") or {})
    if args.device:
        laneatt_cfg["device"] = args.device
    roi_cfg["laneatt"] = laneatt_cfg

    params = LaneATTParams.from_config(roi_cfg)
    detector = LaneATTDetector(LaneATTConfig.from_config(params.laneatt))

    video_path = Path(args.source)
    frames: List[Tuple[int, Any]]
    if args.frame is not None:
        frame_tuple = _read_frame(video_path, args.frame)
        frames = [frame_tuple] if frame_tuple else []
    else:
        frames = _sample_frames(video_path, params.sample_frames)

    debug_dir = Path(args.debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    rightmost_lanes: List[LaneDetection] = []
    all_lane_bottom_xs: List[List[float]] = []
    overlay_source = None

    for frame_idx, frame in frames:
        result = detector.detect(frame)
        lanes = [
            lane
            for lane in getattr(result, "lanes", [])
            if getattr(lane, "score", 1.0) >= params.min_score
        ]
        if not lanes:
            continue
        lanes.sort(key=lambda lane: lane.bottom_point[0])
        all_lane_bottom_xs.append([lane.bottom_point[0] for lane in lanes])
        overlay_source = frame

        vis = _draw_lanes(frame, lanes)
        rightmost = lanes[-1]
        cv2.line(vis, rightmost.points[0], rightmost.points[1], (0, 165, 255), 3)
        cv2.putText(
            vis,
            "rightmost",
            (int(rightmost.bottom_point[0]), int(rightmost.bottom_point[1]) + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(debug_dir / f"lanes_{frame_idx:05d}.jpg"), vis)
        rightmost_lanes.append(rightmost)

    if not rightmost_lanes:
        print("No lanes detected; nothing to save.")
        return

    aggregated = _aggregate_lane(rightmost_lanes)
    lane_widths: List[float] = []
    for xs in all_lane_bottom_xs:
        xs = sorted(xs)
        if len(xs) >= 2:
            lane_widths.extend(xs[i] - xs[i - 1] for i in range(1, len(xs)))
    lane_width_px = None
    if lane_widths:
        lane_width_px = float(statistics.median(lane_widths))

    if aggregated is None or overlay_source is None:
        print("Failed to aggregate lanes for ROI overlay.")
        return

    polygon = _build_polygon(aggregated, overlay_source.shape[:2], params, lane_width_px=lane_width_px)
    overlay = _render_overlay(overlay_source, polygon, aggregated)
    overlay_name = save_dir / f"laneatt_overlay_{rightmost_lanes[-1].bottom_point[0]:.0f}.jpg"
    cv2.imwrite(str(overlay_name), overlay)

    print(f"Saved per-frame overlays to: {debug_dir}")
    print(f"Saved final overlay to: {overlay_name}")
    if lane_width_px is not None:
        print(f"Estimated lane width (px): {lane_width_px:.2f}")


if __name__ == "__main__":
    main()
