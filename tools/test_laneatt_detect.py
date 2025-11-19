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
    _rescale_lane_points,
    _resolve_lane_width,
    _sample_frames,
    _should_accept_width_sample,
)
from src.utils.paths import OUTPUTS_DIR

Point = Tuple[int, int]
LANE_COLOR = (0, 255, 0)


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
    height, width = frame.shape[:2]
    for idx, lane in enumerate(lanes):
        out_of_bounds = False
        for point in lane.points:
            x, y = point
            if not (0 <= x < width) or not (0 <= y < height):
                print(
                    (
                        "[LaneATT] WARNING: lane with score "
                        f"{lane.score:.3f} has out-of-bounds point {point}"
                        f" for frame size {width}x{height}."
                    )
                )
                out_of_bounds = True
                break
        cv2.line(vis, lane.points[0], lane.points[1], LANE_COLOR, 3)
        bottom = lane.bottom_point
        label = f"{idx}:{int(bottom[0])}"
        cv2.circle(vis, bottom, 4, LANE_COLOR, -1)
        cv2.putText(
            vis,
            label,
            (int(bottom[0]), int(bottom[1]) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            LANE_COLOR,
            1,
            cv2.LINE_AA,
        )
        if out_of_bounds:
            cv2.putText(
                vis,
                "warn",
                (int(bottom[0]), int(bottom[1]) - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
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

    reference_lanes: List[LaneDetection] = []
    lane_width_samples: List[float] = []
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
        scaled_lanes = _rescale_lane_points(lanes, frame.shape[:2])
        if not scaled_lanes:
            continue
        scaled_lanes.sort(key=lambda lane: lane.bottom_point[0])
        scores = [lane.score for lane in scaled_lanes]
        if scores:
            print(
                (
                    f"[LaneATT] Frame {frame_idx}: {len(scores)} lanes | score min="
                    f"{min(scores):.3f}, max={max(scores):.3f}, avg={statistics.mean(scores):.3f}"
                )
            )
        overlay_source = frame

        vis = _draw_lanes(frame, scaled_lanes)
        reference_idx = len(scaled_lanes) - 2 if len(scaled_lanes) >= 2 else len(scaled_lanes) - 1
        reference_lane = scaled_lanes[reference_idx]
        right_lane = scaled_lanes[-1]
        if reference_idx != len(scaled_lanes) - 1:
            cv2.line(vis, reference_lane.points[0], reference_lane.points[1], (0, 165, 255), 3)
            cv2.putText(
                vis,
                "baseline",
                (
                    int(reference_lane.bottom_point[0]),
                    int(reference_lane.bottom_point[1]) + 16,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )
        if len(scaled_lanes) >= 2:
            width_sample = right_lane.bottom_point[0] - reference_lane.bottom_point[0]
            if width_sample > 0 and _should_accept_width_sample(
                width_sample, frame.shape[1], params, f"Frame {frame_idx}"
            ):
                lane_width_samples.append(width_sample)
        cv2.imwrite(str(debug_dir / f"lanes_{frame_idx:05d}.jpg"), vis)
        reference_lanes.append(reference_lane)

    if not reference_lanes:
        print("No lanes detected; nothing to save.")
        return

    aggregated = _aggregate_lane(reference_lanes)
    if lane_width_samples:
        lane_width_raw_px = float(statistics.median(lane_width_samples))
        print(
            (
                "[LaneATT] lane width samples -> "
                f"count={len(lane_width_samples)}, "
                f"min={min(lane_width_samples):.2f}, "
                f"max={max(lane_width_samples):.2f}, "
                f"median={lane_width_raw_px:.2f}"
            )
        )
    else:
        lane_width_raw_px = None
        print("[LaneATT] No valid lane-width samples; will fall back to default width.")
    lane_width_px = None
    if overlay_source is not None:
        frame_width = overlay_source.shape[1]
        lane_width_px = _resolve_lane_width(lane_width_raw_px, frame_width, params)

    if aggregated is None or overlay_source is None:
        print("Failed to aggregate lanes for ROI overlay.")
        return

    polygon = _build_polygon(
        aggregated,
        overlay_source.shape[:2],
        params,
        lane_width_px=lane_width_px,
    )
    overlay = _render_overlay(overlay_source, polygon, aggregated)
    overlay_name = save_dir / f"laneatt_overlay_{reference_lanes[-1].bottom_point[0]:.0f}.jpg"
    cv2.imwrite(str(overlay_name), overlay)

    print(f"Saved per-frame overlays to: {debug_dir}")
    print(f"Saved final overlay to: {overlay_name}")
    if lane_width_raw_px is not None:
        print(f"Raw lane width sample median (px): {lane_width_raw_px:.2f}")
    if lane_width_px is not None:
        print(f"Resolved lane width used for ROI (px): {lane_width_px:.2f}")


if __name__ == "__main__":
    main()
