"""Emergency lane ROI generation using LaneATT lane detection."""
from __future__ import annotations

import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
import numpy as np

from lane_detection.laneatt import LaneATTConfig, LaneATTDetector, LaneDetection

from src.utils.paths import OUTPUTS_DIR
from .auto_cv import AutoCVParams, AutoCVResult, DetectedLine

Point = Tuple[int, int]
Polygon = List[Point]


@dataclass
class LaneATTParams:
    """LaneATT powered ROI generation configuration."""

    sample_frames: int = 18
    top_ratio: float = 0.33
    bottom_margin: int = 18
    buffer: int = 10
    min_score: float = 0.05
    min_lane_frames: int = 4
    emergency_width_factor: float = 1.0
    min_roi_width_px: int = 80
    fallback_lane_width_ratio: float = 0.10
    min_width_sample_px: int = 40
    max_width_sample_ratio: float = 0.7
    allow_auto_cv_fallback: bool = True
    save_debug: bool = True
    laneatt: Dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, object]]) -> "LaneATTParams":
        data = dict(cfg or {})
        laneatt_cfg = data.get("laneatt") or {}
        if isinstance(laneatt_cfg, (str, Path)):
            laneatt_cfg = {"weights": str(laneatt_cfg)}
        params = cls(
            sample_frames=int(data.get("sample_frames", cls.sample_frames)),
            top_ratio=float(np.clip(data.get("top_ratio", cls.top_ratio), 0.05, 0.9)),
            bottom_margin=int(max(0, data.get("bottom_margin", cls.bottom_margin))),
            buffer=int(max(0, data.get("buffer", cls.buffer))),
            min_score=float(max(0.0, data.get("min_score", cls.min_score))),
            min_lane_frames=int(max(1, data.get("min_lane_frames", cls.min_lane_frames))),
            emergency_width_factor=float(
                max(0.3, float(data.get("emergency_width_factor", cls.emergency_width_factor)))
            ),
            min_roi_width_px=int(max(20, int(data.get("min_roi_width_px", cls.min_roi_width_px)))),
            fallback_lane_width_ratio=float(
                max(
                    0.02,
                    min(
                        0.5,
                        float(
                            data.get(
                                "fallback_lane_width_ratio", cls.fallback_lane_width_ratio
                            )
                        ),
                    ),
                )
            ),
            min_width_sample_px=int(
                max(10, int(data.get("min_width_sample_px", cls.min_width_sample_px)))
            ),
            max_width_sample_ratio=float(
                max(
                    0.1,
                    min(
                        0.95,
                        float(
                            data.get(
                                "max_width_sample_ratio", cls.max_width_sample_ratio
                            )
                        ),
                    ),
                )
            ),
            allow_auto_cv_fallback=bool(
                data.get("allow_auto_cv_fallback", cls.allow_auto_cv_fallback)
            ),
            save_debug=bool(data.get("save_debug", cls.save_debug)),
            laneatt=dict(laneatt_cfg),
        )
        return params



def _sample_frames(video_path: Path, sample_frames: int) -> List[Tuple[int, np.ndarray]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {video_path}")

    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = sample_frames * 3
    max_range = max(sample_frames, min(frame_count - 1, sample_frames * 4))
    indices = sorted({int(round(x)) for x in np.linspace(0, max_range, sample_frames)})

    frames: List[Tuple[int, np.ndarray]] = []
    for idx in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
        ret, frame = capture.read()
        if not ret or frame is None:
            continue
        frames.append((idx, frame))
    capture.release()
    return frames


def _rescale_lane_points(
    lanes: Sequence[LaneDetection], frame_shape: Tuple[int, int]
) -> List[LaneDetection]:
    """Convert LaneATT coordinates to pixel space when needed."""

    if not lanes:
        return []
    height, width = frame_shape
    if height <= 0 or width <= 0:
        return list(lanes)
    xs = [abs(float(pt[0])) for lane in lanes for pt in lane.points]
    ys = [abs(float(pt[1])) for lane in lanes for pt in lane.points]
    max_x = max(xs) if xs else 0.0
    max_y = max(ys) if ys else 0.0
    normalized_x = max_x <= 3.0
    normalized_y = max_y <= 3.0
    if not normalized_x and not normalized_y:
        return list(lanes)

    def clamp(value: float, limit: int) -> int:
        if limit <= 1:
            return int(round(value))
        return int(max(0.0, min(limit - 1, value)))

    scaled: List[LaneDetection] = []
    for lane in lanes:
        points: List[Point] = []
        for x, y in lane.points:
            scaled_x = float(x) * width if normalized_x else float(x)
            scaled_y = float(y) * height if normalized_y else float(y)
            points.append((clamp(scaled_x, width), clamp(scaled_y, height)))
        scaled.append(
            LaneDetection(
                points=tuple(points),
                length=lane.length,
                angle=lane.angle,
                score=lane.score,
            )
        )
    print(
        (
            f"[LaneATT] Rescaled {len(lanes)} lane(s) using frame size "
            f"{width}x{height} (normalized_x={normalized_x}, normalized_y={normalized_y})."
        )
    )
    return scaled


def _should_accept_width_sample(
    width_sample: float, frame_width: int, params: LaneATTParams, context: str
) -> bool:
    """Check if a lane-width sample falls inside the configured range."""

    min_sample = float(params.min_width_sample_px)
    max_sample = float(frame_width * params.max_width_sample_ratio)
    if min_sample <= width_sample <= max_sample:
        print(
            (
                f"[LaneATT] {context}: width sample accepted = {width_sample:.2f}px "
                f"(range {min_sample:.1f}~{max_sample:.1f})."
            )
        )
        return True
    print(
        (
            f"[LaneATT] {context}: ignore suspicious width sample="
            f"{width_sample:.2f}px (min={min_sample:.1f}, max={max_sample:.1f})."
        )
    )
    return False


def _aggregate_lane(lanes: Iterable[LaneDetection]) -> Optional[DetectedLine]:
    lanes = list(lanes)
    if not lanes:
        return None
    bottom_points = [lane.bottom_point for lane in lanes]
    top_points = [lane.top_point for lane in lanes]
    xs_bottom = [pt[0] for pt in bottom_points]
    ys_bottom = [pt[1] for pt in bottom_points]
    xs_top = [pt[0] for pt in top_points]
    ys_top = [pt[1] for pt in top_points]
    bottom_x = statistics.median(xs_bottom)
    bottom_y = statistics.median(ys_bottom)
    top_x = statistics.median(xs_top)
    top_y = statistics.median(ys_top)
    if top_y == bottom_y:
        return None
    m = (bottom_y - top_y) / max(1e-3, (bottom_x - top_x))
    b = bottom_y - m * bottom_x
    angle = math.degrees(math.atan2(bottom_y - top_y, bottom_x - top_x))
    length = math.hypot(bottom_x - top_x, bottom_y - top_y)
    score = statistics.mean(lane.score for lane in lanes)
    points = ((int(top_x), int(top_y)), (int(bottom_x), int(bottom_y)))
    return DetectedLine(
        m=float(m),
        b=float(b),
        score=float(score),
        length=float(length),
        angle=float(abs(angle)),
        x_bottom=float(bottom_x),
        frame_index=-1,
        points=points,
    )


def _build_polygon(
    line: DetectedLine,
    frame_shape: Tuple[int, int],
    params: LaneATTParams,
    lane_width_px: Optional[float] = None,
) -> Polygon:
    height, width = frame_shape
    top_candidate = max(0, min(height - 1, int(line.points[0][1])))
    max_top = max(0, min(height - 1, int(height * params.top_ratio)))
    top_y = min(top_candidate, max_top)
    bottom_y = height - 1 - params.bottom_margin
    bottom_y = max(top_y + 10, min(height - 1, bottom_y))

    def clamp_x(x: float) -> int:
        return int(max(0, min(width - 1, x)))

    top_x = clamp_x(line.points[0][0] + params.buffer)
    bottom_x = clamp_x(line.points[1][0] + params.buffer)
    if lane_width_px is not None and lane_width_px > 0:
        lane_w = max(params.min_roi_width_px, lane_width_px)
    else:
        default_lane_w = width * params.fallback_lane_width_ratio
        lane_w = max(params.min_roi_width_px, default_lane_w)

    right_x = clamp_x(bottom_x + lane_w)
    roi_width = right_x - bottom_x
    desired_width = int(min(max(lane_w, params.min_roi_width_px), width))
    if roi_width < params.min_roi_width_px and desired_width > roi_width:
        bottom_x = clamp_x(right_x - desired_width)
        roi_width = right_x - bottom_x
    polygon = [
        (bottom_x, bottom_y),
        (right_x, bottom_y),
        (right_x, top_y),
        (top_x, top_y),
    ]
    width_float = float(max(1, width))
    print(
        (
            f"[LaneATT] ROI bounds -> left: {bottom_x}px ({bottom_x / width_float:.3f}), "
            f"right: {right_x}px ({right_x / width_float:.3f}), "
            f"width: {roi_width}px ({roi_width / width_float:.3f})."
        )
    )
    return polygon


def _resolve_lane_width(
    lane_width_raw_px: Optional[float], frame_width: int, params: LaneATTParams
) -> float:
    """Return the final lane width while guarding against implausible values."""

    raw_for_calc = lane_width_raw_px
    min_valid = float(params.min_width_sample_px)
    max_valid = float(frame_width * params.max_width_sample_ratio)
    if raw_for_calc is not None and not (min_valid <= raw_for_calc <= max_valid):
        print(
            (
                f"[LaneATT] WARNING: lane_width_raw_px={raw_for_calc:.2f} outside "
                "expected range; falling back to default width."
            )
        )
        raw_for_calc = None
    if raw_for_calc is not None:
        lane_width_final_px = max(
            params.min_roi_width_px, raw_for_calc * params.emergency_width_factor
        )
    else:
        default_lane_w = frame_width * params.fallback_lane_width_ratio
        lane_width_final_px = max(params.min_roi_width_px, default_lane_w)
    return lane_width_final_px


def _render_overlay(frame: np.ndarray, polygon: Polygon, line: Optional[DetectedLine]) -> np.ndarray:
    overlay = frame.copy()
    if line is not None:
        cv2.line(overlay, line.points[0], line.points[1], (0, 255, 0), 2)
    if polygon:
        cv2.polylines(overlay, [np.array(polygon, dtype=np.int32)], True, (0, 0, 255), 2)
        cv2.fillPoly(
            overlay,
            [np.array(polygon, dtype=np.int32)],
            color=(0, 0, 255),
            lineType=cv2.LINE_AA,
        )
    return overlay


def estimate_roi_laneatt(
    video_path: Path,
    params: LaneATTParams,
    auto_cv_params: Optional[AutoCVParams] = None,
    overlay: bool = False,
    overlay_dir: Optional[Path] = None,
    frames: Optional[Sequence[np.ndarray]] = None,
) -> AutoCVResult:
    """Estimate the emergency-lane ROI using LaneATT detections."""

    if cv2 is None:
        raise RuntimeError("OpenCV is required for LaneATT ROI generation.")
    start = time.time()
    lane_cfg = LaneATTConfig.from_config(params.laneatt)
    detector = LaneATTDetector(lane_cfg)
    sampled: List[Tuple[int, np.ndarray]]
    if frames is not None:
        sampled = list(enumerate(frames))
    else:
        sampled = _sample_frames(video_path, params.sample_frames)
    used_frames: List[int] = []
    detections: List[LaneDetection] = []
    lane_width_samples: List[float] = []
    overlay_image: Optional[np.ndarray] = None
    lane_width_raw_px: Optional[float] = None
    lane_width_final_px: Optional[float] = None
    for index, frame in sampled:
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
        bottom_xs = [lane.bottom_point[0] for lane in scaled_lanes]
        bottom_str = ", ".join(f"{idx}:{x:.1f}" for idx, x in enumerate(bottom_xs))
        print(f"[LaneATT] Frame {index}: lane bottom_x values -> {bottom_str}")
        if len(bottom_xs) >= 2:
            width_sample = bottom_xs[-1] - bottom_xs[-2]
            frame_width = frame.shape[1]
            if width_sample > 0 and _should_accept_width_sample(
                width_sample, frame_width, params, f"Frame {index}"
            ):
                lane_width_samples.append(width_sample)
        reference_idx = len(scaled_lanes) - 2 if len(scaled_lanes) >= 2 else len(scaled_lanes) - 1
        reference_lane = scaled_lanes[reference_idx]
        print(
            f"[LaneATT] Frame {index}: selected lane idx={reference_idx}, "
            f"bottom_x={reference_lane.bottom_point[0]:.1f}"
        )
        detections.append(reference_lane)
        used_frames.append(index)
        overlay_image = frame
    if len(detections) < params.min_lane_frames:
        message = (
            "laneatt_failed_not_enough_lanes"
            if detections
            else "laneatt_failed_no_lanes"
        )
        success = False
        polygon = None
        detected_line = None
    else:
        detected_line = _aggregate_lane(detections)
        if detected_line is None:
            success = False
            polygon = None
            message = "laneatt_failed_geometry"
        else:
            if lane_width_samples:
                lane_width_raw_px = statistics.median(lane_width_samples)
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
                print(
                    "[LaneATT] No valid lane-width samples; will fall back to default width."
                )
            _frame_height, frame_width = overlay_image.shape[:2]  # type: ignore[misc]
            lane_width_final_px = _resolve_lane_width(
                lane_width_raw_px, frame_width, params
            )
            polygon = _build_polygon(
                detected_line,
                overlay_image.shape[:2],  # type: ignore[arg-type]
                params,
                lane_width_px=lane_width_final_px,
            )
            success = True
            message = ""
    duration = time.time() - start
    metrics: Dict[str, float] = {
        "duration": float(duration),
        "lanes_detected": float(len(detections)),
        "engine": "laneatt",
        "laneatt_model_loaded": 1.0 if detector.model_loaded else 0.0,
    }
    if detected_line is not None:
        metrics["lane_width_raw_px"] = float(lane_width_raw_px or 0.0)
        if lane_width_final_px is not None:
            metrics["lane_width_px"] = float(lane_width_final_px)
    if overlay and overlay_image is not None and polygon:
        overlay_img = _render_overlay(overlay_image, polygon, detected_line)
        target_dir = overlay_dir or (OUTPUTS_DIR / "laneatt")
        target_dir.mkdir(parents=True, exist_ok=True)
        frame_tag = used_frames[-1] if used_frames else "na"
        overlay_path = target_dir / f"laneatt_{int(time.time())}_f{frame_tag}.jpg"
        cv2.imwrite(str(overlay_path), overlay_img)
        metrics["overlay_path"] = str(overlay_path)
    result = AutoCVResult(
        success=success,
        polygon=polygon,
        base_size=(overlay_image.shape[1], overlay_image.shape[0]) if overlay_image is not None else (0, 0),
        used_frames=used_frames,
        line=detected_line,
        params_used=AutoCVParams() if auto_cv_params is None else auto_cv_params,
        metrics=metrics,
        duration=duration,
        message=message,
        overlay=None,
    )
    return result
