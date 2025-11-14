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
    polygon = [
        (bottom_x, bottom_y),
        (width - 1, bottom_y),
        (width - 1, top_y),
        (top_x, top_y),
    ]
    return polygon


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
    overlay_image: Optional[np.ndarray] = None
    for index, frame in sampled:
        result = detector.detect(frame)
        best = result.best_lane
        if best is None:
            continue
        if best.score < params.min_score:
            continue
        detections.append(best)
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
            polygon = _build_polygon(detected_line, overlay_image.shape[:2], params)  # type: ignore[arg-type]
            success = True
            message = ""
    duration = time.time() - start
    metrics: Dict[str, float] = {
        "duration": float(duration),
        "lanes_detected": float(len(detections)),
        "engine": "laneatt",
        "laneatt_model_loaded": 1.0 if detector.model_loaded else 0.0,
    }
    if overlay and overlay_image is not None and polygon:
        overlay_img = _render_overlay(overlay_image, polygon, detected_line)
        target_dir = overlay_dir or (OUTPUTS_DIR / "laneatt")
        target_dir.mkdir(parents=True, exist_ok=True)
        overlay_path = target_dir / f"laneatt_{int(time.time())}.jpg"
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
