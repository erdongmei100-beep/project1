"""Emergency lane ROI generation using LaneATT lane detection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
import numpy as np

from lane_detection.laneatt import LaneATTConfig, LaneATTDetector

from src.utils.paths import OUTPUTS_DIR
from .auto_cv import AutoCVParams, AutoCVResult, estimate_roi as estimate_roi_auto

Point = Tuple[int, int]
Polygon = List[Point]


@dataclass
class LaneCandidate:
    pts: np.ndarray
    avg_x: float
    frame_index: int
    lanes_on_frame: List[np.ndarray]
    frame: np.ndarray


@dataclass
class LaneATTParams:
    """LaneATT powered ROI generation configuration."""

    sample_frames: int = 18
    bottom_ratio: float = 0.3
    target_lane_index_from_right: int = 2
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
        bottom_ratio_cfg = data.get("bottom_ratio", cls.bottom_ratio)
        if "bottom_ratio" not in data and "top_ratio" in data:
            try:
                bottom_ratio_cfg = 1.0 - float(data["top_ratio"])
            except Exception:
                bottom_ratio_cfg = cls.bottom_ratio

        params = cls(
            sample_frames=int(data.get("sample_frames", cls.sample_frames)),
            bottom_ratio=float(np.clip(bottom_ratio_cfg, 0.05, 0.9)),
            target_lane_index_from_right=int(
                max(1, data.get("target_lane_index_from_right", cls.target_lane_index_from_right))
            ),
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


def _select_target_lane(
    lanes: Iterable[np.ndarray],
    frame_shape: Tuple[int, int],
    params: LaneATTParams,
    frame_index: int,
    frame: np.ndarray,
) -> Optional[LaneCandidate]:
    height, width = frame_shape
    bottom_ratio = float(np.clip(params.bottom_ratio, 0.05, 0.95))
    y_threshold = height * (1.0 - bottom_ratio)
    candidates: List[Tuple[float, np.ndarray]] = []

    for lane in lanes:
        if lane.ndim != 2 or lane.shape[1] != 2:
            continue
        pts = np.asarray(lane, dtype=float)
        pts_bottom = pts[pts[:, 1] >= y_threshold]
        if len(pts_bottom) == 0:
            continue
        avg_x = float(pts_bottom[:, 0].mean())
        candidates.append((avg_x, pts))

    if len(candidates) < params.target_lane_index_from_right:
        return None

    candidates.sort(key=lambda item: item[0])
    avg_x, pts = candidates[-params.target_lane_index_from_right]
    return LaneCandidate(
        pts=pts,
        avg_x=float(avg_x),
        frame_index=frame_index,
        lanes_on_frame=[np.asarray(lane, dtype=float) for lane in lanes],
        frame=frame.copy(),
    )


def _build_roi_polygon(
    lane_pts: np.ndarray, frame_shape: Tuple[int, int], params: LaneATTParams
) -> Tuple[Optional[np.ndarray], Optional[Polygon]]:
    height, width = frame_shape
    bottom_ratio = float(np.clip(params.bottom_ratio, 0.05, 0.95))
    y_top = int(height * (1.0 - bottom_ratio))

    pts = np.asarray(lane_pts, dtype=float)
    lane_segment = pts[pts[:, 1] >= y_top]
    if len(lane_segment) < 2:
        return None, None

    # Sort from bottom to top to construct a polygon that hugs the lane line.
    lane_segment = lane_segment[np.argsort(-lane_segment[:, 1])]

    poly: Polygon = []
    for x, y in lane_segment:
        poly.append((int(round(x)), int(round(y))))

    top_y = int(round(lane_segment[-1][1]))

    # Close the polygon along the right border – avoids the previous bug that
    # produced a vertical strip from an averaged x position.
    poly.append((width - 1, top_y))
    poly.append((width - 1, height - 1))
    poly.append((int(round(lane_segment[0][0])), height - 1))

    poly_np = np.array(poly, dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 1)
    return mask, poly


def _draw_debug_overlays(
    frame: np.ndarray,
    lanes: Iterable[np.ndarray],
    selected_lane: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    polygon: Optional[Polygon],
    target_dir: Path,
) -> Dict[str, str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    raw_path = target_dir / f"laneatt_raw_{timestamp}.jpg"
    lane_path = target_dir / f"laneatt_lanes_{timestamp}.jpg"
    roi_path = target_dir / f"laneatt_roi_{timestamp}.jpg"

    cv2.imwrite(str(raw_path), frame)

    lanes_img = frame.copy()
    color_cycle = [
        (255, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (0, 165, 255),
        (255, 255, 0),
    ]
    for idx, lane in enumerate(lanes):
        pts_int = np.round(lane).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(lanes_img, [pts_int], False, color_cycle[idx % len(color_cycle)], 2)
    if selected_lane is not None:
        pts_int = np.round(selected_lane).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(lanes_img, [pts_int], False, (0, 255, 0), 3)
    cv2.imwrite(str(lane_path), lanes_img)

    roi_img = lanes_img.copy()
    if mask is not None:
        red = np.zeros_like(roi_img)
        red[:, :, 2] = (mask * 200).astype(np.uint8)
        roi_img = cv2.addWeighted(roi_img, 1.0, red, 0.35, 0)
    if polygon:
        cv2.polylines(roi_img, [np.array(polygon, dtype=np.int32)], True, (0, 0, 255), 2)
    cv2.imwrite(str(roi_path), roi_img)

    return {
        "raw": str(raw_path),
        "lanes": str(lane_path),
        "roi": str(roi_path),
    }


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

    candidates: List[LaneCandidate] = []
    for index, frame in sampled:
        lanes = detector.detect_lanes(frame)
        candidate = _select_target_lane(lanes, frame.shape[:2], params, index, frame)
        if candidate is None:
            continue
        candidates.append(candidate)

    overlay_frame: Optional[np.ndarray] = None
    chosen: Optional[LaneCandidate] = None
    polygon: Optional[Polygon] = None
    mask: Optional[np.ndarray] = None
    if len(candidates) < params.min_lane_frames:
        message = (
            "laneatt_failed_not_enough_lanes"
            if candidates
            else "laneatt_failed_no_lanes"
        )
        success = False
    else:
        candidates.sort(key=lambda cand: cand.avg_x)
        chosen = candidates[len(candidates) // 2]
        mask, polygon = _build_roi_polygon(chosen.pts, chosen.frame.shape[:2], params)
        overlay_frame = chosen.frame
        if mask is None or not polygon:
            success = False
            message = "laneatt_failed_geometry"
        else:
            success = True
            message = ""

    duration = time.time() - start
    metrics: Dict[str, float] = {
        "duration": float(duration),
        "lanes_detected": float(len(candidates)),
        "engine": "laneatt",
        "laneatt_model_loaded": 1.0 if detector.model_loaded else 0.0,
    }

    if overlay and overlay_frame is not None:
        debug_candidate = chosen or (candidates[-1] if candidates else None)
        target_dir = overlay_dir or (OUTPUTS_DIR / "laneatt")
        debug_paths = _draw_debug_overlays(
            overlay_frame,
            debug_candidate.lanes_on_frame if debug_candidate else [],
            debug_candidate.pts if debug_candidate else None,
            mask,
            polygon,
            target_dir,
        )
        metrics.update({f"debug_{k}": v for k, v in debug_paths.items()})

    result = AutoCVResult(
        success=success,
        polygon=polygon,
        base_size=(
            (overlay_frame.shape[1], overlay_frame.shape[0]) if overlay_frame is not None else (0, 0)
        ),
        used_frames=[cand.frame_index for cand in candidates],
        line=None,
        params_used=AutoCVParams() if auto_cv_params is None else auto_cv_params,
        metrics=metrics,
        duration=duration,
        message=message,
        overlay=mask,
    )

    if success or not params.allow_auto_cv_fallback:
        return result

    print("[LaneATT ROI] Falling back to auto_cv due to detection failure")
    return estimate_roi_auto(
        video_path,
        auto_cv_params or AutoCVParams(),
        overlay=overlay,
        overlay_dir=overlay_dir,
    )
