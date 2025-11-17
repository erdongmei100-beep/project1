"""Emergency lane ROI generation using LaneATT lane detection."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import logging

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
import numpy as np

from src.lane_detection.laneatt import LaneATTConfig, LaneATTDetector

from src.utils.paths import OUTPUTS_DIR
from .auto_cv import AutoCVParams, AutoCVResult, estimate_roi as estimate_roi_auto

Point = Tuple[int, int]
Polygon = List[Point]


class LaneATTError:
    NO_LANES = "no_lanes"
    TOO_FEW_POINTS = "too_few_points"
    BAD_POLYGON = "bad_polygon"
    SMALL_AREA = "small_area"
    EMPTY_MASK = "empty_mask"
    HOUGH_FALLBACK_USED = "hough_fallback"

logger = logging.getLogger(__name__)


@dataclass
class LaneCandidate:
    pts: np.ndarray
    avg_x: float
    frame_index: int
    lanes_on_frame: List[np.ndarray]
    frame: np.ndarray
    detected_lane_count: Optional[int] = None


@dataclass
class LaneATTParams:
    """LaneATT powered ROI generation configuration."""

    sample_frames: int = 18
    bottom_ratio: float = 0.35
    target_lane_index_from_right: int = 2
    min_bottom_points: int = 6
    min_poly_points: int = 5
    min_roi_area_ratio: float = 0.01
    allow_hough_fallback: bool = True
    min_lane_frames: int = 4
    min_lanes_for_roi: int = 1
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
            min_bottom_points=int(max(1, data.get("min_bottom_points", cls.min_bottom_points))),
            min_poly_points=int(max(3, data.get("min_poly_points", cls.min_poly_points))),
            min_roi_area_ratio=float(max(0.0, data.get("min_roi_area_ratio", cls.min_roi_area_ratio))),
            allow_hough_fallback=bool(data.get("allow_hough_fallback", cls.allow_hough_fallback)),
            min_lane_frames=int(max(1, data.get("min_lane_frames", cls.min_lane_frames))),
            min_lanes_for_roi=int(max(1, data.get("min_lanes_for_roi", cls.min_lanes_for_roi))),
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
) -> Tuple[Optional[LaneCandidate], int]:
    height, width = frame_shape
    bottom_ratio = float(np.clip(params.bottom_ratio, 0.05, 0.95))
    y_threshold = height * (1.0 - bottom_ratio)
    candidates: List[Tuple[float, np.ndarray]] = []

    for lane in lanes:
        if lane.ndim != 2 or lane.shape[1] != 2:
            continue
        pts = np.asarray(lane, dtype=float)
        pts = pts[np.argsort(pts[:, 1])]
        pts_bottom = pts[pts[:, 1] >= y_threshold]
        if len(pts_bottom) < params.min_bottom_points and len(pts) >= 2:
            y_start = max(y_threshold, float(pts[:, 1].min()))
            y_end = float(pts[:, 1].max())
            if y_end > y_start:
                y_interp = np.linspace(y_start, y_end, num=params.min_bottom_points)
                x_interp = np.interp(y_interp, pts[:, 1], pts[:, 0])
                new_pts = np.stack([x_interp, y_interp], axis=1)
                pts_bottom = new_pts
                pts = np.vstack([pts, new_pts])
                pts = pts[np.argsort(pts[:, 1])]
        if len(pts_bottom) < params.min_bottom_points:
            continue
        avg_x = float(pts_bottom[:, 0].mean())
        candidates.append((avg_x, pts))

    lane_count = len(candidates)
    if lane_count == 0:
        return None, lane_count

    candidates.sort(key=lambda item: item[0])
    target_idx = -params.target_lane_index_from_right
    if lane_count < params.target_lane_index_from_right:
        target_idx = -1
    avg_x, pts = candidates[target_idx]
    return (
        LaneCandidate(
            pts=pts,
            avg_x=float(avg_x),
            frame_index=frame_index,
            lanes_on_frame=[np.asarray(lane, dtype=float) for lane in lanes],
            frame=frame.copy(),
            detected_lane_count=lane_count,
        ),
        lane_count,
    )


def _build_roi_polygon(
    lane_pts: np.ndarray, frame_shape: Tuple[int, int], params: LaneATTParams
) -> Tuple[Optional[np.ndarray], Optional[Polygon], Optional[str]]:
    height, width = frame_shape
    bottom_ratio = float(np.clip(params.bottom_ratio, 0.05, 0.95))
    y_top = int(height * (1.0 - bottom_ratio))

    pts = np.asarray(lane_pts, dtype=float)
    lane_segment = pts[pts[:, 1] >= y_top]
    lane_segment = lane_segment[np.argsort(lane_segment[:, 1])]
    if len(lane_segment) < 2:
        return None, None, LaneATTError.TOO_FEW_POINTS

    if len(lane_segment) < params.min_poly_points and len(lane_segment) >= 2:
        bottom_pts = lane_segment[-min(len(lane_segment), 8) :]
        ys = bottom_pts[:, 1]
        xs = bottom_pts[:, 0]
        if len(bottom_pts) >= 2:
            coef = np.polyfit(ys, xs, 1)
            y_new = np.linspace(y_top, ys.max(), num=params.min_poly_points)
            x_new = coef[0] * y_new + coef[1]
            extrapolated = np.stack([x_new, y_new], axis=1)
            lane_segment = np.vstack([lane_segment, extrapolated])
            lane_segment = lane_segment[np.argsort(lane_segment[:, 1])]
    if len(lane_segment) < params.min_poly_points:
        return None, None, LaneATTError.TOO_FEW_POINTS

    lane_points: Polygon = []
    for x, y in lane_segment[::-1]:
        lane_points.append((int(round(x)), int(round(y))))
    top_y = lane_points[-1][1]

    poly: Polygon = []
    for pt in lane_points:
        if not poly or poly[-1] != pt:
            poly.append(pt)
    poly.extend([(width - 1, top_y), (width - 1, height - 1), (poly[0][0], height - 1)])

    poly_np = np.array(poly, dtype=np.int32)
    area = cv2.contourArea(poly_np)
    if area < params.min_roi_area_ratio * float(width * height):
        return None, None, LaneATTError.SMALL_AREA

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [poly_np], 1)
    if mask.sum() == 0:
        return None, None, LaneATTError.EMPTY_MASK
    return mask, poly, None


def _save_debug(
    frame: np.ndarray,
    lanes: Iterable[np.ndarray],
    selected_lane: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    polygon: Optional[Polygon],
    target_dir: Path,
    reason: Optional[str] = None,
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
    if reason:
        cv2.putText(
            roi_img,
            f"FAIL: {reason}" if reason != "ok" else "ok",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if reason != "ok" else (0, 200, 0),
            2,
            cv2.LINE_AA,
        )
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
    last_debug_frame: Optional[np.ndarray] = None
    last_debug_lanes: List[np.ndarray] = []
    last_selected_lane: Optional[np.ndarray] = None
    last_reason: Optional[str] = None

    debug_dir = OUTPUTS_DIR / "laneatt"
    debug_enabled = overlay or params.save_debug

    for index, frame in sampled:
        lanes = detector.detect_lanes(frame)
        candidate, lane_count = _select_target_lane(
            lanes, frame.shape[:2], params, index, frame
        )
        last_debug_frame = frame.copy()
        last_debug_lanes = [np.asarray(lane, dtype=float) for lane in lanes]
        last_selected_lane = candidate.pts if candidate is not None else None
        last_reason = "no_lanes_detected" if lane_count == 0 else None
        if candidate is None:
            if debug_enabled and last_debug_frame is not None:
                _save_debug(
                    last_debug_frame,
                    last_debug_lanes,
                    last_selected_lane,
                    None,
                    None,
                    debug_dir,
                    reason=last_reason,
                )
            continue
        candidates.append(candidate)

    overlay_frame: Optional[np.ndarray] = None
    chosen: Optional[LaneCandidate] = None
    polygon: Optional[Polygon] = None
    mask: Optional[np.ndarray] = None
    reason: Optional[str] = None
    if not candidates:
        message = "laneatt_failed_no_lanes"
        success = False
    else:
        candidates.sort(key=lambda cand: cand.avg_x)
        idx = -params.target_lane_index_from_right
        if len(candidates) < params.target_lane_index_from_right:
            idx = -1
        chosen = candidates[idx]
        if chosen.detected_lane_count < params.min_lanes_for_roi:
            logger.warning(
                "LaneATT ROI: only %d lanes detected (min required %d), trying best-effort ROI",
                chosen.detected_lane_count,
                params.min_lanes_for_roi,
            )
            last_reason = "not_enough_lanes"
        mask, polygon, reason = _build_roi_polygon(chosen.pts, chosen.frame.shape[:2], params)
        overlay_frame = chosen.frame
        if mask is None or not polygon:
            success = False
            message = reason or "laneatt_failed_geometry"
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

    if debug_enabled and overlay_frame is not None:
        debug_candidate = chosen or (candidates[-1] if candidates else None)
        target_dir = overlay_dir or debug_dir
        debug_paths = _save_debug(
            overlay_frame,
            debug_candidate.lanes_on_frame if debug_candidate else last_debug_lanes,
            debug_candidate.pts if debug_candidate else last_selected_lane,
            mask if success else None,
            polygon,
            target_dir,
            reason="ok" if success else (message or last_reason),
        )
        metrics.update({f"debug_{k}": v for k, v in debug_paths.items()})
    elif debug_enabled and last_debug_frame is not None and (not success or last_reason):
        target_dir = overlay_dir or debug_dir
        debug_paths = _save_debug(
            last_debug_frame,
            last_debug_lanes,
            last_selected_lane,
            None,
            None,
            target_dir,
            reason=message or last_reason,
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

    if success or not params.allow_auto_cv_fallback or frames is not None:
        return result

    print("[LaneATT ROI] Falling back to auto_cv due to detection failure")
    return estimate_roi_auto(
        video_path,
        auto_cv_params or AutoCVParams(),
        overlay=overlay,
        overlay_dir=overlay_dir,
    )
