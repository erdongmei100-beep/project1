"""Automatic ROI estimation via computer vision heuristics."""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from src.utils.paths import OUTPUTS_DIR

Point = Tuple[int, int]
Polygon = List[Point]


@dataclass
class DetectedLine:
    """Describes a detected lane line on a frame."""

    m: float
    b: float
    score: float
    length: float
    angle: float
    x_bottom: float
    frame_index: int
    points: Tuple[Tuple[int, int], Tuple[int, int]]


@dataclass
class AutoCVParams:
    """Configuration parameters for the auto ROI detector."""

    sample_frames: int = 18
    crop_right: float = 0.45
    crop_bottom: float = 0.70
    s_max: int = 70
    v_min: int = 190
    angle_min: float = 18.0
    angle_max: float = 78.0
    top_ratio: float = 0.33
    bottom_margin: int = 18
    buffer: int = 6
    min_box_h_px: int = 30
    min_rel_area: float = 0.003
    min_sharpness: float = 60.0
    bbox_aspect_min: float = 0.6
    only_with_plate: bool = False
    allow_tail_fallback: bool = True
    save_debug: bool = True
    track_min_frames: int = 6
    entry_window_frames: int = 12
    per_track: bool = True

    @classmethod
    def from_config(cls, cfg: Dict[str, object] | None) -> "AutoCVParams":
        data = dict(cfg or {})
        numeric_fields = {
            "sample_frames": int,
            "bottom_margin": int,
            "buffer": int,
            "min_box_h_px": int,
            "track_min_frames": int,
            "entry_window_frames": int,
        }
        float_fields = {
            "crop_right",
            "crop_bottom",
            "s_max",
            "v_min",
            "angle_min",
            "angle_max",
            "top_ratio",
            "min_rel_area",
            "min_sharpness",
            "bbox_aspect_min",
        }
        bool_fields = {
            "only_with_plate",
            "allow_tail_fallback",
            "save_debug",
            "per_track",
        }
        parsed: Dict[str, object] = {}
        for key, caster in numeric_fields.items():
            if key in data:
                parsed[key] = caster(data[key])
        for key in float_fields:
            if key in data:
                parsed[key] = float(data[key])
        for key in bool_fields:
            if key in data:
                parsed[key] = bool(data[key])
        for key in [
            "crop_right",
            "crop_bottom",
            "top_ratio",
            "min_rel_area",
        ]:
            if key in parsed:
                parsed[key] = float(np.clip(parsed[key], 0.0, 1.0))
        if "s_max" in parsed:
            parsed["s_max"] = int(np.clip(parsed["s_max"], 0, 255))
        if "v_min" in parsed:
            parsed["v_min"] = int(np.clip(parsed["v_min"], 0, 255))
        return cls(**parsed)

    def evolve(self, **changes: object) -> "AutoCVParams":
        """Return a copy of the params with selective overrides."""

        return replace(self, **changes)


@dataclass
class AutoCVResult:
    """Result returned by the auto ROI detector."""

    success: bool
    polygon: Optional[Polygon]
    base_size: Tuple[int, int]
    used_frames: List[int]
    line: Optional[DetectedLine]
    params_used: AutoCVParams
    metrics: Dict[str, float] = field(default_factory=dict)
    duration: float = 0.0
    message: str = ""
    overlay: Optional[np.ndarray] = None


def _sample_frames(video_path: Path, params: AutoCVParams) -> Tuple[List[Tuple[int, np.ndarray]], float]:
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = params.sample_frames * 3
    max_range = min(frame_count - 1, int((fps if fps > 0 else 25.0) * 5)) if frame_count > 0 else params.sample_frames
    max_range = max(max_range, params.sample_frames)
    indices = sorted({int(round(x)) for x in np.linspace(0, max_range, params.sample_frames)})
    frames: List[Tuple[int, np.ndarray]] = []

    try:
        target_iter = iter(indices)
        target = next(target_iter)
    except StopIteration:
        return frames, fps

    current = 0
    while True:
        success, frame = capture.read()
        if not success or frame is None:
            break
        if current == target:
            frames.append((current, frame.copy()))
            try:
                target = next(target_iter)
            except StopIteration:
                break
        current += 1
        if current > indices[-1] + 5:
            break

    capture.release()
    return frames, fps


def _detect_lines_for_frames(
    frames: Sequence[Tuple[int, np.ndarray]], params: AutoCVParams
) -> Tuple[List[DetectedLine], List[int]]:
    detected: List[DetectedLine] = []
    used: List[int] = []
    for frame_index, frame in frames:
        line = _detect_lane_line(frame, frame_index, params)
        if line is not None:
            detected.append(line)
            used.append(frame_index)
    return detected, used


def _generate_relaxed_variants(params: AutoCVParams) -> List[AutoCVParams]:
    variants: List[AutoCVParams] = []
    variants.append(
        params.evolve(
            crop_right=min(params.crop_right + 0.18, 0.95),
            crop_bottom=min(params.crop_bottom + 0.18, 0.98),
        )
    )
    variants.append(
        params.evolve(
            angle_min=max(params.angle_min - 10.0, 3.0),
            angle_max=min(params.angle_max + 8.0, 88.0),
        )
    )
    variants.append(
        params.evolve(
            v_min=max(params.v_min - 35, 110),
            s_max=min(params.s_max + 40, 180),
        )
    )
    variants.append(
        params.evolve(
            crop_right=min(params.crop_right + 0.12, 0.98),
            crop_bottom=min(params.crop_bottom + 0.22, 0.99),
            angle_min=max(params.angle_min - 14.0, 2.0),
            angle_max=min(params.angle_max + 12.0, 89.5),
        )
    )
    return variants


def _detect_lane_line(frame: np.ndarray, frame_index: int, params: AutoCVParams) -> Optional[DetectedLine]:
    height, width = frame.shape[:2]
    crop_x0 = int(width * (1.0 - params.crop_right)) if params.crop_right > 0 else 0
    crop_y0 = int(height * (1.0 - params.crop_bottom)) if params.crop_bottom > 0 else 0
    crop = frame[crop_y0:height, crop_x0:width]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, params.v_min], dtype=np.uint8)
    upper = np.array([179, params.s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(mask, 50, 150)
    min_line_len = int(max(crop.shape[0], crop.shape[1]) * 0.3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=max(min_line_len, 30),
        maxLineGap=30,
    )
    if lines is None:
        return None

    best: Optional[DetectedLine] = None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        full_x1 = x1 + crop_x0
        full_y1 = y1 + crop_y0
        full_x2 = x2 + crop_x0
        full_y2 = y2 + crop_y0
        dx = float(full_x2 - full_x1)
        dy = float(full_y2 - full_y1)
        length = math.hypot(dx, dy)
        if length < 20.0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        if not (params.angle_min <= angle <= params.angle_max):
            continue
        if abs(dx) < 1e-3:
            continue
        m = dy / dx
        b = full_y1 - m * full_x1
        if abs(m) < 1e-4:
            continue
        x_bottom = (height - 1 - b) / m
        if not np.isfinite(x_bottom):
            continue
        score_bottom = np.clip(x_bottom / max(width - 1, 1), 0.0, 1.5)
        alignment = 1.0 - abs(angle - 60.0) / 60.0
        score = score_bottom * (1.0 + max(alignment, 0.0)) * (length / max(width, 1))
        if best is None or score > best.score:
            best = DetectedLine(
                m=m,
                b=b,
                score=score,
                length=length,
                angle=angle,
                x_bottom=x_bottom,
                frame_index=frame_index,
                points=((full_x1, full_y1), (full_x2, full_y2)),
            )
    return best


def _fuse_lines(lines: Sequence[DetectedLine], frame_height: int) -> DetectedLine:
    slopes = np.array([line.m for line in lines], dtype=np.float32)
    bottoms = np.array([line.x_bottom for line in lines], dtype=np.float32)
    lengths = np.array([line.length for line in lines], dtype=np.float32)
    angles = np.array([line.angle for line in lines], dtype=np.float32)

    m_med = float(np.median(slopes))
    x_bottom_med = float(np.median(bottoms))
    b_med = (frame_height - 1) - m_med * x_bottom_med
    length_med = float(np.median(lengths))
    angle_med = float(np.median(angles))

    base = lines[0]
    return DetectedLine(
        m=m_med,
        b=b_med,
        score=float(np.median([line.score for line in lines])),
        length=length_med,
        angle=angle_med,
        x_bottom=x_bottom_med,
        frame_index=base.frame_index,
        points=base.points,
    )


def _clamp_polygon(polygon: Polygon, width: int, height: int) -> Polygon:
    clamped: Polygon = []
    for x, y in polygon:
        clamped.append((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))))
    return clamped


def _polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for idx in range(len(polygon)):
        x1, y1 = polygon[idx]
        x2, y2 = polygon[(idx + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _build_roi_polygon(line: DetectedLine, width: int, height: int, params: AutoCVParams) -> Optional[Polygon]:
    y_top = int(height * params.top_ratio)
    y_bottom = height - 1 - params.bottom_margin
    y_top = int(np.clip(y_top, 0, height - 1))
    y_bottom = int(np.clip(y_bottom, 0, height - 1))
    if y_bottom <= y_top:
        y_bottom = min(height - 1, y_top + max(params.min_box_h_px, 10))
    m = line.m
    b = line.b
    if abs(m) < 1e-4:
        return None

    def project_x(y: int) -> int:
        x = (y - b) / m
        return int(round(x))

    left_bottom = project_x(y_bottom) + params.buffer
    left_top = project_x(y_top) + params.buffer
    right = width - 1
    polygon = [(left_bottom, y_bottom), (right, y_bottom), (right, y_top), (left_top, y_top)]
    polygon = _clamp_polygon(polygon, width, height)
    return polygon


def _compute_metrics(polygon: Polygon, frame: np.ndarray) -> Dict[str, float]:
    width = frame.shape[1]
    height = frame.shape[0]
    total_area = float(width * height) if width > 0 and height > 0 else 1.0
    area = _polygon_area(polygon)
    mask = np.zeros((height, width), dtype=np.uint8)
    pts = np.array(polygon, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    masked = cv2.bitwise_and(gray, gray, mask=mask)
    lap = cv2.Laplacian(masked, cv2.CV_64F)
    sharpness = float(lap.var())
    x_coords = np.array([p[0] for p in polygon], dtype=np.float32)
    y_coords = np.array([p[1] for p in polygon], dtype=np.float32)
    height_roi = float(y_coords.max() - y_coords.min()) if len(y_coords) else 0.0
    width_roi = float(x_coords.max() - x_coords.min()) if len(x_coords) else 0.0
    aspect = width_roi / height_roi if height_roi > 0 else 0.0
    rel_area = area / total_area if total_area > 0 else 0.0
    return {
        "area": area,
        "rel_area": rel_area,
        "sharpness": sharpness,
        "roi_height": height_roi,
        "roi_width": width_roi,
        "roi_aspect": aspect,
    }


def _draw_overlay(frame: np.ndarray, polygon: Polygon, line: Optional[DetectedLine], metrics: Dict[str, float]) -> np.ndarray:
    vis = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.fillPoly(vis, [pts], (0, 255, 0,))
    overlay = np.zeros_like(vis)
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    vis = cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)
    if line is not None:
        y_vals = [metrics.get("roi_height", 0.0), 0.0]
        x_coords = np.array([p[0] for p in polygon])
        y_coords = np.array([p[1] for p in polygon])
        y_top = int(y_coords.min())
        y_bottom = int(y_coords.max())
        for y in (y_top, y_bottom):
            x = int(round((y - line.b) / line.m))
            cv2.circle(vis, (x, y), 6, (0, 128, 255), -1, cv2.LINE_AA)
        cv2.line(vis, (int(x_coords.min()), y_bottom), (int(x_coords.max()), y_top), (0, 200, 255), 2, cv2.LINE_AA)
    text_lines = [
        f"area_ratio={metrics.get('rel_area', 0.0):.4f}",
        f"sharpness={metrics.get('sharpness', 0.0):.1f}",
        f"aspect={metrics.get('roi_aspect', 0.0):.2f}",
    ]
    for idx, text in enumerate(text_lines):
        cv2.putText(
            vis,
            text,
            (20, 40 + idx * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def estimate_roi(
    video_path: Path,
    params: AutoCVParams,
    *,
    overlay: bool = False,
    overlay_dir: Optional[Path] = None,
) -> AutoCVResult:
    start_time = time.time()
    frames, fps = _sample_frames(video_path, params)
    if not frames:
        return AutoCVResult(
            success=False,
            polygon=None,
            base_size=(0, 0),
            used_frames=[],
            line=None,
            params_used=params,
            metrics={},
            duration=time.time() - start_time,
            message="No frames sampled for auto ROI",
        )

    base_frame = frames[0][1]
    active_params = params
    detected, used_indices = _detect_lines_for_frames(frames, active_params)
    fallback_variant = 0

    if not detected and params.allow_tail_fallback:
        for idx, variant in enumerate(_generate_relaxed_variants(params), start=1):
            detected_variant, used_variant = _detect_lines_for_frames(frames, variant)
            if detected_variant:
                detected = detected_variant
                used_indices = used_variant
                active_params = variant
                fallback_variant = idx
                break

    if not detected:
        return AutoCVResult(
            success=False,
            polygon=None,
            base_size=(base_frame.shape[1], base_frame.shape[0]),
            used_frames=used_indices,
            line=None,
            params_used=active_params,
            metrics={},
            duration=time.time() - start_time,
            message="No lane line detected",
        )

    fused = _fuse_lines(detected, base_frame.shape[0])
    polygon = _build_roi_polygon(fused, base_frame.shape[1], base_frame.shape[0], active_params)
    if polygon is None:
        return AutoCVResult(
            success=False,
            polygon=None,
            base_size=(base_frame.shape[1], base_frame.shape[0]),
            used_frames=used_indices,
            line=fused,
            params_used=active_params,
            metrics={},
            duration=time.time() - start_time,
            message="Unable to derive polygon from detected line",
        )

    metrics = _compute_metrics(polygon, base_frame)
    height_ok = metrics.get("roi_height", 0.0) >= active_params.min_box_h_px
    area_ok = metrics.get("rel_area", 0.0) >= active_params.min_rel_area
    aspect_ok = metrics.get("roi_aspect", 0.0) >= active_params.bbox_aspect_min
    sharp_ok = metrics.get("sharpness", 0.0) >= active_params.min_sharpness
    success = height_ok and area_ok and aspect_ok and (
        not active_params.only_with_plate or sharp_ok
    )

    vis = None
    if overlay and active_params.save_debug:
        target_dir = overlay_dir or (OUTPUTS_DIR / "auto_cv")
        target_dir.mkdir(parents=True, exist_ok=True)
        vis = _draw_overlay(base_frame, polygon, fused, metrics)
        overlay_path = target_dir / f"{video_path.stem}_overlay.png"
        if cv2.imwrite(str(overlay_path), vis):
            metrics["overlay_path"] = str(overlay_path)

    duration = time.time() - start_time
    metrics["fps"] = fps
    metrics["duration_s"] = duration
    metrics["frames"] = len(frames)
    metrics["fallback_variant"] = fallback_variant

    message = ""
    if not success:
        failed_reasons = []
        if not height_ok:
            failed_reasons.append("roi_height")
        if not area_ok:
            failed_reasons.append("rel_area")
        if not aspect_ok:
            failed_reasons.append("roi_aspect")
        if params.only_with_plate and not sharp_ok:
            failed_reasons.append("sharpness")
        message = "; ".join(failed_reasons) or "Validation failed"

    return AutoCVResult(
        success=success,
        polygon=polygon,
        base_size=(base_frame.shape[1], base_frame.shape[0]),
        used_frames=used_indices,
        line=fused,
        params_used=active_params,
        metrics=metrics,
        duration=duration,
        message=message,
        overlay=vis,
    )


def save_roi_json(path: Path, base_size: Tuple[int, int], polygon: Polygon, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_size": [int(base_size[0]), int(base_size[1])],
        "polygon": [[int(x), int(y)] for x, y in polygon],
        "meta": meta,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


__all__ = [
    "AutoCVParams",
    "AutoCVResult",
    "DetectedLine",
    "estimate_roi",
    "save_roi_json",
]
