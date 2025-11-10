"""Utilities for reproducible plate cropping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np


class PlateCropError(RuntimeError):
    """Raised when fine plate cropping fails."""


@dataclass
class CropDebugInfo:
    frame_path: Path
    frame_bbox: Tuple[int, int, int, int]
    final_bbox: Tuple[int, int, int, int]
    margin_bbox: Tuple[int, int, int, int]


def _ensure_path(path: Union[str, Path]) -> Path:
    path_obj = Path(path)
    if not path_obj.exists():
        raise PlateCropError(f"frame not found: {path_obj}")
    return path_obj


def _to_int_box(coords: Sequence[Union[int, float]]) -> Tuple[int, int, int, int]:
    if len(coords) != 4:
        raise PlateCropError(f"bbox must have 4 values, got {coords}")
    try:
        x1, y1, x2, y2 = [int(round(float(v))) for v in coords]
    except Exception as exc:  # pragma: no cover - defensive
        raise PlateCropError(f"invalid bbox values: {coords}") from exc
    if x2 <= x1 or y2 <= y1:
        raise PlateCropError(f"bbox has non-positive area: {coords}")
    return x1, y1, x2, y2


def _clip_box(box: Tuple[int, int, int, int], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    width = int(width)
    height = int(height)
    if width <= 1 or height <= 1:
        raise PlateCropError("frame has invalid size")
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return x1, y1, x2, y2


def _expand_box(
    box: Tuple[int, int, int, int],
    margin: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    margin = max(float(margin), 0.0)
    dx = int(round((x2 - x1) * margin))
    dy = int(round((y2 - y1) * margin))
    expanded = (x1 - dx, y1 - dy, x2 + dx, y2 + dy)
    return _clip_box(expanded, width, height)


def _ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    raise PlateCropError("unsupported image shape for crop")


def _draw_debug(debug: CropDebugInfo, output_path: Path) -> None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    frame = cv2.imread(str(debug.frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        return
    overlay = frame.copy()
    x1, y1, x2, y2 = debug.margin_bbox
    cv2.rectangle(overlay, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 255), 2)
    fx1, fy1, fx2, fy2 = debug.final_bbox
    cv2.rectangle(overlay, (fx1, fy1), (fx2 - 1, fy2 - 1), (0, 128, 255), 2)
    alpha = 0.35
    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    cv2.imwrite(str(output_path), frame)


def crop_tail_and_plate(
    frame_path: Union[str, Path],
    bbox_xyxy: Sequence[Union[int, float]],
    *,
    margin: float = 0.12,
    out_tail_path: Union[str, Path],
    out_plate_path: Union[str, Path],
    redetect: bool = False,
    plate_model: Optional[object] = None,
    min_h: int = 48,
    aspect_range: Tuple[float, float] = (2.0, 5.0),
    debug_draw: bool = False,
    debug_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Crop the vehicle tail and a tight plate patch from a frame.

    Parameters follow the project specification. All coordinates are treated as
    pixel-space xyxy boxes relative to the original frame.
    """

    frame_path = _ensure_path(frame_path)
    frame = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
    if frame is None:
        raise PlateCropError(f"failed to read frame: {frame_path}")

    height, width = frame.shape[:2]
    base_box = _clip_box(_to_int_box(bbox_xyxy), width, height)

    tail_box = _expand_box(base_box, 0.05, width, height)
    tail_img = _ensure_color(frame[tail_box[1] : tail_box[3], tail_box[0] : tail_box[2]])
    if tail_img.size == 0:
        raise PlateCropError("tail crop has zero size")

    out_tail_path = Path(out_tail_path)
    out_tail_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_tail_path), tail_img):
        raise PlateCropError(f"failed to save tail crop: {out_tail_path}")

    if redetect:
        if plate_model is None:
            raise PlateCropError("redetect=True requires plate_model")
        try:
            detect_result = plate_model.fine_crop(tail_img)
        except Exception as exc:  # pragma: no cover - detector failures
            raise PlateCropError(f"plate re-detect failed: {exc}") from exc
        if not detect_result:
            raise PlateCropError("plate re-detect returned empty result")
        plate_candidate, det_bbox, _ = detect_result
        if plate_candidate is None or plate_candidate.size == 0:
            raise PlateCropError("plate re-detect produced empty crop")
        local_box = _clip_box(_to_int_box(det_bbox), tail_img.shape[1], tail_img.shape[0])
        margin_box = _expand_box(local_box, margin, tail_img.shape[1], tail_img.shape[0])
        plate_img = _ensure_color(
            tail_img[margin_box[1] : margin_box[3], margin_box[0] : margin_box[2]]
        )
        final_box = (
            tail_box[0] + margin_box[0],
            tail_box[1] + margin_box[1],
            tail_box[0] + margin_box[2],
            tail_box[1] + margin_box[3],
        )
        local_xyxy = [int(v) for v in margin_box]
    else:
        margin_box = _expand_box(base_box, margin, width, height)
        plate_img = _ensure_color(
            frame[margin_box[1] : margin_box[3], margin_box[0] : margin_box[2]]
        )
        final_box = margin_box
        local_xyxy = [int(v) for v in margin_box]

    plate_h, plate_w = plate_img.shape[:2]
    if plate_h <= 0 or plate_w <= 0:
        raise PlateCropError("plate crop has zero size")
    if plate_h < int(min_h):
        raise PlateCropError(f"plate crop height {plate_h} < min_h {min_h}")

    aspect = plate_w / float(plate_h)
    min_aspect, max_aspect = aspect_range
    if aspect < min_aspect or aspect > max_aspect:
        raise PlateCropError(
            f"plate aspect {aspect:.2f} outside [{min_aspect}, {max_aspect}]"
        )

    out_plate_path = Path(out_plate_path)
    out_plate_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_plate_path), plate_img):
        raise PlateCropError(f"failed to save plate crop: {out_plate_path}")

    if debug_draw:
        debug_base = Path(debug_dir) if debug_dir else out_plate_path.parent / "_debug_crops"
        debug_path = debug_base / f"{out_plate_path.stem}_viz.jpg"
        debug_info = CropDebugInfo(
            frame_path=frame_path,
            frame_bbox=base_box,
            final_bbox=final_box,
            margin_bbox=margin_box,
        )
        _draw_debug(debug_info, debug_path)

    return {
        "tail_path": str(out_tail_path),
        "plate_path": str(out_plate_path),
        "plate_xyxy_local": local_xyxy,
        "w": int(plate_w),
        "h": int(plate_h),
        "frame_box": [int(v) for v in base_box],
    }


__all__ = ["PlateCropError", "crop_tail_and_plate"]
