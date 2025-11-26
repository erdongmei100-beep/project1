"""ROI management utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.geometry import point_in_polygon, scale_polygon

Point = Tuple[float, float]


@dataclass
class ROIConfig:
    base_width: float
    base_height: float
    polygon: List[Point]


@dataclass
class ROIStatus:
    polygon: List[Point]
    valid: bool
    reason: str = ""


class ROIManager:
    """Load/track ROI polygons and validate segmentation masks."""

    def __init__(
        self,
        roi_path: Path | str | None = None,
        *,
        min_area_ratio: float = 0.005,
        max_area_ratio: float = 0.4,
        expected_centroid: Tuple[float, float] = (0.55, 0.55),
        stability_ratio: float = 0.25,
    ) -> None:
        self.roi_path = Path(roi_path) if roi_path is not None else None
        self.min_area_ratio = min_area_ratio
        self.max_area_ratio = max_area_ratio
        self.expected_centroid = expected_centroid
        self.stability_ratio = stability_ratio

        self._config: Optional[ROIConfig] = None
        self._scaled_polygon: Optional[List[Point]] = None
        self._current_polygon: List[Point] = []
        self._last_valid_centroid: Optional[Tuple[float, float]] = None
        self._last_status: Optional[ROIStatus] = None

    def load(self) -> None:
        if self.roi_path is None:
            raise RuntimeError("roi_path is not configured for ROIManager")
        data = json.loads(self.roi_path.read_text(encoding="utf-8"))
        base_size = data.get("base_size") or data.get("baseSize")
        if not base_size or len(base_size) != 2:
            raise ValueError("ROI JSON must define base_size as [width, height]")
        polygon = data.get("polygon")
        if not polygon or len(polygon) < 3:
            raise ValueError("ROI JSON must define polygon with >= 3 points")
        points = [(float(x), float(y)) for x, y in polygon]
        self._config = ROIConfig(
            base_width=float(base_size[0]),
            base_height=float(base_size[1]),
            polygon=points,
        )
        self._scaled_polygon = None

    def scale_to_frame(self, frame_size: Tuple[int, int]) -> None:
        if not self._config:
            raise RuntimeError("ROI configuration has not been loaded")
        width, height = frame_size
        sx = width / self._config.base_width
        sy = height / self._config.base_height
        self._scaled_polygon = scale_polygon(self._config.polygon, sx, sy)

    @property
    def polygon(self) -> List[Point]:
        if self._current_polygon:
            return self._current_polygon
        if self._scaled_polygon is not None:
            return self._scaled_polygon
        if not self._config:
            raise RuntimeError("ROI configuration has not been loaded")
        return self._config.polygon

    def point_in_roi(self, point: Point) -> bool:
        polygon = self._current_polygon
        if not polygon:
            polygon = self._scaled_polygon or (self._config.polygon if self._config else [])
        if not polygon:
            return False
        if self._current_polygon and self._last_status and not self._last_status.valid:
            return False
        return point_in_polygon(point, polygon)

    def ensure_ready(self, frame_size: Tuple[int, int]) -> None:
        if self._config is None:
            self.load()
        if self._scaled_polygon is None:
            self.scale_to_frame(frame_size)

    def _mask_to_polygon(self, mask: np.ndarray) -> List[Point]:
        mask_uint8 = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return []
        largest = max(contours, key=cv2.contourArea)
        pts = largest.reshape(-1, 2)
        return [(float(x), float(y)) for x, y in pts]

    def _centroid(self, polygon: List[Point]) -> Optional[Tuple[float, float]]:
        if not polygon:
            return None
        arr = np.array(polygon, dtype=np.float32)
        m = cv2.moments(arr)
        if m["m00"] == 0:
            return None
        return (float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"]))

    def _ego_covered(self, mask: np.ndarray, frame_size: Tuple[int, int]) -> bool:
        width, height = frame_size
        x = int(width * 0.5)
        y = int(height * 0.95)
        if x < 0 or y < 0 or y >= mask.shape[0] or x >= mask.shape[1]:
            return False
        return bool(mask[y, x])

    def update_from_segmentation(
        self, mask: Optional[np.ndarray], frame_size: Tuple[int, int]
    ) -> ROIStatus:
        width, height = frame_size
        reason = ""
        valid = False
        polygon: List[Point] = []

        if mask is None:
            reason = "no_mask"
        else:
            if mask.shape[:2] != (height, width):
                mask = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST)
                mask = mask.astype(bool)

            polygon = self._mask_to_polygon(mask)
            if not polygon:
                reason = "no_contour"
            else:
                area = float(cv2.contourArea(np.array(polygon, dtype=np.float32)))
                area_ratio = area / float(width * height)
                if area_ratio < self.min_area_ratio or area_ratio > self.max_area_ratio:
                    reason = "area_out_of_range"
                else:
                    centroid = self._centroid(polygon)
                    if centroid is None:
                        reason = "no_centroid"
                    else:
                        cx, cy = centroid
                        exp_x = width * self.expected_centroid[0]
                        exp_y = height * self.expected_centroid[1]
                        if cx < exp_x or cy < exp_y:
                            reason = "centroid_out_of_position"
                        elif self._ego_covered(mask, frame_size):
                            reason = "ego_vehicle"
                        else:
                            if self._last_valid_centroid is not None:
                                dx = cx - self._last_valid_centroid[0]
                                dy = cy - self._last_valid_centroid[1]
                                dist = float(np.hypot(dx, dy))
                                diag = float(np.hypot(width, height))
                                if dist > diag * self.stability_ratio:
                                    reason = "unstable_centroid"
                            if not reason:
                                valid = True
                                reason = "ok"
                                self._last_valid_centroid = centroid

        self._current_polygon = polygon
        status = ROIStatus(polygon=polygon, valid=valid, reason=reason)
        self._last_status = status
        return status

    @property
    def status(self) -> ROIStatus:
        if self._last_status is not None:
            return self._last_status
        return ROIStatus(polygon=[], valid=False, reason="uninitialized")

