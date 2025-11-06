"""ROI management utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Sequence

from src.utils.geometry import point_in_polygon, scale_polygon

Point = Tuple[float, float]


@dataclass
class ROIConfig:
    base_width: Optional[float]
    base_height: Optional[float]
    polygon: List[Point]


class ROIManager:
    """Load and scale ROI polygons for different frame sizes."""

    def __init__(self, roi_path: Path | str):
        self.roi_path = Path(roi_path)
        self._config: Optional[ROIConfig] = None
        self._scaled_polygon: Optional[List[Point]] = None

    def load(self) -> None:
        data = json.loads(self.roi_path.read_text(encoding="utf-8"))
        polygon: Optional[Sequence[Sequence[float]]] = data.get("polygon")
        if not polygon or len(polygon) < 3:
            raise ValueError("ROI JSON must define polygon with >= 3 points")
        points = [(float(pt[0]), float(pt[1])) for pt in polygon]

        base_size = data.get("base_size") or data.get("baseSize")
        width = height = None
        if isinstance(base_size, (list, tuple)) and len(base_size) == 2:
            try:
                width = float(base_size[0])
                height = float(base_size[1])
            except (TypeError, ValueError):
                width = height = None
        if width is None or height is None:
            width = data.get("base_width") or data.get("width") or data.get("frame_width")
            height = data.get("base_height") or data.get("height") or data.get("frame_height")
            try:
                width = float(width) if width is not None else None
                height = float(height) if height is not None else None
            except (TypeError, ValueError):
                width = height = None
        self._config = ROIConfig(
            base_width=width,
            base_height=height,
            polygon=points,
        )
        self._scaled_polygon = None

    def scale_to_frame(self, frame_size: Tuple[int, int]) -> None:
        if not self._config:
            raise RuntimeError("ROI configuration has not been loaded")
        width, height = frame_size
        base_w = self._config.base_width
        base_h = self._config.base_height
        if base_w and base_h and base_w > 0 and base_h > 0:
            sx = width / base_w
            sy = height / base_h
            self._scaled_polygon = scale_polygon(self._config.polygon, sx, sy)
        else:
            self._scaled_polygon = list(self._config.polygon)

    @property
    def polygon(self) -> List[Point]:
        if self._scaled_polygon is not None:
            return self._scaled_polygon
        if not self._config:
            raise RuntimeError("ROI configuration has not been loaded")
        return self._config.polygon

    def point_in_roi(self, point: Point) -> bool:
        return point_in_polygon(point, self.polygon)

    def ensure_ready(self, frame_size: Tuple[int, int]) -> None:
        if self._config is None:
            self.load()
        if self._scaled_polygon is None:
            self.scale_to_frame(frame_size)

