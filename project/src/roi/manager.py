"""ROI management utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from src.utils.geometry import point_in_polygon, scale_polygon

Point = Tuple[float, float]


@dataclass
class ROIConfig:
    base_width: float
    base_height: float
    polygon: List[Point]


class ROIManager:
    """Load and scale ROI polygons for different frame sizes."""

    def __init__(self, roi_path: Path | str):
        self.roi_path = Path(roi_path)
        self._config: Optional[ROIConfig] = None
        self._scaled_polygon: Optional[List[Point]] = None

    def load(self) -> None:
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

