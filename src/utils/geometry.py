"""Geometry utilities for ROI operations."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

Point = Tuple[float, float]
Polygon = Sequence[Point]


def scale_polygon(polygon: Polygon, scale_x: float, scale_y: float) -> list[Point]:
    """Scale polygon coordinates by the provided factors."""
    return [(x * scale_x, y * scale_y) for x, y in polygon]


def point_in_polygon(point: Point, polygon: Polygon) -> bool:
    """Return True if point lies inside polygon using ray casting."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False

    x0, y0 = polygon[0]
    for i in range(1, n + 1):
        x1, y1 = polygon[i % n]
        intersects = ((y0 > y) != (y1 > y)) and (
            x < (x1 - x0) * (y - y0) / (y1 - y0 + 1e-12) + x0
        )
        if intersects:
            inside = not inside
        x0, y0 = x1, y1
    return inside


def polygon_bounds(polygon: Polygon) -> Tuple[float, float, float, float]:
    """Return minx, miny, maxx, maxy."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)

