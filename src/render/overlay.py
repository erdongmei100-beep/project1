"""Rendering helpers for occupancy visualization."""
from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np

Point = Tuple[float, float]


def draw_overlays(
    frame,
    roi_polygon: Sequence[Point] | None,
    track_records: Iterable[dict],
    show_tracks: bool = True,
    show_footpoints: bool = True,
):
    """Draw ROI polygon, tracked boxes, and footpoints onto the frame."""
    canvas = frame
    if roi_polygon:
        pts = np.array([[int(x), int(y)] for x, y in roi_polygon], dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=(0, 215, 255), thickness=2)
        overlay = canvas.copy()
        cv2.fillPoly(overlay, [pts], color=(0, 215, 255))
        cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, dst=canvas)

    if not show_tracks:
        return canvas

    for record in track_records:
        bbox = record.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = map(int, bbox)
        inside = bool(record.get("inside"))
        track_id = record.get("track_id", "?")
        color = (0, 0, 255) if inside else (46, 204, 113)
        thickness = 3 if inside else 1
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)
        label_suffix = " VIOLATION" if inside else ""
        label = f"ID {track_id}{label_suffix}"
        cv2.putText(canvas, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if show_footpoints and record.get("footpoint"):
            fx, fy = record["footpoint"]
            cv2.circle(canvas, (int(fx), int(fy)), 4, color, -1)
    return canvas

