"""Utility helpers for secondary plate cropping."""
from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def ensure_xyxy_int(xyxy: Iterable[float]) -> Tuple[int, int, int, int]:
    """Return a clamped integer xyxy tuple from an iterable of coordinates."""

    values = list(xyxy)
    if len(values) != 4:
        raise ValueError("xyxy iterable must contain exactly four values")
    x1, y1, x2, y2 = [int(round(float(v))) for v in values]
    if x2 <= x1:
        x2 = x1 + 1
    if y2 <= y1:
        y2 = y1 + 1
    return x1, y1, x2, y2


def expand_with_margin(
    bbox: Tuple[int, int, int, int],
    margin: float,
    width: int,
    height: int,
) -> Tuple[int, int, int, int]:
    """Expand a bounding box by the given relative margin and clamp to image bounds."""

    x1, y1, x2, y2 = bbox
    box_w = max(x2 - x1, 1)
    box_h = max(y2 - y1, 1)
    pad_x = int(round(box_w * max(margin, 0.0)))
    pad_y = int(round(box_h * max(margin, 0.0)))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width, x2 + pad_x)
    y2 = min(height, y2 + pad_y)
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def crop_by_xyxy(
    image: np.ndarray, bbox: Tuple[int, int, int, int]
) -> np.ndarray:
    """Return the crop specified by bbox; empty array if invalid."""

    x1, y1, x2, y2 = bbox
    x1 = max(int(x1), 0)
    y1 = max(int(y1), 0)
    x2 = max(int(x2), x1 + 1)
    y2 = max(int(y2), y1 + 1)
    h, w = image.shape[:2]
    x2 = min(x2, w)
    y2 = min(y2, h)
    if x1 >= x2 or y1 >= y2:
        return np.empty((0, 0, 3), dtype=image.dtype)
    return image[y1:y2, x1:x2].copy()


__all__ = ["ensure_xyxy_int", "expand_with_margin", "crop_by_xyxy"]
