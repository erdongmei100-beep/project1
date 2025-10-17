"""Plate crop quality utilities."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np


def laplacian_var(img: np.ndarray) -> float:
    """Compute Laplacian variance for an image (BGR or grayscale)."""
    if img.size == 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return float(laplacian.var())


def laplacian_sharpness(bgr_img: np.ndarray) -> float:
    """Compute Laplacian variance as a focus measure."""
    return laplacian_var(bgr_img)


def bbox_area(xyxy: Sequence[float]) -> float:
    x1, y1, x2, y2 = xyxy
    return max(float(x2) - float(x1), 0.0) * max(float(y2) - float(y1), 0.0)


def relative_area(xyxy: Sequence[float], parent_xyxy: Sequence[float]) -> float:
    parent = bbox_area(parent_xyxy)
    if parent <= 0.0:
        return 0.0
    return bbox_area(xyxy) / parent


def angle_penalty(_xyxy: Sequence[float]) -> float:
    return 1.0


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(value, max_value)))


def score_plate(det_conf: float, rel_area: float, sharpness: float, angle_factor: float) -> float:
    conf = max(float(det_conf), 0.0)
    area_term = _clamp(rel_area, 0.0, 0.2)
    sharp_term = _clamp(sharpness / 300.0, 0.0, 1.0)
    return conf * area_term * sharp_term * float(angle_factor)
