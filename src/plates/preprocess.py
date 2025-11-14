"""Plate crop enhancement pipeline."""
from __future__ import annotations

from typing import Any, Dict, Optional

import cv2
import numpy as np

from .quality import laplacian_var


_AR_PLATE = 440.0 / 140.0  # approximate width / height aspect ratio for CN plates


def _ensure_bgr(img: np.ndarray) -> np.ndarray:
    if img is None:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img.copy()


def _upscale(img: np.ndarray, target_min_side: int) -> np.ndarray:
    if img.size == 0:
        return img
    h, w = img.shape[:2]
    min_side = max(min(h, w), 1)
    scale = float(target_min_side) / float(min_side)
    if scale <= 1.0:
        return img
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)


def _deskew(img: np.ndarray, max_deg: float) -> np.ndarray:
    if img.size == 0:
        return img
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    ys, xs = np.nonzero(edges)
    if xs.size < 20:
        return img
    pts = np.column_stack((xs.astype(np.float32), ys.astype(np.float32)))
    rect = cv2.minAreaRect(pts)
    angle = rect[-1]
    if angle < -45.0:
        angle += 90.0
    if abs(angle) < 1e-3 or abs(angle) > max_deg:
        return img
    center = (img.shape[1] / 2.0, img.shape[0] / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    h, w = img.shape[:2]
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(
        img,
        matrix,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def _rectify(
    img: np.ndarray, quad_pts: Optional[np.ndarray], target_min_side: int, enable: bool
) -> np.ndarray:
    if not enable or quad_pts is None:
        return img
    pts = np.asarray(quad_pts, dtype=np.float32)
    if pts.shape != (4, 2):
        return img
    width_target = max(int(round(target_min_side * 1.5)), 2)
    width_target += width_target % 2
    height_target = max(int(round(width_target / _AR_PLATE)), 2)
    dst = np.array(
        [
            [0.0, 0.0],
            [width_target - 1.0, 0.0],
            [width_target - 1.0, height_target - 1.0],
            [0.0, height_target - 1.0],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(pts, dst)
    return cv2.warpPerspective(
        img,
        matrix,
        (width_target, height_target),
        flags=cv2.INTER_CUBIC,
    )


def _apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    clip = max(float(clip), 0.1)
    grid = max(int(grid), 1)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def _apply_denoise(img: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "none").lower()
    if mode == "bilateral":
        return cv2.bilateralFilter(img, d=5, sigmaColor=75, sigmaSpace=75)
    if mode == "nlmeans":
        return cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
    return img


def _unsharp(img: np.ndarray, amount: float, radius: int) -> np.ndarray:
    amount = max(float(amount), 0.0)
    radius = max(int(radius), 0)
    if amount == 0.0:
        return img
    ksize = radius * 2 + 1
    if ksize <= 1:
        ksize = 3
    blur = cv2.GaussianBlur(img, (ksize, ksize), 0)
    return cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)


def _binarize(gray: np.ndarray) -> np.ndarray:
    block_size = max(int(round(min(gray.shape[:2]) / 8) * 2 + 1), 3)
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    return cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        5,
    )


def _fallback_pipeline(bgr: np.ndarray) -> Dict[str, Any]:
    stages: Dict[str, np.ndarray] = {"raw": bgr.copy()}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    stages["gray"] = gray
    clahe = _apply_clahe(gray, 2.0, 8)
    stages["clahe"] = clahe
    enhanced = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    sharpened = _unsharp(enhanced, 0.5, 3)
    stages["denoise"] = enhanced
    stages["sharp"] = sharpened
    stages["deskew"] = bgr.copy()
    stages["rectified"] = bgr.copy()
    stages["binary"] = cv2.cvtColor(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    final = sharpened
    sharpness = laplacian_var(final)
    return {"final": final, "stages": stages, "sharpness": sharpness}


def enhance_plate(
    bgr_img: np.ndarray, cfg: Dict[str, Any], quad_pts: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    cfg = dict(cfg or {})
    base = _ensure_bgr(bgr_img)
    stages: Dict[str, np.ndarray] = {"raw": base.copy()}

    if not cfg.get("enable", True):
        sharpness = laplacian_var(base)
        stages["deskew"] = base.copy()
        stages["rectified"] = base.copy()
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        stages["gray"] = gray
        clahe = _apply_clahe(gray, cfg.get("clahe_clip", 2.0), cfg.get("clahe_grid", 8))
        stages["clahe"] = clahe
        bgr_clahe = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
        stages["denoise"] = bgr_clahe
        stages["sharp"] = base.copy()
        stages["binary"] = cv2.cvtColor(cv2.cvtColor(base, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        return {"final": base, "stages": stages, "sharpness": sharpness}

    try:
        upscale_side = int(cfg.get("upscale_min_side", 320))
        deskew_deg = float(cfg.get("deskew_max_deg", 12.0))
        use_perspective = bool(cfg.get("use_perspective", True))
        clahe_clip = cfg.get("clahe_clip", 2.0)
        clahe_grid = cfg.get("clahe_grid", 8)
        denoise_mode = cfg.get("denoise", "bilateral")
        unsharp_amount = cfg.get("unsharp_amount", 0.6)
        unsharp_radius = cfg.get("unsharp_radius", 3)
        do_binarize = bool(cfg.get("binarize", False))

        working = _upscale(base, upscale_side)
        deskewed = _deskew(working, deskew_deg)
        stages["deskew"] = deskewed.copy()
        rectified = _rectify(deskewed, quad_pts, upscale_side, use_perspective)
        stages["rectified"] = rectified.copy()

        gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
        stages["gray"] = gray
        clahe = _apply_clahe(gray, clahe_clip, clahe_grid)
        stages["clahe"] = clahe
        clahe_bgr = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)

        denoised = _apply_denoise(clahe_bgr, denoise_mode)
        stages["denoise"] = denoised.copy()

        sharpened = _unsharp(denoised, unsharp_amount, int(unsharp_radius))
        stages["sharp"] = sharpened.copy()

        final_img = sharpened
        if do_binarize:
            binary = _binarize(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY))
            stages["binary"] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            final_img = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        else:
            stages["binary"] = cv2.cvtColor(cv2.cvtColor(sharpened, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)

        sharpness = laplacian_var(final_img)
        return {"final": final_img, "stages": stages, "sharpness": sharpness}
    except Exception:
        return _fallback_pipeline(base)
