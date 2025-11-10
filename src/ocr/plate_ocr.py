"""RapidOCR wrapper for plate text recognition."""
from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

try:  # pragma: no cover - runtime dependency
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    RapidOCR = None
    _IMPORT_ERROR = repr(exc)
else:
    _IMPORT_ERROR = ""


class PlateOCR:
    """Perform OCR on cropped license plate images using RapidOCR."""

    def __init__(self, min_conf: float = 0.15) -> None:
        if RapidOCR is None:
            raise RuntimeError(f"RapidOCR 未安装或初始化失败: {_IMPORT_ERROR}")
        self.ocr = RapidOCR()
        self.min_conf = float(min_conf)
        print("[plate] Initialized RapidOCR engine")

    def read(self, img_bgr: np.ndarray | None) -> Tuple[str, float]:
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return "null", 0.0

        h, w = img_bgr.shape[:2]
        if max(h, w) < 200:
            new_w = max(int(w * 2), 1)
            new_h = max(int(h * 2), 1)
            img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        try:
            result, _ = self.ocr(img)
        except Exception:
            return "null", 0.0

        if not result:
            return "null", 0.0

        text = str(result[0][0]).strip()
        try:
            conf = float(result[0][1])
        except Exception:
            conf = 0.0

        if not text or conf < self.min_conf:
            return "null", conf
        return text, conf


__all__ = ["PlateOCR"]
