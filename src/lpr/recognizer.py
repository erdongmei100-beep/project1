from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VALID_PROVINCES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领")


def calculate_blur_score(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_valid_plate_format(text: str) -> bool:
    if not text:
        return False
    if not (7 <= len(text) <= 8):
        return False
    if text[0] not in VALID_PROVINCES:
        return False
    return True


@dataclass
class LPRResult:
    plate_text: str
    plate_score: float
    plate_bbox: Optional[Tuple[int, int, int, int]]
    status: str


class HyperLPR3Recognizer:
    def __init__(
        self,
        *,
        backend: str = "onnxruntime-gpu",
        model_dir: Optional[str] = None,
        quality_filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.backend = backend
        self.model_dir = model_dir
        quality_cfg = quality_filter or {}
        self.quality_enable = bool(quality_cfg.get("enable", True))
        self.blur_threshold = float(quality_cfg.get("blur_var_threshold", 60))
        self.format_check = bool(quality_cfg.get("format_check", True))
        self._catcher = None
        self._available = False
        self._init_model()

    def _init_model(self) -> None:
        module_spec = importlib.util.find_spec("hyperlpr3")
        if module_spec is None:
            logger.error("hyperlpr3 is not installed; skip LPR initialization")
            return
        lpr3 = importlib.import_module("hyperlpr3")  # type: ignore

        kwargs: Dict[str, Any] = {"detect_level": getattr(lpr3, "DETECT_LEVEL_LOW", 0)}
        if self.backend:
            kwargs["backend_name"] = self.backend
        if self.model_dir:
            kwargs["model_path"] = str(self.model_dir)
        try:
            self._catcher = lpr3.LicensePlateCatcher(**kwargs)
            self._available = True
        except Exception as exc:  # pragma: no cover - runtime dependency
            logger.error("Failed to initialize hyperlpr3 catcher: %s", exc)
            self._catcher = None
            self._available = False

    def recognize(self, image_path: str) -> LPRResult:
        path = Path(image_path)
        if not path.exists():
            return LPRResult("", 0.0, None, "missing_file")

        if not self._available or self._catcher is None:
            return LPRResult("", 0.0, None, "fail")

        try:
            image = cv2.imread(str(path))
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("Failed to read image %s: %s", path, exc)
            return LPRResult("", 0.0, None, "fail")

        if image is None:
            return LPRResult("", 0.0, None, "fail")

        try:
            results = self._catcher(image)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("HyperLPR inference error on %s: %s", path, exc)
            return LPRResult("", 0.0, None, "fail")

        if not results:
            return LPRResult("", 0.0, None, "empty")

        best = max(results, key=lambda r: r[1])
        plate_text = best[0]
        plate_score = float(best[1])
        bbox = tuple(map(int, best[3])) if len(best) > 3 else None

        if self.format_check and not is_valid_plate_format(plate_text):
            return LPRResult("", 0.0, bbox, "fail")

        if bbox is not None and self.quality_enable:
            x1, y1, x2, y2 = bbox
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            crop = image[y1:y2, x1:x2]
            blur_score = calculate_blur_score(crop)
            if blur_score < self.blur_threshold:
                logger.warning("Blur score %.2f below threshold %.2f for %s", blur_score, self.blur_threshold, path)
                return LPRResult("", 0.0, bbox, "bad_quality")

            province_width = max(1, int(crop.shape[1] * 0.2))
            province_crop = crop[:, :province_width]
            province_score = calculate_blur_score(province_crop)
            if province_score < self.blur_threshold:
                logger.warning(
                    "Province blur score %.2f below threshold %.2f for %s", province_score, self.blur_threshold, path
                )
                return LPRResult("", 0.0, bbox, "bad_quality")

        return LPRResult(plate_text, plate_score, bbox, "ok")
