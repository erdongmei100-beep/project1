"""Secondary plate detection for fine cropping."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np

try:  # pragma: no cover - runtime dependency
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - optional dependency guard
    YOLO = None


class PlateDetector:
    """Run a lightweight YOLO model to obtain a tight crop of the plate."""

    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        imgsz: int = 640,
        margin: float = 0.25,
        device: str = "cpu",
    ) -> None:
        self.weights = Path(weights)
        self.conf = float(conf)
        self.imgsz = int(imgsz)
        self.margin = float(margin)
        self.device = str(device or "cpu")
        self._model: Optional[YOLO] = None
        self._load_model()

    def _load_model(self) -> None:
        if YOLO is None:
            print("[plate] Ultralytics package unavailable; fine plate detector disabled.")
            return
        if not self.weights.exists():
            print(f"[plate] Fine detector weights missing: {self.weights}")
            return
        try:
            self._model = YOLO(str(self.weights))
            print(f"[plate] Loaded fine plate detector weights: {self.weights}")
        except Exception as exc:
            print(f"[plate] Failed to initialize fine plate detector: {exc}")
            self._model = None

    def _clip(self, x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
        dx = int((x2 - x1) * self.margin)
        dy = int((y2 - y1) * self.margin)
        x1 = max(0, x1 - dx)
        x2 = min(width, x2 + dx)
        y1 = max(0, y1 - dy)
        y2 = min(height, y2 + dy)
        if x2 <= x1:
            x2 = min(width, x1 + 1)
        if y2 <= y1:
            y2 = min(height, y1 + 1)
        return x1, y1, x2, y2

    def fine_crop(self, img_bgr: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, int, int, int], float]]:
        if self._model is None:
            return None
        if img_bgr is None or img_bgr.size == 0:
            return None
        height, width = img_bgr.shape[:2]
        try:
            result = self._model.predict(
                img_bgr,
                imgsz=self.imgsz,
                conf=self.conf,
                verbose=False,
                device=self.device,
            )[0]
        except Exception as exc:
            print(f"[plate] Fine crop prediction failed: {exc}")
            return None
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return None
        confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
        if confs.size == 0 or xyxy.size == 0:
            return None
        best_idx = int(np.argmax(confs))
        x1, y1, x2, y2 = [int(v) for v in xyxy[best_idx]]
        x1, y1, x2, y2 = self._clip(x1, y1, x2, y2, width, height)
        crop = img_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        det_conf = float(confs[best_idx])
        return crop, (x1, y1, x2, y2), det_conf


__all__ = ["PlateDetector"]
