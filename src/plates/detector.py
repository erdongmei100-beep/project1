"""YOLO-based plate detector wrapper."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

try:  # pragma: no cover - optional torch check
    import torch
except Exception:  # pragma: no cover - torch may be absent
    torch = None

try:  # pragma: no cover - optional dependency handling
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover - runtime import guard
    YOLO = None


class PlateDetector:
    """Lightweight wrapper around an Ultralytics YOLO model for plate detection."""

    def __init__(
        self,
        weights: str,
        device: str = "cpu",
        imgsz: int = 320,
        conf: float = 0.25,
        iou: float = 0.45,
    ) -> None:
        self.weights = Path(weights)
        self.device = self._resolve_device(device)
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self._model: YOLO | None = None
        self._load_model()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def _resolve_device(self, device: str) -> str:
        requested = (device or "cpu").strip()
        if not requested:
            requested = "cpu"
        if requested.startswith("cuda") and torch is not None:
            if not torch.cuda.is_available():  # pragma: no cover - runtime environment dependent
                print("CUDA not available for plate detector; falling back to CPU.")
                return "cpu"
        if requested.startswith("cuda") and torch is None:
            print("Torch not available; plate detector using CPU.")
            return "cpu"
        return requested

    def _load_model(self) -> None:
        if YOLO is None:
            print(
                "Ultralytics package not available; plate detector disabled."
            )
            return
        if not self.weights.exists():
            print(f"Plate detector weights not found: {self.weights}")
            return
        try:
            self._model = YOLO(str(self.weights))
        except Exception as exc:  # pragma: no cover - runtime dependent
            print(f"Failed to load plate detector weights: {exc}")
            self._model = None

    def detect(self, bgr_img: np.ndarray) -> List[Dict[str, object]]:
        """Run inference on the provided BGR crop and return detections."""
        if self._model is None:
            return []
        try:
            results = self._model.predict(
                source=bgr_img,
                imgsz=self.imgsz,
                conf=self.conf,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - runtime dependent
            print(f"Plate detection failed: {exc}")
            return []

        detections: List[Dict[str, object]] = []
        if not results:
            return detections
        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return detections

        xyxy = boxes.xyxy
        conf = boxes.conf
        if hasattr(xyxy, "cpu"):
            xyxy = xyxy.cpu().numpy()
        if hasattr(conf, "cpu"):
            conf = conf.cpu().numpy()

        xyxy_array = np.asarray(xyxy)
        conf_array = np.asarray(conf)
        for coords, score in zip(xyxy_array, conf_array):
            detections.append({"xyxy": coords.tolist(), "conf": float(score)})
        return detections

