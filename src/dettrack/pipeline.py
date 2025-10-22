"""Detection and tracking utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from ultralytics import YOLO


class DetectorTracker:
    """Thin wrapper around Ultralytics YOLO tracking interface."""

    def __init__(
        self,
        weights: Path | str,
        device: str = "cpu",
        imgsz: int = 640,
        tracker_cfg: Path | str | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        max_det: int = 1000,
    ) -> None:
        self.model = YOLO(str(weights))
        self.args: Dict[str, object] = {
            "device": device,
            "imgsz": imgsz,
            "conf": conf,
            "iou": iou,
            "max_det": max_det,
            "stream": True,
            "verbose": False,
        }
        if tracker_cfg is not None:
            self.args["tracker"] = str(tracker_cfg)

    def track(self, source: Path | str) -> Iterator:
        return self.model.track(source=str(source), **self.args)

