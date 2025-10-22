"""Video IO helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2


@dataclass
class VideoMetadata:
    path: Path
    fps: float
    frame_count: int
    frame_size: Tuple[int, int]


def probe_video(path: Path | str) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video source: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    return VideoMetadata(path=Path(path), fps=fps, frame_count=frame_count, frame_size=(width, height))

