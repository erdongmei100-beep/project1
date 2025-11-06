from __future__ import annotations

from typing import Generator, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(
        self,
        weights: str,
        imgsz: int = 1280,
        use_sahi: bool = True,
        tile_size: int = 1024,
        tile_overlap: float = 0.2,
        device: str = "",
        conf: float = 0.25,
    ):
        self.model = YOLO(weights)
        self.model.to(device if device else "cpu")
        self.imgsz = imgsz
        self.use_sahi = use_sahi
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.conf = float(conf)

    def _tile_positions(self, length: int) -> List[int]:
        """Generate tile start indices ensuring coverage of the tail region."""

        ts = self.tile_size
        if length <= ts:
            return [0]

        step = int(ts * (1 - self.tile_overlap))
        step = max(1, step)

        positions = list(range(0, max(1, length - ts + 1), step))
        tail = max(0, length - ts)
        if not positions or positions[-1] != tail:
            positions.append(tail)
        return sorted(set(positions))

    def _tiles(self, img: np.ndarray) -> Generator[Tuple[Tuple[int, int], np.ndarray], None, None]:
        h, w = img.shape[:2]
        for y in self._tile_positions(h):
            for x in self._tile_positions(w):
                tile = img[y : y + self.tile_size, x : x + self.tile_size]
                if tile.size == 0:
                    continue
                yield (x, y), tile

    def detect(self, frame_bgr: np.ndarray) -> List[List[float]]:
        H, W = frame_bgr.shape[:2]
        results = []

        if self.use_sahi and max(H, W) > self.imgsz:
            for (x0, y0), tile in self._tiles(frame_bgr):
                r = self.model(tile, imgsz=self.imgsz, conf=self.conf)[0]
                if r.boxes is None:
                    continue
                for b in r.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b[:4]
                    results.append([x0 + x1, y0 + y1, x0 + x2, y0 + y2])
        else:
            r = self.model(frame_bgr, imgsz=self.imgsz, conf=self.conf)[0]
            if r.boxes is not None:
                for b in r.boxes.xyxy.cpu().numpy():
                    results.append(b[:4].tolist())

        def iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1, iy1, ix2, iy2 = (
                max(ax1, bx1),
                max(ay1, by1),
                min(ax2, bx2),
                min(ay2, by2),
            )
            iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
            inter = iw * ih
            area = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
            return inter / max(1e-6, area)

        keep = []
        for b in sorted(
            results, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]), reverse=True
        ):
            if all(iou(b, k) < 0.5 for k in keep):
                keep.append(b)
        return keep
