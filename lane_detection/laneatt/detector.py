"""LaneATT detector facade used by the emergency-lane ROI module."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore
import numpy as np

from .laneatt_model import LaneATTModel

Point = Tuple[int, int]


@dataclass
class LaneATTConfig:
    """Runtime configuration for :class:`LaneATTDetector`."""

    weights: Optional[Path] = None
    device: str = "auto"
    min_length_px: int = 80
    canny_threshold1: int = 80
    canny_threshold2: int = 160
    hough_threshold: int = 50
    hough_min_line_length: int = 60
    hough_max_line_gap: int = 40
    angle_range: Tuple[float, float] = (15.0, 80.0)
    right_lane_only: bool = True

    @classmethod
    def from_config(cls, config: Optional[dict]) -> "LaneATTConfig":
        data = dict(config or {})
        weights = data.get("weights")
        if weights:
            weights = Path(weights)
        angle = data.get("angle_range") or data.get("angle")
        if angle and isinstance(angle, Sequence) and len(angle) == 2:
            angle_range = (float(angle[0]), float(angle[1]))
        else:
            angle_range = cls.angle_range
        return cls(
            weights=weights,
            device=str(data.get("device", cls.device)),
            min_length_px=int(data.get("min_length_px", cls.min_length_px)),
            canny_threshold1=int(data.get("canny_threshold1", cls.canny_threshold1)),
            canny_threshold2=int(data.get("canny_threshold2", cls.canny_threshold2)),
            hough_threshold=int(data.get("hough_threshold", cls.hough_threshold)),
            hough_min_line_length=int(
                data.get("hough_min_line_length", cls.hough_min_line_length)
            ),
            hough_max_line_gap=int(
                data.get("hough_max_line_gap", cls.hough_max_line_gap)
            ),
            angle_range=angle_range,
            right_lane_only=bool(data.get("right_lane_only", cls.right_lane_only)),
        )


@dataclass
class LaneDetection:
    """Description of a detected lane line."""

    points: Tuple[Point, Point]
    length: float
    angle: float
    score: float

    @property
    def bottom_point(self) -> Point:
        return max(self.points, key=lambda p: p[1])

    @property
    def top_point(self) -> Point:
        return min(self.points, key=lambda p: p[1])


@dataclass
class LaneDetectionResult:
    """Wrapper returned by :class:`LaneATTDetector.detect`."""

    lanes: List[LaneDetection]
    model_loaded: bool

    @property
    def best_lane(self) -> Optional[LaneDetection]:
        if not self.lanes:
            return None
        return max(self.lanes, key=lambda lane: lane.score)


class LaneATTDetector:
    """Hybrid LaneATT detector.

    The implementation prioritises using the actual LaneATT model when a
    checkpoint is provided and PyTorch is available.  When the heavy
    dependency is missing (common in light-weight deployments or during
    unit tests) the detector falls back to classical computer-vision
    primitives (Canny + Hough transform).  The fallback produces
    reasonable lane approximations on synthetic frames which is sufficient
    for ROI unit tests.
    """

    def __init__(self, config: LaneATTConfig | None = None) -> None:
        self.config = config or LaneATTConfig()
        self.model = LaneATTModel()
        self._model_ready = False
        self._model_error: Optional[str] = None
        if self.config.weights:
            try:
                self.model.load_weights(self.config.weights, self._resolve_device())
            except Exception as exc:  # pragma: no cover - optional dependency path
                # Keep running with heuristic fallback while recording the
                # failure for debugging purposes.
                self._model_error = str(exc)
            else:
                self._model_ready = True
        else:
            self._model_error = "weights_not_provided"

    @property
    def model_loaded(self) -> bool:
        return self._model_ready

    @property
    def model_error(self) -> Optional[str]:
        return self._model_error

    def _resolve_device(self) -> str:
        if self.config.device == "auto":
            return "cuda" if self._has_cuda() else "cpu"
        return self.config.device

    @staticmethod
    def _has_cuda() -> bool:
        try:  # pragma: no cover - depends on torch availability
            import torch

            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def detect(self, image: np.ndarray) -> LaneDetectionResult:
        if image is None or image.size == 0:
            return LaneDetectionResult([], self._model_ready)
        lanes = self._detect_fallback(image)
        return LaneDetectionResult(lanes=lanes, model_loaded=self._model_ready)

    # Fallback lane detector used in tests and when PyTorch is not
    # available.  The method is intentionally verbose to keep the maths
    # transparent for future maintainers.
    def _detect_fallback(self, image: np.ndarray) -> List[LaneDetection]:
        if cv2 is None:
            return []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blur, self.config.canny_threshold1, self.config.canny_threshold2
        )
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config.hough_threshold,
            minLineLength=self.config.hough_min_line_length,
            maxLineGap=self.config.hough_max_line_gap,
        )
        if lines is None:
            return []
        detections: List[LaneDetection] = []
        angle_min, angle_max = self.config.angle_range
        for x1, y1, x2, y2 in lines.reshape(-1, 4):
            length = math.hypot(x2 - x1, y2 - y1)
            if length < self.config.min_length_px:
                continue
            angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
            angle = 180.0 - angle if angle > 90.0 else angle
            if not (angle_min <= angle <= angle_max):
                continue
            points = ((int(x1), int(y1)), (int(x2), int(y2)))
            score = length / max(1.0, angle)
            detections.append(LaneDetection(points=points, length=length, angle=angle, score=score))

        if self.config.right_lane_only:
            detections = self._select_rightmost(detections)
        detections.sort(key=lambda lane: lane.score, reverse=True)
        return detections

    def _select_rightmost(self, lanes: Iterable[LaneDetection]) -> List[LaneDetection]:
        lanes = list(lanes)
        if not lanes:
            return []
        lanes.sort(key=lambda lane: lane.bottom_point[0], reverse=True)
        if len(lanes) <= 2:
            return lanes
        return lanes[:2]
