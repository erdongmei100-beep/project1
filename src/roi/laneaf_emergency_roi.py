"""Generate emergency lane ROI using LaneAF lane detections."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from lane_detection.laneaf import LaneAFDetector

Point = Tuple[int, int]
Polygon = List[Point]


class LaneAFEstimationError(RuntimeError):
    """Raised when the LaneAF ROI estimation cannot produce a valid polygon."""


@dataclass
class LaneAFROIConfig:
    """配置 LaneAF ROI 生成逻辑的阈值。"""

    bottom_ratio: float = 0.35
    min_lane_points: int = 12
    min_lane_height_ratio: float = 0.45
    min_bottom_span_px: float = 20.0
    target_lane_index_from_right: int = 1
    simplify_step: int = 2
    fallback_to_rightmost: bool = True
    right_margin_px: int = 0

    @classmethod
    def from_config(cls, cfg: Optional[Dict[str, object]]) -> "LaneAFROIConfig":
        if not cfg:
            return cls()
        data = dict(cfg)
        parsed: Dict[str, object] = {}
        for key in (
            "bottom_ratio",
            "min_lane_height_ratio",
        ):
            if key in data:
                parsed[key] = float(data[key])
        for key in (
            "min_lane_points",
            "simplify_step",
            "target_lane_index_from_right",
        ):
            if key in data:
                parsed[key] = int(data[key])
        if "min_bottom_span_px" in data:
            parsed["min_bottom_span_px"] = float(data["min_bottom_span_px"])
        if "fallback_to_rightmost" in data:
            parsed["fallback_to_rightmost"] = bool(data["fallback_to_rightmost"])
        if "right_margin_px" in data:
            parsed["right_margin_px"] = int(data["right_margin_px"])
        return cls(**parsed)  # type: ignore[arg-type]


class LaneAFEmergencyROI:
    """利用 LaneAF 检测结果推断应急车道 ROI。"""

    def __init__(
        self,
        laneaf_detector: LaneAFDetector,
        config: LaneAFROIConfig | Dict[str, object] | None = None,
    ) -> None:
        if isinstance(config, LaneAFROIConfig):
            self.config = config
        else:
            self.config = LaneAFROIConfig.from_config(config)
        self.detector = laneaf_detector

    def get_roi(self, image_bgr: np.ndarray) -> Polygon:
        lanes = self.detector.detect_lanes(image_bgr)
        if not lanes:
            raise LaneAFEstimationError("LaneAF 未检测到任何车道线，无法生成 ROI。")

        height, width = image_bgr.shape[:2]
        bottom_threshold = height * (1.0 - float(np.clip(self.config.bottom_ratio, 0.0, 1.0)))
        min_height = height * float(np.clip(self.config.min_lane_height_ratio, 0.0, 1.0))

        candidates: List[Tuple[float, np.ndarray]] = []
        for lane in lanes:
            if lane.shape[0] < max(self.config.min_lane_points, 3):
                continue
            lane = lane[np.argsort(lane[:, 1])]
            if lane[-1, 1] - lane[0, 1] < min_height:
                continue
            bottom_mask = lane[:, 1] >= bottom_threshold
            bottom_points = lane[bottom_mask]
            if bottom_points.size == 0:
                continue
            if bottom_points.shape[0] >= 2:
                bottom_span = float(bottom_points[:, 0].max() - bottom_points[:, 0].min())
            else:
                bottom_span = 0.0
            if bottom_span < self.config.min_bottom_span_px:
                continue
            avg_x = float(np.mean(bottom_points[:, 0]))
            candidates.append((avg_x, lane))

        if not candidates:
            raise LaneAFEstimationError("LaneAF 未找到满足高度与底部覆盖条件的车道线。")

        candidates.sort(key=lambda item: item[0])
        target_offset = max(int(self.config.target_lane_index_from_right), 0)
        target_idx = len(candidates) - 1 - target_offset
        if target_idx < 0:
            if self.config.fallback_to_rightmost:
                target_idx = len(candidates) - 1
            else:
                raise LaneAFEstimationError("LaneAF 检测的车道线数量不足，无法定位右侧分界线。")
        target_lane = candidates[target_idx][1]

        polygon = self._build_polygon(target_lane, width, height)
        if len(polygon) < 3:
            raise LaneAFEstimationError("LaneAF 生成的 ROI 顶点不足。")
        return polygon

    def _build_polygon(self, lane: np.ndarray, width: int, height: int) -> Polygon:
        lane = lane[np.argsort(lane[:, 1])]
        simplify = max(int(self.config.simplify_step), 1)
        sampled: List[Point] = []
        for idx in range(0, lane.shape[0], simplify):
            x = int(np.clip(round(lane[idx, 0]), 0, width - 1))
            y = int(np.clip(round(lane[idx, 1]), 0, height - 1))
            sampled.append((x, y))

        last_point = (
            int(np.clip(round(lane[-1, 0]), 0, width - 1)),
            int(np.clip(round(lane[-1, 1]), 0, height - 1)),
        )
        if not sampled or sampled[-1] != last_point:
            sampled.append(last_point)

        deduped: List[Point] = []
        for pt in sampled:
            if not deduped or pt != deduped[-1]:
                deduped.append(pt)
        sampled = deduped

        margin = int(self.config.right_margin_px)
        if margin >= 0:
            right_x = max(0, width - 1 - margin)
        else:
            right_x = min(width - 1, width - 1 - margin)
        top_y = sampled[0][1]
        bottom_y = height - 1

        polygon: Polygon = list(sampled)
        polygon.append((right_x, bottom_y))
        polygon.append((right_x, max(0, top_y)))
        return polygon


__all__ = ["LaneAFEmergencyROI", "LaneAFROIConfig", "LaneAFEstimationError"]
