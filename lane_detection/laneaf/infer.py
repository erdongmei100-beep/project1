"""LaneAF inference wrapper."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Sequence

import cv2
import numpy as np
import torch

from .affinity import decode_afs
from .model import get_pose_net

LOGGER = logging.getLogger(__name__)


class LaneAFDetector:
    """封装 LaneAF 推理流程，便于在项目中复用。"""

    def __init__(
        self,
        weights_path: str | Path,
        device: str = "cuda",
        input_size: tuple[int, int] = (288, 832),
        confidence_threshold: float = 0.35,
        min_points: int = 6,
    ) -> None:
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"LaneAF 权重文件不存在: {weights_path}")

        device_resolved = str(device or "cpu").lower()
        if device_resolved.startswith("cuda") and not torch.cuda.is_available():
            LOGGER.warning("CUDA 不可用，LaneAF 自动回退 CPU。")
            device_resolved = "cpu"
        self.device = torch.device(device_resolved)

        self.input_size = (int(input_size[0]), int(input_size[1]))
        self.conf_threshold = float(confidence_threshold)
        self.min_points = int(max(min_points, 3))

        heads = {"hm": 1, "vaf": 2, "haf": 1}
        self.model = get_pose_net(num_layers=34, heads=heads, head_conv=256, down_ratio=4)
        state = torch.load(weights_path, map_location="cpu")
        state_dict = state.get("state_dict") if isinstance(state, dict) else None
        if state_dict is None:
            state_dict = state
        cleaned = {k.replace("module.", ""): v for k, v in state_dict.items()}
        missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
        if missing:
            LOGGER.warning("LaneAF 权重缺少参数: %s", ", ".join(missing))
        if unexpected:
            LOGGER.warning("LaneAF 权重存在未使用参数: %s", ", ".join(unexpected))

        self.model.to(self.device)
        self.model.eval()

        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    @torch.no_grad()
    def detect_lanes(self, image_bgr: np.ndarray) -> List[np.ndarray]:
        """对单帧图像执行 LaneAF 推理，返回若干车道线折线。"""

        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("输入图像为空，无法执行 LaneAF 推理。")

        orig_h, orig_w = image_bgr.shape[:2]
        resized = cv2.resize(
            image_bgr,
            (self.input_size[1], self.input_size[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - self.mean) / self.std
        tensor = torch.from_numpy(rgb.transpose(2, 0, 1)).unsqueeze(0).to(self.device)

        outputs = self.model(tensor)[-1]
        heatmap = torch.sigmoid(outputs["hm"]).squeeze(0).squeeze(0).cpu().numpy()
        vaf = outputs["vaf"].squeeze(0).permute(1, 2, 0).cpu().numpy()
        haf = outputs["haf"].squeeze(0).permute(1, 2, 0).cpu().numpy()

        mask = (heatmap * 255.0).astype(np.uint8)
        seg = decode_afs(mask, vaf, haf, fg_thresh=128.0, err_thresh=5.0)

        lanes = self._segmentation_to_lanes(seg, heatmap, (orig_h, orig_w))
        return lanes

    def _segmentation_to_lanes(
        self,
        segmentation: np.ndarray,
        heatmap: np.ndarray,
        original_hw: Sequence[int],
    ) -> List[np.ndarray]:
        lane_ids = np.unique(segmentation)
        lane_ids = lane_ids[lane_ids > 0]
        if lane_ids.size == 0:
            return []

        out_h, out_w = segmentation.shape
        scale_x = self.input_size[1] / float(out_w)
        scale_y = self.input_size[0] / float(out_h)
        scale_x_full = float(original_hw[1]) / float(self.input_size[1])
        scale_y_full = float(original_hw[0]) / float(self.input_size[0])

        results: List[np.ndarray] = []

        for lane_id in lane_ids:
            coords = np.column_stack(np.where(segmentation == lane_id))
            if coords.shape[0] < self.min_points:
                continue
            coords = coords[np.argsort(coords[:, 0])]
            rows: dict[int, List[float]] = {}
            for row, col in coords:
                rows.setdefault(int(row), []).append(float(col))

            sorted_rows = sorted(rows.items())
            lane_points: List[tuple[float, float]] = []
            confidences: List[float] = []
            for row, cols in sorted_rows:
                col_mean = float(np.mean(cols))
                x_model = (col_mean + 0.5) * scale_x
                y_model = (row + 0.5) * scale_y
                x_full = x_model * scale_x_full
                y_full = y_model * scale_y_full
                lane_points.append((x_full, y_full))
                hm_row = min(max(int(round(row)), 0), heatmap.shape[0] - 1)
                hm_col = min(max(int(round(col_mean)), 0), heatmap.shape[1] - 1)
                confidences.append(float(heatmap[hm_row, hm_col]))

            if len(lane_points) < self.min_points:
                continue

            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            if avg_conf < self.conf_threshold:
                continue

            lane_arr = np.array(lane_points, dtype=np.float32)
            results.append(lane_arr)

        return results


__all__ = ["LaneAFDetector"]
