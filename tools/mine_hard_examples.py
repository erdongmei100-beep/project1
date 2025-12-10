"""Mine hard examples for emergency lane segmentation using YOLOv8-seg."""
from __future__ import annotations

import sys
from pathlib import Path

# --- 【新增代码开始】 ---
# 强制将项目根目录加入 Python 搜索路径
# 获取当前脚本的绝对路径 -> 找到父级(tools) -> 再找父级(project1)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # project root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add root to PATH
# --- 【新增代码结束】 ---

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# 现在这一行就能正常运行了，因为 ROOT 已经在 path 里了
from src.utils.config import load_config, resolve_path
from src.utils.paths import PROJECT_ROOT, project_path


@dataclass
class FrameRecord:
    index: int
    area_ratio: float
    status: str  # one of {valid, miss, anomaly, low_conf}


DEFAULT_CONFIG = project_path("configs", "default.yaml").relative_to(PROJECT_ROOT)
DEFAULT_WEIGHTS = project_path("weights", "lane_seg_v2.pt").relative_to(PROJECT_ROOT)
DEFAULT_OUTPUT = project_path("data", "mined_images").relative_to(PROJECT_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine hard examples from raw videos for active learning.")
    parser.add_argument("--source", required=True, help="Path to input video file.")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help="Path to default.yaml configuration file (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--weights",
        default=str(DEFAULT_WEIGHTS),
        help="Path to YOLOv8-seg weights (default: weights/lane_seg_v2.pt).",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory to save mined frames and report (default: data/mined_images).",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=30,
        help="Sampling interval in frames (process every Nth frame).",
    )
    return parser.parse_args()


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _prepare_output_dirs(root: Path) -> Dict[str, Path]:
    categories = {
        "miss": root / "miss",
        "weird_left": root / "weird" / "left_side",
        "weird_tiny": root / "weird" / "tiny",
        "low_conf": root / "low_conf",
    }
    for path in categories.values():
        path.mkdir(parents=True, exist_ok=True)
    return categories


def _build_mask(result, frame_shape: Tuple[int, int], target_class: int) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """Return boolean mask for target class and the max confidence among its detections."""
    if result is None or result.masks is None or result.boxes is None:
        return None, None

    height, width = frame_shape
    polygons = result.masks.xy or []
    class_ids = result.boxes.cls
    confidences = result.boxes.conf

    cls_array = class_ids.cpu().numpy() if hasattr(class_ids, "cpu") else np.asarray(class_ids) if class_ids is not None else None
    conf_array = confidences.cpu().numpy() if hasattr(confidences, "cpu") else np.asarray(confidences) if confidences is not None else None

    if cls_array is None:
        return None, None

    mask_img = np.zeros((height, width), dtype=np.uint8)
    max_conf: Optional[float] = None

    for idx, (polygon, cls_id) in enumerate(zip(polygons, cls_array)):
        if int(cls_id) != int(target_class):
            continue
        cv2.fillPoly(mask_img, [polygon.astype(np.int32)], 1)
        if conf_array is not None and idx < len(conf_array):
            conf_value = float(conf_array[idx])
            max_conf = conf_value if max_conf is None else max(max_conf, conf_value)

    if not np.any(mask_img):
        return None, max_conf
    return mask_img.astype(bool), max_conf


def _mask_metrics(mask: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[float, Optional[float]]:
    height, width = frame_shape
    total_pixels = float(height * width) if height > 0 and width > 0 else 1.0
    area = float(np.count_nonzero(mask))
    area_ratio = area / total_pixels

    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return area_ratio, None
    cx = moments["m10"] / moments["m00"]
    return area_ratio, float(cx / width) if width > 0 else None


def _save_frame(
    frame: np.ndarray,
    mask: Optional[np.ndarray],
    save_path: Path,
    reason: str,
) -> None:
    canvas = frame.copy()
    if mask is not None:
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, (0, 255, 255), 2)
    cv2.putText(canvas, reason, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imwrite(str(save_path), canvas)


def _status_color(status: str) -> str:
    if status == "miss":
        return "red"
    if status in {"anomaly", "low_conf"}:
        return "yellow"
    return "green"


def generate_plot(records: List[FrameRecord], fps: float, output_path: Path) -> None:
    if not records:
        print("No records to plot; skipping lane_stats.png generation.")
        return

    times = [rec.index / fps if fps > 0 else rec.index for rec in records]
    ratios = [rec.area_ratio for rec in records]
    statuses = [rec.status for rec in records]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(times, ratios, label="Lane Area Ratio", color="blue")

    for i in range(len(times)):
        start_t = times[i]
        end_t = times[i + 1] if i + 1 < len(times) else (times[i] + (1.0 / fps if fps > 0 else 1.0))
        color = _status_color(statuses[i])
        ax.axvspan(start_t, end_t, color=color, alpha=0.15)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Lane area ratio")
    ax.set_title("Lane detection stability")
    ax.legend()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def process_video(args: argparse.Namespace) -> None:
    config_path = _resolve_project_path(args.config)
    weights_path = _resolve_project_path(args.weights)
    output_root = _resolve_project_path(args.output)
    output_dirs = _prepare_output_dirs(output_root)

    config = load_config(config_path)
    roi_cfg = dict(config.get("roi") or {})
    dynamic_filters = dict(roi_cfg.get("dynamic_filters") or {})
    min_centroid_x_ratio = float(dynamic_filters.get("min_centroid_x_ratio", 0.45))

    dynamic_seg_cfg = dict(roi_cfg.get("dynamic_seg") or {})
    lane_class_id = int(dynamic_seg_cfg.get("class_id", 0))
    lane_conf = dynamic_seg_cfg.get("conf")
    lane_imgsz = dynamic_seg_cfg.get("imgsz")
    lane_device = dynamic_seg_cfg.get("device")

    model = YOLO(str(weights_path))
    predict_args: Dict[str, object] = {"verbose": False, "retina_masks": False}
    if lane_conf is not None:
        predict_args["conf"] = float(lane_conf)
    if lane_imgsz is not None:
        predict_args["imgsz"] = int(lane_imgsz)
    if lane_device is not None:
        predict_args["device"] = lane_device

    video_path = resolve_path(PROJECT_ROOT, args.source)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    records: List[FrameRecord] = []
    frame_idx = 0

    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break

        if frame_idx % max(1, args.interval) != 0:
            frame_idx += 1
            continue

        height, width = frame.shape[:2]
        results = model.predict(frame, **predict_args)
        result = results[0] if results else None

        mask, max_conf = _build_mask(result, (height, width), lane_class_id)
        area_ratio = float(np.count_nonzero(mask)) / float(height * width) if mask is not None else 0.0

        save_path: Optional[Path] = None
        reason = ""
        status = "valid"

        if mask is None:
            status = "miss"
            reason = "No mask detected"
            save_path = output_dirs["miss"] / f"frame_{frame_idx:06d}.jpg"
        else:
            area_ratio, cx_ratio = _mask_metrics(mask, (height, width))
            tiny_threshold = max(500.0, 0.01 * height * width)
            if cx_ratio is None:
                status = "anomaly"
                reason = "Invalid centroid"
                save_path = output_dirs["weird_tiny"] / f"frame_{frame_idx:06d}.jpg"
            elif cx_ratio < min_centroid_x_ratio:
                status = "anomaly"
                reason = f"Left Side: Cx={cx_ratio:.3f}"
                save_path = output_dirs["weird_left"] / f"frame_{frame_idx:06d}.jpg"
            elif area_ratio * height * width < tiny_threshold:
                status = "anomaly"
                reason = f"Tiny Area: {int(area_ratio * height * width)}px"
                save_path = output_dirs["weird_tiny"] / f"frame_{frame_idx:06d}.jpg"
            elif max_conf is not None and max_conf < 0.5:
                status = "low_conf"
                reason = f"Low confidence: {max_conf:.2f}"
                save_path = output_dirs["low_conf"] / f"frame_{frame_idx:06d}.jpg"

        records.append(FrameRecord(index=frame_idx, area_ratio=area_ratio, status=status))

        if save_path is not None:
            _save_frame(frame, mask, save_path, reason)
            print(f"Saved frame {frame_idx} to {save_path} ({reason})")

        frame_idx += 1

    cap.release()
    plot_path = output_root / "lane_stats.png"
    generate_plot(records, fps, plot_path)


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()
