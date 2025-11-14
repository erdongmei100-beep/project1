"""Quick visual sanity check for the LaneAF based ROI generator."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable, Tuple

import cv2
import numpy as np

from lane_detection.laneaf import LaneAFDetector
from src.roi.laneaf_emergency_roi import LaneAFEmergencyROI, LaneAFEstimationError, LaneAFROIConfig
from src.utils.paths import OUTPUTS_DIR, ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("LaneAF ROI visualisation helper")
    parser.add_argument("--source", required=True, help="视频路径或包含图片的目录")
    parser.add_argument("--weights", required=True, help="LaneAF 权重文件路径")
    parser.add_argument("--device", default="cuda", help="推理设备，默认 cuda 自动回退")
    parser.add_argument("--stride", type=int, default=5, help="视频抽帧间隔，单位帧")
    parser.add_argument(
        "--limit", type=int, default=30, help="最多处理多少帧/图片，0 表示不限制"
    )
    parser.add_argument(
        "--output",
        default=str(OUTPUTS_DIR / "laneaf_roi_vis"),
        help="输出目录，默认 outputs/laneaf_roi_vis",
    )
    return parser.parse_args()


def build_estimator(weights: Path, device: str) -> LaneAFEmergencyROI:
    detector = LaneAFDetector(weights, device=device)
    config = LaneAFROIConfig()
    return LaneAFEmergencyROI(detector, config)


def overlay_polygon(image: np.ndarray, polygon: Iterable[Tuple[int, int]]) -> np.ndarray:
    vis = image.copy()
    pts = np.array(list(polygon), dtype=np.int32)
    if pts.size == 0:
        return vis
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    overlay = np.zeros_like(vis)
    cv2.fillPoly(overlay, [pts], (0, 200, 0))
    return cv2.addWeighted(vis, 0.7, overlay, 0.3, 0)


def process_video(path: Path, estimator: LaneAFEmergencyROI, stride: int, limit: int, out_dir: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频: {path}")
    saved = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok or frame is None:
            break
        if stride > 1 and index % stride != 0:
            index += 1
            continue
        try:
            polygon = estimator.get_roi(frame)
        except LaneAFEstimationError as exc:
            print(f"[WARN] LaneAF 在第 {index} 帧失败: {exc}")
            index += 1
            continue
        except Exception as exc:  # pragma: no cover - defensive guard
            print(f"[WARN] LaneAF 在第 {index} 帧异常: {exc}")
            index += 1
            continue
        vis = overlay_polygon(frame, polygon)
        out_path = out_dir / f"{path.stem}_{index:05d}.png"
        cv2.imwrite(str(out_path), vis)
        saved += 1
        index += 1
        if limit > 0 and saved >= limit:
            break
    capture.release()
    return saved


def process_image_folder(path: Path, estimator: LaneAFEmergencyROI, limit: int, out_dir: Path) -> int:
    images = sorted([p for p in path.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg"}])
    saved = 0
    for image_path in images:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        try:
            polygon = estimator.get_roi(frame)
        except LaneAFEstimationError as exc:
            print(f"[WARN] LaneAF 处理 {image_path.name} 失败: {exc}")
            continue
        vis = overlay_polygon(frame, polygon)
        out_path = out_dir / f"{image_path.stem}.png"
        cv2.imwrite(str(out_path), vis)
        saved += 1
        if limit > 0 and saved >= limit:
            break
    return saved


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    weights_path = Path(args.weights)
    if not source_path.exists():
        raise FileNotFoundError(f"Source not found: {source_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    output_dir = Path(args.output)
    if not output_dir.is_absolute():
        output_dir = (ROOT / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    estimator = build_estimator(weights_path, args.device)

    processed = 0
    if source_path.is_dir():
        processed = process_image_folder(source_path, estimator, args.limit, output_dir)
    else:
        processed = process_video(source_path, estimator, args.stride, args.limit, output_dir)

    print(f"LaneAF ROI 可视化已保存 {processed} 帧至 {output_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
