"""LaneATT lane detection quick check utility."""
from __future__ import annotations

import argparse
from pathlib import Path

# --- add project root to sys.path (bootstrap) ---
import sys
ROOT = Path(__file__).resolve().parents[1]  # <repo-root>/project1
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# ------------------------------------------------

import cv2
import numpy as np

from src.lane_detection.laneatt import LaneATTConfig, LaneATTDetector
from src.utils.config import load_config
from src.utils.paths import ROOT as PROJECT_ROOT


def _resolve(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test for LaneATT detection")
    parser.add_argument("--video", required=True, help="Video path relative to repo root")
    parser.add_argument("--frame", type=int, default=0, help="Frame index to sample")
    parser.add_argument("--config", default="configs/default.yaml", help="Config YAML path")
    parser.add_argument("--save-dir", default="data/outputs/laneatt_detect", help="Output folder")
    parser.add_argument("--device", default="", help="Override LaneATT device (e.g. cpu, cuda:0)")
    return parser.parse_args()


def _load_frame(video_path: Path, frame_index: int) -> cv2.Mat:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Unable to open video: {video_path}, using blank frame")
        return _blank_frame()
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print(f"[WARN] Unable to read frame {frame_index} from {video_path}, using blank frame")
        return _blank_frame()
    return frame


def _blank_frame() -> cv2.Mat:
    return np.zeros((480, 640, 3), dtype=np.uint8)


def _draw_lanes(frame: cv2.Mat, lanes: list[cv2.Mat]) -> cv2.Mat:
    overlay = frame.copy()
    for lane in lanes:
        pts = lane.reshape((-1, 1, 2)).astype(int)
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 255), thickness=2)
    cv2.putText(
        overlay,
        f"lanes: {len(lanes)}",
        (16, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return overlay


def main() -> None:
    args = _parse_args()
    video_path = _resolve(args.video)
    config_path = _resolve(args.config)
    save_dir = _resolve(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(config_path)
    lane_cfg = ((cfg.get("roi") or {}).get("laneatt") or {}).get("laneatt") or {}
    if args.device:
        lane_cfg = dict(lane_cfg, device=args.device)
    detector = LaneATTDetector(LaneATTConfig.from_config(lane_cfg))

    frame = _load_frame(video_path, args.frame)
    lanes = detector.detect_lanes(frame)
    overlay = _draw_lanes(frame, lanes)

    out_path = save_dir / f"laneatt_frame_{args.frame:05d}.jpg"
    cv2.imwrite(str(out_path), overlay)
    print(f"Frame saved: {out_path}")
    print(f"Lanes detected: {len(lanes)} | model_loaded={detector.model_loaded}")


if __name__ == "__main__":
    main()
