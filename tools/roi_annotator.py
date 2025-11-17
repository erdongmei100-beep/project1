"""Interactive ROI annotator with zoom/pan controls.

Example usage:
    python tools/roi_annotator.py --image data/frames/2_000123.jpg --save data/rois/2.json
    python tools/roi_annotator.py --video data/videos/2.mp4 --frame 160 --save data/rois/2.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2

from src.roi.manual import ROIAnnotator


def _read_image(image_path: Path) -> Tuple[cv2.Mat, Tuple[int, int]]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Failed to read image: {image_path}")
    h, w = image.shape[:2]
    return image, (w, h)


def _read_video_frame(video_path: Path, frame_index: int) -> Tuple[cv2.Mat, Tuple[int, int]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = cap.read()
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Unable to read frame {frame_index} from {video_path}")
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]
    return frame, (width, height)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manual ROI annotator with zoom/pan")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image", type=Path, help="Image file for annotation")
    group.add_argument("--video", type=Path, help="Video file for extracting a frame")
    parser.add_argument("--frame", type=int, default=0, help="Frame index when using --video")
    parser.add_argument("--save", type=Path, required=True, help="Path to save ROI JSON")
    parser.add_argument("--use-pil-font", action="store_true", help="Use PIL for text rendering if available")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.image:
        frame, base_size = _read_image(args.image)
    else:
        frame, base_size = _read_video_frame(args.video, args.frame)

    annotator = ROIAnnotator(frame, use_pil_font=args.use_pil_font)
    try:
        polygon = annotator.run()
    finally:
        cv2.destroyAllWindows()

    if not polygon:
        print("Annotation cancelled; no ROI saved")
        return

    args.save.parent.mkdir(parents=True, exist_ok=True)
    payload = {"polygon": polygon, "image_wh": list(base_size), "mode": "manual"}
    args.save.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ROI saved to {args.save}")


if __name__ == "__main__":
    main()
