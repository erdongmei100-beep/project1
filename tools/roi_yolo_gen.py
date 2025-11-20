"""YOLOv8-based ROI generation tool."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

Point = Tuple[int, int]
Polygon = List[Point]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8-based ROI generator")
    parser.add_argument("--source", required=True, help="Path to input video or image")
    parser.add_argument("--out", required=True, help="Path to output ROI JSON")
    parser.add_argument(
        "--model", default="yolov8n-seg.pt", help="Path to YOLOv8 model weights (.pt)"
    )
    parser.add_argument("--class-id", type=int, default=0, help="Target class id")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--show", action="store_true", help="Show visualization window")
    return parser.parse_args(argv)


def load_sample_frame(source: Path, max_frames: int = 5) -> Tuple[np.ndarray, float]:
    if not source.exists():
        raise FileNotFoundError(f"Input source not found: {source}")

    if source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        frame = cv2.imread(str(source))
        if frame is None:
            raise RuntimeError(f"Failed to read image: {source}")
        return frame, 0.0

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Failed to open video: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: List[np.ndarray] = []
    for _ in range(max_frames):
        success, frame = capture.read()
        if not success or frame is None:
            break
        frames.append(frame)
    capture.release()

    if not frames:
        raise RuntimeError("Failed to read frames from video.")

    if len(frames) == 1:
        return frames[0], fps

    avg = np.mean(np.stack(frames, axis=0), axis=0)
    return avg.astype(np.uint8), fps


def clamp_polygon(polygon: Iterable[Tuple[float, float]], width: int, height: int) -> Polygon:
    clamped: Polygon = []
    for x, y in polygon:
        clamped.append((int(np.clip(round(x), 0, width - 1)), int(np.clip(round(y), 0, height - 1))))
    return clamped


def select_best_mask(result, target_class: int) -> Optional[np.ndarray]:
    masks = result.masks
    boxes = result.boxes
    if masks is None or boxes is None:
        return None

    best_idx: Optional[int] = None
    best_score: float = float("-inf")
    for idx, (cls, conf) in enumerate(zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())):
        if int(cls) != target_class:
            continue
        if conf > best_score and idx < len(masks.xy):
            best_score = float(conf)
            best_idx = idx
    if best_idx is None:
        return None
    return masks.xy[best_idx]


def select_best_box(result, target_class: int) -> Optional[np.ndarray]:
    boxes = result.boxes
    if boxes is None:
        return None
    best_idx: Optional[int] = None
    best_score: float = float("-inf")
    for idx, (cls, conf) in enumerate(zip(boxes.cls.cpu().numpy(), boxes.conf.cpu().numpy())):
        if int(cls) != target_class:
            continue
        if conf > best_score:
            best_score = float(conf)
            best_idx = idx
    if best_idx is None:
        return None
    return boxes.xyxy[best_idx].cpu().numpy()


class ManualPolygonPicker:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.clone = image.copy()
        self.points: Polygon = []
        self.window = "ROI YOLO Manual"
        self.message = "Click polygon vertices (>=3). S-save, R-reset, Q/ESC-quit"

    def reset(self) -> None:
        self.points.clear()
        self.clone = self.image.copy()

    def mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.points.append((x, y))

    def pick(self) -> Optional[Polygon]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)
        while True:
            display = self.clone.copy()
            for idx, (x, y) in enumerate(self.points):
                cv2.circle(display, (x, y), 5, (0, 255, 255), -1, cv2.LINE_AA)
                cv2.putText(
                    display,
                    f"P{idx+1}",
                    (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            if len(self.points) >= 2:
                cv2.polylines(display, [np.array(self.points, dtype=np.int32)], False, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(
                display,
                self.message,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(self.window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(self.window)
                return None
            if key in (ord("r"), ord("R")):
                self.reset()
            if key in (ord("s"), ord("S")) and len(self.points) >= 3:
                cv2.destroyWindow(self.window)
                return self.points[:]


def visualize(frame: np.ndarray, polygon: Polygon) -> None:
    vis = frame.copy()
    if polygon:
        pts = np.array(polygon, dtype=np.int32)
        cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.namedWindow("ROI YOLO Result", cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow("ROI YOLO Result", vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyWindow("ROI YOLO Result")


def save_roi_json(path: Path, frame: np.ndarray, polygon: Polygon) -> None:
    height, width = frame.shape[:2]
    payload = {
        "base_size": [int(width), int(height)],
        "polygon": [[int(x), int(y)] for x, y in polygon],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def extract_polygon(result, frame_shape: Tuple[int, int], target_class: int) -> Optional[Polygon]:
    height, width = frame_shape
    mask = select_best_mask(result, target_class)
    if mask is not None and len(mask) >= 3:
        return clamp_polygon(mask, width, height)

    box = select_best_box(result, target_class)
    if box is None or len(box) != 4:
        return None
    x1, y1, x2, y2 = box.tolist()
    polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return clamp_polygon(polygon, width, height)


def run_inference(model_path: str, frame: np.ndarray, conf: float) -> object:
    model = YOLO(model_path)
    results = model.predict(frame, conf=conf, verbose=False)
    if not results:
        raise RuntimeError("YOLO returned no results.")
    return results[0]


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    source = Path(args.source)
    output = Path(args.out)

    frame, fps = load_sample_frame(source)
    _ = fps  # reserved for future use

    try:
        result = run_inference(args.model, frame, args.conf)
        polygon = extract_polygon(result, frame.shape[:2], args.class_id)
    except Exception as exc:
        print(f"Inference failed: {exc}")
        polygon = None

    if polygon is None:
        print("No detection found; entering manual ROI selection mode...")
        picker = ManualPolygonPicker(frame)
        polygon = picker.pick()
        if polygon is None:
            raise RuntimeError("Manual ROI selection aborted.")

    save_roi_json(output, frame, polygon)
    print(f"ROI saved to {output}")

    if args.show:
        visualize(frame, polygon)


if __name__ == "__main__":
    main()
