"""半自动应急车道左侧白线 ROI 生成工具。"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

LinePoints = Tuple[Tuple[float, float], Tuple[float, float]]
Polygon = List[Tuple[int, int]]


@dataclass
class DetectedLine:
    """描述在一帧中检测到的直线。"""

    m: float
    b: float
    score: float
    length: float
    angle: float
    x_bottom: float
    frame_index: int
    points: LinePoints


@dataclass
class AutoLineConfig:
    source: Path
    output: Path
    sample_frames: int = 15
    crop_right: float = 0.45
    crop_bottom: float = 0.70
    s_max: int = 60
    v_min: int = 200
    angle_min: float = 15.0
    angle_max: float = 75.0
    top_ratio: float = 0.35
    bottom_margin: int = 20
    buffer: int = 6
    preview: Optional[Path] = None
    show: bool = False


def parse_args(argv: Optional[Sequence[str]] = None) -> AutoLineConfig:
    parser = argparse.ArgumentParser(description="半自动应急车道白线 ROI 生成")
    parser.add_argument("--source", required=True, help="输入视频或图像路径")
    parser.add_argument("--out", required=True, help="输出 ROI JSON 路径")
    parser.add_argument("--sample-frames", type=int, default=15, help="参与融合的帧数")
    parser.add_argument("--crop-right", type=float, default=0.45, help="右侧裁剪比例")
    parser.add_argument("--crop-bottom", type=float, default=0.70, help="下侧裁剪比例")
    parser.add_argument("--s-max", type=int, default=60, help="HSV 饱和度上限")
    parser.add_argument("--v-min", type=int, default=200, help="HSV 亮度下限")
    parser.add_argument("--angle-min", type=float, default=15.0, help="直线角度下限")
    parser.add_argument("--angle-max", type=float, default=75.0, help="直线角度上限")
    parser.add_argument("--top-ratio", type=float, default=0.35, help="ROI 顶部位置比例")
    parser.add_argument("--bottom-margin", type=int, default=20, help="ROI 底部安全边距")
    parser.add_argument("--buffer", type=int, default=6, help="沿白线内缩像素")
    parser.add_argument("--preview", help="保存叠加预览 PNG")
    parser.add_argument("--show", action="store_true", help="显示检测可视化窗口")
    args = parser.parse_args(argv)

    return AutoLineConfig(
        source=Path(args.source),
        output=Path(args.out),
        sample_frames=max(1, args.sample_frames),
        crop_right=float(np.clip(args.crop_right, 0.0, 1.0)),
        crop_bottom=float(np.clip(args.crop_bottom, 0.0, 1.0)),
        s_max=int(np.clip(args.s_max, 0, 255)),
        v_min=int(np.clip(args.v_min, 0, 255)),
        angle_min=float(args.angle_min),
        angle_max=float(args.angle_max),
        top_ratio=float(np.clip(args.top_ratio, 0.0, 1.0)),
        bottom_margin=max(0, int(args.bottom_margin)),
        buffer=int(args.buffer),
        preview=Path(args.preview) if args.preview else None,
        show=bool(args.show),
    )


def load_frames(config: AutoLineConfig) -> Tuple[List[Tuple[int, np.ndarray]], float]:
    if not config.source.exists():
        raise FileNotFoundError(f"无法找到输入文件: {config.source}")

    if config.source.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        frame = cv2.imread(str(config.source))
        if frame is None:
            raise RuntimeError(f"无法读取图像: {config.source}")
        return [(0, frame)], 0.0

    capture = cv2.VideoCapture(str(config.source))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开视频文件: {config.source}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if frame_count <= 0:
        frame_count = config.sample_frames * 3
    max_range = min(frame_count - 1, int((fps if fps > 0 else 25.0) * 5)) if frame_count > 0 else config.sample_frames
    max_range = max(max_range, config.sample_frames)
    indices = sorted({int(round(x)) for x in np.linspace(0, max_range, config.sample_frames)})

    frames: List[Tuple[int, np.ndarray]] = []
    target_iter = iter(indices)
    try:
        target = next(target_iter)
    except StopIteration:
        target = None

    current = 0
    while target is not None:
        success, frame = capture.read()
        if not success or frame is None:
            break
        if current == target:
            frames.append((current, frame.copy()))
            try:
                target = next(target_iter)
            except StopIteration:
                target = None
        current += 1
        if current > indices[-1] + 5:
            break

    capture.release()
    if not frames:
        raise RuntimeError("无法从视频中读取有效帧")
    return frames, fps


def detect_lane_line(frame: np.ndarray, frame_index: int, config: AutoLineConfig) -> Optional[DetectedLine]:
    height, width = frame.shape[:2]
    crop_x0 = int(width * (1.0 - config.crop_right)) if config.crop_right > 0 else 0
    crop_y0 = int(height * (1.0 - config.crop_bottom)) if config.crop_bottom > 0 else 0
    crop = frame[crop_y0:height, crop_x0:width]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 0, config.v_min], dtype=np.uint8)
    upper = np.array([179, config.s_max, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    edges = cv2.Canny(mask, 50, 150)
    min_line_len = int(max(crop.shape[0], crop.shape[1]) * 0.3)
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=60,
        minLineLength=max(min_line_len, 30),
        maxLineGap=30,
    )
    if lines is None:
        return None

    best: Optional[DetectedLine] = None
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        full_x1 = x1 + crop_x0
        full_y1 = y1 + crop_y0
        full_x2 = x2 + crop_x0
        full_y2 = y2 + crop_y0
        dx = float(full_x2 - full_x1)
        dy = float(full_y2 - full_y1)
        length = math.hypot(dx, dy)
        if length < 20.0:
            continue
        angle = abs(math.degrees(math.atan2(dy, dx)))
        if angle > 90.0:
            angle = 180.0 - angle
        if not (config.angle_min <= angle <= config.angle_max):
            continue
        if abs(dx) < 1e-3:
            continue
        m = dy / dx
        b = full_y1 - m * full_x1
        if abs(m) < 1e-4:
            continue
        x_bottom = (height - 1 - b) / m
        if not np.isfinite(x_bottom):
            continue
        score_bottom = np.clip(x_bottom / max(width - 1, 1), 0.0, 1.5)
        alignment = 1.0 - abs(angle - 60.0) / 60.0
        score = score_bottom * (1.0 + max(alignment, 0.0)) * (length / max(width, 1))
        if best is None or score > best.score:
            best = DetectedLine(
                m=m,
                b=b,
                score=score,
                length=length,
                angle=angle,
                x_bottom=x_bottom,
                frame_index=frame_index,
                points=((full_x1, full_y1), (full_x2, full_y2)),
            )
    return best


def fuse_lines(lines: Sequence[DetectedLine], frame_height: int) -> DetectedLine:
    slopes = np.array([line.m for line in lines])
    bottoms = np.array([line.x_bottom for line in lines])
    lengths = np.array([line.length for line in lines])
    angles = np.array([line.angle for line in lines])

    m_med = float(np.median(slopes))
    x_bottom_med = float(np.median(bottoms))
    b_med = (frame_height - 1) - m_med * x_bottom_med
    length_med = float(np.median(lengths))
    angle_med = float(np.median(angles))

    return DetectedLine(
        m=m_med,
        b=b_med,
        score=float(np.median([line.score for line in lines])),
        length=length_med,
        angle=angle_med,
        x_bottom=x_bottom_med,
        frame_index=lines[0].frame_index,
        points=lines[0].points,
    )


def clamp_polygon(polygon: Polygon, width: int, height: int) -> Polygon:
    clamped = []
    for x, y in polygon:
        clamped.append((int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))))
    return clamped


def polygon_area(polygon: Polygon) -> float:
    if len(polygon) < 3:
        return 0.0
    area = 0.0
    for i in range(len(polygon)):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % len(polygon)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def build_roi_polygon(line: DetectedLine, width: int, height: int, config: AutoLineConfig) -> Optional[Polygon]:
    y_top = int(height * config.top_ratio)
    y_bottom = height - 1 - config.bottom_margin
    y_top = int(np.clip(y_top, 0, height - 1))
    y_bottom = int(np.clip(y_bottom, 0, height - 1))
    if y_bottom <= y_top:
        y_bottom = min(height - 1, y_top + 10)
    m = line.m
    b = line.b
    if abs(m) < 1e-4:
        return None

    def project_x(y: int) -> int:
        x = (y - b) / m
        return int(round(x))

    left_bottom = project_x(y_bottom) + config.buffer
    left_top = project_x(y_top) + config.buffer
    right = width - 1
    polygon = [(left_bottom, y_bottom), (right, y_bottom), (right, y_top), (left_top, y_top)]
    polygon = clamp_polygon(polygon, width, height)
    area = polygon_area(polygon)
    total_area = width * height
    if total_area <= 0:
        return None
    if not (0.01 * total_area <= area <= 0.40 * total_area):
        return None
    if polygon[0][0] >= polygon[1][0] or polygon[3][0] >= polygon[2][0]:
        return None
    return polygon


def save_roi_json(path: Path, frame_size: Tuple[int, int], polygon: Polygon, meta: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_size": [int(frame_size[0]), int(frame_size[1])],
        "polygon": [[int(x), int(y)] for x, y in polygon],
        "meta": meta,
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def draw_preview(frame: np.ndarray, polygon: Polygon, line: DetectedLine) -> np.ndarray:
    vis = frame.copy()
    pts = np.array(polygon, dtype=np.int32)
    cv2.polylines(vis, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
    x_coords = np.array([p[0] for p in polygon])
    y_coords = np.array([p[1] for p in polygon])
    x_min = int(x_coords.min())
    x_max = int(x_coords.max())
    y_bottom = int(y_coords.max())
    y_top = int(y_coords.min())
    for y in [y_top, y_bottom]:
        x = int(round((y - line.b) / line.m))
        cv2.circle(vis, (x, y), 5, (255, 0, 0), -1, cv2.LINE_AA)
    cv2.line(vis, (x_min, y_bottom), (x_max, y_top), (0, 200, 255), 2, cv2.LINE_AA)
    return vis


class ManualLinePicker:
    def __init__(self, image: np.ndarray) -> None:
        self.image = image
        self.clone = image.copy()
        self.points: List[Tuple[int, int]] = []
        self.window = "ROI Auto Line"
        self.message = "请依次点击近端/远端白线点；S-保存，R-重置，Q/ESC-退出"

    def reset(self) -> None:
        self.points.clear()
        self.clone = self.image.copy()

    def mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(self.points) >= 2:
            self.points.pop(0)
        self.points.append((x, y))

    def pick(self) -> Optional[LinePoints]:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.mouse)
        while True:
            display = self.image.copy()
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
            if len(self.points) == 2:
                cv2.line(display, self.points[0], self.points[1], (0, 255, 0), 2, cv2.LINE_AA)
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
            if key in (ord("s"), ord("S")) and len(self.points) == 2:
                cv2.destroyWindow(self.window)
                return self.points[0], self.points[1]


def points_to_line(points: LinePoints) -> Optional[DetectedLine]:
    (x1, y1), (x2, y2) = points
    dx = float(x2 - x1)
    dy = float(y2 - y1)
    if abs(dx) < 1e-3:
        return None
    m = dy / dx
    b = y1 - m * x1
    length = math.hypot(dx, dy)
    angle = abs(math.degrees(math.atan2(dy, dx)))
    if angle > 90.0:
        angle = 180.0 - angle
    return DetectedLine(
        m=m,
        b=b,
        score=0.0,
        length=length,
        angle=angle,
        x_bottom=0.0,
        frame_index=0,
        points=points,
    )


def auto_detect(config: AutoLineConfig) -> Tuple[Optional[DetectedLine], List[int], np.ndarray]:
    frames, _ = load_frames(config)
    base_frame = frames[0][1]
    detected: List[DetectedLine] = []
    used_indices: List[int] = []
    for frame_index, frame in frames:
        line = detect_lane_line(frame, frame_index, config)
        if line is not None:
            detected.append(line)
            used_indices.append(frame_index)
    if detected:
        fused = fuse_lines(detected, base_frame.shape[0])
    else:
        fused = None
    return fused, used_indices, base_frame


def ensure_polygon(
    line: Optional[DetectedLine],
    base_frame: np.ndarray,
    config: AutoLineConfig,
    used_indices: Sequence[int],
) -> Tuple[Polygon, DetectedLine, List[int]]:
    height, width = base_frame.shape[:2]
    if line is not None:
        polygon = build_roi_polygon(line, width, height, config)
        if polygon is not None:
            return polygon, line, list(used_indices)

    if not config.show:
        raise RuntimeError("自动检测失败且未开启手动模式，请使用 --show 进行手动修正。")

    picker = ManualLinePicker(base_frame)
    picked = picker.pick()
    if picked is None:
        raise RuntimeError("手动模式未完成，ROI 生成失败。")
    manual_line = points_to_line(picked)
    if manual_line is None:
        raise RuntimeError("手动模式选择的点无法形成有效直线。")
    polygon = build_roi_polygon(manual_line, width, height, config)
    if polygon is None:
        raise RuntimeError("手动模式生成的多边形无效，请重新尝试。")
    return polygon, manual_line, list(used_indices)


def show_result(frame: np.ndarray, polygon: Polygon, line: DetectedLine) -> None:
    vis = draw_preview(frame, polygon, line)
    cv2.namedWindow("ROI Auto Line Result", cv2.WINDOW_NORMAL)
    while True:
        cv2.imshow("ROI Auto Line Result", vis)
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("s"), ord("S")):
            cv2.imwrite("roi_auto_line_preview.png", vis)
            print("已将当前窗口截图保存为 roi_auto_line_preview.png")
        if key in (ord("q"), ord("Q"), 27):
            break
    cv2.destroyWindow("ROI Auto Line Result")


def main(argv: Optional[Sequence[str]] = None) -> None:
    config = parse_args(argv)
    line, used_indices, base_frame = auto_detect(config)
    try:
        polygon, final_line, used_indices = ensure_polygon(line, base_frame, config, used_indices)
    except RuntimeError as exc:
        print(f"错误：{exc}")
        exit(1)

    frame_height, frame_width = base_frame.shape[:2]
    meta = {
        "mode": "auto_line",
        "frame_indices": used_indices,
        "params": {
            "sample_frames": config.sample_frames,
            "crop_right": config.crop_right,
            "crop_bottom": config.crop_bottom,
            "s_max": config.s_max,
            "v_min": config.v_min,
            "angle_min": config.angle_min,
            "angle_max": config.angle_max,
            "top_ratio": config.top_ratio,
            "bottom_margin": config.bottom_margin,
            "buffer": config.buffer,
        },
    }
    save_roi_json(config.output, (frame_width, frame_height), polygon, meta)
    print(f"ROI 已保存至 {config.output}")

    preview_image = draw_preview(base_frame, polygon, final_line)
    if config.preview:
        config.preview.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(config.preview), preview_image)
        print(f"预览图已保存至 {config.preview}")

    if config.show:
        show_result(base_frame, polygon, final_line)


if __name__ == "__main__":
    main()
