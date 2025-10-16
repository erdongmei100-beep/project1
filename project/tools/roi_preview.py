import argparse
import json

import cv2
import numpy as np


# 读取并展示 ROI 区域的预览工具
VIDEO_EXTENSIONS = (
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".m4v",
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="ROI 预览工具")
    parser.add_argument("--source", required=True, help="源视频或图片路径")
    parser.add_argument("--roi", required=True, help="ROI JSON 路径")
    parser.add_argument("--out", help="可选的 PNG 保存路径")
    return parser.parse_args()


def load_source_frame(path):
    """加载图像或视频首帧"""
    lower = path.lower()
    if lower.endswith(VIDEO_EXTENSIONS):
        return _load_video_first_frame(path)
    return _load_image(path)


def _load_image(path):
    """加载图片文件"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"未找到源文件: {path}") from error
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("无法读取图片内容，请检查文件格式")
    return image


def _load_video_first_frame(path):
    """提取视频首帧"""
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"无法打开视频: {path}")
    success, frame = capture.read()
    capture.release()
    if not success or frame is None:
        raise ValueError("无法读取视频首帧")
    return frame


def load_roi(path):
    """加载并校验 ROI JSON"""
    try:
        with open(path, "r", encoding="utf-8") as file:
            data = json.load(file)
    except FileNotFoundError as error:
        raise FileNotFoundError(f"未找到 ROI 文件: {path}") from error
    except json.JSONDecodeError as error:
        raise ValueError(f"ROI JSON 解析失败: {error}") from error

    if "base_size" not in data or "polygon" not in data:
        raise ValueError("ROI JSON 缺少 base_size 或 polygon 字段")

    base_size = data["base_size"]
    polygon = data["polygon"]

    if (
        not isinstance(base_size, list)
        or len(base_size) != 2
        or not all(isinstance(value, (int, float)) for value in base_size)
    ):
        raise ValueError("base_size 字段格式错误，应为 [宽, 高]")

    base_w, base_h = float(base_size[0]), float(base_size[1])
    if base_w <= 0 or base_h <= 0:
        raise ValueError("base_size 数值必须大于 0")

    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError("polygon 至少需要三个点")

    points = []
    for index, raw_point in enumerate(polygon, start=1):
        if (
            not isinstance(raw_point, (list, tuple))
            or len(raw_point) != 2
            or not all(isinstance(value, (int, float)) for value in raw_point)
        ):
            raise ValueError(f"第 {index} 个顶点格式错误")
        points.append((float(raw_point[0]), float(raw_point[1])))

    return (base_w, base_h), points


def scale_polygon(points, base_size, target_size):
    """将多边形按分辨率比例缩放"""
    base_w, base_h = base_size
    target_w, target_h = target_size
    scale_x = target_w / base_w
    scale_y = target_h / base_h
    scaled = np.array(points, dtype=np.float32)
    scaled[:, 0] *= scale_x
    scaled[:, 1] *= scale_y
    return scaled


def ensure_points_in_bounds(points, width, height):
    """确认所有点均在画面范围内"""
    for index, (x, y) in enumerate(points, start=1):
        if not (0 <= x < width and 0 <= y < height):
            raise ValueError(f"第 {index} 个顶点越界: ({x:.1f}, {y:.1f})")


def draw_polygon(frame, points):
    """在图像上绘制多边形及顶点"""
    preview = frame.copy()
    int_points = np.round(points).astype(np.int32)
    cv2.polylines(preview, [int_points.reshape(-1, 1, 2)], True, (0, 255, 0), 2, cv2.LINE_AA)
    for index, (x, y) in enumerate(int_points, start=1):
        cv2.circle(preview, (int(x), int(y)), 5, (0, 0, 255), -1, cv2.LINE_AA)
        cv2.putText(
            preview,
            str(index),
            (int(x) + 5, int(y) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return preview


def show_preview(image):
    """窗口展示叠加结果"""
    window = "ROI Preview"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, image)
    cv2.waitKey(0)
    cv2.destroyWindow(window)


def main():
    """程序入口"""
    args = parse_args()
    try:
        frame = load_source_frame(args.source)
        (base_w, base_h), roi_points = load_roi(args.roi)
        height, width = frame.shape[:2]
        scaled_points = scale_polygon(roi_points, (base_w, base_h), (width, height))
        ensure_points_in_bounds(scaled_points, width, height)
        preview = draw_polygon(frame, scaled_points)
        if args.out:
            if not args.out.lower().endswith(".png"):
                raise ValueError("仅支持保存为 PNG 格式，请使用 .png 后缀")
            if not cv2.imwrite(args.out, preview):
                raise ValueError(f"无法保存预览图: {args.out}")
        show_preview(preview)
    except Exception as error:  # noqa: BLE001
        print(f"错误: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
