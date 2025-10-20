"""ROI 多边形标注工具。

此脚本使用 OpenCV 提供一个可视化窗口，允许用户通过鼠标点击定义 ROI 多边形，
并将结果保存为 JSON 文件。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import List, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]
VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".mpg",
    ".mpeg",
    ".m4v",
}


class ROIPolygonSelector:
    """在图像上收集多边形 ROI 顶点的交互式工具。"""

    def __init__(self, window_name: str, image: np.ndarray) -> None:
        """初始化工具。

        Args:
            window_name: OpenCV 窗口名称。
            image: 用于标注的图像数据。
        """
        self.window_name = window_name
        self.base_image = image
        self.points: List[Point] = []
        self.message: str = "左键添加点 | U-撤销 | R-重置 | S-保存 | ESC-退出"
        self.height, self.width = image.shape[:2]

    def mouse_callback(self, event: int, x: int, y: int, flags: int, userdata: object) -> None:
        """处理鼠标点击事件，记录合法的顶点。

        Args:
            event: OpenCV 鼠标事件码。
            x: 鼠标 x 坐标。
            y: 鼠标 y 坐标。
            flags: 事件附加标志（未使用）。
            userdata: 回调附带数据（未使用）。
        """
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if 0 <= x < self.width and 0 <= y < self.height:
            self.points.append((x, y))
            self.message = f"已添加点 ({x}, {y})"
        else:
            self.message = "点击超出图像范围，已忽略"

    def undo_point(self) -> None:
        """撤销最后一个添加的顶点。"""
        if self.points:
            removed = self.points.pop()
            self.message = f"已撤销点 {removed}"
        else:
            self.message = "无可撤销的点"

    def reset_points(self) -> None:
        """清空所有已添加顶点。"""
        if self.points:
            self.points.clear()
            self.message = "已清空所有点"
        else:
            self.message = "当前无点可清空"

    def can_save(self) -> bool:
        """判断当前点集是否满足保存要求。"""
        return len(self.points) >= 3

    def set_message(self, text: str) -> None:
        """更新状态提示文字。"""
        self.message = text

    def get_points(self) -> List[Point]:
        """获取当前收集的顶点列表，返回副本。"""
        return list(self.points)

    def render(self) -> np.ndarray:
        """绘制当前状态的图像帧。"""
        frame = self.base_image.copy()
        self._draw_polygon(frame)
        self._draw_overlay(frame)
        return frame

    def _draw_polygon(self, frame: np.ndarray) -> None:
        """在给定帧上绘制多边形轮廓和顶点标号。"""
        if len(self.points) > 1:
            pts = np.array(self.points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (0, 255, 0), 2, cv2.LINE_AA)
        for idx, (x, y) in enumerate(self.points, start=1):
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                str(idx),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

    def _draw_overlay(self, frame: np.ndarray) -> None:
        """绘制提示文字和状态信息。"""
        info_text = "左键添加点 | U-撤销 | R-重置 | S-保存 | ESC-退出"
        cv2.putText(
            frame,
            info_text,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.message,
            (10, max(35, self.height - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="ROI 多边形标注工具")
    parser.add_argument("--source", required=True, help="输入图片或视频路径")
    parser.add_argument("--out", required=True, help="输出 JSON 文件路径")
    return parser.parse_args()


def load_source_image(path: str) -> np.ndarray:
    """加载图像或视频第一帧。

    Args:
        path: 输入资源路径。

    Returns:
        用于标注的图像。

    Raises:
        FileNotFoundError: 输入路径不存在。
        RuntimeError: 无法读取图像或视频帧。
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到输入文件: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext in VIDEO_EXTENSIONS:
        return _load_video_first_frame(path)
    return _load_image(path)


def _load_video_first_frame(path: str) -> np.ndarray:
    """从视频中获取第一帧。"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频文件: {path}")
    success, frame = cap.read()
    cap.release()
    if not success or frame is None:
        raise RuntimeError("无法读取视频第一帧")
    return frame


def _load_image(path: str) -> np.ndarray:
    """加载图片文件，支持含空格或中文的路径。"""
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"无法读取图像文件: {path}")
    return image


def save_polygon(path: str, width: int, height: int, points: List[Point]) -> None:
    """将标注结果保存为 JSON 文件。"""
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
    payload = {
        "base_size": [int(width), int(height)],
        "polygon": [[int(x), int(y)] for x, y in points],
    }
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _interactive_select(window_name: str, image: np.ndarray, out_path: str) -> bool:
    """运行交互式 ROI 选择流程并保存结果。"""
    height, width = image.shape[:2]
    selector = ROIPolygonSelector(window_name, image)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, selector.mouse_callback)

    saved_once = False

    while True:
        frame = selector.render()
        cv2.imshow(window_name, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == 0xFF:  # 无按键
            continue
        if key in (ord("u"), ord("U")):
            selector.undo_point()
        elif key in (ord("r"), ord("R")):
            selector.reset_points()
        elif key in (ord("s"), ord("S")):
            points = selector.get_points()
            if not selector.can_save():
                selector.set_message("至少需要三个点才能保存")
                continue
            if not _points_in_bounds(points, width, height):
                selector.set_message("存在越界点，保存失败")
                continue
            try:
                save_polygon(out_path, width, height, points)
            except OSError as error:
                selector.set_message(f"保存失败: {error}")
            else:
                selector.set_message(f"保存成功: {out_path}")
                saved_once = True
        elif key == 27:  # ESC
            break

    cv2.destroyAllWindows()
    return saved_once


def launch_roi_selector(source: str, out_path: str, window_name: str | None = None) -> bool:
    """加载资源并启动交互式 ROI 标注。"""
    image = load_source_image(source)
    window = window_name or "ROI Maker"
    saved = _interactive_select(window, image, out_path)
    if saved:
        print(f"ROI 已保存至 {out_path}")
    return saved


def main() -> None:
    """脚本主入口。"""
    args = parse_args()
    saved = launch_roi_selector(args.source, args.out)
    if not saved:
        print("未保存 ROI，多边形标注未生效。")


def _points_in_bounds(points: List[Point], width: int, height: int) -> bool:
    """确认所有点都落在图像范围内。"""
    for x, y in points:
        if not (0 <= x < width and 0 <= y < height):
            return False
    return True


if __name__ == "__main__":
    main()
