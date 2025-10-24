"""Interactive ROI drawing helpers used as a fallback when auto_cv fails."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Polygon = List[Point]


@dataclass
class ManualROIResult:
    polygon: Optional[Polygon]
    base_size: Tuple[int, int]


def _read_first_frame(video_path: Path) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"无法打开视频源: {video_path}")
    ok, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if not ok or frame is None:
        return None, (width, height)
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]
    return frame, (width, height)


def draw_roi_interactively(video_path: Path) -> ManualROIResult:
    """Launch an interactive OpenCV window for manual ROI drawing."""

    frame, base_size = _read_first_frame(video_path)
    if frame is None:
        raise RuntimeError("无法读取首帧，无法进行手动 ROI 标注。")

    window_name = "手动 ROI 标注"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    except cv2.error as exc:  # pragma: no cover - GUI availability varies
        raise RuntimeError(
            "当前环境不支持图形界面，无法弹出手动 ROI 标注窗口。"
        ) from exc

    points: Polygon = []

    instructions = [
        "自动 ROI 失败，请手动画出感兴趣区域。",
        "操作提示：左键添加顶点，右键撤销，按 R 重置，按 S 保存，按 Q 退出。",
    ]

    colors = {
        "text": (255, 255, 255),
        "shadow": (0, 0, 0),
        "line": (0, 200, 0),
        "point": (0, 255, 255),
    }

    def refresh_canvas() -> None:
        canvas = frame.copy()
        for idx, text in enumerate(instructions):
            org = (20, 40 + idx * 36)
            cv2.putText(
                canvas,
                text,
                (org[0] + 2, org[1] + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colors["shadow"],
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                colors["text"],
                2,
                cv2.LINE_AA,
            )
        if len(points) >= 1:
            cv2.polylines(canvas, [np.array(points, dtype=np.int32)], False, colors["line"], 2, cv2.LINE_AA)
        for pt in points:
            cv2.circle(canvas, pt, 6, colors["point"], -1, cv2.LINE_AA)
        cv2.imshow(window_name, canvas)

    def on_mouse(event: int, x: int, y: int, flags: int, param: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            if points:
                points.pop()
        refresh_canvas()

    cv2.setMouseCallback(window_name, on_mouse)
    refresh_canvas()

    try:
        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                cv2.destroyWindow(window_name)
                return ManualROIResult(polygon=None, base_size=base_size)
            if key in (ord("r"), ord("R")):
                points.clear()
                refresh_canvas()
            if key in (ord("s"), ord("S")):
                if len(points) >= 3:
                    cv2.destroyWindow(window_name)
                    return ManualROIResult(polygon=list(points), base_size=base_size)
                else:
                    if hasattr(cv2, "displayOverlay"):
                        try:
                            cv2.displayOverlay(
                                window_name,
                                "至少需要三个顶点才能保存 ROI。",
                                1500,
                            )
                        except cv2.error:
                            pass
    finally:
        cv2.setMouseCallback(window_name, lambda *args: None)


__all__ = ["ManualROIResult", "draw_roi_interactively"]
