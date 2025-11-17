"""Interactive ROI drawing helpers used as a fallback when auto_cv fails."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

try:  # pragma: no cover - GUI availability varies per environment
    import cv2
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore
import numpy as np

try:  # pragma: no cover - optional for Chinese text rendering
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = None  # type: ignore
    ImageDraw = None  # type: ignore
    ImageFont = None  # type: ignore


Point = Tuple[int, int]
Polygon = List[Point]


@dataclass
class ManualROIResult:
    polygon: Optional[Polygon]
    base_size: Tuple[int, int]


class ROIAnnotator:
    """Interactive ROI annotator with zoom/pan support."""

    HELP_TEXT = (
        "LMB: add point | RMB/Bksp: undo | C: clear | Enter/Space: save | F: fit | Wheel: zoom | Drag: pan"
    )

    def __init__(
        self,
        image: np.ndarray,
        window_name: str = "ROI Annotator",
        use_pil_font: bool = False,
    ) -> None:
        if cv2 is None:
            raise RuntimeError("OpenCV is required for manual ROI annotation.")

        self.image = image
        self.window_name = window_name
        self.use_pil_font = use_pil_font
        self.points: Polygon = []
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.dragging = False
        self.drag_button: Optional[int] = None
        self.last_mouse: Tuple[int, int] = (0, 0)
        self.drag_moved = False
        self.font = self._load_font()

        self.view_w, self.view_h = self._init_window()
        self.fit_to_window()
        cv2.setMouseCallback(self.window_name, self._on_mouse)

    def _load_font(self):  # pragma: no cover - GUI rendering
        if not self.use_pil_font or ImageFont is None:
            return None
        font_candidates = [
            "C:\\Windows\\Fonts\\msyh.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        ]
        for path in font_candidates:
            try:
                return ImageFont.truetype(path, 18)
            except Exception:
                continue
        return None

    def _get_screen_size(self) -> Tuple[int, int]:  # pragma: no cover - platform specific
        try:
            import tkinter as tk

            root = tk.Tk()
            root.withdraw()
            width = root.winfo_screenwidth()
            height = root.winfo_screenheight()
            root.destroy()
            return int(width), int(height)
        except Exception:
            return 1920, 1080

    def _init_window(self) -> Tuple[int, int]:
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except cv2.error as exc:  # pragma: no cover - GUI availability varies
            raise RuntimeError("当前环境不支持图形界面，无法弹出手动 ROI 窗口。") from exc

        screen_w, screen_h = self._get_screen_size()
        h, w = self.image.shape[:2]
        initial_scale = min((screen_h * 0.8) / max(1, h), (screen_w * 0.8) / max(1, w))
        self.scale = max(0.1, min(initial_scale, 4.0))
        view_w = int(max(1, round(w * self.scale)))
        view_h = int(max(1, round(h * self.scale)))
        cv2.resizeWindow(self.window_name, view_w, view_h)
        return view_w, view_h

    @property
    def transform(self) -> np.ndarray:
        return np.array([[self.scale, 0.0, self.offset_x], [0.0, self.scale, self.offset_y]])

    def fit_to_window(self) -> None:
        h, w = self.image.shape[:2]
        self.scale = min(self.view_w / max(1.0, w), self.view_h / max(1.0, h))
        self.scale = float(np.clip(self.scale, 0.1, 4.0))
        scaled_w, scaled_h = w * self.scale, h * self.scale
        self.offset_x = (self.view_w - scaled_w) / 2.0
        self.offset_y = (self.view_h - scaled_h) / 2.0

    def _clamp_offset(self) -> None:
        h, w = self.image.shape[:2]
        scaled_w, scaled_h = w * self.scale, h * self.scale
        if scaled_w < self.view_w:
            self.offset_x = (self.view_w - scaled_w) / 2.0
        else:
            self.offset_x = float(min(0.0, max(self.offset_x, self.view_w - scaled_w)))
        if scaled_h < self.view_h:
            self.offset_y = (self.view_h - scaled_h) / 2.0
        else:
            self.offset_y = float(min(0.0, max(self.offset_y, self.view_h - scaled_h)))

    def _view_to_image(self, x: int, y: int) -> Point:
        ix = int(round((x - self.offset_x) / self.scale))
        iy = int(round((y - self.offset_y) / self.scale))
        h, w = self.image.shape[:2]
        return int(np.clip(ix, 0, w - 1)), int(np.clip(iy, 0, h - 1))

    def _on_mouse(self, event: int, x: int, y: int, flags: int, param: object) -> None:  # pragma: no cover - GUI
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN:
            self.points.append(self._view_to_image(x, y))
        elif event in (cv2.EVENT_RBUTTONDOWN, cv2.EVENT_MBUTTONDOWN):
            self.dragging = True
            self.drag_button = event
            self.last_mouse = (x, y)
            self.drag_moved = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            if dx or dy:
                self.drag_moved = True
            self.offset_x += dx
            self.offset_y += dy
            self.last_mouse = (x, y)
            self._clamp_offset()
        elif event in (cv2.EVENT_RBUTTONUP, cv2.EVENT_MBUTTONUP):
            if event == cv2.EVENT_RBUTTONUP and not self.drag_moved and self.points:
                self.points.pop()
            self.dragging = False
            self.drag_button = None
        elif event == cv2.EVENT_MOUSEWHEEL:
            self._handle_wheel(x, y, event)

    def _handle_wheel(self, x: int, y: int, event: int) -> None:
        direction = 1 if event > 0 else -1
        step = 0.1 * direction
        new_scale = float(np.clip(self.scale + step, 0.1, 4.0))
        if math.isclose(new_scale, self.scale, rel_tol=1e-3):
            return
        ix, iy = (x - self.offset_x) / self.scale, (y - self.offset_y) / self.scale
        self.scale = new_scale
        self.offset_x = x - ix * self.scale
        self.offset_y = y - iy * self.scale
        self._clamp_offset()

    def _draw_polygon(self, canvas: np.ndarray) -> np.ndarray:
        if not self.points:
            return canvas
        pts_view = [
            (
                int(round(pt[0] * self.scale + self.offset_x)),
                int(round(pt[1] * self.scale + self.offset_y)),
            )
            for pt in self.points
        ]
        overlay = canvas.copy()
        color_line = (0, 255, 255)
        color_fill = (0, 0, 255)
        if len(pts_view) >= 2:
            cv2.polylines(overlay, [np.array(pts_view, dtype=np.int32)], False, color_line, 2, cv2.LINE_AA)
        if len(pts_view) >= 3:
            cv2.fillPoly(overlay, [np.array(pts_view, dtype=np.int32)], color_fill)
            canvas = cv2.addWeighted(canvas, 0.7, overlay, 0.3, 0)
        for pt in pts_view:
            cv2.circle(canvas, pt, 5, (0, 255, 0), -1, cv2.LINE_AA)
        return canvas

    def _draw_text(self, canvas: np.ndarray, text: str, org: Tuple[int, int]) -> np.ndarray:
        if self.font and Image is not None and ImageDraw is not None:  # pragma: no cover - GUI path
            img_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(img_pil)
            draw.text(org, text, font=self.font, fill=(255, 255, 0))
            return np.array(img_pil)
        cv2.putText(canvas, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
        return canvas

    def render(self) -> np.ndarray:
        canvas = cv2.warpAffine(self.image, self.transform, (self.view_w, self.view_h))
        canvas = self._draw_polygon(canvas)
        canvas = self._draw_text(canvas, self.HELP_TEXT, (10, self.view_h - 12))
        return canvas

    def run(self) -> Optional[Polygon]:  # pragma: no cover - interactive loop
        while True:
            cv2.imshow(self.window_name, self.render())
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                return None
            if key in (13, 32):  # Enter or Space
                if len(self.points) >= 3:
                    return list(self.points)
            if key in (ord("c"), ord("C")):
                self.points.clear()
            if key == 8 and self.points:  # Backspace
                self.points.pop()
            if key in (ord("f"), ord("F")):
                self.fit_to_window()


def _read_frame(video_path: Path, frame_index: Optional[int] = None) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open video source: {video_path}")
    if frame_index is not None:
        capture.set(cv2.CAP_PROP_POS_FRAMES, float(frame_index))
    ok, frame = capture.read()
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()
    if not ok or frame is None:
        return None, (width, height)
    if width <= 0 or height <= 0:
        height, width = frame.shape[:2]
    return frame, (width, height)


def draw_roi_interactively(video_path: Path, frame_index: Optional[int] = None, use_pil_font: bool = False) -> ManualROIResult:
    """Launch an interactive OpenCV window for manual ROI drawing.

    This function mirrors the behaviour of the standalone tools/roi_annotator.py
    script so that the main pipeline can reuse the same zoom/pan capable UI.
    """

    frame, base_size = _read_frame(video_path, frame_index)
    if frame is None:
        raise RuntimeError("无法读取帧，无法进行手动 ROI 标注。")
    annotator = ROIAnnotator(frame, window_name="手动 ROI 标注", use_pil_font=use_pil_font)
    try:
        polygon = annotator.run()
    finally:
        if cv2 is not None:
            cv2.setMouseCallback(annotator.window_name, lambda *args: None)
            cv2.destroyWindow(annotator.window_name)
    return ManualROIResult(polygon=polygon, base_size=base_size)


__all__ = ["ManualROIResult", "ROIAnnotator", "draw_roi_interactively"]
