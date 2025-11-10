"""Unified OCR facade for license plates."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

try:  # pragma: no cover - optional dependency
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    _RapidOCR = None

try:  # pragma: no cover - optional dependency
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore
except Exception:  # pragma: no cover - dependency may be missing
    _PaddleOCR = None


class PlateOCRError(RuntimeError):
    """Raised when OCR initialisation fails."""


@dataclass
class OCRResult:
    text: str
    conf: float
    points: Optional[List[List[float]]]


def _ensure_model_dir(model_dir: Union[str, Path]) -> None:
    path = Path(model_dir)
    required = ["inference.pdmodel", "inference.pdiparams"]
    missing = [name for name in required if not (path / name).exists()]
    if missing:
        raise PlateOCRError(
            "PaddleOCR 模型目录不完整: {} 缺少 {}. 请将官方 inference 模型解压到该目录。".format(
                path, ", ".join(missing)
            )
        )


class PlateOCR:
    """Unified OCR entry-point supporting PaddleOCR and RapidOCR."""

    _PLATE_REGEX = re.compile(r"^[\u4e00-\u9fff][A-Z][A-Z0-9]{5}$")

    def __init__(
        self,
        engine: str = "paddle",
        use_gpu: bool = False,
        rec_model_dir: Optional[Union[str, Path]] = None,
        min_conf: float = 0.20,
        normalize_plate: bool = True,
    ) -> None:
        self.engine = engine.lower().strip() or "paddle"
        self.use_gpu = bool(use_gpu)
        self.rec_model_dir = Path(rec_model_dir) if rec_model_dir else None
        self.min_conf = float(min_conf)
        self.normalize_plate = bool(normalize_plate)
        self._reader: Any = None
        self._engine_type = ""

        if self.engine == "paddle":
            self._init_paddle()
        elif self.engine == "rapid":
            self._init_rapid()
        else:
            raise PlateOCRError(f"unsupported OCR engine: {engine}")

    # ------------------------------------------------------------------
    # Engine initialisation helpers
    def _init_paddle(self) -> None:
        if _PaddleOCR is None:
            raise PlateOCRError("PaddleOCR 未安装，请先执行 pip install paddleocr")
        if self.rec_model_dir is None:
            raise PlateOCRError(
                "PaddleOCR 需要提供 rec_model_dir，例如 weights/ppocr/ch_PP-OCRv4_rec_infer"
            )
        _ensure_model_dir(self.rec_model_dir)
        try:
            self._reader = _PaddleOCR(
                det=False,
                rec=True,
                use_gpu=self.use_gpu,
                rec_model_dir=str(self.rec_model_dir),
            )
        except Exception as exc:  # pragma: no cover - runtime guard
            raise PlateOCRError(f"PaddleOCR 初始化失败: {exc}") from exc
        self._engine_type = "paddle"

    def _init_rapid(self) -> None:
        if _RapidOCR is None:
            raise PlateOCRError(
                "RapidOCR 未安装。请执行 pip install rapidocr-onnxruntime 或切换 --engine paddle"
            )
        try:
            self._reader = _RapidOCR()
        except Exception as exc:  # pragma: no cover - runtime guard
            raise PlateOCRError(f"RapidOCR 初始化失败: {exc}") from exc
        self._engine_type = "rapid"

    # ------------------------------------------------------------------
    def __call__(self, image: Union[str, Path, np.ndarray]) -> dict:
        image_bgr = self._load_image(image)
        if image_bgr is None:
            return {
                "text": "",
                "conf": 0.0,
                "points": None,
                "engine": self._engine_type,
                "elapsed_ms": 0.0,
            }

        t0 = time.perf_counter()
        result = self._recognize(image_bgr)
        elapsed = (time.perf_counter() - t0) * 1000.0

        if result.conf < self.min_conf or not result.text:
            return {
                "text": "",
                "conf": float(result.conf),
                "points": result.points,
                "engine": self._engine_type,
                "elapsed_ms": elapsed,
            }

        text = result.text
        if self.normalize_plate:
            text = self._normalise_text(text)
            if text and not self._PLATE_REGEX.match(text):
                print(f"[WARN] OCR 文本不符合车牌格式: {text}")

        return {
            "text": text,
            "conf": float(result.conf),
            "points": result.points,
            "engine": self._engine_type,
            "elapsed_ms": elapsed,
        }

    # ------------------------------------------------------------------
    def _load_image(self, image: Union[str, Path, np.ndarray]) -> Optional[np.ndarray]:
        if isinstance(image, (str, Path)):
            path = Path(image)
            if not path.exists():
                print(f"[WARN] OCR 输入文件不存在: {path}")
                return None
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                print(f"[WARN] OCR 无法读取图片: {path}")
            return img
        array = np.asarray(image)
        if array.ndim == 2:
            return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        if array.ndim == 3 and array.shape[2] == 3:
            return array.copy()
        if array.ndim == 3 and array.shape[2] == 4:
            return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
        print("[WARN] OCR 输入图片格式不支持")
        return None

    def _recognize(self, image_bgr: np.ndarray) -> OCRResult:
        if self._engine_type == "paddle":
            return self._run_paddle(image_bgr)
        if self._engine_type == "rapid":
            return self._run_rapid(image_bgr)
        return OCRResult(text="", conf=0.0, points=None)

    def _run_paddle(self, image_bgr: np.ndarray) -> OCRResult:
        if self._reader is None:
            return OCRResult(text="", conf=0.0, points=None)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        try:
            lines = self._reader.ocr(image_rgb, cls=False)
        except Exception:
            return OCRResult(text="", conf=0.0, points=None)
        candidates: List[OCRResult] = []
        for line in lines or []:
            if not line or len(line) < 2:
                continue
            points = line[0]
            meta = line[1]
            if not isinstance(meta, (list, tuple)) or len(meta) < 2:
                continue
            text = str(meta[0]).strip()
            try:
                conf = float(meta[1])
            except Exception:
                conf = 0.0
            norm_points = self._normalise_points(points)
            candidates.append(OCRResult(text=text, conf=conf, points=norm_points))
        return self._pick_best(candidates)

    def _run_rapid(self, image_bgr: np.ndarray) -> OCRResult:
        if self._reader is None:
            return OCRResult(text="", conf=0.0, points=None)
        try:
            result = self._reader(image_bgr)
        except Exception:
            return OCRResult(text="", conf=0.0, points=None)
        data: Iterable[Any]
        if isinstance(result, tuple) and len(result) >= 1:
            data = result[0] or []
        else:
            data = result or []
        candidates: List[OCRResult] = []
        for entry in data:
            if not entry:
                continue
            text: str = ""
            conf: float = 0.0
            points: Optional[List[List[float]]] = None
            if isinstance(entry, (list, tuple)) and len(entry) >= 3 and isinstance(entry[0], (list, tuple, np.ndarray)):
                points = self._normalise_points(entry[0])
                text = str(entry[1]).strip()
                try:
                    conf = float(entry[2])
                except Exception:
                    conf = 0.0
            elif isinstance(entry, (list, tuple)):
                text = str(entry[0]).strip()
                if len(entry) > 1:
                    try:
                        conf = float(entry[1])
                    except Exception:
                        conf = 0.0
            else:
                text = str(entry).strip()
            candidates.append(OCRResult(text=text, conf=conf, points=points))
        return self._pick_best(candidates)

    def _pick_best(self, candidates: Sequence[OCRResult]) -> OCRResult:
        if not candidates:
            return OCRResult(text="", conf=0.0, points=None)
        best = max(candidates, key=lambda item: item.conf if item.text else -1.0)
        return best

    @staticmethod
    def _normalise_points(points: Any) -> Optional[List[List[float]]]:
        try:
            arr = np.asarray(points, dtype=float)
        except Exception:
            return None
        if arr.ndim != 2 or arr.shape[1] != 2:
            return None
        return arr.tolist()

    @staticmethod
    def _normalise_text(text: str) -> str:
        for token in [" ", "·", "•", ".", "-"]:
            cleaned = cleaned.replace(token, "")
        cleaned = cleaned.upper()
        return cleaned


__all__ = ["PlateOCR", "PlateOCRError"]
