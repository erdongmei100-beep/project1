"""License plate OCR wrapper supporting RapidOCR and PaddleOCR."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

try:  # pragma: no cover - optional dependency guard
    import cv2
except Exception as exc:  # pragma: no cover
    cv2 = None
    _CV_ERROR = repr(exc)
else:
    _CV_ERROR = ""

try:  # pragma: no cover - runtime dependency
    from rapidocr_onnxruntime import RapidOCR as _RapidOCR  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency guard
    _RapidOCR = None
    _RAPID_ERROR = repr(exc)
else:
    _RAPID_ERROR = ""

try:  # pragma: no cover - runtime dependency
    from paddleocr import PaddleOCR as _RawPaddleOCR  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency guard
    _RawPaddleOCR = None
    _PADDLE_ERROR = repr(exc)
else:
    _PADDLE_ERROR = ""


class _SafePaddleOCR(_RawPaddleOCR if _RawPaddleOCR else object):  # type: ignore[misc]
    """PaddleOCR proxy that rejects full-frame inputs for safety."""

    def ocr(self, img, *args, **kwargs):  # type: ignore[override]
        try:
            height = int(img.shape[0])
            width = int(img.shape[1])
            if height > 512 or width > 512:
                return []
        except Exception:
            pass
        return super().ocr(img, *args, **kwargs)  # type: ignore[misc]


_WARNED_MODEL_DIRS: set[str] = set()


def _check_paddle_dir(model_dir: str) -> Tuple[bool, str]:
    path = Path(model_dir)
    core_files = ["inference.pdmodel", "inference.pdiparams"]
    if not path.exists():
        return False, "dir_not_exist"
    if not all((path / name).exists() for name in core_files):
        return False, "core_files_missing"
    has_meta = any(
        (path / name).exists()
        for name in ("inference.json", "inference.yml", "inference.yaml")
    )
    return True, ("ok" if has_meta else "ok_no_meta")


class PlateOCR:
    """Unified OCR wrapper offering RapidOCR by default with Paddle fallback."""

    def __init__(
        self,
        lang: str = "ch",
        det: bool = False,
        rec: bool = True,
        use_angle_cls: bool = False,
        ocr_model_dir: Optional[str] = None,
        use_gpu: bool = False,
        log_csv_path: str = "runs/plates/plate_logs.csv",
        crops_dir: str = "runs/plates/crops",
        min_height: int = 96,
        write_empty: bool = True,
        min_conf: float = 0.25,
        engine: str = "rapidocr",
        **_: object,
    ) -> None:
        self.lang = lang
        self.det = bool(det)
        self.rec = bool(rec)
        self.use_angle_cls = bool(use_angle_cls)
        self.rec_model_dir = str(ocr_model_dir) if ocr_model_dir else ""
        self.use_gpu = bool(use_gpu)
        self.log_csv_path = log_csv_path
        self.crops_dir = crops_dir
        self.min_height = int(min_height)
        self.min_conf = float(min_conf)
        self.write_empty = bool(write_empty)
        self.ocr_engine = str(engine or "rapidocr").lower()

        self._reader = None
        self._reader_type = ""

        if cv2 is None:
            raise RuntimeError(f"OpenCV 未安装: {_CV_ERROR}")

        if self.ocr_engine == "rapidocr" and _RapidOCR is not None:
            self._init_rapidocr()
        elif self.ocr_engine == "paddle" and _RawPaddleOCR is not None:
            self._init_paddle()
        else:
            # Attempt automatic fallback.
            if _RapidOCR is not None:
                self.ocr_engine = "rapidocr"
                self._init_rapidocr()
            elif _RawPaddleOCR is not None:
                self.ocr_engine = "paddle"
                self._init_paddle()
            else:
                raise RuntimeError(
                    "OCR 引擎不可用：RapidOCR 初始化失败"
                    f"({_RAPID_ERROR}) 且 PaddleOCR 不可用({_PADDLE_ERROR})."
                )

    # ------------------------------------------------------------------
    # Engine initialisation helpers
    def _init_rapidocr(self) -> None:
        if _RapidOCR is None:
            raise RuntimeError(f"RapidOCR 未安装: {_RAPID_ERROR}")
        try:
            self._reader = _RapidOCR()
            self._reader_type = "rapidocr"
            print("[plate] Initialized RapidOCR engine")
        except Exception as exc:
            raise RuntimeError(f"RapidOCR 初始化失败: {exc}")

    def _init_paddle(self) -> None:
        if _RawPaddleOCR is None:
            raise RuntimeError(f"PaddleOCR 未安装: {_PADDLE_ERROR}")
        if not self.rec_model_dir:
            raise RuntimeError("PaddleOCR 需要提供 ocr_model_dir")
        ok, status = _check_paddle_dir(self.rec_model_dir)
        if not ok:
            raise RuntimeError(
                f"PaddleOCR 模型目录不完整({status})：{self.rec_model_dir}"
            )
        if status == "ok_no_meta" and self.rec_model_dir not in _WARNED_MODEL_DIRS:
            print(
                f"[WARN] OCR 模型缺少 meta 文件（json/yml），按默认配置加载：{self.rec_model_dir}"
            )
            _WARNED_MODEL_DIRS.add(self.rec_model_dir)
        try:
            self._reader = _SafePaddleOCR(
                lang=self.lang,
                det=self.det,
                rec=self.rec,
                use_angle_cls=self.use_angle_cls,
                rec_model_dir=self.rec_model_dir,
                use_gpu=self.use_gpu,
            )
            self._reader_type = "paddle"
            print("[plate] Initialized PaddleOCR engine")
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR 初始化失败: {exc}")

    # ------------------------------------------------------------------
    # Public API
    def read(self, img_bgr: Optional[np.ndarray]) -> Tuple[str, float]:
        """Return text/confidence tuple from a plate crop."""

        text, conf = self._recognize_internal(img_bgr)
        if not text:
            return "null", conf
        if conf < self.min_conf:
            return "null", conf
        return text, conf

    def recognize_crop(self, crop_bgr: Optional[np.ndarray]) -> Tuple[str, float]:
        """Legacy helper used by batch tools; returns raw text/conf pair."""

        return self._recognize_internal(crop_bgr)

    def recognize(
        self,
        crop_bgr: Optional[np.ndarray],
        bbox_xyxy: Optional[Sequence[int]] = None,
        frame_idx: Optional[int] = None,
        video_time_ms: Optional[int] = None,
        cam_id: Optional[str] = None,
        roi_flag: Optional[bool] = None,
    ) -> dict:
        text, conf = self._recognize_internal(crop_bgr)
        ok = bool(text) and conf >= self.min_conf
        result = {
            "ok": ok,
            "text": text if ok else "",
            "conf": float(conf),
            "bbox": tuple(bbox_xyxy) if bbox_xyxy is not None else None,
            "frame_idx": frame_idx,
            "video_time_ms": video_time_ms,
            "cam_id": cam_id,
            "roi_flag": roi_flag,
        }
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    def _recognize_internal(self, img_bgr: Optional[np.ndarray]) -> Tuple[str, float]:
        if img_bgr is None or getattr(img_bgr, "size", 0) == 0:
            return "", 0.0

        image = np.asarray(img_bgr)
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

        if self._reader_type == "rapidocr":
            return self._run_rapidocr(image)
        if self._reader_type == "paddle":
            return self._run_paddle(image)
        return "", 0.0

    def _run_rapidocr(self, image: np.ndarray) -> Tuple[str, float]:
        if _RapidOCR is None or self._reader is None:
            return "", 0.0
        height, width = image.shape[:2]
        if max(height, width) < max(self.min_height, 200):
            scale = max(1, int(round(max(self.min_height, 200) / max(height, width))))
            new_w = max(int(width * scale), 1)
            new_h = max(int(height * scale), 1)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        try:
            result, _ = self._reader(enhanced_bgr)
        except Exception:
            return "", 0.0
        if not result:
            return "", 0.0
        text = str(result[0][0]).strip()
        try:
            conf = float(result[0][1])
        except Exception:
            conf = 0.0
        return text, conf

    def _run_paddle(self, image: np.ndarray) -> Tuple[str, float]:
        if _RawPaddleOCR is None or self._reader is None:
            return "", 0.0
        if image.shape[0] < self.min_height:
            return "", 0.0
        try:
            result = self._reader.ocr(image, cls=False)
        except Exception:
            return "", 0.0
        if not result or not result[0]:
            return "", 0.0
        text = str(result[0][1][0]).strip()
        try:
            conf = float(result[0][1][1])
        except Exception:
            conf = 0.0
        return text, conf


__all__ = ["PlateOCR"]

