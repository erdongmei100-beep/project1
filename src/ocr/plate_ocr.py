# src/ocr/plate_ocr.py
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

# ---- import PaddleOCR (raw) ----
_IMPORT_ERROR = ""
try:
    from paddleocr import PaddleOCR as _RawPaddleOCR  # type: ignore
except Exception as e:
    _RawPaddleOCR = None  # type: ignore
    _IMPORT_ERROR = repr(e)


# ---- guard: forbid frame-level OCR by size ----
class _SafePaddleOCR(_RawPaddleOCR if _RawPaddleOCR else object):  # type: ignore
    """
    安全代理：如果传入图像任一边 > 512，认为是“整帧/大图”，直接返回空结果，避免误用。
    """

    def ocr(self, img, *args, **kwargs):  # type: ignore[override]
        try:
            h = int(img.shape[0])
            w = int(img.shape[1])
            if h > 512 or w > 512:
                return []  # 拒绝大图
        except Exception:
            # 不是ndarray也放行给上游处理
            pass
        return super().ocr(img, *args, **kwargs)  # type: ignore[misc]


# ---- model-dir check (allow no json) ----
def _check_dir_ok(d: str) -> Tuple[bool, str]:
    """
    必须包含: inference.pdmodel + inference.pdiparams
    meta 文件(json/yml/yaml)可选；缺失时给出软警告但允许继续。
    """
    p = Path(d)
    core = ["inference.pdmodel", "inference.pdiparams"]
    if not p.exists():
        return False, "dir_not_exist"
    if not all((p / f).exists() for f in core):
        return False, "core_files_missing"
    has_meta = any(
        (p / f).exists() for f in ["inference.json", "inference.yml", "inference.yaml"]
    )
    return True, ("ok" if has_meta else "ok_no_meta")


class PlateOCR:
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
    ):
        # 依赖检查
        if _RawPaddleOCR is None:
            raise RuntimeError(f"PaddleOCR 未安装: {_IMPORT_ERROR}")

        # 记录配置
        self.rec_model_dir = str(ocr_model_dir) if ocr_model_dir is not None else ""
        self.min_height = int(min_height)
        self.min_conf = float(min_conf)
        self.write_empty = bool(write_empty)
        self.log_csv_path = log_csv_path
        self.crops_dir = crops_dir

        # 模型目录校验（允许无 json）
        ok, why = _check_dir_ok(self.rec_model_dir)
        if not ok:
            raise RuntimeError(f"PaddleOCR 模型目录不完整({why})：{self.rec_model_dir}")
        if why == "ok_no_meta":
            print(
                f"[WARN] OCR 模型缺少 meta 文件（json/yml），按默认配置加载：{self.rec_model_dir}"
            )

        # 用安全代理，误传整帧也会被拦截为空
        self.ocr = _SafePaddleOCR(
            lang=lang,
            det=det,
            rec=rec,
            use_angle_cls=use_angle_cls,
            rec_model_dir=self.rec_model_dir,
            use_gpu=use_gpu,
        )

    # 仅对“已裁剪的车牌小图”做识别
    def recognize_crop(self, crop_bgr) -> Tuple[str, float]:
        """
        输入: BGR裁剪图 (ndarray)
        输出: (text, conf). 失败/太小返回("", 0.0)
        """
        if crop_bgr is None:
            return "", 0.0
        try:
            if getattr(crop_bgr, "size", 0) <= 0:
                return "", 0.0
            h = int(crop_bgr.shape[0])
            if h < self.min_height:
                return "", 0.0
            res = self.ocr.ocr(crop_bgr, cls=False)  # [[ [box], (text, score) ]]
            if not res or not res[0]:
                return "", 0.0
            item = res[0][0]
            text, conf = item[1][0], float(item[1][1])
            return text, conf
        except Exception:
            return "", 0.0
