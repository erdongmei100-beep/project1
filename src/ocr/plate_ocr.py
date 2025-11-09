import csv
import inspect
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2

try:
    from paddleocr import PaddleOCR as _RawPaddleOCR
except Exception as e:
    _RawPaddleOCR = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


# --- BEGIN: hard guard to ban frame-level OCR ---
class _SafePaddleOCR(_RawPaddleOCR if _RawPaddleOCR else object):
    """
    拦截对大图(整帧)的 OCR 调用：若图像任一边 > 512 像素，直接返回空结果。
    这样即使上层误传整帧，也不会耗时/产出垃圾。
    """

    def ocr(self, img, *args, **kwargs):  # type: ignore[override]
        try:
            h, w = int(img.shape[0]), int(img.shape[1])
            if h > 512 or w > 512:
                return []  # 拒绝大图
        except Exception:
            pass
        if _RawPaddleOCR is None:
            return []
        return super().ocr(img, *args, **kwargs)  # type: ignore[misc]


# --- END ---


def _check_dir_ok(d: str):
    if not d:
        return False, "dir_not_exist"
    p = Path(d)
    core_files = ["inference.pdmodel", "inference.pdiparams"]
    if not p.exists():
        return False, "dir_not_exist"
    if not all((p / f).exists() for f in core_files):
        return False, "core_files_missing"
    has_meta = any((p / f).exists() for f in ["inference.json", "inference.yml", "inference.yaml"])
    return True, ("ok" if has_meta else "ok_no_meta")


def _ensure_dir(p):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)


def _unsharp(img):
    blur = cv2.GaussianBlur(img, (0, 0), 1.0)
    return cv2.addWeighted(img, 1.5, blur, -0.5, 0)


def _clahe(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


class PlateOCR:
    _warned_missing_meta = False

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
        if _RawPaddleOCR is None:
            raise RuntimeError(f"PaddleOCR 未安装：{_IMPORT_ERROR}")

        self.rec_model_dir = str(ocr_model_dir) if ocr_model_dir is not None else ""
        ok, why = _check_dir_ok(self.rec_model_dir)
        if not ok:
            raise RuntimeError(
                f"PaddleOCR 模型目录不完整({why})：{self.rec_model_dir}"
            )
        if why == "ok_no_meta" and not PlateOCR._warned_missing_meta:
            print(
                f"[WARN] OCR 模型缺少 meta 文件（json/yml），将按默认配置加载：{self.rec_model_dir}"
            )
            PlateOCR._warned_missing_meta = True

        init_params = inspect.signature(_RawPaddleOCR).parameters
        kwargs = dict(
            lang=lang,
            det=det,
            rec=rec,
            use_angle_cls=use_angle_cls,
            rec_model_dir=self.rec_model_dir,
            use_gpu=bool(use_gpu),
        )
        kwargs = {k: v for k, v in kwargs.items() if k in init_params}
        self.ocr = _SafePaddleOCR(**kwargs)

        self.log_csv_path = log_csv_path
        self.crops_dir = crops_dir
        _ensure_dir(os.path.dirname(self.log_csv_path))
        _ensure_dir(self.crops_dir)

        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        "event_id",
                        "timestamp",
                        "video_time_ms",
                        "frame_idx",
                        "plate_text",
                        "rec_confidence",
                        "image_path",
                        "bbox_x1",
                        "bbox_y1",
                        "bbox_x2",
                        "bbox_y2",
                        "cam_id",
                        "roi_flag",
                        "reason",
                    ]
                )

        self.min_height = int(min_height)
        self.min_conf = float(min_conf)
        self.write_empty = bool(write_empty)

    def _preprocess(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        if h < self.min_height:
            scale = self.min_height / max(1, h)
            crop_bgr = cv2.resize(
                crop_bgr,
                (int(w * scale), int(h * scale)),
                interpolation=cv2.INTER_CUBIC,
            )
        crop_bgr = _clahe(crop_bgr)
        crop_bgr = _unsharp(crop_bgr)
        return crop_bgr

    def recognize_crop(self, crop_bgr):
        """
        仅对一张已裁剪的车牌图做识别；不保存文件，不写日志。
        返回: (text:str, conf:float)
        """
        if crop_bgr is None or getattr(crop_bgr, "size", 0) == 0:
            return "", 0.0
        h = int(crop_bgr.shape[0])
        if h < self.min_height:
            return "", 0.0
        try:
            res = self.ocr.ocr(crop_bgr, cls=False)
            if not res or not res[0]:
                return "", 0.0
            item = res[0][0]  # [[box], (text, score)]
            text, conf = item[1][0], float(item[1][1])
            return text, conf
        except Exception:
            return "", 0.0

    def recognize(
        self,
        crop_bgr,
        bbox_xyxy: Tuple[int, int, int, int],
        frame_idx: int = -1,
        video_time_ms: Optional[int] = None,
        cam_id: Optional[str] = None,
        roi_flag: Optional[bool] = None,
        save_crop: bool = True,
    ) -> Dict[str, Any]:
        reason = ""
        if crop_bgr is None or crop_bgr.size == 0:
            reason = "empty_crop"
            return self._log(
                "",
                0.0,
                "",
                bbox_xyxy,
                frame_idx,
                video_time_ms,
                cam_id,
                roi_flag,
                reason,
            )

        crop_bgr = self._preprocess(crop_bgr)
        rec = self.ocr.ocr(crop_bgr, cls=False)
        text, conf = "", 0.0
        if rec and rec[0]:
            text = rec[0][0][1][0]
            conf = float(rec[0][0][1][1])
            if conf < self.min_conf:
                reason = "low_conf"
        else:
            reason = "no_text"

        image_path = ""
        if save_crop:
            eid = str(uuid.uuid4())[:8]
            ts = int(time.time())
            safe_txt = text if text else "plate"
            safe_txt = re.sub(r"[^0-9A-Za-z\u4e00-\u9fa5]+", "", safe_txt) or "plate"
            image_path = os.path.join(
                self.crops_dir, f"{ts}_{eid}_{safe_txt}.jpg"
            )
            cv2.imwrite(image_path, crop_bgr)

        return self._log(
            text,
            conf,
            image_path,
            bbox_xyxy,
            frame_idx,
            video_time_ms,
            cam_id,
            roi_flag,
            reason,
        )

    def _log(
        self,
        text,
        conf,
        image_path,
        bbox_xyxy,
        frame_idx,
        video_time_ms,
        cam_id,
        roi_flag,
        reason,
    ):
        if self.write_empty or text:
            with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    [
                        str(uuid.uuid4())[:8],
                        int(time.time()),
                        int(video_time_ms) if video_time_ms is not None else "",
                        int(frame_idx) if frame_idx is not None else -1,
                        text,
                        round(conf, 6),
                        image_path,
                        int(bbox_xyxy[0]),
                        int(bbox_xyxy[1]),
                        int(bbox_xyxy[2]),
                        int(bbox_xyxy[3]),
                        str(cam_id) if cam_id is not None else "",
                        str(roi_flag) if roi_flag is not None else "Unknown",
                        reason,
                    ]
                )
        return {
            "ok": bool(text),
            "text": text,
            "conf": conf,
            "image_path": image_path,
            "reason": reason,
        }
