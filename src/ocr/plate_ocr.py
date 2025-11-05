# src/ocr/plate_ocr.py
import os
import csv
import time
import uuid
from typing import Optional, Tuple, Dict, Any
try:
    import cv2
except Exception as exc:  # pragma: no cover - environment specific dependency
    cv2 = None
    _CV_IMPORT_ERROR = exc
else:
    _CV_IMPORT_ERROR = None

try:
    from paddleocr import PaddleOCR
except Exception as e:
    PaddleOCR = None
    _IMPORT_ERROR = e
else:
    _IMPORT_ERROR = None


class PlateOCR:
    """
    封装 PaddleOCR 的车牌文本识别，并提供同时保存图片与记录 CSV 日志的便捷接口。
    """
    def __init__(
        self,
        lang: str = "ch",
        use_angle_cls: bool = False,
        rec: bool = True,
        det: bool = False,
        ocr_model_dir: Optional[str] = None,
        log_csv_path: str = "runs/plates/plate_logs.csv",
        crops_dir: str = "runs/plates/crops",
        ensure_dirs: bool = True
    ):
        if cv2 is None:
            raise RuntimeError(
                "OpenCV 未安装或缺少运行时依赖：{}".format(_CV_IMPORT_ERROR)
            )

        if PaddleOCR is None:
            raise RuntimeError(
                "PaddleOCR 未安装：{}。请先安装 paddleocr 与 paddlepaddle / paddlepaddle-gpu".format(_IMPORT_ERROR)
            )

        # 只做识别（det=False），定位由外部检测器完成
        self.ocr = PaddleOCR(
            lang=lang,
            det=det,
            rec=rec,
            use_angle_cls=use_angle_cls,
            rec_model_dir=ocr_model_dir if ocr_model_dir else None,
            show_log=False
        )
        self.log_csv_path = log_csv_path
        self.crops_dir = crops_dir
        if ensure_dirs:
            os.makedirs(os.path.dirname(self.log_csv_path), exist_ok=True)
            os.makedirs(self.crops_dir, exist_ok=True)

        # 若日志文件不存在，写入表头
        if not os.path.exists(self.log_csv_path):
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "event_id",          # 唯一事件ID（每次检测一次生成）
                    "timestamp",         # UNIX时间戳(s)
                    "video_time_ms",     # 可选：视频毫秒时间（没用到可留空）
                    "frame_idx",         # 帧号（无则写-1）
                    "plate_text",        # 识别文本
                    "rec_confidence",    # 识别置信度(0~1)
                    "image_path",        # 裁剪图保存路径
                    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",   # 车牌检测框（像素）
                    "cam_id",            # 摄像头/通道ID
                    "roi_flag",          # 是否处于ROI（应急车道），True/False/Unknown
                ])

    @staticmethod
    def _preprocess(crop_bgr):
        """可选的前处理：放大、去噪、增强等。默认仅保证最小高度。"""
        if crop_bgr is None or crop_bgr.size == 0 or cv2 is None:
            return crop_bgr
        h, w = crop_bgr.shape[:2]
        min_h = 48  # OCR 对小文字更友好
        if h < min_h:
            scale = min_h / max(1, h)
            crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
        # 可选：添加 CLAHE、锐化等
        return crop_bgr

    def recognize(
        self,
        crop_bgr,
        bbox_xyxy: Tuple[int, int, int, int],
        frame_idx: int = -1,
        video_time_ms: Optional[int] = None,
        cam_id: Optional[str] = None,
        roi_flag: Optional[bool] = None,
        save_crop: bool = True
    ) -> Dict[str, Any]:
        """
        对单个车牌裁剪图进行识别，并可选保存图片与记录日志。
        返回 dict，包含文本、置信度、保存路径等信息。
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {"ok": False, "err": "empty_crop"}

        crop_bgr = self._preprocess(crop_bgr)
        # PaddleOCR 接收BGR/路径均可；这里直接传 ndarray
        rec = self.ocr.ocr(crop_bgr, cls=False)
        plate_text, plate_conf = "", 0.0
        if rec and rec[0]:
            # 取首个候选
            plate_text = rec[0][0][1][0]
            plate_conf = float(rec[0][0][1][1])

        # 保存裁剪图
        image_path = ""
        if save_crop:
            event_id = str(uuid.uuid4())[:8]
            ts = int(time.time())
            image_name = f"{ts}_{event_id}_{plate_text if plate_text else 'plate'}.jpg"
            image_path = os.path.join(self.crops_dir, image_name)
            cv2.imwrite(image_path, crop_bgr)

        # 记录日志
        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                str(uuid.uuid4())[:8],          # event_id
                int(time.time()),               # timestamp
                int(video_time_ms) if video_time_ms is not None else "",   # video_time_ms
                int(frame_idx) if frame_idx is not None else -1,           # frame_idx
                plate_text,
                round(plate_conf, 6),
                image_path,
                int(bbox_xyxy[0]), int(bbox_xyxy[1]), int(bbox_xyxy[2]), int(bbox_xyxy[3]),
                str(cam_id) if cam_id is not None else "",
                str(roi_flag) if roi_flag is not None else "Unknown",
            ])

        return {
            "ok": True,
            "text": plate_text,
            "conf": plate_conf,
            "image_path": image_path
        }
