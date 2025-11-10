# modules/lp_pipeline.py
import glob
import os
import re
import tempfile
import urllib.request
from pathlib import Path
from typing import BinaryIO, Iterable, List, Tuple, Optional

import cv2
import numpy as np
import torch

PROV = r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼]"
PLATE_PATTERNS = [
    re.compile(rf"{PROV}[A-Z][A-HJ-NP-Z0-9]{{5}}"),   # 常规6位（去除I/O）
    re.compile(rf"{PROV}[A-Z][A-HJ-NP-Z0-9]{{6}}"),   # 新能源等7位（宽松）
]

_ENV_URL_KEYS = ("PLATE_YOLOV5_URL", "YOLOV5_PLATE_URL", "PLATE_WEIGHTS_URL")
DEFAULT_LOCAL_WEIGHTS_URL = (
    "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt"
)

_CHUNK_SIZE = 1 << 15  # 32 KiB


def _iter_download_chunks(response: BinaryIO) -> Iterable[bytes]:
    while True:
        chunk = response.read(_CHUNK_SIZE)
        if not chunk:
            break
        yield chunk


def is_cn_plate(text: str) -> bool:
    t = (text or "").strip().upper().replace("·", "").replace(" ", "")
    return any(p.fullmatch(t) for p in PLATE_PATTERNS)


def _resolve_download_url(candidate: Optional[str]) -> str:
    if candidate and str(candidate).strip():
        return str(candidate).strip()
    for key in _ENV_URL_KEYS:
        val = os.getenv(key)
        if val:
            return val.strip()
    return DEFAULT_LOCAL_WEIGHTS_URL


def _ensure_local_weights(weights: str, download_url: Optional[str]) -> Path:
    weight_path = Path(weights)
    if weight_path.exists():
        return weight_path

    resolved_url = _resolve_download_url(download_url)
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    if not resolved_url:
        raise RuntimeError(
            "本地 YOLOv5 权重缺失且未提供下载链接。"
            " 请通过 --download_url 或环境变量 PLATE_YOLOV5_URL 指定。"
        )

    print(
        f"[lp_pipeline] 本地 YOLOv5 权重缺失，将从 {resolved_url} 下载至 {weight_path}"  # noqa: T201
    )
    try:
        with urllib.request.urlopen(resolved_url) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            with tempfile.NamedTemporaryFile(
                dir=str(weight_path.parent), delete=False
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                try:
                    for chunk in _iter_download_chunks(response):
                        tmp_file.write(chunk)
                        downloaded += len(chunk)
                        if total:
                            percent = downloaded / total * 100
                            print(
                                f"[lp_pipeline] 下载进度: {percent:5.1f}%",
                                end="\r",
                                flush=True,
                            )
                except Exception:
                    tmp_path.unlink(missing_ok=True)
                    raise
            if total:
                print(" " * 40, end="\r", flush=True)
            if weight_path.exists():
                weight_path.unlink(missing_ok=True)
            tmp_path.replace(weight_path)
    except Exception as exc:  # pragma: no cover - 网络异常
        raise RuntimeError(
            f"下载 YOLOv5 权重失败: {exc}. 请检查网络或手动放置 {weight_path}"
        ) from exc
    if weight_path.stat().st_size == 0:
        weight_path.unlink(missing_ok=True)
        raise RuntimeError("下载的 YOLOv5 权重大小为 0，请检查下载链接是否正确。")
    return weight_path


def load_yolo(
    use_hub: bool,
    weights: str,
    img_size: int,
    conf: float,
    iou: float,
    download_url: Optional[str] = None,
):
    """
    返回可调用的 YOLO 模型对象。Hub 方式首选；若 use_hub=False 则尝试本地 yolov5。
    """
    if use_hub:
        model = torch.hub.load(
            "ultralytics/yolov5", "custom", path=weights, source="github"
        )  # 首次需网络
        model.conf = conf
        model.iou = iou
        model.max_det = 10
        model.classes = None  # 只要权重是单类plate则不必指定
        return model, "hub"
    # 本地导入
    import sys

    local_repo = Path("yolov5")
    if not local_repo.exists():
        raise RuntimeError("未找到本地 yolov5/ 目录。请执行 git submodule 或设置 --use_hub 1")
    if str(local_repo) not in sys.path:
        sys.path.append(str(local_repo))

    from models.common import DetectMultiBackend  # type: ignore
    from utils.augmentations import letterbox  # type: ignore
    from utils.general import check_img_size  # type: ignore
    from utils.torch_utils import select_device  # type: ignore

    weights_path = _ensure_local_weights(weights, download_url)
    device = select_device("")
    model = DetectMultiBackend(str(weights_path), device=device, dnn=False, fp16=False)
    stride = int(model.stride.max()) if hasattr(model, "stride") else 32
    checked_img_size = check_img_size(img_size, s=stride)
    model.warmup(imgsz=(1, 3, checked_img_size, checked_img_size))
    # 记录配置，供下游前后处理使用
    model._img_size = checked_img_size
    model._conf = conf
    model._iou = iou
    model._letterbox = letterbox
    return model, "local"


def detect_plates(model, img_bgr: np.ndarray, img_size: int) -> List[List[int]]:
    """
    返回 [x1,y1,x2,y2,conf] 列表（像素坐标，int）。
    """
    if hasattr(model, "names") and "detectmultibackend" not in str(type(model)).lower():
        res = model(img_bgr, size=img_size)
        det = res.xyxy[0].cpu().numpy() if len(res.xyxy) else np.empty((0, 6))
        out = []
        for x1, y1, x2, y2, conf, _cls in det:
            out.append([int(x1), int(y1), int(x2), int(y2), float(conf)])
        return out

    model_type = str(type(model)).lower()
    if "detectmultibackend" in model_type:
        from utils.general import non_max_suppression, scale_boxes  # type: ignore

        stride = int(getattr(model, "stride", [32])[0]) if hasattr(model, "stride") else 32
        conf = float(getattr(model, "_conf", 0.25))
        iou = float(getattr(model, "_iou", 0.45))
        target_size = int(getattr(model, "_img_size", img_size))
        letterbox = getattr(model, "_letterbox", None)
        if letterbox is None:
            from utils.augmentations import letterbox as _letterbox  # type: ignore

            letterbox = _letterbox
        img = letterbox(img_bgr, target_size, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(model.device)
        im = im.half() if getattr(model, "fp16", False) else im.float()
        im /= 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        preds = non_max_suppression(pred, conf, iou, max_det=10)
        out: List[List[int]] = []
        det = preds[0]
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], img_bgr.shape).round()
            for x1, y1, x2, y2, conf_score, _cls in det.tolist():
                out.append([int(x1), int(y1), int(x2), int(y2), float(conf_score)])
        return out

    raise RuntimeError("Unsupported YOLO model type for plate detection")


def expand_box(x1, y1, x2, y2, w, h, ratio=0.10):
    bw, bh = x2 - x1, y2 - y1
    dx, dy = int(bw * ratio), int(bh * ratio)
    nx1 = max(0, x1 - dx)
    ny1 = max(0, y1 - dy)
    nx2 = min(w - 1, x2 + dx)
    ny2 = min(h - 1, y2 + dy)
    return nx1, ny1, nx2, ny2


def deskew_by_minAreaRect(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    coords = np.column_stack(np.where(edges > 0))
    if coords.size < 10:
        return plate_bgr
    rect = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = plate_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv2.warpAffine(
        plate_bgr,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return rotated


def _standardize_plate(crop: np.ndarray, target_h=64, aspect=3.2) -> np.ndarray:
    scale = target_h / max(1, crop.shape[0])
    nw, nh = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
    resized = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_LINEAR)
    target_w = int(target_h * aspect)
    canvas = np.full((target_h, target_w, 3), 255, dtype=np.uint8)
    x_off = max(0, (target_w - nw) // 2)
    y_off = max(0, (target_h - nh) // 2)
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def load_hyperlpr():
    import hyperlpr3 as lpr3

    return lpr3.LicensePlateCatcher()


def recognize_plate(lpr, plate_bgr: np.ndarray) -> Tuple[Optional[str], float]:
    try:
        results = lpr(plate_bgr)
        if not results:
            return None, 0.0
        best = results[0]
        if isinstance(best, (list, tuple)) and len(best) >= 2:
            text, conf = str(best[0]), float(best[1])
        elif isinstance(best, dict):
            text = str(best.get("text", ""))
            conf = float(best.get("confidence", best.get("score", 0.0)))
        else:
            text, conf = str(best), 0.0
        text = (text or "").strip()
        if is_cn_plate(text):
            conf += 0.05  # 合规加权
        return text, conf
    except Exception:
        return None, 0.0


def process_vehicle_folder(
    vehicle_dir: str,
    out_dir: str = "runs/plates",
    yolo_weights: str = "weights/plate_best.pt",
    img_size: int = 640,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    expand_ratio: float = 0.10,
    save_candidates: bool = True,
    use_hub: bool = True,
    download_url: Optional[str] = None,
):
    os.makedirs(out_dir, exist_ok=True)
    yolo, _ = load_yolo(
        use_hub,
        yolo_weights,
        img_size,
        conf_thres,
        iou_thres,
        download_url=download_url,
    )
    lpr = load_hyperlpr()

    out_csv = os.path.join(out_dir, "results.csv")
    lines = ["filename,plate_text,conf,box\n"]

    imgs = sorted(glob.glob(os.path.join(vehicle_dir, "*.*")))
    for p in imgs:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        dets = detect_plates(yolo, img, img_size)
        if not dets:
            lines.append(f"{os.path.basename(p)},,0,\n")
            continue

        best: Tuple[str, float, Optional[Tuple[int, int, int, int]]] = ("", 0.0, None)
        for i, (x1, y1, x2, y2, s) in enumerate(dets):
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, ratio=expand_ratio)
            crop = img[y1:y2, x1:x2].copy()
            crop = deskew_by_minAreaRect(crop)
            std = _standardize_plate(crop)

            text, conf = recognize_plate(lpr, std)
            if text and conf > best[1]:
                best = (text, conf, (x1, y1, x2, y2))

            if save_candidates:
                cv2.imwrite(os.path.join(out_dir, f"{os.path.basename(p)}_cand{i}.jpg"), std)

        plate_text, plate_conf, box = best
        box_str = "" if box is None else f"{box[0]} {box[1]} {box[2]} {box[3]}"
        lines.append(f"{os.path.basename(p)},{plate_text},{plate_conf:.4f},{box_str}\n")

        if box is not None:
            x1, y1, x2, y2 = box
            vis = img.copy()
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"{plate_text} {plate_conf:.2f}",
                (x1, max(0, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )
            cv2.imwrite(os.path.join(out_dir, f"{os.path.basename(p)}_vis.jpg"), vis)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.writelines(lines)
    return out_csv
