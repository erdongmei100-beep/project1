"""通用工具函数集合，适配 HyperLPR3 官方 API，增加【分区域】模糊检测与强规则清洗。"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple, Optional

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# 尝试导入 HyperLPR3
try:
    import hyperlpr3 as lpr3
    # 初始化一个全局的 catcher 实例
    _CATCHER = lpr3.LicensePlateCatcher(detect_level=lpr3.DETECT_LEVEL_LOW)
    HAS_HYPERLPR = True
except ImportError:
    HAS_HYPERLPR = False
    _CATCHER = None

logger = logging.getLogger(__name__)

# --- 阈值设置 (可根据实际效果微调) ---
# 全局清晰度阈值 (整体不能太糊)
GLOBAL_BLUR_THRESHOLD = 70.0
# 【新增】省份区域(左侧1/7) 清晰度阈值
# 因为汉字笔画复杂，如果清晰，方差应该很高。如果糊了，方差会掉得很快。
# 建议比全局阈值稍低一点或持平，因为区域小。
PROVINCE_BLUR_THRESHOLD = 65.0

# 规则校验：有效省份 (必须是这些汉字开头)
VALID_PROVINCES = set("京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领")


def natural_sort_key(path: Path) -> List:
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", path.stem)]


def list_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        logger.warning(f"图片目录不存在: {image_dir}")
        return []
    images = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".png", ".jpeg", ".bmp"}],
        key=natural_sort_key,
    )
    return images


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def calculate_blur_score(image: np.ndarray) -> float:
    """计算图像拉普拉斯方差 (清晰度得分)。"""
    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def is_valid_plate_format(text: str) -> bool:
    """
    强规则校验：
    1. 长度 7-8 位
    2. 第一位必须是合法汉字省份
    """
    if not text:
        return False
    # 长度校验
    if not (7 <= len(text) <= 8):
        return False
    # 首字符校验 (拦截 AMCP759 这种纯字母开头的)
    if text[0] not in VALID_PROVINCES:
        return False
    return True


def run_hyperlpr(image: np.ndarray) -> Tuple[str, float, Optional[Tuple[int, int, int, int]]]:
    """
    运行识别 -> 扣图 -> 分区域测模糊 -> 强校验 -> 返回结果
    """
    if not HAS_HYPERLPR or _CATCHER is None:
        raise ImportError("hyperlpr3 未安装或初始化失败。")

    # 1. 模型推理
    results = _CATCHER(image)
    if not results:
        return "", 0.0, None

    best = max(results, key=lambda r: r[1])
    plate_text = best[0]
    plate_conf = float(best[1])
    bbox = tuple(map(int, best[3])) # [x1, y1, x2, y2]

    # --- 阶段 1：利用 bbox 扣取车牌小图 ---
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    # 稍微做一点边界检查
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    plate_crop = image[y1:y2, x1:x2]
    if plate_crop.size == 0:
        return "", 0.0, None

    # --- 阶段 2：强规则校验 (优先拦截格式不对的，比如没有汉字的) ---
    if not is_valid_plate_format(plate_text):
        logger.warning(f"⛔ 拦截非法格式: {plate_text} (首字非省份汉字或长度不对)")
        return "", 0.0, None

    # --- 阶段 3：分区域模糊检测 ---
    # 3.1 全局模糊检测
    global_score = calculate_blur_score(plate_crop)
    
    # 3.2 【核心改进】左侧省份区域模糊检测
    # 假设车牌左侧 1/7 区域是汉字 (约 14%)
    crop_w = plate_crop.shape[1]
    split_idx = int(crop_w * 0.16) # 取前 16%
    province_crop = plate_crop[:, :split_idx]
    
    province_score = calculate_blur_score(province_crop)

    # 日志输出两个分数，方便调试
    logger.info(f"分析: {plate_text} | 全局分: {global_score:.1f} | 汉字区分: {province_score:.1f}")

    # 3.3 联合判定
    # 如果全局太糊，或者 【汉字区特别糊】
    if global_score < GLOBAL_BLUR_THRESHOLD:
        logger.warning(f"⛔ 拦截(全局模糊): {plate_text} (Score: {global_score:.1f})")
        return "", 0.0, None
        
    if province_score < PROVINCE_BLUR_THRESHOLD:
        logger.warning(f"⛔ 拦截(汉字模糊): {plate_text} (汉字区分数 {province_score:.1f} 过低)")
        return "", 0.0, None

    # 全部通过
    return plate_text, plate_conf, bbox


def draw_chinese_text(image: np.ndarray, text: str, position: Tuple[int, int], font_path: Path, color: Tuple[int, int, int] = (0, 255, 0), size: int = 30) -> np.ndarray:
    if not isinstance(image, np.ndarray): return image
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    try:
        font = ImageFont.truetype(str(font_path), size)
    except OSError:
        font = ImageFont.load_default()
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.asarray(img_pil), cv2.COLOR_RGB2BGR)


def crop_plate(image: np.ndarray, bbox: Tuple[int, int, int, int], output_path: Path) -> None:
    if bbox is None: return
    x1, y1, x2, y2 = bbox
    h, w = image.shape[:2]
    pad_h, pad_w = int((y2 - y1) * 0.1), int((x2 - x1) * 0.1)
    x1, y1, x2, y2 = max(0, x1 - pad_w), max(0, y1 - pad_h), min(w, x2 + pad_w), min(h, y2 + pad_h)
    crop = image[y1:y2, x1:x2]
    if crop.size > 0:
        ensure_dir(output_path.parent)
        cv2.imwrite(str(output_path), crop)