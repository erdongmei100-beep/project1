"""Core pipeline for running HyperLPR3 license plate recognition on CSV files."""

from __future__ import annotations

import logging
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# 扁平化引用
from config import (
    DEFAULT_PLATE_COLUMN,
    DEFAULT_PLATE_IMAGE_COLUMN,
    DEFAULT_PROGRESS_INTERVAL,
)
from utils import (
    list_images,     # 使用 utils 中现有的排序逻辑
    run_hyperlpr,    # 使用 utils 中现有的识别函数
    crop_plate,      # 使用 utils 中现有的裁剪函数
    ensure_dir,
)

logger = logging.getLogger("plate_recognition")
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def recognize_plates_for_csv(
    csv_path: str,
    images_dir: str,
    output_csv_path: str,
    *,
    # 兼容性参数
    image_column: str = "image_name",
    plate_column: str = DEFAULT_PLATE_COLUMN,
    log_path: str | None = None,
    device: str = "auto", # utils.py 暂不支持手动选设备，此参数仅作占位
    save_plate_images: bool = False,
    plate_image_dir: str = "outputs/plates",
    plate_image_column: str = DEFAULT_PLATE_IMAGE_COLUMN,
    progress_interval: int = DEFAULT_PROGRESS_INTERVAL,
) -> None:
    """
    读取 CSV 与图片目录，通过【自然排序】将图片与 CSV 行一一对应。
    """
    
    # 1. 读取 CSV
    logger.info(f"读取 CSV 文件: {csv_path}")
    df = pd.read_csv(csv_path)
    total_rows = len(df)

    # 2. 读取并【排序】图片 (直接调用 utils.py 的逻辑)
    images_dir_path = Path(images_dir)
    logger.info(f"扫描图片目录: {images_dir}")
    
    # list_images 内部已经包含了 natural_sort_key 排序
    sorted_images = list_images(images_dir_path) 
    total_images = len(sorted_images)
    
    logger.info(f"找到 {total_images} 张图片 (已按自然顺序排序)")

    # 3. 【关键】对齐校验
    if total_rows != total_images:
        logger.error(f"严重错误：CSV行数 ({total_rows}) 与 图片数量 ({total_images}) 不一致！")
        raise AssertionError(f"Data count mismatch: CSV has {total_rows} rows, Images dir has {total_images} files.")
    else:
        logger.info("校验通过：CSV 行数与图片数量一致，准备开始合并处理。")

    # 准备新列
    df["source_image_filename"] = "" 
    df[plate_column] = ""
    if save_plate_images:
        ensure_dir(Path(plate_image_dir))
        df[plate_image_column] = ""

    success_count = 0
    failure_count = 0

    # 4. 遍历处理
    for idx, image_path_obj in zip(df.index, sorted_images):
        image_filename = image_path_obj.name
        full_image_path = str(image_path_obj)
        
        # 记录图片文件名
        df.at[idx, "source_image_filename"] = image_filename

        try:
            # 读取图片
            image = cv2.imread(full_image_path)
            if image is None:
                logger.warning(f"无法读取图片: {full_image_path}")
                continue

            # 识别 (调用 utils.py 的 run_hyperlpr)
            # 返回值: plate_text, plate_conf, bbox
            text, conf, bbox = run_hyperlpr(image)

            if text:
                df.at[idx, plate_column] = text
                success_count += 1
                
                # 保存裁剪小图
                if save_plate_images and bbox:
                    try:
                        stem = image_path_obj.stem
                        crop_filename = f"{stem}_plate_{text}.jpg"
                        crop_path = Path(plate_image_dir) / crop_filename
                        
                        # 调用 utils.py 的 crop_plate
                        crop_plate(image, bbox, crop_path)
                        
                        df.at[idx, plate_image_column] = str(crop_path)
                    except Exception as e:
                        logger.warning(f"裁剪保存失败: {e}")
            else:
                failure_count += 1

        except Exception as e:
            logger.exception(f"处理异常 (行 {idx}): {e}")
            failure_count += 1
            continue

        if progress_interval and (idx + 1) % progress_interval == 0:
            logger.info(f"已处理 {idx + 1}/{total_rows} 行...")

    # 5. 保存结果
    df.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    logger.info("-" * 40)
    logger.info(f"处理完成！结果已保存至: {output_csv_path}")
    logger.info(f"成功识别: {success_count} | 未识别: {failure_count}")