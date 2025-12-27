"""Streamlit review terminal for emergency lane violations."""
from __future__ import annotations

import os
import ast
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

# ================= 配置区域 =================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
CSV_NAME = "events_with_plate.csv"
SESSION_KEY = "review_tasks"

# 核心字段映射
COLS = {
    "img_path": "best_frame_path",
    "plate_crop": "plate_crop",       
    "plate_text": "plate_text",
    "plate_score": "plate_score",
    "lpr_status": "lpr_status",
    "bbox": "plate_bbox"             # 读取坐标用于动态裁切
}

# ================= 样式注入 (CSS) =================
def inject_custom_css():
    st.markdown("""
        <style>
        /* 状态标签样式 */
        .status-badge-ok {
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            border: 1px solid #c3e6cb;
        }
        .status-badge-fail {
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            border: 1px solid #f5c6cb;
        }
        /* 复核状态指示器 */
        .review-status-yes {
            color: #28a745;
            font-size: 1.2em;
            font-weight: bold;
        }
        .review-status-no {
            color: #dc3545;
            font-size: 1.2em;
            font-weight: bold;
        }
        .review-status-exclude {
            color: #004085;
            font-size: 1.2em;
            font-weight: bold;
        }
        /* 不可编辑的灰字样式 */
        .read-only-text {
            color: #6c757d;
            font-size: 14px;
            margin-top: 5px;
            font-family: monospace;
        }
        </style>
    """, unsafe_allow_html=True)

# ================= 功能函数 =================

def discover_tasks(outputs_dir: Path) -> List[Tuple[str, Path]]:
    tasks = []
    if not outputs_dir.exists():
        return tasks
    for entry in sorted(outputs_dir.iterdir()):
        if not entry.is_dir(): continue
        csv_path = entry / CSV_NAME
        if csv_path.is_file():
            tasks.append((entry.name, csv_path))
    return tasks

def initialize_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    
    # 兼容处理
    if "plate_crop_path" in df.columns and COLS["plate_crop"] not in df.columns:
        df.rename(columns={"plate_crop_path": COLS["plate_crop"]}, inplace=True)

    # 确保置信度列为数值型
    if COLS["plate_score"] in df.columns:
        df[COLS["plate_score"]] = pd.to_numeric(df[COLS["plate_score"]], errors='coerce')

    # 初始化辅助列
    if "reviewed" not in df.columns:
        df["reviewed"] = False
    else:
        df["reviewed"] = df["reviewed"].astype(bool).fillna(False)

    if "manual_plate" not in df.columns:
        df["manual_plate"] = df.get(COLS["plate_text"], "").fillna("")
    
    if "is_excluded" not in df.columns:
        df["is_excluded"] = False
    else:
        df["is_excluded"] = df["is_excluded"].astype(bool).fillna(False)

    return df

def get_task_data(task_name: str, csv_path: Path) -> Dict:
    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = {}
    
    if task_name not in st.session_state[SESSION_KEY]:
        st.session_state[SESSION_KEY][task_name] = {
            "csv_path": csv_path,
            "df": initialize_dataframe(csv_path),
            "index": 0,
        }
    return st.session_state[SESSION_KEY][task_name]

def load_image_robust(path_str):
    if not path_str or str(path_str).lower() == 'nan': return None
    clean_path = str(path_str).strip().replace('"', '')
    path_obj = Path(clean_path)
    if path_obj.is_file():
        try:
            return Image.open(path_obj)
        except:
            return None
    return None

def crop_plate_dynamic(full_img, bbox_str):
    """如果硬盘上没有特写图，就根据坐标现场切一个"""
    try:
        bbox = ast.literal_eval(bbox_str)
        if isinstance(bbox, list) and len(bbox) == 4:
            x1, y1, x2, y2 = map(int, bbox)
            padding = 5
            width, height = full_img.size
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            return full_img.crop((x1, y1, x2, y2))
    except Exception:
        return None
    return None

# ================= 主程序 =================
def main():
    st.set_page_config(page_title="违规复核终端", page_icon="🚓", layout="wide")
    inject_custom_css()
    st.title("🚓 应急车道违规复核终端")

    tasks = discover_tasks(OUTPUTS_DIR)
    if not tasks:
        st.error("未找到任务数据 (data/