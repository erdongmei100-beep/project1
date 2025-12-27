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
    "bbox": "plate_bbox"             # 新增：读取坐标用于动态裁切
}

# ================= 样式注入 (CSS) =================
# Streamlit 原生不支持按钮变蓝变绿，需要注入 CSS 魔法
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

    # 初始化辅助列
    if "reviewed" not in df.columns:
        df["reviewed"] = False
    else:
        df["reviewed"] = df["reviewed"].astype(bool).fillna(False)

    if "manual_plate" not in df.columns:
        df["manual_plate"] = df.get(COLS["plate_text"], "").fillna("")
    
    # 初始化排除列 (Exclude)
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
        # bbox_str 格式通常是 "[x1, y1, x2, y2]"
        bbox = ast.literal_eval(bbox_str)
        if isinstance(bbox, list) and len(bbox) == 4:
            # PIL crop 接受 (left, top, right, bottom)
            # 注意：如果坐标是浮点数，需要转int
            x1, y1, x2, y2 = map(int, bbox)
            # 增加一点点padding防止切太紧
            padding = 5
            width, height = full_img.size
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            
            return full_img.crop((x1, y1, x2, y2))
    except Exception as e:
        print(f"Cropping error: {e}")
        return None
    return None

# ================= 主程序 =================
def main():
    st.set_page_config(page_title="违规复核终端", page_icon="🚓", layout="wide")
    inject_custom_css() # 注入样式
    st.title("🚓 应急车道违规复核终端")

    tasks = discover_tasks(OUTPUTS_DIR)
    if not tasks:
        st.error("未找到任务数据 (data/outputs)")
        st.stop()

    # 侧边栏
    task_names = [t[0] for t in tasks]
    selected_task = st.sidebar.selectbox("选择任务", task_names)
    csv_path = dict(tasks)[selected_task]
    
    # 加载数据
    task_data = get_task_data(selected_task, csv_path)
    df = task_data["df"]
    
    # 翻页逻辑
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    idx = task_data["index"]
    
    # 顶部统计
    reviewed_count = df["reviewed"].sum()
    total = len(df)
    col_stat1.metric("总事件", total)
    col_stat2.metric("复核进度", f"{reviewed_count}/{total}")
    
    # 翻页按钮
    c_prev, c_curr, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("⬅️ 上一条", key="btn_prev", use_container_width=True):
            task_data["index"] = max(0, idx - 1)
            st.rerun()
    with c_next:
        if st.button("下一条 ➡️", key="btn_next", use_container_width=True):
            task_data["index"] = min(total - 1, idx + 1)
            st.rerun()

    # --- 核心内容区 ---
    if total == 0:
        st.info("数据为空")
        st.stop()

    row = df.iloc[task_data["index"]]
    
    c_img, c_detail = st.columns([2, 1])
    
    # 1. 左侧大图
    with c_img:
        full_img = load_image_robust(row.get(COLS["img_path"]))
        if full_img:
            st.image(full_img, use_container_width=True, caption="占用画面 (Evidence)")
        else:
            st.warning("原始证据图丢失")

    # 2. 右侧详情与操作
    with c_detail:
        st.subheader("🔎 详情面板")
        
        # --- 动态车牌显示 ---
        # 优先读硬盘上的小图，如果没有，就用大图切
        crop_img = load_image_robust(row.get(COLS["plate_crop"]))
        
        if crop_img is None and full_img is not None and pd.notna(row.get(COLS["bbox"])):
            # 现场裁切！
            crop_img = crop_plate_dynamic(full_img, row[COLS["bbox"]])
            caption_txt = "车牌截图 (动态裁切)"
        else:
            caption_txt = "车牌截图 (文件)"

        if crop_img:
            st.image(crop_img, width=250, caption=caption_txt)
        else:
            st.info("无法获取车牌图像")

        # --- 识别状态 ---
        lpr_val = str(row.get(COLS["lpr_status"], "unknown"))
        st.markdown("**车牌文本识别**")
        if lpr_val.lower() == 'ok':
            st.markdown('<span class="status-badge-ok">✅ 成功运行</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge-fail">⚠️ {lpr_val}</span>', unsafe_allow_html=True)
        
        st.divider()

        # --- 复核操作表单 ---
        
        # 显示当前的复核状态
        is_reviewed = row.get("reviewed", False)
        is_excluded = row.get("is_excluded", False)
        
        if is_excluded:
            st.markdown("当前状态：<span style='color:blue;font-weight:bold'>🚫 已排除 (非违规)</span>", unsafe_allow_html=True)
        elif is_reviewed:
            st.markdown("当前状态：<span class='review-status-yes'>✅ 已复核</span>", unsafe_allow_html=True)
        else:
            st.markdown("当前状态：<span class='review-status-no'>🔴 未复核</span>", unsafe_allow_html=True)

        manual_val = row.get("manual_plate", "")
        # 如果是空的，默认填入 AI 识别的结果
        if pd.isna(manual_val) or manual_val == "":
            manual_val = row.get(COLS["plate_text"], "")

        new_plate = st.text_input("人工校正车牌", value=str(manual_val))
        
        # --- 按钮区 (保存 & 排除) ---
        # 使用列来横向排列按钮
        b_col1, b_col2 = st.columns(2)
        
        with b_col1:
            # 绿色按钮 (Primary)
            if st.button("✅ 保存并通过", type="primary", use_container_width=True):
                # 写入数据
                df.at[task_data["index"], "manual_plate"] = new_plate
                df.at[task_data["index"], "reviewed"] = True
                df.at[task_data["index"], "is_excluded"] = False # 如果保存通过，就取消排除状态
                # 存盘
                df.to_csv(task_data["csv_path"], index=False)
                task_data["df"] = df
                st.toast("✅ 已保存为【已复核】")
                st.rerun()

        with b_col2:
            # 普通按钮 (代表蓝色/排除)
            if st.button("🚫 排除此记录", use_container_width=True):
                df.at[task_data["index"], "is_excluded"] = True
                df.at[task_data["index"], "reviewed"] = True # 排除也算复核过的一种
                # 存盘
                df.to_csv(task_data["csv_path"], index=False)
                task_data["df"] = df
                st.toast("🚫 已标记为【排除】")
                st.rerun()

if __name__ == "__main__":
    main()