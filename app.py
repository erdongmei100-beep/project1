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
        st.error("未找到任务数据 (data/outputs)")
        st.stop()

    task_names = [t[0] for t in tasks]
    selected_task = st.sidebar.selectbox("选择任务", task_names)
    csv_path = dict(tasks)[selected_task]
    
    task_data = get_task_data(selected_task, csv_path)
    df = task_data["df"]
    
    # --- 顶部统计 (恢复 AI识别置信度) ---
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    idx = task_data["index"]
    
    reviewed_count = df["reviewed"].sum()
    total = len(df)
    
    # 计算平均置信度
    avg_conf = 0.00
    if COLS["plate_score"] in df.columns and not df[COLS["plate_score"]].empty:
        avg_conf = df[COLS["plate_score"]].mean()

    col_stat1.metric("总事件", total)
    col_stat2.metric("AI识别置信度", f"{avg_conf:.2f}") # 恢复此项
    col_stat3.metric("复核进度", f"{reviewed_count}/{total}")
    
    c_prev, c_curr, c_next = st.columns([1, 2, 1])
    with c_prev:
        if st.button("⬅️ 上一条", key="btn_prev", use_container_width=True):
            task_data["index"] = max(0, idx - 1)
            st.rerun()
    with c_next:
        if st.button("下一条 ➡️", key="btn_next", use_container_width=True):
            task_data["index"] = min(total - 1, idx + 1)
            st.rerun()

    if total == 0:
        st.info("数据为空")
        st.stop()

    row = df.iloc[task_data["index"]]
    
    c_img, c_detail = st.columns([2, 1])
    
    with c_img:
        full_img = load_image_robust(row.get(COLS["img_path"]))
        if full_img:
            st.image(full_img, use_container_width=True, caption="占用画面 (Evidence)")
        else:
            st.warning("原始证据图丢失")

    with c_detail:
        st.subheader("🔎 详情面板")
        
        # --- 动态车牌显示 ---
        crop_img = load_image_robust(row.get(COLS["plate_crop"]))
        
        if crop_img is None and full_img is not None and pd.notna(row.get(COLS["bbox"])):
            crop_img = crop_plate_dynamic(full_img, row[COLS["bbox"]])

        caption_txt = "车牌截图"

        if crop_img:
            st.image(crop_img, width=250, caption=caption_txt)
        else:
            st.info("无法获取车牌图像")

        # --- 识别状态 & 结果展示 ---
        lpr_val = str(row.get(COLS["lpr_status"], "unknown"))
        ai_plate_text = row.get(COLS["plate_text"], "未知")
        
        st.markdown("**车牌文本识别**")
        if lpr_val.lower() == 'ok':
            st.markdown('<span class="status-badge-ok">✅ 成功运行</span>', unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-badge-fail">⚠️ {lpr_val}</span>', unsafe_allow_html=True)
            
        # [修改点]：新增不可编辑的灰色字
        st.markdown(f'<div class="read-only-text">识别结果：{ai_plate_text}</div>', unsafe_allow_html=True)
        
        st.divider()

        # --- 复核操作表单 ---
        is_reviewed = row.get("reviewed", False)
        is_excluded = row.get("is_excluded", False)
        
        if is_excluded:
            st.markdown("当前状态：<span class='review-status-exclude'>🚫 已复核，非违规占用</span>", unsafe_allow_html=True)
        elif is_reviewed:
            st.markdown("当前状态：<span class='review-status-yes'>✅ 已复核</span>", unsafe_allow_html=True)
        else:
            st.markdown("当前状态：<span class='review-status-no'>🔴 未复核</span>", unsafe_allow_html=True)

        manual_val = row.get("manual_plate", "")
        if pd.isna(manual_val) or manual_val == "":
            manual_val = row.get(COLS["plate_text"], "")

        new_plate = st.text_input("人工校正车牌", value=str(manual_val))
        
        b_col1, b_col2 = st.columns(2)
        
        with b_col1:
            if st.button("✅ 保存并通过", type="primary", use_container_width=True):
                df.at[task_data["index"], "manual_plate"] = new_plate
                df.at[task_data["index"], "reviewed"] = True
                df.at[task_data["index"], "is_excluded"] = False
                df.to_csv(task_data["csv_path"], index=False)
                task_data["df"] = df
                st.toast("✅ 已保存为【已复核】")
                st.rerun()

        with b_col2:
            if st.button("🚫 排除此记录", use_container_width=True):
                df.at[task_data["index"], "is_excluded"] = True
                df.at[task_data["index"], "reviewed"] = True
                df.to_csv(task_data["csv_path"], index=False)
                task_data["df"] = df
                st.toast("🚫 已标记为【排除】")
                st.rerun()

if __name__ == "__main__":
    main()