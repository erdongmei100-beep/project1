"""Streamlit review terminal for emergency lane violations.

This app scans processed video tasks under ``data/outputs/`` and provides a
human-in-the-loop workflow to review AI-detected license plates. Users can
navigate events, correct plate values, mark reviews, and persist updates back to
the source CSV files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "data" / "outputs"
CSV_NAME = "events_with_plate.csv"
SESSION_KEY = "review_tasks"


def discover_tasks(outputs_dir: Path) -> List[Tuple[str, Path]]:
    """Return list of (task_name, csv_path) where the csv exists under outputs."""

    tasks: List[Tuple[str, Path]] = []
    if not outputs_dir.exists():
        return tasks

    for entry in sorted(outputs_dir.iterdir()):
        if not entry.is_dir():
            continue
        csv_path = entry / CSV_NAME
        if csv_path.is_file():
            tasks.append((entry.name, csv_path))
    return tasks


def initialize_dataframe(csv_path: Path) -> pd.DataFrame:
    """Load the CSV and ensure review helper columns exist."""

    df = pd.read_csv(csv_path)
    # Keep expected columns as strings where appropriate to avoid type surprises.
    for col in ["best_frame", "plate_crop", "plate_text", "lpr_status"]:
        if col in df.columns:
            df[col] = df[col].astype(str)

    if "plate_score" in df.columns:
        df["plate_score"] = pd.to_numeric(df["plate_score"], errors="coerce")

    if "reviewed" not in df.columns:
        df["reviewed"] = False
    else:
        df["reviewed"] = df["reviewed"].astype(bool).fillna(False)

    if "manual_plate" not in df.columns:
        df["manual_plate"] = df.get("plate_text", pd.Series(["" for _ in range(len(df))])).fillna("")
    else:
        df["manual_plate"] = df["manual_plate"].fillna(df.get("plate_text", "")).fillna("")

    return df


def set_active_task(task_name: str) -> None:
    """Update session state with the currently selected task."""

    st.session_state["active_task"] = task_name


def ensure_task_loaded(task_name: str, csv_path: Path) -> None:
    """Load task data into session state if missing."""

    if SESSION_KEY not in st.session_state:
        st.session_state[SESSION_KEY] = {}

    if task_name not in st.session_state[SESSION_KEY]:
        st.session_state[SESSION_KEY][task_name] = {
            "csv_path": csv_path,
            "df": initialize_dataframe(csv_path),
            "index": 0,
        }


def get_task_data(task_name: str) -> Dict:
    return st.session_state[SESSION_KEY][task_name]


def clamp_index(idx: int, total: int) -> int:
    if total == 0:
        return 0
    return max(0, min(idx, total - 1))


def display_image(image_path: str | os.PathLike[str], caption: str) -> None:
    """Render an image if possible; otherwise show a warning placeholder."""

    if not image_path:
        st.warning(f"缺少图片：{caption}")
        return

    path_obj = Path(image_path)
    if not path_obj.is_file():
        st.warning(f"未找到图片文件：{path_obj}")
        return

    try:
        image = Image.open(path_obj)
    except Exception as exc:  # noqa: BLE001 - display error to user directly
        st.warning(f"无法加载图片 {path_obj.name}: {exc}")
        return

    st.image(image, use_column_width=True, caption=caption)


def main() -> None:
    st.set_page_config(page_title="应急车道违规复核终端", page_icon="🚧", layout="wide")
    st.title("应急车道违规复核终端")
    st.caption("查看模型检测结果，人工核查车牌并实时写回 CSV。")

    tasks = discover_tasks(OUTPUTS_DIR)
    if not tasks:
        st.error("未在 data/outputs/ 中找到包含 events_with_plate.csv 的任务目录。")
        st.stop()

    task_names = [name for name, _ in tasks]
    default_idx = 0
    current_task = st.session_state.get("active_task")
    if current_task in task_names:
        default_idx = task_names.index(current_task)

    selected_task = st.sidebar.selectbox("选择视频任务", task_names, index=default_idx)
    set_active_task(selected_task)

    csv_path = dict(tasks)[selected_task]
    ensure_task_loaded(selected_task, csv_path)
    task_data = get_task_data(selected_task)

    df = task_data["df"]
    total_events = len(df)
    current_index = clamp_index(task_data.get("index", 0), total_events)
    task_data["index"] = current_index

    if total_events == 0:
        st.info("当前任务的 CSV 为空。")
        st.stop()

    reviewed_count = int(df["reviewed"].sum()) if "reviewed" in df.columns else 0
    avg_confidence = (
        df["plate_score"].mean()
        if "plate_score" in df.columns and not df["plate_score"].empty
        else float("nan")
    )

    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("总违规事件", f"{total_events}")
    metric_col2.metric("AI 识别置信度", f"{avg_confidence:.3f}" if pd.notna(avg_confidence) else "N/A")
    metric_col3.metric("复核进度", f"{reviewed_count}/{total_events}")

    navigation_left, navigation_right = st.columns([1, 1])
    with navigation_left:
        if st.button("上一条", disabled=current_index <= 0):
            task_data["index"] = clamp_index(current_index - 1, total_events)
            st.experimental_rerun()

    with navigation_right:
        if st.button("下一条", disabled=current_index >= total_events - 1):
            task_data["index"] = clamp_index(current_index + 1, total_events)
            st.experimental_rerun()

    current_row = df.iloc[task_data["index"]]

    left_col, right_col = st.columns([2, 1])
    with left_col:
        display_image(current_row.get("best_frame", ""), caption="违规证据帧")

    with right_col:
        st.subheader("车牌区域")
        display_image(current_row.get("plate_crop", ""), caption="车牌截图")

        st.markdown("**AI 识别结果**")
        st.write(f"车牌号：{current_row.get('plate_text', '未知')}")
        if pd.notna(current_row.get("plate_score")):
            st.write(f"置信度：{float(current_row.get('plate_score')):.3f}")
        st.write(f"状态：{current_row.get('lpr_status', '未知')}")

        manual_default = current_row.get("manual_plate") or current_row.get("plate_text") or ""
        with st.form(key=f"review_form_{selected_task}_{task_data['index']}"):
            manual_plate = st.text_input("人工校正车牌", value=str(manual_default))
            reviewed_flag = st.checkbox("标记为已复核", value=bool(current_row.get("reviewed", False)))
            save_clicked = st.form_submit_button("保存", type="primary")

        if save_clicked:
            df.at[task_data["index"], "manual_plate"] = manual_plate.strip()
            df.at[task_data["index"], "reviewed"] = reviewed_flag

            df.to_csv(task_data["csv_path"], index=False)
            task_data["df"] = df

            st.toast("已保存并写回 CSV。")


if __name__ == "__main__":
    main()
