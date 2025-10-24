"""Streamlit launcher for the emergency lane occupancy detection pipeline."""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional

import streamlit as st

from src.utils.paths import ROOT
DEFAULT_CONFIG = "configs/default.yaml"

st.set_page_config(page_title="应急车道占用检测", page_icon="🚗", layout="centered")
st.title("应急车道占用检测 MVP")
st.caption("上传或指定视频，一键完成检测、事件导出、车牌截图等任务。")

video_mode = st.radio("视频来源", ["上传文件", "本地路径"], horizontal=True)
uploaded_video = st.file_uploader(
    "上传视频文件", type=["mp4", "mov", "avi", "mkv"], accept_multiple_files=False, disabled=video_mode != "上传文件"
)
video_path_input = st.text_input(
    "视频路径（例如 data/videos/sample.mp4）", value="data/videos/sample.mp4", disabled=video_mode != "本地路径"
)

roi_mode = st.radio("ROI 选择方式", ["自动同名", "上传 ROI", "本地路径"], horizontal=True)
uploaded_roi = st.file_uploader(
    "上传 ROI JSON", type=["json"], accept_multiple_files=False, disabled=roi_mode != "上传 ROI", key="roi_file"
)
roi_path_input = st.text_input(
    "ROI 路径（可选）", value="data/rois/sample.json", disabled=roi_mode != "本地路径"
)

clip_flag = st.checkbox("导出事件剪辑 (--clip)", value=True)
plate_flag = st.checkbox("导出车牌截图 (--plate)", value=True)
save_video_flag = st.checkbox("导出标注视频 (--save-video)", value=True)
save_csv_flag = st.checkbox("导出事件 CSV (--save-csv)", value=True)

config_path_input = st.text_input("配置文件路径", value=DEFAULT_CONFIG)

if st.button("开始处理"):
    errors: List[str] = []
    if video_mode == "上传文件" and uploaded_video is None:
        errors.append("请先上传视频文件。")
    if video_mode == "本地路径" and not video_path_input.strip():
        errors.append("请填写有效的视频路径。")
    if roi_mode == "上传 ROI" and uploaded_roi is None:
        errors.append("请上传 ROI JSON，或切换到自动模式。")

    if errors:
        for message in errors:
            st.warning(message)
    else:
        with tempfile.TemporaryDirectory(prefix="lane_app_") as tmpdir:
            tmp_root = Path(tmpdir)

            if video_mode == "上传文件" and uploaded_video is not None:
                video_path = tmp_root / uploaded_video.name
                video_path.write_bytes(uploaded_video.getbuffer())
            else:
                video_path = Path(video_path_input.strip()).expanduser()
                if not video_path.is_absolute():
                    video_path = (ROOT / video_path).resolve()

            roi_override: Optional[Path] = None
            if roi_mode == "上传 ROI" and uploaded_roi is not None:
                roi_temp = tmp_root / uploaded_roi.name
                roi_temp.write_bytes(uploaded_roi.getbuffer())
                roi_override = roi_temp
            elif roi_mode == "本地路径" and roi_path_input.strip():
                roi_override = Path(roi_path_input.strip()).expanduser()
                if not roi_override.is_absolute():
                    roi_override = (ROOT / roi_override).resolve()

            command = [sys.executable, str(ROOT / "run.py"), "--source", str(video_path)]
            if config_path_input.strip():
                command.extend(["--config", config_path_input.strip()])
            if roi_override is not None:
                command.extend(["--roi", str(roi_override)])
            if clip_flag:
                command.append("--clip")
            if plate_flag:
                command.append("--plate")
            if save_video_flag:
                command.append("--save-video")
            if save_csv_flag:
                command.append("--save-csv")

            outputs_dir = ROOT / "data" / "outputs"
            before = set()
            if outputs_dir.exists():
                before = {p for p in outputs_dir.glob("**/*") if p.is_file()}

            st.info("正在执行检测流程，请稍候……")
            result = subprocess.run(
                command,
                cwd=ROOT,
                capture_output=True,
                text=True,
            )

            command_preview = " ".join(command)
            st.code(f"$ {command_preview}")
            if result.stdout:
                st.text_area("标准输出", result.stdout, height=240)
            if result.stderr:
                st.text_area("标准错误", result.stderr, height=200)

            if result.returncode != 0:
                st.error(f"检测流程失败（退出码 {result.returncode}）。请检查日志。")
            else:
                st.success("检测完成！")
                outputs_dir.mkdir(parents=True, exist_ok=True)
                after = {p for p in outputs_dir.glob("**/*") if p.is_file()}
                new_files = sorted(after - before)
                if new_files:
                    st.subheader("新增输出文件")
                    for file_path in new_files:
                        try:
                            rel_path = file_path.relative_to(ROOT)
                        except ValueError:
                            rel_path = file_path
                        st.write(f"- {rel_path}")
                else:
                    st.write("未检测到新的输出文件。")
