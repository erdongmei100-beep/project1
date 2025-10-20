# 应急车道占用检测 MVP

本仓库提供应急车道占用检测的完整离线流程，包括检测、跟踪、ROI 判定、事件导出、片段/车牌剪裁等能力。目录扁平化后，所有脚本均可直接在仓库根目录运行。

## 获取代码与权重

- Windows 可执行 `get_work.bat` 同步 `work` 分支的最新代码及 Git LFS 大文件（含示例视频与车牌权重）。
- 其他平台请手动执行：
  ```bash
  git fetch origin
  git switch work
  git pull --rebase
  git lfs install
  git lfs pull
  ```

## 快速开始

1. （可选）运行 `setup_env.bat` 在仓库根目录创建并初始化 `./.venv` 虚拟环境。
2. 激活虚拟环境并安装依赖：
   ```powershell
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. 执行推理：
   ```powershell
   python run.py --source "data/videos/ambulance.mp4" --roi "data/rois/ambulance.json" --save-video --save-csv --clip --plate
   ```

运行结束后将在 `data/outputs/` 下生成：

- `ambulance.mp4` 与 `ambulance.csv`（带标注视频与事件 CSV）。
- `clips/` 子目录（含事件前后缓冲片段 MP4）。
- `plates/` 子目录（事件最佳帧的车牌截图或尾部截图）。

若希望快速体验，可直接运行 Windows 批处理脚本：

```bat
run_one.bat
```

脚本会自动切换到仓库根目录、可选地激活 `./.venv`，并调用 `run.py` 生成完整输出。

## 车牌权重获取策略

程序会在启动时确保车牌检测权重就绪：

1. 优先使用 `--weights-plate` 指定的路径。
2. 其次使用环境变量 `PLATE_WEIGHTS`。
3. 若上述均未提供，则使用配置文件中的路径。若该路径指向仓库默认位置 `weights/plate/yolov8n-plate.pt` 且文件缺失，程序会尝试从以下链接自动下载：
   - https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
   - https://github.com/keremberke/yolov8-license-plate/releases/download/v0.0.0/yolov8n-license-plate.pt

成功下载后会在终端输出 `[weights] plate weights ready: ...`。若下载失败，程序将抛出可操作的错误信息，提示用户手动下载并放置到指定路径。

## 目录结构

```
configs/
data/
  videos/
  rois/
  outputs/
logs/
src/
tests/
tools/
weights/
  plate/
    yolov8n-plate.pt
run.py
app.py
requirements.txt
run_one.bat
setup_env.bat
get_work.bat
```

- `configs/default.yaml` 已使用仓库根目录进行路径拼装，可直接修改。
- 输出统一写入 `data/outputs/`，已在 `.gitignore` 中忽略。
- `tools/` 下包含 ROI 标注与调试相关脚本，可按需使用。

## 常见问题

- **缺少 Python 依赖**：运行 `setup_env.bat` 或手动创建虚拟环境后执行 `pip install -r requirements.txt`。
- **权重下载失败**：检查网络访问，或自行下载权重放到 `weights/plate/yolov8n-plate.pt`，也可通过 `PLATE_WEIGHTS` 或 `--weights-plate` 指定其它路径。
- **Git LFS 未安装**：执行 `git lfs install` 后再运行 `git lfs pull`。若使用 `get_work.bat` 会自动处理。
