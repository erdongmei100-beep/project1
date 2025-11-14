# Emergency Lane Occupancy Detection

基于 YOLOv8 与 ByteTrack 的应急车道占用检测原型。仓库已扁平化，所有脚本、配置与数据均以仓库根目录为基准，避免 `project/project` 之类的嵌套。

## 目录结构
```
<repo-root>/
  run.py                # 统一入口
  setup_env.(bat|sh)    # 一键创建 .venv 并安装依赖
  run_one.(bat|sh)      # 一键运行示例，使用 data/videos/ambulance.mp4
  requirements.txt
  configs/
    default.yaml
    tracker/bytetrack.yaml
  data/
    videos/ambulance.mp4      # 需要 Git LFS 或按 README 拉取
    rois/ambulance.json
    outputs/                  # 运行结果输出目录（含 .gitkeep）
  weights/plate/              # 车牌检测权重放置于此
  src/                        # 业务逻辑与工具模块
```

## 准备工作

1. **克隆并同步 Git LFS**
   ```bash
   git clone <your-fork-url>
   cd project1
   git lfs install
   git lfs pull
   ```

2. **创建虚拟环境并安装依赖**
   - Windows
     ```powershell
     .\setup_env.bat
     .\.venv\Scripts\activate
     ```
   - Linux/macOS
     ```bash
     ./setup_env.sh
     source .venv/bin/activate
     ```

   若已设置 `PIP_INDEX_URL`，脚本会沿用；否则默认使用清华镜像。

### 车牌裁剪说明

推理阶段会导出车辆 ROI 与车牌裁剪图，保存在 `data/outputs/<run_name>/plates/` 目录下。项目已移除所有车牌文字识别组件，仅保留检测、裁剪与事件记录流程。如需后续做文字识别，可根据自身需求在裁剪结果基础上集成外部识别工具。

## 运行示例

- Windows
  ```powershell
  .\run_one.bat
  ```
- Linux/macOS
  ```bash
  ./run_one.sh
  ```

或手动执行：
```bash
python run.py \
  --source data/videos/ambulance.mp4 \
  --config configs/default.yaml \
  --save-video --save-csv --clip --plate
```

`run.py` 会根据配置中的 `roi.mode=auto_cv` 自动估计 ROI 并写入 `data/rois/<视频名>.json`。
若需使用手工 ROI，可添加 `--roi data/rois/xxx.json` 覆盖自动结果。

程序会将输出写入 `data/outputs/<视频名>/` 下的子目录，终端会打印每个生成文件的完整路径。

## 车牌检测权重

默认权重路径为 `weights/plate/yolov8n-plate.pt`。若文件缺失：

1. 首先执行：
   ```bash
   git lfs install
   git lfs pull
   ```
2. 若仍未获得权重，`run.py` 会尝试从镜像下载：
   ```
   https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt
   ```
   下载失败会给出明确报错，可根据提示手动放置到 `weights/plate/` 目录。

## 常见问题

- **缺少依赖**：运行脚本会在导入失败时提示“请先运行 setup 脚本或 pip install -r requirements.txt”。
- **输出位置**：所有视频、剪辑、截图与 CSV 均保存在 `data/outputs/`，可安全清理或忽略。
- **重新运行**：删除 `data/outputs/<视频名>/` 下的旧结果后再执行 `run.py` 即可重新生成。
- **仓库默认状态**：为保持仓库整洁，`data/outputs/` 仅保留 `.gitkeep` 占位文件；实际运行时会重新生成所需的 CSV、可视化视频、叠加图等产物。

## 自动 ROI 调参与快速诊断

自动 ROI 逻辑在 `configs/default.yaml` 的 `roi.auto_cv` 下配置，关键阈值的推荐起点：

| 参数 | 作用 |
| --- | --- |
| `min_box_h_px` | ROI 多边形的最小高度，过小会导致检测到的区域过窄 |
| `min_rel_area` | ROI 占整帧的最小面积比例，避免误检到细小噪声 |
| `min_sharpness` | ROI 区域内的清晰度阈值（拉普拉斯方差），确保车道线明显 |
| `bbox_aspect_min` | ROI 宽高比下限，限制区域过于瘦长 |
| `save_debug` | 开启后在 `data/outputs/auto_cv/` 保存叠加图，便于肉眼确认 |

调参与批量诊断示例：

```bash
python tools/eval_auto_cv.py \
  --videos data/videos/ambulance.mp4 data/videos/broken_down_vehicle.mp4 data/videos/exceptional_case.mp4 \
  --config configs/default.yaml \
  --out reports/auto_cv_tune.csv \
  --save-overlays
```

脚本会输出：

- `reports/auto_cv_tune.csv`：记录每段视频的耗时、ROI 质量指标与是否成功。
- `data/outputs/auto_cv/<video>_overlay.png`：自动 ROI 叠加图。
- 更新后的 `data/rois/<video>.json` 与配置中的默认阈值。

若想单独验证某段素材，可使用：

```bash
python -m project.tools.roi_auto_cv \
  --source data/videos/ambulance.mp4 \
  --config configs/default.yaml \
  --out data/rois/ambulance.json \
  --save-overlay
```

如出现 “Auto ROI failed” 提示，可优先检查：

1. `min_rel_area` 与 `bbox_aspect_min` 是否过大导致 ROI 被过滤。
2. 原视频是否过暗，可尝试降低 `v_min` 或 `min_sharpness`。
3. 叠加图是否明显偏离车道线，据此手动调整 `crop_right`、`crop_bottom` 等裁剪参数。
4. 极端素材（例如 `exceptional_case.mp4`）中车道线往往位于画面更右侧且对比度较低，脚本会自动尝试放宽裁剪区域与角度阈值；若仍失败，将弹出手动 ROI 标注窗口。

手动模式说明：自动检测失败时会弹出一个窗口，左上角会展示中文提示。按以下操作即可快速标注：

- 鼠标左键：按顺序点击多边形顶点。
- 鼠标右键：撤销上一个顶点。
- 按 `R`：清空当前标注重新开始。
- 按 `S`：保存当前多边形（至少三个点）。
- 按 `Q` 或 `Esc`：退出手动标注。

欢迎提交问题或改进建议。
