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

### 车牌识别依赖

新增的车牌文字识别模块依赖 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) 及其底层框架 PaddlePaddle。
安装依赖时请确保：

- 先执行 `pip install paddleocr>=2.7.0`；
- 根据硬件环境二选一安装 PaddlePaddle：CPU 环境使用 `pip install paddlepaddle>=2.5.0`，若使用 GPU 请根据本机 CUDA 版本选择匹配的 `paddlepaddle-gpu` 版本（例如 `paddlepaddle-gpu==2.5.2.post120`）。

如未安装或版本不匹配，车牌 OCR 初始化会给出明确错误提示。

### 批量车牌 OCR 使用说明

视频推理阶段默认只导出车牌裁剪图，不再实时做 OCR。需要进行批量识别时，请执行独立脚本：

```bash
python tools/ocr_plates.py \
  --input data/outputs/<run_name>/plates \
  --rec-model-dir weights/ppocr/ch_PP-OCRv4_rec_infer \
  --use-gpu false \
  --min-height 64 \
  --min-conf 0.20 \
  --num-workers 4 \
  --dry-run false
```

参数说明：

- `--input`：必填，指向 `data/outputs/<run_name>/plates/` 目录，脚本会递归遍历其中的裁剪图。
- `--rec-model-dir`：PaddleOCR 识别模型目录，需至少包含 `inference.pdmodel` 与 `inference.pdiparams`。如缺少 `inference.json` 等元数据，会打印一次 WARN 并继续。
- `--use-gpu`：如环境已安装 GPU 版 Paddle，可设为 `true`；默认 CPU（`false`）。
- `--min-height`：像素高度阈值，小于该值的裁剪图直接跳过。
- `--min-conf`：置信度阈值，识别分数低于该值时文本置空但仍保留记录。
- `--num-workers`：并行线程数，用于加速批量识别。
- `--dry-run`：设为 `true` 时仅统计将要处理的图像数量，不写入 CSV、不执行 OCR。

脚本会在 `plates/` 目录下生成（或增量更新）`plate_ocr_results.csv`，列顺序固定为：

`image_path, plate_text, rec_confidence, width, height, ocr_engine, used_gpu, elapsed_ms`

其中 `image_path` 为仓库根目录的相对路径；若识别置信度不足或失败，`plate_text` 为空字符串、`rec_confidence` 记录原始分数（无分数时记 0.0）。脚本具备幂等性，已写入 CSV 的图像不会重复处理；若裁剪图尺寸任一边大于 512 像素，会被安全跳过以防误将整帧送入 OCR。

若终端摘要显示存在“低于置信度阈值”的图片，可适当降低 `--min-conf` 后重新运行。

### YOLOv5 车牌检测 + HyperLPR 识别

车辆裁剪阶段会在 `data/outputs/<run_name>/plates/` 下生成 `*_tail.jpg` 车辆 ROI。可使用新增脚本完成车牌检测、精确裁剪与 HyperLPR 识别：

```bash
python tools/run_plate_ocr.py \
  --vehicle_dir data/outputs/<run_name>/plates/vehicle_roi \
  --yolo_weights weights/plate_best.pt \
  --out_dir runs/plates \
  --download_url https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5n.pt
```

如需手动准备权重，可将下载好的 `*.pt` 文件放到仓库根目录下的 `weights/plate_best.pt`（或在配置中的 `plate.lp_pipeline.yolo_weights` 指定新的路径）。脚本也会在本地模式下自动创建目录并把在线下载的权重写入同一位置，方便后续离线复用。

> 📦 **关于 YOLOv5 本地模式：** `weights/plate_best.pt` 只需要模型权重文件本身，不要把整个 YOLOv5 仓库解压到 `weights/`。若想使用 `--use_hub 0`，请把解压后的 `ultralytics-yolov5-*/` 内容拷贝到仓库根目录的 `yolov5/` 文件夹（保持 `models/、utils/` 等子目录结构），而权重文件仍应单独放在 `weights/plate_best.pt`。

脚本会自动同步已有 `*_tail.jpg` 至 `vehicle_roi/` 子目录。默认通过 PyTorch Hub 加载 YOLOv5；如需使用本地 `yolov5/`，可追加 `--use_hub 0`。当本地权重缺失时，脚本会自动按 `--download_url`（留空则使用默认链接，亦可通过环境变量 `PLATE_YOLOV5_URL` 覆盖）下载模型并保存到指定路径。

执行后会生成：

- `runs/plates/results.csv`
- `runs/plates/*_vis.jpg`
- `runs/plates/*_cand{i}.jpg`

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
- **RapidOCR 初次运行报错 `Cannot load model ... huggingface_hub is not installed`**：这是 `rapidocr-onnxruntime` 在下载 ONNX 检测/识别模型时未检测到 `huggingface_hub`。按提示执行 `pip install huggingface_hub` 或重新运行 `setup_env` 即可补齐依赖。
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
