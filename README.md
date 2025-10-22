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
  --roi data/rois/ambulance.json \
  --config configs/default.yaml \
  --save-video --save-csv --clip --plate
```

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

欢迎提交问题或改进建议。
