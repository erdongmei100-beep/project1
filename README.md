# Emergency Lane Occupancy Detection

Offline computer-vision pipeline for detecting and tracking vehicles that occupy an emergency lane. The system uses a YOLOv8 detector with ByteTrack-style multi-object tracking, applies a user-defined ROI polygon to decide whether a vehicle is inside the emergency lane, and exports per-event metadata and optional video clips.

## Features
- YOLOv8 detection with tracker-driven IDs for stable vehicle tracks.
- ROI polygon management with automatic lookup and optional interactive creation.
- Event accumulation with CSV export (start/end frames and timestamps).
- Optional per-event clip export with configurable pre/post buffers.

## Setting up a GPU environment (recommended)
PyPI provides CPU wheels for PyTorch by default. Install a CUDA build first, then the rest of the dependencies:

```bash
# Example for CUDA 11.8; adjust the URL/version to match your driver
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

If a compatible GPU is available, Ultralytics will automatically run the model on it. The default configuration also selects device `0` (first GPU). Override `model.device` in `configs/default.yaml` or via a custom config if needed.

## Installation
1. Create and activate a Python 3.10+ virtual environment.
2. Install the CUDA-enabled PyTorch as shown above, then install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Place your YOLOv8 weights file at `yolov8n.pt` in the repository root or update `model.weights` in the config.

## Usage
Run inference on a video with an ROI file:

```bash
python run.py --source data/videos/ambulance.mp4 --roi data/rois/ambulance.json --save-video --save-csv --clip
```

Outputs are written to `data/outputs/` by default:
- Annotated video (`<source_stem>.mp4`) when `--save-video` is set.
- Event CSV (`<source_stem>.csv`) when `--save-csv` is set.
- Per-event clips under `data/outputs/clips/` when `--clip` is set.

If the ROI file is missing, the runner will attempt automatic ROI generation (when enabled in config) or launch the interactive ROI tool.

## Configuration
- Default configuration: `configs/default.yaml` (device, detector weights, ROI path, thresholds, clip buffers, etc.).
- Tracker settings: `configs/tracker/bytetrack.yaml`.
- ROI files: `data/rois/*.json` (one per video is recommended).

## Project structure
```
configs/
  default.yaml
  tracker/
data/
  videos/
  rois/
  outputs/
logs/
src/
  dettrack/
  io/
  logic/
  render/
  roi/
  utils/
tests/
tools/
run.py
app.py
requirements.txt
yolov8n.pt
```

## Troubleshooting
- **CUDA not used**: verify the GPU build of PyTorch is installed and `model.device` is set to `0` (or another GPU index).
- **ROI missing**: provide `--roi` explicitly or ensure `data/rois/<video_stem>.json` exists; the tool can generate one interactively if permitted by config.
- **Ultralytics model download issues**: manually download the desired YOLOv8 weights and point `model.weights` to the local path.
