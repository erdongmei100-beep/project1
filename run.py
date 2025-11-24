"""Emergency lane occupancy detection MVP runner."""
from __future__ import annotations

import argparse
import copy
import math
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.dettrack.pipeline import DetectorTracker
from src.io.video import VideoMetadata, probe_video
from src.logic.events import EventAccumulator, FrameOccupancy, OccupancyEvent
from src.render.overlay import draw_overlays
from src.roi.manager import ROIManager
from src.utils.config import load_config, resolve_path
from src.utils.paths import PROJECT_ROOT, project_path, resolve_project_path


DEFAULT_CONFIG_REL = project_path("configs", "default.yaml").relative_to(PROJECT_ROOT)
DEFAULT_TRACKER_CFG_REL = project_path("configs", "tracker", "bytetrack.yaml").relative_to(PROJECT_ROOT)
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency lane occupancy detection MVP")
    parser.add_argument("--source", required=True, help="Path to input video or directory of videos")
    parser.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG_REL),
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--save-video",
        action="store_true",
        help="Save annotated video to outputs directory",
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save occupancy events as CSV",
    )
    parser.add_argument(
        "--roi",
        default=None,
        help="Optional ROI JSON file that overrides configuration and automatic lookup.",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Export per-event video clips with buffered context.",
    )
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        joined = " ".join(unknown)
        print(f"Warning: Ignoring unknown arguments: {joined}")
    return args


def prepare_tracker(config: Dict[str, object], tracker_cfg_path: Path) -> DetectorTracker:
    model_cfg = config.get("model", {})
    tracking_cfg = config.get("tracking", {})
    return DetectorTracker(
        weights=model_cfg.get("weights", "yolov8n.pt"),
        device=model_cfg.get("device", 0),
        imgsz=int(model_cfg.get("imgsz", 640)),
        tracker_cfg=tracker_cfg_path,
        conf=float(tracking_cfg.get("conf", 0.25)),
        iou=float(tracking_cfg.get("iou", 0.45)),
        max_det=int(tracking_cfg.get("max_det", 1000)),
    )


def export_events(events, output_csv: Path, fps: float) -> None:
    if not events:
        print("No occupancy events detected; skipping CSV export.")
        return
    rows = []
    for event in events:
        start_time = event.start_frame / fps if fps else None
        end_time = event.end_frame / fps if fps else None
        row: Dict[str, object] = {
            "track_id": event.track_id,
            "start_frame": event.start_frame,
            "end_frame": event.end_frame,
            "duration_frames": event.duration_frames,
            "start_time_s": start_time,
            "end_time_s": end_time,
        }
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved CSV to {output_csv}")


def _compute_clip_range(
    event: OccupancyEvent,
    fps: float,
    frame_count: Optional[int],
    pre_seconds: float,
    post_seconds: float,
) -> Tuple[int, int]:
    safe_fps = fps if fps > 0 else 25.0
    pre_frames = int(max(pre_seconds, 0.0) * safe_fps)
    post_frames = int(max(post_seconds, 0.0) * safe_fps)
    start = max(event.start_frame - pre_frames, 0)
    end = event.end_frame + post_frames
    if frame_count is not None and frame_count > 0:
        end = min(end, frame_count - 1)
    if end < start:
        end = start
    return start, end


def _write_video_segment(
    video_path: Path,
    output_path: Path,
    start_frame: int,
    end_frame: int,
    fps: float,
    frame_size: Tuple[int, int],
) -> bool:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return False
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(start_frame, 0))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer: Optional[cv2.VideoWriter] = None
    frame_idx = start_frame
    success = False
    width, height = frame_size

    while frame_idx <= end_frame:
        has_frame, frame = capture.read()
        if not has_frame or frame is None:
            break
        if writer is None:
            frame_height, frame_width = frame.shape[:2]
            if width <= 0 or height <= 0:
                width, height = frame_width, frame_height
            writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 25.0, (width, height))
            if not writer.isOpened():
                capture.release()
                return False
        writer.write(frame)
        frame_idx += 1
        success = True

    capture.release()
    if writer is not None:
        writer.release()
    return success


def export_event_clips(
    events: List[OccupancyEvent],
    metadata: VideoMetadata,
    output_dir: Path,
    fps: float,
    pre_seconds: float,
    post_seconds: float,
    clip_subdir: str,
) -> None:
    if not events:
        print("No occupancy events detected; skipping clip export.")
        return

    clip_dir = output_dir / clip_subdir
    clip_dir.mkdir(parents=True, exist_ok=True)

    effective_fps = fps if fps > 0 else (metadata.fps if metadata.fps > 0 else 25.0)
    frame_count = metadata.frame_count if metadata.frame_count > 0 else None

    for index, event in enumerate(events, start=1):
        clip_start, clip_end = _compute_clip_range(event, effective_fps, frame_count, pre_seconds, post_seconds)
        clip_name = f"{metadata.path.stem}_event{index:02d}_track{event.track_id}_{clip_start:06d}-{clip_end:06d}.mp4"
        clip_path = clip_dir / clip_name
        if _write_video_segment(metadata.path, clip_path, clip_start, clip_end, effective_fps, metadata.frame_size):
            print(f"Saved clip to {clip_path}")
        else:
            print(f"Failed to export clip for track {event.track_id} ({clip_start}-{clip_end}).")


def _classify_event_motion(event: OccupancyEvent, metadata: VideoMetadata) -> str:
    if len(event.frames) < 2:
        return "unknown"

    y_values = [frame.footpoint[1] for frame in event.frames]
    dy_total = y_values[-1] - y_values[0]
    step_deltas = [abs(y_values[i + 1] - y_values[i]) for i in range(len(y_values) - 1)]
    max_step = max(step_deltas, default=0.0)

    heights = [frame.bbox[3] - frame.bbox[1] for frame in event.frames]
    avg_height = sum(heights) / len(heights) if heights else 1.0
    avg_height = max(avg_height, 1.0)

    total_threshold = avg_height * 0.35
    step_threshold = avg_height * 0.25

    if abs(dy_total) <= total_threshold and max_step <= step_threshold:
        return "front_static"

    if dy_total >= 0:
        return "rear_approach"

    return "unknown"


def _get_float_config(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _iter_video_files(directory: Path) -> Iterable[Path]:
    supported = {".mp4", ".avi", ".mov", ".mkv"}
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in supported:
            yield path


def process_video(source_path: Path, base_config: Dict[str, object], args: argparse.Namespace) -> None:
    config = copy.deepcopy(base_config)
    tracking_cfg = dict(config.get("tracking", {}) or {})
    tracker_cfg_rel = tracking_cfg.get("tracker_config", str(DEFAULT_TRACKER_CFG_REL))
    tracker_cfg = resolve_path(PROJECT_ROOT, tracker_cfg_rel)

    metadata = probe_video(source_path)
    print(
        f"Video: {metadata.path.name} | FPS: {metadata.fps:.2f} | Frames: {metadata.frame_count} | Size: {metadata.frame_size}"
    )

    roi_cfg = dict(config.get("roi", {}) or {})
    roi_path = resolve_project_path(None, project_path("data", "rois")) / f"{source_path.stem}.json"
    if args.roi:
        roi_path = resolve_path(PROJECT_ROOT, str(args.roi))
    roi_path.parent.mkdir(parents=True, exist_ok=True)

    auto_mode = str(roi_cfg.get("mode", "")).lower()
    if auto_mode == "yolo_gen" and roi_path.exists():
        roi_path.unlink()

    if not roi_path.exists():
        if auto_mode == "yolo_gen":
            auto_script = PROJECT_ROOT / "tools" / "roi_yolo_gen.py"
            if auto_script.exists():
                auto_cfg = dict(roi_cfg.get("yolo_gen", {}) or {})
                cmd = [
                    sys.executable,
                    str(auto_script),
                    "--source",
                    str(source_path),
                    "--out",
                    str(roi_path),
                ]
                flag_map = {
                    "model": "--model",
                    "class_id": "--class-id",
                    "conf": "--conf",
                }
                for key, flag in flag_map.items():
                    if key in auto_cfg and auto_cfg[key] is not None:
                        cmd.extend([flag, str(auto_cfg[key])])
                if bool(auto_cfg.get("show")):
                    cmd.append("--show")
                try:
                    print("未找到 ROI 文件，正在执行 YOLO 自动生成...")
                    subprocess.run(cmd, check=True, cwd=str(PROJECT_ROOT))
                except subprocess.CalledProcessError as exc:
                    print(f"YOLO 自动 ROI 生成失败（退出码 {exc.returncode}）：{exc}")
                except Exception as exc:
                    print(f"YOLO 自动 ROI 生成执行失败：{exc}")
        if not roi_path.exists():
            try:
                from tools.roi_make import launch_roi_selector

                print("未找到 ROI 文件，正在启动交互式标注工具...")
                created = launch_roi_selector(str(source_path), str(roi_path))
            except Exception as exc:
                raise RuntimeError(f"无法创建 ROI：{exc}") from exc
            if not created or not roi_path.exists():
                raise RuntimeError("ROI 标注未完成，无法继续运行。")

    print(f"Using ROI file: {roi_path}")
    roi_manager = ROIManager(roi_path)

    events_cfg = config.get("events", {})
    accumulator = EventAccumulator(
        min_frames_in=int(events_cfg.get("min_frames_in", 5)),
        min_frames_out=int(events_cfg.get("min_frames_out", 5)),
    )

    render_cfg = config.get("render", {})
    show_tracks = bool(render_cfg.get("show_tracks", True))
    show_footpoints = bool(render_cfg.get("show_footpoints", True))

    outputs_cfg = dict(config.get("outputs") or config.get("output") or {})
    output_dir = project_path("data", "outputs", source_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs_cfg["video_filename"] = f"{source_path.stem}.mp4"
    outputs_cfg["csv_filename"] = f"{source_path.stem}.csv"

    clip_pre_seconds = _get_float_config(outputs_cfg.get("clip_pre_seconds"), 1.0)
    clip_post_seconds = _get_float_config(outputs_cfg.get("clip_post_seconds"), 1.0)
    clip_subdir = str(outputs_cfg.get("clip_dir") or "clips")

    tracker = prepare_tracker(config, tracker_cfg)
    video_writer = None
    fps = metadata.fps or 25.0
    video_output_path = output_dir / f"{source_path.stem}.mp4"
    csv_output_path = output_dir / f"{source_path.stem}.csv"

    try:
        for frame_idx, result in enumerate(tracker.track(source_path)):
            frame = result.orig_img
            if frame is None:
                continue
            if frame_idx == 0:
                roi_manager.ensure_ready((frame.shape[1], frame.shape[0]))
                if args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_output_path),
                        fourcc,
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )

            boxes = getattr(result, "boxes", None)
            track_entries: List[Dict[str, object]] = []
            if boxes is not None and boxes.id is not None:
                ids = boxes.id
                if hasattr(ids, "cpu"):
                    ids = ids.cpu().numpy()
                ids_list = np.array(ids).flatten().tolist()

                coords = boxes.xyxy
                if hasattr(coords, "cpu"):
                    coords = coords.cpu().numpy()
                coords_list = np.array(coords).tolist()

                confs = boxes.conf
                if confs is not None and hasattr(confs, "cpu"):
                    confs = confs.cpu().numpy()
                conf_list = (
                    np.array(confs).flatten().tolist() if confs is not None else [None] * len(coords_list)
                )

                total = min(len(ids_list), len(coords_list))
                for idx in range(total):
                    track_id = ids_list[idx]
                    if track_id is None or (isinstance(track_id, float) and math.isnan(track_id)):
                        continue
                    track_id_int = int(track_id)

                    bbox = coords_list[idx] if idx < len(coords_list) else None
                    if bbox is None or len(bbox) < 4:
                        continue
                    conf = conf_list[idx] if idx < len(conf_list) else None
                    x1, y1, x2, y2 = bbox[:4]

                    footpoint = ((x1 + x2) / 2.0, y2)
                    inside = roi_manager.point_in_roi(footpoint)
                    track_entries.append(
                        {
                            "track_id": track_id_int,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "footpoint": [float(footpoint[0]), float(footpoint[1])],
                            "inside": inside,
                            "confidence": float(conf) if conf is not None else None,
                        }
                    )

            new_events = accumulator.update(frame_idx, track_entries)
            for event in new_events:
                duration = event.duration_frames
                print(
                    f"Completed event: track {event.track_id} | frames {event.start_frame}-{event.end_frame} | duration {duration}"
                )

            annotated = frame.copy()
            draw_overlays(annotated, roi_manager.polygon, track_entries, show_tracks, show_footpoints)
            if args.save_video and video_writer is not None:
                video_writer.write(annotated)

    finally:
        if video_writer is not None:
            video_writer.release()

    accumulator.flush()
    events = list(accumulator.completed)
    if args.save_csv:
        export_events(events, csv_output_path, fps)

    if args.clip:
        export_event_clips(events, metadata, output_dir, fps, clip_pre_seconds, clip_post_seconds, clip_subdir)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = load_config(config_path)

    source_path = resolve_path(PROJECT_ROOT, args.source)
    if source_path.is_dir():
        video_files = list(_iter_video_files(source_path))
        if not video_files:
            print(f"No supported video files found in {source_path}")
            return
        for video_file in video_files:
            process_video(video_file, config, args)
    else:
        process_video(source_path, config, args)


if __name__ == "__main__":
    main()
