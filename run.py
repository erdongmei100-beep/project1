"""Emergency lane occupancy detection MVP runner."""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from src.dettrack.pipeline import DetectorTracker
from src.io.video import VideoMetadata, probe_video
from src.logic.events import EventAccumulator, FrameOccupancy, OccupancyEvent
from src.render.overlay import draw_overlays
from src.roi.manager import ROIManager
from src.utils.config import load_config, resolve_path, save_config
from src.utils.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    OUTPUTS_DIR,
    PROJECT_ROOT,
    project_path,
    resolve_project_path,
)
from src.plates import VehicleFilter
from src.utils.weights import ensure_plate_weights


DEFAULT_CONFIG_REL = project_path("configs", "default.yaml").relative_to(PROJECT_ROOT)
DEFAULT_TRACKER_CFG_REL = project_path("configs", "tracker", "bytetrack.yaml").relative_to(PROJECT_ROOT)
DEFAULT_ROIS_DIR = project_path("data", "rois")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Emergency lane occupancy detection MVP")
    parser.add_argument("--source", required=True, help="Path to input video")
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
    parser.add_argument(
        "--plate",
        action="store_true",
        help="Export per-event plate crops (best frame snapshot).",
    )
    parser.add_argument(
        "--weights-plate",
        default=None,
        help="Optional override for plate detector weights path.",
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
        device=model_cfg.get("device", "cpu"),
        imgsz=int(model_cfg.get("imgsz", 640)),
        tracker_cfg=tracker_cfg_path,
        conf=float(tracking_cfg.get("conf", 0.25)),
        iou=float(tracking_cfg.get("iou", 0.45)),
        max_det=int(tracking_cfg.get("max_det", 1000)),
    )


def export_events(
    events, output_csv: Path, fps: float, plate_metadata: Optional[List[Dict[str, object]]] = None
) -> None:
    if not events:
        print("No occupancy events detected; skipping CSV export.")
        return
    include_plate = plate_metadata is not None
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
        if include_plate:
            meta = {}
            if plate_metadata and len(plate_metadata) > len(rows):
                meta = plate_metadata[len(rows)]
            row.update(
                {
                    "plate_img": meta.get("plate_img", ""),
                    "plate_conf": meta.get("plate_conf"),
                    "plate_score": meta.get("plate_score"),
                    "plate_text": meta.get("plate_text"),
                    "tail_img": meta.get("tail_img", ""),
                    "plate_sharp_post": meta.get("plate_sharp_post"),
                }
            )
        rows.append(row)
    df = pd.DataFrame(rows)
    if include_plate:
        for column in [
            "plate_img",
            "plate_conf",
            "plate_score",
            "plate_text",
            "tail_img",
            "plate_sharp_post",
        ]:
            if column not in df.columns:
                df[column] = None
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


def _select_plate_frame(
    event: OccupancyEvent,
    metadata: VideoMetadata,
    fps: float,
    pre_seconds: float,
    post_seconds: float,
) -> Optional[Tuple[int, FrameOccupancy]]:
    if not event.frames:
        return None

    effective_fps = fps if fps > 0 else (metadata.fps if metadata.fps > 0 else 25.0)
    pre_frames = int(max(pre_seconds, 0.0) * effective_fps)
    post_frames = int(max(post_seconds, 0.0) * effective_fps)

    motion_type = _classify_event_motion(event, metadata)

    if motion_type == "front_static":
        reference = event.frames[0]
        frame_idx = max(reference.frame_idx - pre_frames, 0)
        return frame_idx, reference

    candidates: List[FrameOccupancy] = [
        frame for frame in event.frames if frame.frame_idx >= event.start_frame + post_frames
    ]
    if not candidates:
        candidates = event.frames[:]

    best_frame: Optional[FrameOccupancy] = None
    best_score: Optional[Tuple[float, float]] = None
    for frame_record in candidates:
        confidence = frame_record.confidence if frame_record.confidence is not None else float("-inf")
        proximity = -abs(event.end_frame - frame_record.frame_idx)
        score = (confidence, proximity)
        if best_score is None or score > best_score:
            best_score = score
            best_frame = frame_record

    if best_frame is None:
        best_frame = event.frames[-1]
    return best_frame.frame_idx, best_frame


def _read_frame(video_path: Path, frame_idx: int) -> Optional[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        return None
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(frame_idx, 0))
    success, frame = capture.read()
    capture.release()
    if not success:
        return None
    return frame


def _crop_with_padding(frame: np.ndarray, bbox: Tuple[float, float, float, float], padding: float) -> Optional[np.ndarray]:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    pad_ratio = max(padding, 0.0)
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio

    x_min = max(int(math.floor(x1 - pad_x)), 0)
    y_min = max(int(math.floor(y1 - pad_y)), 0)
    x_max = min(int(math.ceil(x2 + pad_x)), width)
    y_max = min(int(math.ceil(y2 + pad_y)), height)

    if x_max <= x_min or y_max <= y_min:
        return None
    return frame[y_min:y_max, x_min:x_max].copy()


def export_plate_crops(
    events: List[OccupancyEvent],
    metadata: VideoMetadata,
    output_dir: Path,
    fps: float,
    padding: float,
    plate_subdir: str,
    pre_seconds: float,
    post_seconds: float,
) -> None:
    if not events:
        print("No occupancy events detected; skipping plate export.")
        return

    plate_dir = output_dir / plate_subdir
    plate_dir.mkdir(parents=True, exist_ok=True)

    for index, event in enumerate(events, start=1):
        selection = _select_plate_frame(event, metadata, fps, pre_seconds, post_seconds)
        if selection is None:
            continue
        frame_idx, frame_info = selection
        frame = _read_frame(metadata.path, frame_idx)
        if frame is None:
            print(f"Failed to read frame {frame_idx} for plate capture (track {event.track_id}).")
            continue

        crop = _crop_with_padding(frame, frame_info.bbox, padding)
        if crop is None:
            print(f"Unable to crop plate for track {event.track_id}; bounding box out of range.")
            continue

        plate_name = f"{metadata.path.stem}_event{index:02d}_track{event.track_id}_{frame_idx:06d}.jpg"
        plate_path = plate_dir / plate_name
        if cv2.imwrite(str(plate_path), crop):
            print(f"Saved plate crop to {plate_path}")
        else:
            print(f"Failed to save plate crop for track {event.track_id}.")


def _get_float_config(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def main() -> None:
    args = parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (PROJECT_ROOT / config_path).resolve()
    config = load_config(config_path)

    source_path = resolve_path(PROJECT_ROOT, args.source)
    tracking_cfg = dict(config.get("tracking", {}) or {})
    tracker_cfg_rel = tracking_cfg.get("tracker_config", str(DEFAULT_TRACKER_CFG_REL))
    tracker_cfg = resolve_path(PROJECT_ROOT, tracker_cfg_rel)

    data_cfg = dict(config.get("data", {}) or {})
    weights_cfg = dict(config.get("weights", {}) or {})

    metadata = probe_video(source_path)
    print(
        f"Video: {metadata.path.name} | FPS: {metadata.fps:.2f} | Frames: {metadata.frame_count} | Size: {metadata.frame_size}"
    )

    roi_cfg = dict(config.get("roi", {}) or {})
    roi_candidates: List[Path] = []
    roi_path: Optional[Path] = None

    rois_dir = resolve_project_path(data_cfg.get("rois_dir"), DEFAULT_ROIS_DIR)
    fallback_roi_path = rois_dir / f"{source_path.stem}.json"
    roi_candidates.append(fallback_roi_path)

    if args.roi:
        roi_override = resolve_path(PROJECT_ROOT, str(args.roi))
        roi_candidates.append(roi_override)
        roi_path = roi_override
    else:
        roi_path_value = roi_cfg.get("path")
        if isinstance(roi_path_value, (str, Path)) and str(roi_path_value).strip():
            config_roi_path = resolve_path(PROJECT_ROOT, str(roi_path_value))
            roi_candidates.append(config_roi_path)
            roi_path = config_roi_path

    if roi_path is None:
        roi_path = fallback_roi_path

    if roi_path is None:
        attempted = ", ".join(str(path) for path in roi_candidates)
        raise FileNotFoundError(f"ROI path could not be resolved. Checked: {attempted}")

    if not roi_path.exists():
        # >>> 自动 ROI 生成
        auto_mode = str(roi_cfg.get("mode", "")).lower()
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
        # <<< 自动 ROI 生成
        if not roi_path.exists():
            try:
                from tools.roi_make import launch_roi_selector

                print("未找到 ROI 文件，正在启动交互式标注工具...")
                created = launch_roi_selector(str(source_path), str(roi_path))
            except Exception as exc:
                raise RuntimeError(f"无法创建 ROI：{exc}") from exc
            if not created or not roi_path.exists():
                attempted = ", ".join(str(path) for path in roi_candidates)
                raise RuntimeError(
                    "ROI 标注未完成，无法继续运行。已尝试路径: " f"{attempted}"
                )
        try:
            relative_path = roi_path.relative_to(PROJECT_ROOT)
            roi_cfg["path"] = str(relative_path)
        except ValueError:
            roi_cfg["path"] = str(roi_path)
        config["roi"] = roi_cfg
        try:
            save_config(config_path, config)
            print(f"已更新配置文件 {config_path} 的 ROI 路径。")
        except Exception as exc:
            print(f"警告：无法写入配置文件更新 ROI 路径：{exc}")
    else:
        try:
            relative_path = roi_path.relative_to(PROJECT_ROOT)
            roi_cfg.setdefault("path", str(relative_path))
        except ValueError:
            roi_cfg.setdefault("path", str(roi_path))
        config["roi"] = roi_cfg

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
    output_dir = resolve_project_path(outputs_cfg.get("root") or outputs_cfg.get("dir"), OUTPUTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    source_stem = metadata.path.stem
    # >>> 默认输出文件名（基于输入视频 stem）
    if not outputs_cfg.get("video_filename"):
        outputs_cfg["video_filename"] = f"{source_stem}.mp4"
    if not outputs_cfg.get("csv_filename"):
        outputs_cfg["csv_filename"] = f"{source_stem}.csv"
    # <<< 默认输出文件名

    clip_pre_seconds = _get_float_config(outputs_cfg.get("clip_pre_seconds"), 1.0)
    clip_post_seconds = _get_float_config(outputs_cfg.get("clip_post_seconds"), 1.0)
    clip_subdir = str(outputs_cfg.get("clip_dir") or "clips")

    plate_cfg = dict(config.get("plate", {}) or {})
    plate_enabled = bool(args.plate or plate_cfg.get("enable", False))
    plate_metadata: List[Dict[str, object]] = []
    collector = None
    plate_every_n_frames = 1
    vehicle_filter = VehicleFilter(
        plate_cfg.get("classes_allow"),
        float(plate_cfg.get("bbox_aspect_min", 0.9)),
        int(plate_cfg.get("min_box_h_px", 60)),
    )
    track_label_votes: Dict[int, Dict[str, Tuple[int, float]]] = {}

    if plate_enabled:
        from src.plates.detector_yolo import PlateDetector
        from src.plates.collector import PlateCollector

        plate_dirname = str(outputs_cfg.get("plate_dir") or "plates")
        plates_out_dir = output_dir / plate_dirname
        plates_out_dir.mkdir(parents=True, exist_ok=True)

        configured_weight = plate_cfg.get("det_weights") or weights_cfg.get("plate")
        env_plate_value = os.getenv("PLATE_WEIGHTS") or None
        configured_value = str(configured_weight) if configured_weight else None
        plate_weights_path = ensure_plate_weights(
            args.weights_plate,
            env_plate_value,
            configured_value,
        )

        plate_cfg_resolved = dict(plate_cfg)

        plate_cfg_resolved["det_weights"] = str(plate_weights_path)
        try:
            detector = PlateDetector(
                weights=plate_cfg_resolved["det_weights"],
                device=str(plate_cfg_resolved.get("device", "cpu")),
                imgsz=int(plate_cfg_resolved.get("imgsz", 320)),
                conf=float(plate_cfg_resolved.get("conf", 0.25)),
                iou=float(plate_cfg_resolved.get("iou", 0.45)),
            )
            collector = PlateCollector(plate_cfg_resolved, plates_out_dir, detector=detector)
            plate_every_n_frames = max(getattr(collector, "every_n_frames", 1), 1)
            if getattr(collector, "detector_available", False):
                print(
                    f"Plate detection enabled (weights: {plate_cfg_resolved['det_weights']}, device: {detector.device})."
                )
            else:
                print(
                    "Plate detector inactive; continuing with tail snapshot fallback only."
                )
        except Exception as exc:
            print(f"Failed to initialize plate detection; continuing without it: {exc}")
            collector = None
            plate_enabled = False

    if not plate_enabled:
        print("Plate detection disabled.")

    tracker = prepare_tracker(config, tracker_cfg)
    video_writer = None
    fps = metadata.fps or 25.0
    event_counter = 0

    def _default_plate_meta() -> Dict[str, object]:
        return {
            "plate_img": "",
            "plate_conf": None,
            "plate_score": None,
            "plate_text": "null",
            "tail_img": "",
            "plate_sharp_post": None,
        }

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
                        str(output_dir / outputs_cfg.get("video_filename", "occupancy.mp4")),
                        fourcc,
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )

            boxes = getattr(result, "boxes", None)
            track_entries: List[Dict[str, object]] = []
            present_vote_ids: set[int] = set()
            frame_height = frame.shape[0]
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

                cls_ids = getattr(boxes, "cls", None)
                if cls_ids is not None and hasattr(cls_ids, "cpu"):
                    cls_ids = cls_ids.cpu().numpy()
                cls_list = (
                    np.array(cls_ids).flatten().tolist() if cls_ids is not None else [None] * len(coords_list)
                )

                names_map = getattr(result, "names", None)

                total = min(len(ids_list), len(coords_list))
                for idx in range(total):
                    track_id = ids_list[idx]
                    if track_id is None or (isinstance(track_id, float) and math.isnan(track_id)):
                        continue
                    track_id_int = int(track_id)
                    present_vote_ids.add(track_id_int)

                    bbox = coords_list[idx]
                    conf = conf_list[idx] if idx < len(conf_list) else None
                    cls_id = cls_list[idx] if idx < len(cls_list) else None

                    class_name = ""
                    cls_index: Optional[int] = None
                    if cls_id is not None and not (isinstance(cls_id, float) and math.isnan(cls_id)):
                        try:
                            cls_index = int(cls_id)
                        except (TypeError, ValueError):
                            cls_index = None
                    if cls_index is not None:
                        if isinstance(names_map, dict):
                            class_name = str(names_map.get(cls_index, ""))
                        elif isinstance(names_map, (list, tuple)):
                            if 0 <= cls_index < len(names_map):
                                class_name = str(names_map[cls_index])
                        if not class_name:
                            class_name = str(cls_index)
                    class_name = class_name.lower() if class_name else ""

                    votes = track_label_votes.setdefault(track_id_int, {})
                    if class_name:
                        prev_count, prev_conf = votes.get(class_name, (0, 0.0))
                        conf_val = float(conf) if conf is not None else 0.0
                        votes[class_name] = (prev_count + 1, max(prev_conf, conf_val))

                    x1, y1, x2, y2 = bbox
                    width = max(float(x2) - float(x1), 0.0)
                    height = max(float(y2) - float(y1), 0.0)
                    if not vehicle_filter.is_vehicle(votes, (width, height), frame_height):
                        continue

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

            missing_vote_ids = set(track_label_votes.keys()) - present_vote_ids
            for missing_id in list(missing_vote_ids):
                track_label_votes.pop(missing_id, None)

            if plate_enabled and collector is not None and frame_idx % plate_every_n_frames == 0:
                for entry in track_entries:
                    if not entry.get("inside"):
                        continue
                    bbox_entry = entry["bbox"]
                    car_xyxy_full = [
                        int(math.floor(bbox_entry[0])),
                        int(math.floor(bbox_entry[1])),
                        int(math.ceil(bbox_entry[2])),
                        int(math.ceil(bbox_entry[3])),
                    ]
                    try:
                        collector.update(
                            track_id=int(entry["track_id"]),
                            frame_idx=frame_idx,
                            car_xyxy_full=car_xyxy_full,
                            frame_bgr=frame,
                        )
                    except Exception as exc:
                        print(f"Plate collector update failed for track {entry['track_id']}: {exc}")

            new_events = accumulator.update(frame_idx, track_entries)
            for event in new_events:
                duration = event.duration_frames
                print(
                    f"Completed event: track {event.track_id} | frames {event.start_frame}-{event.end_frame} | duration {duration}"
                )
                event_counter += 1
                if plate_enabled:
                    meta = _default_plate_meta()
                    if collector is not None:
                        try:
                            meta = collector.finalize_and_save(
                                event_counter, event.track_id, start_frame=event.start_frame
                            )
                        except Exception as exc:
                            print(f"Plate finalize failed for track {event.track_id}: {exc}")
                        finally:
                            try:
                                collector.clear(event_counter, event.track_id)
                            except Exception as clear_exc:
                                print(
                                    f"Failed to clear plate collector for track {event.track_id}: {clear_exc}"
                                )
                    plate_metadata.append(meta)

            annotated = frame.copy()
            draw_overlays(annotated, roi_manager.polygon, track_entries, show_tracks, show_footpoints)
            if args.save_video and video_writer is not None:
                video_writer.write(annotated)

    finally:
        if video_writer is not None:
            video_writer.release()

    remaining_events = accumulator.flush()
    for event in remaining_events:
        event_counter += 1
        if plate_enabled:
            meta = _default_plate_meta()
            if collector is not None:
                try:
                    meta = collector.finalize_and_save(
                        event_counter, event.track_id, start_frame=event.start_frame
                    )
                except Exception as exc:
                    print(f"Plate finalize failed for track {event.track_id}: {exc}")
                finally:
                    try:
                        collector.clear(event_counter, event.track_id)
                    except Exception as clear_exc:
                        print(
                            f"Failed to clear plate collector for track {event.track_id}: {clear_exc}"
                        )
            plate_metadata.append(meta)

    events = list(accumulator.completed)
    if args.save_csv:
        csv_path = output_dir / outputs_cfg.get("csv_filename", "occupancy.csv")
        export_events(events, csv_path, fps, plate_metadata if plate_enabled else None)

    if args.clip:
        export_event_clips(events, metadata, output_dir, fps, clip_pre_seconds, clip_post_seconds, clip_subdir)


if __name__ == "__main__":
    main()

