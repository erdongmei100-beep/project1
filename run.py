"""Emergency lane occupancy detection MVP runner."""
from __future__ import annotations

import argparse
import copy
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import torch

from src.dettrack.pipeline import DetectorTracker
from src.io.video import VideoMetadata, probe_video
from src.logic.events import EventAccumulator, FrameOccupancy, OccupancyEvent
from src.roi.manager import ROIManager
from src.render.overlay import draw_overlays
from src.utils.config import load_config, resolve_path
from src.utils.geometry import point_in_polygon
from src.utils.paths import PROJECT_ROOT, project_path


DEFAULT_CONFIG_REL = project_path("configs", "default.yaml").relative_to(PROJECT_ROOT)
DEFAULT_TRACKER_CFG_REL = project_path("configs", "tracker", "bytetrack.yaml").relative_to(PROJECT_ROOT)
DEFAULT_LANE_WEIGHTS_REL = project_path("weights", "lane_seg.pt").relative_to(PROJECT_ROOT)


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
        "--lane-weights",
        default=str(DEFAULT_LANE_WEIGHTS_REL),
        help="Path to emergency lane segmentation weights.",
    )
    parser.add_argument(
        "--clip",
        action="store_true",
        help="Export per-event video clips with buffered context.",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Enable diagnostic logging and failure snapshots for ROI debugging.",
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


def _extract_emergency_lane_mask(
    result,
    frame_shape: Tuple[int, int],
    class_name: str = "emergency_lane",
    fallback_class_id: int = 0,
) -> Optional[np.ndarray]:
    if result is None or result.masks is None or result.boxes is None:
        return None

    frame_height, frame_width = frame_shape

    target_class_id: Optional[int] = None
    names = getattr(result, "names", None) or {}
    for cls_id, name in names.items():
        if name == class_name:
            target_class_id = int(cls_id)
            break

    if target_class_id is None:
        target_class_id = int(fallback_class_id)

    class_ids = result.boxes.cls
    if class_ids is None:
        return None

    cls_ids = class_ids.cpu().numpy() if hasattr(class_ids, "cpu") else np.asarray(class_ids)
    polygons = getattr(result.masks, "xy", None) or []
    mask_img = np.zeros((frame_height, frame_width), dtype=np.uint8)

    for polygon, cls_id in zip(polygons, cls_ids):
        if int(cls_id) == target_class_id:
            cv2.fillPoly(mask_img, [polygon.astype(np.int32)], 1)

    return mask_img.astype(bool) if np.any(mask_img) else None


def _point_in_mask(point: Tuple[float, float], mask: Optional[np.ndarray]) -> bool:
    if mask is None:
        return False
    h, w = mask.shape[:2]
    x, y = int(point[0]), int(point[1])
    if x < 0 or y < 0 or x >= w or y >= h:
        return False
    return bool(mask[y, x])


def _overlay_lane_mask(frame: np.ndarray, mask: Optional[np.ndarray], color=(255, 255, 0), alpha: float = 0.3) -> np.ndarray:
    if mask is None:
        return frame
    overlay = frame.copy()
    overlay[mask] = color
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)
    return frame


def _bbox_quality(bbox: Tuple[float, float, float, float], confidence: Optional[float]) -> float:
    x1, y1, x2, y2 = bbox
    area = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    conf = confidence if confidence is not None else 1.0
    return float(area * conf)


def _lane_metrics(mask: Optional[np.ndarray], frame_size: Tuple[int, int]) -> Tuple[float, Optional[float]]:
    width, height = frame_size
    total_pixels = float(width * height) if width > 0 and height > 0 else 1.0
    if mask is None:
        return 0.0, None

    area = float(np.count_nonzero(mask))
    area_ratio = area / total_pixels

    moments = cv2.moments(mask.astype(np.uint8))
    if moments["m00"] == 0:
        return area_ratio, None
    cx = moments["m10"] / moments["m00"]
    return area_ratio, float(cx / width) if width > 0 else None


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
    model_cfg = config.get("model", {})
    tracking_cfg = dict(config.get("tracking", {}) or {})
    tracker_cfg_rel = tracking_cfg.get("tracker_config", str(DEFAULT_TRACKER_CFG_REL))
    tracker_cfg = resolve_path(PROJECT_ROOT, tracker_cfg_rel)

    roi_cfg = config.get("roi", {}) or {}
    dynamic_seg_cfg = dict(roi_cfg.get("dynamic_seg") or {})
    lane_conf = dynamic_seg_cfg.get("conf")
    lane_imgsz = dynamic_seg_cfg.get("imgsz")
    lane_class_id = int(dynamic_seg_cfg.get("class_id", 0))
    lane_class_name = str(dynamic_seg_cfg.get("class_name", "emergency_lane"))

    lane_weights_path = resolve_path(PROJECT_ROOT, args.lane_weights)
    lane_model = YOLO(str(lane_weights_path))
    print(f"Lane model classes: {lane_model.names}")
    lane_predict_args: Dict[str, object] = {"verbose": False}
    lane_predict_args["retina_masks"] = False
    if model_cfg.get("device") is not None:
        lane_predict_args["device"] = model_cfg.get("device", 0)
    if lane_conf is not None:
        lane_predict_args["conf"] = float(lane_conf)
    if lane_imgsz is not None:
        lane_predict_args["imgsz"] = int(lane_imgsz)

    filters_cfg = dict(roi_cfg.get("dynamic_filters") or {})
    roi_manager = ROIManager(
        min_area_ratio=float(filters_cfg.get("min_area_ratio", 0.005)),
        max_area_ratio=float(filters_cfg.get("max_area_ratio", 0.4)),
        expected_centroid=(
            float(filters_cfg.get("min_centroid_x_ratio", 0.55)),
            float(
                filters_cfg.get(
                    "min_centroid_y_ratio",
                    filters_cfg.get("max_centroid_y_ratio", 0.55),
                )
            ),
        ),
        stability_ratio=float(filters_cfg.get("max_centroid_jump", 0.25)),
        good_streak_thresh=int(filters_cfg.get("good_streak", 5)),
        bad_streak_tolerance=int(filters_cfg.get("bad_streak_tolerance", 5)),
    )

    metadata = probe_video(source_path)
    print(
        f"Video: {metadata.path.name} | FPS: {metadata.fps:.2f} | Frames: {metadata.frame_count} | Size: {metadata.frame_size}"
    )

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
    snapshot_dir = output_dir / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    failure_dir = output_dir / "failures"
    debug_log_path = output_dir / "debug_log.csv"
    if args.diagnose:
        failure_dir.mkdir(parents=True, exist_ok=True)

    clip_pre_seconds = _get_float_config(outputs_cfg.get("clip_pre_seconds"), 1.0)
    clip_post_seconds = _get_float_config(outputs_cfg.get("clip_post_seconds"), 1.0)
    clip_subdir = str(outputs_cfg.get("clip_dir") or "clips")

    tracker = prepare_tracker(config, tracker_cfg)
    video_writer = None
    fps = metadata.fps or 25.0
    video_output_path = output_dir / f"{source_path.stem}.mp4"
    csv_output_path = output_dir / f"{source_path.stem}.csv"
    best_snapshots: Dict[int, Dict[str, object]] = {}
    debug_rows: List[Dict[str, object]] = []
    last_failure_frame = -int(math.ceil(fps))
    roi_active = False
    roi_reason = "uninitialized"
    active_polygon: List[Tuple[float, float]] = []

    try:
        for frame_idx, result in enumerate(tracker.track(source_path)):
            frame = result.orig_img
            if frame is None:
                continue
            if frame_idx == 0:
                if args.save_video:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(
                        str(video_output_path),
                        fourcc,
                        fps,
                        (frame.shape[1], frame.shape[0]),
                    )

            lane_results = lane_model.predict(source=frame, **lane_predict_args)
            frame_height, frame_width = frame.shape[:2]
            lane_mask: Optional[np.ndarray] = None
            if lane_results and lane_results[0].masks is not None:
                lane_result = lane_results[0]
                lane_mask_img = np.zeros(frame.shape[:2], dtype=np.uint8)
                polygons = lane_result.masks.xy or []
                class_ids = lane_result.boxes.cls if lane_result.boxes is not None else None
                cls_ids = class_ids.cpu().numpy() if hasattr(class_ids, "cpu") else np.asarray(class_ids) if class_ids is not None else None
                if cls_ids is not None:
                    for polygon, cls_id in zip(polygons, cls_ids):
                        if int(cls_id) == lane_class_id:
                            cv2.fillPoly(lane_mask_img, [polygon.astype(np.int32)], 1)
                if np.any(lane_mask_img):
                    lane_mask = lane_mask_img.astype(bool)

            if lane_mask is None:
                print(f"[DEBUG] Frame {frame_idx}: YOLO detected no mask for class {lane_class_id}")

            roi_status = roi_manager.update_from_segmentation(lane_mask, (frame_width, frame_height))
            if not roi_status.valid:
                print(f"[DEBUG] Frame {frame_idx}: ROI rejected by Manager. Reason: {roi_status.reason}")

            roi_active = roi_status.valid
            active_polygon = roi_status.polygon if roi_active else []

            roi_reason = roi_status.reason or ("ok" if roi_active else "unknown")
            ego_point = (frame_width / 2.0, frame_height - 1)
            ego_in_lane = _point_in_mask(ego_point, lane_mask)
            lane_area_ratio, lane_centroid_x = _lane_metrics(lane_mask, (frame_width, frame_height))

            boxes = getattr(result, "boxes", None)
            track_entries: List[Dict[str, object]] = []
            num_vehicles_on_right = 0
            coords_list: List[List[float]] = []
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

                    center_x = (float(x1) + float(x2)) / 2.0
                    if frame_width > 0 and center_x / float(frame_width) > 0.6:
                        num_vehicles_on_right += 1

                    footpoint = ((x1 + x2) / 2.0, y2)
                    inside = False
                    if roi_active and not ego_in_lane and active_polygon:
                        inside = point_in_polygon(footpoint, active_polygon)
                    track_entries.append(
                        {
                            "track_id": track_id_int,
                            "bbox": [float(x1), float(y1), float(x2), float(y2)],
                            "footpoint": [float(footpoint[0]), float(footpoint[1])],
                            "inside": inside,
                            "confidence": float(conf) if conf is not None else None,
                        }
                    )

                    if inside and roi_active:
                        quality = _bbox_quality((float(x1), float(y1), float(x2), float(y2)), conf)
                        prev = best_snapshots.get(track_id_int)
                        if not prev or quality > prev.get("quality", 0):
                            best_snapshots[track_id_int] = {
                                "frame": frame.copy(),
                                "mask": lane_mask.copy() if lane_mask is not None else None,
                                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                                "quality": quality,
                            }

            new_events = accumulator.update(frame_idx, track_entries, roi_active=roi_active)
            completed_start = len(accumulator.completed) - len(new_events)
            for offset, event in enumerate(new_events):
                duration = event.duration_frames
                event_id = completed_start + offset + 1
                print(
                    f"Completed event: track {event.track_id} | frames {event.start_frame}-{event.end_frame} | duration {duration}"
                )
                best = best_snapshots.pop(event.track_id, None)
                if best:
                    snapshot = best["frame"].copy()
                    _overlay_lane_mask(snapshot, best.get("mask"))
                    bx1, by1, bx2, by2 = map(int, best["bbox"])
                    cv2.rectangle(snapshot, (bx1, by1), (bx2, by2), (0, 0, 255), 3)
                    cv2.putText(
                        snapshot,
                        f"Event {event_id} | Track {event.track_id}",
                        (max(5, bx1), max(20, by1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    snap_path = snapshot_dir / f"event_{event_id:03d}.jpg"
                    cv2.imwrite(str(snap_path), snapshot)

            if args.diagnose:
                debug_rows.append(
                    {
                        "frame_idx": frame_idx,
                        "timestamp_s": float(frame_idx / fps) if fps else 0.0,
                        "roi_active": int(roi_active),
                        "roi_reason": roi_reason,
                        "lane_area_ratio": lane_area_ratio,
                        "lane_centroid_x": lane_centroid_x,
                        "num_vehicles_on_right": num_vehicles_on_right,
                    }
                )

                should_snapshot = (
                    not roi_active
                    and num_vehicles_on_right > 0
                    and frame_idx - last_failure_frame >= int(max(1, round(fps)))
                )
                if should_snapshot:
                    failure_frame = frame.copy()
                    if lane_mask is not None:
                        overlay = failure_frame.copy()
                        overlay[lane_mask] = (255, 0, 0)
                        cv2.addWeighted(overlay, 0.4, failure_frame, 0.6, 0, dst=failure_frame)
                    if coords_list:
                        for bbox in coords_list:
                            if bbox is None or len(bbox) < 4:
                                continue
                            bx1, by1, bx2, by2 = map(int, bbox[:4])
                            cv2.rectangle(failure_frame, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        failure_frame,
                        f"Frame {frame_idx} | reason: {roi_reason}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )
                    failure_path = failure_dir / f"frame_{frame_idx:06d}_{roi_reason}.jpg"
                    cv2.imwrite(str(failure_path), failure_frame)
                    last_failure_frame = frame_idx

            annotated = frame.copy()
            _overlay_lane_mask(annotated, lane_mask if roi_active else None)
            draw_overlays(
                annotated,
                active_polygon if roi_active else None,
                track_entries,
                show_tracks,
                show_footpoints,
            )
            if args.save_video and video_writer is not None:
                video_writer.write(annotated)

    finally:
        if video_writer is not None:
            video_writer.release()

    accumulator.flush()
    events = list(accumulator.completed)
    if args.diagnose and debug_rows:
        debug_df = pd.DataFrame(debug_rows)
        debug_df.to_csv(debug_log_path, index=False)
        print(f"Saved diagnostic log to {debug_log_path}")

        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            times = debug_df["timestamp_s"].tolist()
            ratios = debug_df["lane_area_ratio"].tolist()
            roi_states = debug_df["roi_active"].tolist()
            ax1.plot(times, ratios, label="lane_area_ratio", color="blue")
            if times:
                for idx in range(len(times) - 1):
                    start_t = times[idx]
                    end_t = times[idx + 1]
                    color = "green" if roi_states[idx] else "red"
                    ax1.axvspan(start_t, end_t, color=color, alpha=0.1)
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Lane area ratio")
            ax1.set_title("Lane area ratio over time")
            ax1.legend()

            reasons = Counter(debug_df["roi_reason"].tolist())
            ax2.bar(reasons.keys(), reasons.values(), color="purple")
            ax2.set_ylabel("Count")
            ax2.set_title("ROI rejection reasons")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()

            report_path = output_dir / "report_analysis.png"
            plt.savefig(report_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved diagnostic report to {report_path}")
        except Exception as exc:  # pragma: no cover - best effort plotting
            print(f"Skipping diagnostic plots due to error: {exc}")

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
