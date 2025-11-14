"""Collect plate detection candidates across events."""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from src.utils.paths import OUTPUTS_DIR, WEIGHTS_DIR

from .detector import PlateDetector as CandidatePlateDetector
from .fine_crop import crop_by_xyxy, ensure_xyxy_int, expand_with_margin
from .plate_det import PlateDetector as FinePlateDetector
from .preprocess import enhance_plate
from .quality import (
    angle_penalty,
    laplacian_sharpness,
    laplacian_var,
    relative_area,
    score_plate,
)


class VehicleFilter:
    """Lightweight helper to decide whether a tracked object is a vehicle."""

    def __init__(
        self,
        classes_allow: Optional[List[str]] = None,
        bbox_aspect_min: float = 0.9,
        min_box_h_px: int = 60,
    ) -> None:
        allow = classes_allow or ["car", "truck", "bus"]
        self.classes_allow = {str(name).lower() for name in allow}
        self.bbox_aspect_min = float(bbox_aspect_min)
        self.min_box_h_px = int(min_box_h_px)
        self.min_height_ratio = 0.06

    def is_vehicle(
        self,
        track_label_votes: Dict[str, Tuple[int, float]],
        bbox_wh: Tuple[float, float],
        frame_h: int,
    ) -> bool:
        """Return True when the track appears to be a valid vehicle."""

        if not track_label_votes:
            return False

        best_label: Optional[str] = None
        best_count = -1
        best_conf = -1.0
        for label, (count, conf) in track_label_votes.items():
            if count > best_count or (count == best_count and conf > best_conf):
                best_label = label
                best_count = count
                best_conf = conf

        if best_label is None:
            return False

        if self.classes_allow and best_label not in self.classes_allow:
            return False

        width, height = bbox_wh
        if height <= 0 or width <= 0:
            return False

        aspect = width / height
        if aspect < self.bbox_aspect_min:
            return False

        adaptive_min = max(self.min_box_h_px, int(round(self.min_height_ratio * max(frame_h, 1))))
        if height < adaptive_min:
            return False

        return True


@dataclass
class PlateCandidate:
    frame_idx: int
    xyxy_full: List[int]
    xyxy_crop: List[int]
    conf: float
    rel_area: float
    sharpness: float
    score: float
    crop_bgr: np.ndarray
    plate_crop_bgr: np.ndarray
    frame_bgr: np.ndarray
    roi_flag: Optional[bool] = None


class PlateCollector:
    """Maintain plate detection candidates per track/event."""

    def __init__(
        self,
        cfg_plate: dict,
        out_dir: Path | None,
        detector: Optional[PlateDetector] = None,
        fine_detector: Optional[FinePlateDetector] = None,
        video_fps: Optional[float] = None,
        cam_id: Optional[str] = None,
    ) -> None:
        self.cfg = dict(cfg_plate or {})
        self.out_dir = Path(out_dir) if out_dir else OUTPUTS_DIR / "plates"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.debug_dir = self.out_dir / "debug"
        self.save_debug = bool(self.cfg.get("save_debug", False))
        self.preproc_cfg = dict(self.cfg.get("preproc", {}))
        self.preproc_debug = bool(self.preproc_cfg.get("save_debug", False))

        classes_allow = self.cfg.get("classes_allow")
        aspect_min = float(self.cfg.get("bbox_aspect_min", 0.9))
        min_box_h_px = int(self.cfg.get("min_box_h_px", 60))
        self.vehicle_filter = VehicleFilter(classes_allow, aspect_min, min_box_h_px)

        self.only_with_plate = bool(self.cfg.get("only_with_plate", False))
        self.allow_tail_fallback = bool(self.cfg.get("allow_tail_fallback", True))
        self.pick_mode = str(self.cfg.get("pick_mode", "full")).lower()
        self.entry_window_frames = max(int(self.cfg.get("entry_window_frames", 0)), 0)

        det_weights_default = WEIGHTS_DIR / "plate" / "yolov8n-plate.pt"
        weights_cfg = self.cfg.get("det_weights")
        weights_path = Path(str(weights_cfg)) if weights_cfg else det_weights_default
        device = str(self.cfg.get("device", "cpu"))
        imgsz = int(self.cfg.get("imgsz", 320))
        conf = float(self.cfg.get("conf", 0.25))
        iou = float(self.cfg.get("iou", 0.45))

        self.detector = detector or CandidatePlateDetector(
            weights=str(weights_path),
            device=device,
            imgsz=imgsz,
            conf=conf,
            iou=iou,
        )
        self.detector_available = bool(getattr(self.detector, "ready", True))

        self.save_fine_plate = bool(self.cfg.get("save_fine_plate", True))
        self.save_gray_plate = bool(self.cfg.get("save_gray_plate", False))
        self.fine_mode = str(self.cfg.get("fine_crop_mode", "reuse_bbox")).lower()
        if self.fine_mode not in {"reuse_bbox", "redetect"}:
            self.fine_mode = "reuse_bbox"
        self.fine_margin = float(
            self.cfg.get("plate_crop_margin", self.cfg.get("plate_margin", 0.12))
        )
        self.fine_area_fail_ratio = float(self.cfg.get("plate_crop_area_fail_ratio", 0.92))
        self.rel_plate_dir = Path("plates")
        self.rel_fine_dir = Path("plates_fine")
        self.fine_out_dir = self.out_dir.parent / self.rel_fine_dir
        if self.save_fine_plate:
            self.fine_out_dir.mkdir(parents=True, exist_ok=True)

        self.fine_detector: Optional[FinePlateDetector] = None
        if self.fine_mode == "redetect":
            self.fine_detector = fine_detector
            if self.fine_detector is None:
                fine_weights_cfg = self.cfg.get("plate_crop_weights") or self.cfg.get("plate_det_weights")
                fine_weights = (
                    Path(str(fine_weights_cfg))
                    if fine_weights_cfg
                    else WEIGHTS_DIR / "plate" / "yolov8n-plate.pt"
                )
                fine_conf = float(
                    self.cfg.get("plate_crop_conf", self.cfg.get("plate_det_conf", 0.2))
                )
                fine_imgsz = int(
                    self.cfg.get("plate_crop_imgsz", self.cfg.get("plate_det_imgsz", 960))
                )
                try:
                    self.fine_detector = FinePlateDetector(
                        weights=str(fine_weights),
                        conf=fine_conf,
                        imgsz=fine_imgsz,
                        margin=self.fine_margin,
                        device=str(self.cfg.get("device", "cpu")),
                    )
                except Exception as exc:
                    print(
                        f"[plate] Fine detector unavailable; continuing without fine crop: {exc}"
                    )
                    self.fine_detector = None

        self.cam_id = str(cam_id) if cam_id is not None else None

        self.every_n_frames = max(int(self.cfg.get("every_n_frames", 3)), 1)
        self.padding = float(self.cfg.get("crop_padding", 0.15))
        self.min_rel_area = float(self.cfg.get("min_rel_area", 0.005))
        self.min_sharpness = float(self.cfg.get("min_sharpness", 120.0))

        self._candidates: Dict[int, List[PlateCandidate]] = {}
        self._best_candidate: Dict[int, PlateCandidate] = {}
        self._tail_best: Dict[int, Tuple[float, int, np.ndarray]] = {}

        self.debug_preproc_dir = self.debug_dir / "preproc"

        if not self.detector_available:
            print(
                "Plate detector unavailable; falling back to tail image export only."
            )

        _ = video_fps  # retained for API compatibility


    def update(
        self,
        track_id: int,
        frame_idx: int,
        car_xyxy_full: List[int],
        frame_bgr: np.ndarray,
        roi_flag: Optional[bool] = None,
    ) -> Optional[PlateCandidate]:
        crop_info = self._crop_with_padding(frame_bgr, car_xyxy_full)
        if crop_info is None:
            return None
        crop_bgr, crop_x1, crop_y1 = crop_info
        crop_context = crop_bgr.copy()

        car_area = (car_xyxy_full[2] - car_xyxy_full[0]) * (car_xyxy_full[3] - car_xyxy_full[1])
        if car_area <= 0:
            car_area = 1.0
        tail_sharpness = laplacian_sharpness(crop_bgr)
        tail_score = car_area * tail_sharpness
        tail_entry = self._tail_best.get(track_id)
        if tail_entry is None or tail_score > tail_entry[0]:
            self._tail_best[track_id] = (tail_score, frame_idx, crop_bgr.copy())

        if not self.detector_available:
            return None

        detections = self.detector.detect(crop_bgr)
        if not detections:
            print(f"[plate] no detection at frame {frame_idx}")
            return self._best_candidate.get(track_id)

        best_for_track = self._best_candidate.get(track_id)
        frame_height, frame_width = frame_bgr.shape[:2]

        for det in detections:
            xyxy = det.get("xyxy")
            conf = float(det.get("conf", 0.0))
            if xyxy is None:
                continue
            px1, py1, px2, py2 = [float(v) for v in xyxy]
            px1_i = max(int(math.floor(px1)), 0)
            py1_i = max(int(math.floor(py1)), 0)
            px2_i = max(int(math.ceil(px2)), px1_i + 1)
            py2_i = max(int(math.ceil(py2)), py1_i + 1)

            plate_crop = crop_bgr[py1_i:py2_i, px1_i:px2_i]
            if plate_crop.size == 0:
                continue

            full_xyxy = [
                int(px1_i + crop_x1),
                int(py1_i + crop_y1),
                int(px2_i + crop_x1),
                int(py2_i + crop_y1),
            ]
            rel_area = relative_area(full_xyxy, car_xyxy_full)
            if rel_area < self.min_rel_area:
                continue

            sharpness = laplacian_sharpness(plate_crop)
            if sharpness < self.min_sharpness:
                continue

            angle_factor = angle_penalty(full_xyxy)
            score = score_plate(conf, rel_area, sharpness, angle_factor)

            candidate = PlateCandidate(
                frame_idx=frame_idx,
                xyxy_full=full_xyxy,
                xyxy_crop=[px1_i, py1_i, px2_i, py2_i],
                conf=conf,
                rel_area=rel_area,
                sharpness=sharpness,
                score=score,
                crop_bgr=crop_context,
                plate_crop_bgr=plate_crop.copy(),
                frame_bgr=frame_bgr.copy(),
                roi_flag=roi_flag,
            )
            self._candidates.setdefault(track_id, []).append(candidate)
            if best_for_track is None or candidate.score > best_for_track.score:
                best_for_track = candidate
                self._best_candidate[track_id] = candidate
            if self.save_debug:
                self._save_debug_crop(track_id, frame_idx, candidate)

        return best_for_track

    def select_best(self, event_id: int, track_id: int) -> Optional[PlateCandidate]:
        _ = event_id  # placeholder for future use
        return self._best_candidate.get(track_id)

    def is_vehicle(
        self, track_label_votes: Dict[str, Tuple[int, float]], bbox_wh: Tuple[float, float], frame_h: int
    ) -> bool:
        return self.vehicle_filter.is_vehicle(track_label_votes, bbox_wh, frame_h)

    def finalize_and_save(
        self, event_id: int, track_id: int, start_frame: Optional[int] = None
    ) -> Dict[str, object]:
        meta: Dict[str, object] = {
            "plate_img": "",
            "plate_conf": None,
            "plate_score": None,
            "tail_img": "",
            "plate_sharp_post": None,
            "fine_img": "",
            "plate_det_conf": None,
            "plate_det_bbox": "",
            "plate_det_success": False,
            "plate_bbox_xyxy": "",
            "best_frame_img": "",
            "best_frame_w": None,
            "best_frame_h": None,
            "plate_frame_idx": None,
        }

        candidates = list(self._candidates.get(track_id, []))
        best = self._select_candidate(track_id, candidates, start_frame)

        if best is None:
            if self.allow_tail_fallback and not self.only_with_plate:
                tail_entry = self._tail_best.get(track_id)
                if tail_entry is not None:
                    _, tail_frame_idx, tail_crop = tail_entry
                    tail_name = f"event{event_id:02d}_track{track_id}_tail.jpg"
                    tail_path = self.out_dir / tail_name
                    self.out_dir.mkdir(parents=True, exist_ok=True)
                    if cv2.imwrite(str(tail_path), tail_crop):
                        rel_tail = self._normalize_rel_path(self.rel_plate_dir / tail_name)
                        meta["tail_img"] = rel_tail
                        print(
                            f"[plate] fallback tail saved for track {track_id} frame {tail_frame_idx}"
                        )
            return meta

        meta["plate_conf"] = float(best.conf)
        meta["plate_score"] = float(best.score)
        meta["plate_sharp_post"] = float(best.sharpness)
        meta["plate_frame_idx"] = int(best.frame_idx)
        meta["plate_bbox_xyxy"] = ",".join(str(int(v)) for v in best.xyxy_full)

        frame_bgr = self._ensure_bgr(best.frame_bgr)
        frame_h, frame_w = frame_bgr.shape[:2]
        meta["best_frame_w"] = int(frame_w)
        meta["best_frame_h"] = int(frame_h)

        frame_dir = self.out_dir.parent / "plates_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_name = f"event{event_id:02d}_track{track_id}_frame{best.frame_idx:06d}.jpg"
        frame_path = frame_dir / frame_name
        if cv2.imwrite(str(frame_path), frame_bgr):
            meta["best_frame_img"] = self._normalize_rel_path(Path("plates_frames") / frame_name)
        else:
            print(f"[plate] failed to save best frame image: {frame_path}")

        tail_bgr = self._ensure_bgr(best.crop_bgr)
        tail_name = f"event{event_id:02d}_track{track_id}_tail.jpg"
        tail_path = self.out_dir / tail_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        if cv2.imwrite(str(tail_path), tail_bgr):
            meta["tail_img"] = self._normalize_rel_path(self.rel_plate_dir / tail_name)
        else:
            print(f"[plate] failed to save tail crop: {tail_path}")

        plate_crop_bgr = self._ensure_bgr(best.plate_crop_bgr)
        plate_name = f"event{event_id:02d}_track{track_id}_det.jpg"
        plate_path = self.out_dir / plate_name
        plate_rel = ""
        if cv2.imwrite(str(plate_path), plate_crop_bgr):
            plate_rel = self._normalize_rel_path(self.rel_plate_dir / plate_name)
            meta["plate_img"] = plate_rel
        else:
            print(f"[plate] failed to save coarse plate crop: {plate_path}")

        fine_result = self._run_fine_crop(
            event_id,
            track_id,
            best,
            plate_crop_bgr,
            plate_crop_bgr,
            plate_rel,
            f"event{event_id:02d}_track{track_id}",
        )
        if fine_result:
            meta.update({k: v for k, v in fine_result.items() if k in meta})
            det_bbox = fine_result.get("plate_det_bbox")
            if det_bbox and isinstance(det_bbox, (tuple, list)):
                meta["plate_det_bbox"] = ",".join(str(int(v)) for v in det_bbox)
            elif not det_bbox:
                meta["plate_det_bbox"] = ""
            if "plate_det_conf" in fine_result and fine_result["plate_det_conf"] is not None:
                meta["plate_det_conf"] = float(fine_result["plate_det_conf"])

        print(
            "[plate] event=%02d track=%d frame=%d bbox=%s" % (
                event_id,
                track_id,
                best.frame_idx,
                meta["plate_bbox_xyxy"],
            )
        )

        return meta

    def _run_fine_crop(
        self,
        event_id: int,
        track_id: int,
        candidate: PlateCandidate,
        plate_bgr: np.ndarray,
        fallback_plate_bgr: np.ndarray,
        plate_rel_path: str,
        plate_name_stem: str,
    ) -> Dict[str, object]:
        result: Dict[str, object] = {
            "fine_img": "",
            "plate_det_conf": None,
            "plate_det_bbox": None,
            "plate_det_success": False,
        }

        base_plate = plate_bgr if plate_bgr is not None and plate_bgr.size > 0 else None
        fallback_crop = (
            fallback_plate_bgr if fallback_plate_bgr is not None and fallback_plate_bgr.size > 0 else None
        )
        if base_plate is None:
            base_plate = fallback_crop
        if base_plate is None:
            return result

        det_conf: float = 0.0
        det_bbox: Optional[Tuple[int, int, int, int]] = None
        det_success = False
        fine_crop: Optional[np.ndarray] = None

        raw_bbox: Optional[Tuple[int, int, int, int]] = None
        if getattr(candidate, "xyxy_crop", None):
            try:
                raw_bbox = ensure_xyxy_int(candidate.xyxy_crop)
            except Exception:
                raw_bbox = None

        if self.fine_mode == "reuse_bbox" and raw_bbox is not None:
            height, width = base_plate.shape[:2]
            det_bbox = expand_with_margin(raw_bbox, self.fine_margin, width, height)
            fine_crop = crop_by_xyxy(base_plate, det_bbox)
            if fine_crop is not None and fine_crop.size > 0:
                det_bbox_valid: Optional[Tuple[int, int, int, int]] = det_bbox
                plate_area = max(width * height, 1)
                det_area = max((det_bbox[2] - det_bbox[0]) * (det_bbox[3] - det_bbox[1]), 1)
                area_ratio = det_area / float(plate_area)
                if area_ratio >= self.fine_area_fail_ratio:
                    det_bbox_raw = expand_with_margin(raw_bbox, 0.0, width, height)
                    raw_area = max(
                        (det_bbox_raw[2] - det_bbox_raw[0]) * (det_bbox_raw[3] - det_bbox_raw[1]),
                        1,
                    )
                    raw_ratio = raw_area / float(plate_area)
                    if raw_ratio < self.fine_area_fail_ratio:
                        det_bbox_valid = det_bbox_raw
                    else:
                        det_bbox_valid = None
                if det_bbox_valid is not None:
                    det_bbox = det_bbox_valid
                    fine_crop = crop_by_xyxy(base_plate, det_bbox)
                else:
                    print(
                        f"[plate] event={event_id:02d} track={track_id} fine crop bbox ratio>=threshold"
                    )
                    det_bbox = None
                    fine_crop = None
                    if fallback_crop is not None and fallback_crop.size > 0:
                        fine_crop = fallback_crop
            if fine_crop is not None and fine_crop.size > 0:
                det_success = True
                det_conf = float(candidate.conf)
        elif self.fine_mode == "redetect" and self.fine_detector is not None:
            try:
                fine_result = self.fine_detector.fine_crop(base_plate)
            except Exception as exc:
                print(
                    f"[plate] event={event_id:02d} track={track_id} fine crop error: {exc}"
                )
                fine_result = None

            if fine_result is not None:
                crop_candidate, det_bbox, det_conf = fine_result
                plate_h, plate_w = base_plate.shape[:2]
                det_w = max(det_bbox[2] - det_bbox[0], 1)
                det_h = max(det_bbox[3] - det_bbox[1], 1)
                det_area = det_w * det_h
                plate_area = max(plate_h * plate_w, 1)
                area_ratio = det_area / float(plate_area)
                if area_ratio >= self.fine_area_fail_ratio and fallback_crop is not None:
                    print(
                        "[plate] fine crop bbox ~full image; fallback to primary plate crop"
                    )
                    det_bbox = None
                    det_conf = 0.0
                else:
                    fine_crop = crop_candidate
                    det_success = True

            if not det_success and raw_bbox is not None:
                height, width = base_plate.shape[:2]
                det_bbox = expand_with_margin(raw_bbox, self.fine_margin, width, height)
                fine_crop = crop_by_xyxy(base_plate, det_bbox)
                if fine_crop is not None and fine_crop.size > 0:
                    det_conf = float(candidate.conf)
                    det_success = True

            if not det_success and plate_rel_path:
                print(f"[plate] event={event_id:02d} track={track_id} fine crop det_fail")
        elif self.fine_mode == "redetect" and self.fine_detector is None and raw_bbox is not None:
            height, width = base_plate.shape[:2]
            det_bbox = expand_with_margin(raw_bbox, self.fine_margin, width, height)
            fine_crop = crop_by_xyxy(base_plate, det_bbox)
            if fine_crop is not None and fine_crop.size > 0:
                det_conf = float(candidate.conf)
                det_success = True

        if det_success and fine_crop is not None and fine_crop.size > 0:
            if self.save_fine_plate:
                fine_name = f"{plate_name_stem}_plate.jpg"
                fine_path = self.fine_out_dir / fine_name
                if cv2.imwrite(str(fine_path), fine_crop):
                    result["fine_img"] = self._normalize_rel_path(self.rel_fine_dir / fine_name)
                else:
                    print(f"[plate] Failed to save fine plate image: {fine_path}")

        result["plate_det_conf"] = det_conf if det_success else None
        result["plate_det_bbox"] = det_bbox
        result["plate_det_success"] = det_success
        return result

    def _select_candidate(
        self,
        track_id: int,
        candidates: List[PlateCandidate],
        start_frame: Optional[int],
    ) -> Optional[PlateCandidate]:
        if not candidates:
            return None

        window_candidates: List[PlateCandidate] = []
        if (
            self.pick_mode == "entry"
            and start_frame is not None
            and self.entry_window_frames >= 0
        ):
            window_end = start_frame + self.entry_window_frames
            window_candidates = [
                cand
                for cand in candidates
                if start_frame <= cand.frame_idx <= window_end
            ]

        pool = window_candidates if window_candidates else candidates
        best = max(pool, key=lambda cand: cand.score, default=None)
        if best is not None:
            self._best_candidate[track_id] = best
        else:
            self._best_candidate.pop(track_id, None)
        return best

    def clear(self, event_id: int, track_id: int) -> None:
        _ = event_id  # placeholder for future use
        self._candidates.pop(track_id, None)
        self._best_candidate.pop(track_id, None)
        self._tail_best.pop(track_id, None)

    def _crop_with_padding(
        self, frame_bgr: np.ndarray, bbox: List[int]
    ) -> Optional[Tuple[np.ndarray, int, int]]:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [float(v) for v in bbox]
        pad_x = (x2 - x1) * max(self.padding, 0.0)
        pad_y = (y2 - y1) * max(self.padding, 0.0)

        x_min = max(int(math.floor(x1 - pad_x)), 0)
        y_min = max(int(math.floor(y1 - pad_y)), 0)
        x_max = min(int(math.ceil(x2 + pad_x)), width)
        y_max = min(int(math.ceil(y2 + pad_y)), height)

        if x_max <= x_min or y_max <= y_min:
            return None
        return frame_bgr[y_min:y_max, x_min:x_max].copy(), x_min, y_min

    def _save_debug_crop(self, track_id: int, frame_idx: int, candidate: PlateCandidate) -> None:
        try:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return
        filename = f"track{track_id}_f{frame_idx}_s{candidate.score:.3f}.jpg"
        crop = candidate.plate_crop_bgr if hasattr(candidate, "plate_crop_bgr") else candidate.crop_bgr
        cv2.imwrite(str(self.debug_dir / filename), crop)

    # >>> 修改开始: 预处理调试图拼接
    def _save_preproc_debug(
        self, event_id: int, track_id: int, stages: Optional[Dict[str, np.ndarray]]
    ) -> None:
        if not isinstance(stages, dict):
            return
        try:
            self.debug_preproc_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        keys = ["raw", "deskew", "rectified", "clahe", "denoise", "sharp", "binary"]
        tiles = []
        for key in keys:
            img = stages.get(key)
            if img is None:
                continue
            tiles.append((key, self._ensure_bgr(img)))
        if not tiles:
            return

        tile_h, tile_w = 180, 280
        cols = 3
        rows = int(math.ceil(len(tiles) / cols))
        montage = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)

        for idx, (label, img) in enumerate(tiles):
            resized = cv2.resize(img, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
            r = idx // cols
            c = idx % cols
            y0, x0 = r * tile_h, c * tile_w
            montage[y0 : y0 + tile_h, x0 : x0 + tile_w] = resized
            cv2.putText(
                montage,
                label,
                (x0 + 8, y0 + tile_h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
                cv2.LINE_AA,
            )

        debug_path = self.debug_preproc_dir / f"event{event_id:02d}_track{track_id}.jpg"
        cv2.imwrite(str(debug_path), montage)

    def _ensure_bgr(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return np.zeros((1, 1, 3), dtype=np.uint8)
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.ndim == 3:
            if img.shape[2] == 1:
                return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[2] == 3:
                return img.copy()
            if img.shape[2] == 4:
                return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        clipped = np.clip(img, 0, 255).astype(np.uint8)
        if clipped.ndim == 2:
            return cv2.cvtColor(clipped, cv2.COLOR_GRAY2BGR)
        return clipped

    def _normalize_rel_path(self, rel_path: Path | str) -> str:
        if not rel_path:
            return ""
        path_obj = Path(rel_path)
        if not path_obj.parts:
            return ""
        return str(Path(*path_obj.parts))
    # <<< 修改结束

