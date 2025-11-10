#!/usr/bin/env python3
"""Batch OCR pipeline for fine plate crops."""
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np

from src.plates.plate_ocr import PlateOCR, PlateOCRError

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass
class OCRJob:
    path: Path
    rel_path: str
    name: str


@dataclass
class OCRRecord:
    image_path: str
    plate_text: str
    rec_confidence: float
    width: int
    height: int
    ocr_engine: str
    used_gpu: bool
    elapsed_ms: float
    ocr_img: str


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recognise text on cropped plate images")
    parser.add_argument("--input", required=True, help="Directory containing plate crops")
    parser.add_argument(
        "--rec-model-dir",
        default="weights/ppocr/ch_PP-OCRv4_rec_infer",
        help="PaddleOCR recognition model directory",
    )
    parser.add_argument(
        "--engine",
        choices=["paddle", "rapid"],
        default="paddle",
        help="OCR engine to use",
    )
    parser.add_argument("--use-gpu", action="store_true", help="Enable GPU for PaddleOCR")
    parser.add_argument("--min-conf", type=float, default=0.20, help="Minimum confidence threshold")
    parser.add_argument("--num-workers", type=int, default=4, help="Thread count for OCR")
    parser.add_argument("--csv", help="Optional CSV file to update with OCR results")
    parser.add_argument("--force", action="store_true", help="Re-run OCR even if results exist")
    return parser.parse_args(argv)


def _scan_images(input_dir: Path) -> List[OCRJob]:
    jobs: List[OCRJob] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        jobs.append(OCRJob(path=path, rel_path=path.name, name=path.name))
    return jobs


def _load_csv(csv_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    rows: List[Dict[str, str]] = []
    with csv_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = list(reader.fieldnames or [])
        for row in reader:
            rows.append(dict(row))
    return rows, header


def _save_csv(csv_path: Path, rows: List[Dict[str, str]], header: List[str]) -> None:
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _load_existing_results(results_path: Path) -> Dict[str, Dict[str, str]]:
    if not results_path.exists():
        return {}
    with results_path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["image_path"]: dict(row) for row in reader if row.get("image_path")}


def _normalise_path(path: Path) -> str:
    return path.as_posix()


def _draw_annotation(
    image: np.ndarray, text: str, conf: float, points: Optional[List[List[float]]]
) -> np.ndarray:
    annotated = image.copy()
    if points:
        pts = [(int(round(p[0])), int(round(p[1]))) for p in points]
        if len(pts) >= 4:
            cv2.polylines(annotated, [np.array(pts, dtype=int)], isClosed=True, color=(0, 255, 255), thickness=2)
    label = text if text else "<empty>"
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 32), (0, 0, 0), thickness=-1)
    cv2.putText(
        annotated,
        f"{label} | {conf:.2f}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return annotated


def _match_rows(rows: List[Dict[str, str]]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for idx, row in enumerate(rows):
        for key in ("fine_img", "plate_img", "tail_img"):
            value = row.get(key, "").strip()
            if not value:
                continue
            name = Path(value).name
            if name and name not in mapping:
                mapping[name] = idx
    return mapping


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    base_root = input_dir.parent
    try:
        input_prefix = input_dir.relative_to(base_root)
    except ValueError:
        input_prefix = Path(input_dir.name)

    jobs = _scan_images(input_dir)
    if not jobs:
        print("未找到可处理的图片。")
        return 0

    try:
        ocr = PlateOCR(
            engine=args.engine,
            use_gpu=args.use_gpu,
            rec_model_dir=args.rec_model_dir if args.engine == "paddle" else None,
            min_conf=args.min_conf,
        )
    except PlateOCRError as exc:
        raise SystemExit(str(exc))

    ocr_dir = input_dir / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    results_path = input_dir / "plate_ocr_results.csv"
    existing_results = {} if args.force else _load_existing_results(results_path)

    csv_rows: List[Dict[str, str]] = []
    csv_header: List[str] = []
    row_mapping: Dict[str, int] = {}
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
        csv_rows, csv_header = _load_csv(csv_path)
        row_mapping = _match_rows(csv_rows)
    else:
        csv_path = None

    processed_records: Dict[str, OCRRecord] = {}

    def submit(job: OCRJob) -> Optional[OCRRecord]:
        image_key = _normalise_path(input_prefix / job.rel_path)
        if not args.force and image_key in existing_results:
            previous = existing_results[image_key]
            if (previous.get("plate_text") or "").strip():
                print(f"[SKIP] {job.rel_path} 已有结果")
                return None
        image = cv2.imread(str(job.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            print(f"[WARN] 无法读取图片: {job.path}")
            return OCRRecord(
                image_path=job.rel_path,
                plate_text="",
                rec_confidence=0.0,
                width=0,
                height=0,
                ocr_engine=args.engine,
                used_gpu=args.use_gpu,
                elapsed_ms=0.0,
                ocr_img="",
            )
        result = ocr(image)
        text = result.get("text", "").strip()
        conf = float(result.get("conf", 0.0))
        points = result.get("points")
        elapsed = float(result.get("elapsed_ms", 0.0))
        if conf < args.min_conf:
            text = ""
        annotated = _draw_annotation(image, text, conf, points)
        ocr_rel = input_prefix / "ocr" / f"{job.path.stem}_ocr.jpg"
        ocr_path = ocr_dir / f"{job.path.stem}_ocr.jpg"
        cv2.imwrite(str(ocr_path), annotated)
        width, height = int(image.shape[1]), int(image.shape[0])
        return OCRRecord(
            image_path=image_key,
            plate_text=text,
            rec_confidence=conf,
            width=width,
            height=height,
            ocr_engine=args.engine,
            used_gpu=args.use_gpu,
            elapsed_ms=elapsed,
            ocr_img=_normalise_path(ocr_rel),
        )

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        future_map = {executor.submit(submit, job): job for job in jobs}
        for future in as_completed(future_map):
            job = future_map[future]
            try:
                record = future.result()
            except Exception as exc:
                print(f"[WARN] OCR 异常 {job.rel_path}: {exc}")
                continue
            if record is None:
                continue
            processed_records[record.image_path] = record
            print(
                f"[OK] {job.rel_path} text='{record.plate_text}' conf={record.rec_confidence:.3f}"
            )
            if csv_rows:
                basename = job.name
                if basename in row_mapping:
                    row = csv_rows[row_mapping[basename]]
                    row["plate_text"] = record.plate_text
                    row["plate_ocr_conf"] = f"{record.rec_confidence:.6f}"
                    row["plate_ocr_img"] = record.ocr_img

    combined = existing_results
    for rec in processed_records.values():
        combined[rec.image_path] = {
            "image_path": rec.image_path,
            "plate_text": rec.plate_text,
            "rec_confidence": f"{rec.rec_confidence:.6f}",
            "width": str(rec.width),
            "height": str(rec.height),
            "ocr_engine": rec.ocr_engine,
            "used_gpu": "true" if rec.used_gpu else "false",
            "elapsed_ms": f"{rec.elapsed_ms:.2f}",
            "ocr_img": rec.ocr_img,
        }

    if combined:
        fieldnames = [
            "image_path",
            "plate_text",
            "rec_confidence",
            "width",
            "height",
            "ocr_engine",
            "used_gpu",
            "elapsed_ms",
            "ocr_img",
        ]
        with results_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for key in sorted(combined.keys()):
                writer.writerow(combined[key])

    if csv_rows and csv_path is not None:
        if csv_header:
            if "plate_text" not in csv_header:
                csv_header.append("plate_text")
            for extra in ["plate_ocr_conf", "plate_ocr_img"]:
                if extra not in csv_header:
                    csv_header.append(extra)
        else:
            csv_header = ["plate_text", "plate_ocr_conf", "plate_ocr_img"]
        _save_csv(csv_path, csv_rows, csv_header)

    print(f"完成 OCR，共处理 {len(processed_records)} 张图片。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
