#!/usr/bin/env python3
"""Batch OCR for plate crops under data/outputs/<run>/plates."""
from __future__ import annotations

import argparse
import csv
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ocr import PlateOCR  # noqa: E402

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png"}
CSV_HEADER = [
    "image_path",
    "plate_text",
    "rec_confidence",
    "width",
    "height",
    "ocr_engine",
    "used_gpu",
    "elapsed_ms",
]


@dataclass
class ImageJob:
    abs_path: Path
    rel_path: str
    width: int
    height: int


@dataclass
class JobResult:
    rel_path: str
    text: str
    confidence: float
    width: int
    height: int
    elapsed_ms: int
    low_confidence: bool = False


class BatchRunner:
    def __init__(
        self,
        input_dir: Path,
        rec_model_dir: Path,
        use_gpu: bool,
        min_height: int,
        min_conf: float,
        num_workers: int,
    ) -> None:
        self.input_dir = input_dir
        self.rec_model_dir = rec_model_dir
        self.use_gpu = use_gpu
        self.min_height = min_height
        self.min_conf = min_conf
        self.num_workers = max(1, num_workers)
        self._thread_local: threading.local = threading.local()

    def _get_ocr(self) -> PlateOCR:
        ocr = getattr(self._thread_local, "ocr", None)
        if ocr is None:
            ocr = PlateOCR(
                lang="ch",
                det=False,
                rec=True,
                use_angle_cls=False,
                ocr_model_dir=str(self.rec_model_dir),
                use_gpu=self.use_gpu,
                log_csv_path="",
                crops_dir="",
                min_height=self.min_height,
                write_empty=True,
                min_conf=self.min_conf,
            )
            self._thread_local.ocr = ocr
        return ocr

    def _process(self, job: ImageJob) -> Optional[JobResult]:
        if job.height < self.min_height:
            return None
        if job.width > 512 or job.height > 512:
            return None

        try:
            with Image.open(job.abs_path) as im:
                crop_rgb = im.convert("RGB")
                crop_np = np.asarray(crop_rgb)
        except (OSError, UnidentifiedImageError):
            return None
        if crop_np.size == 0:
            return None
        # Pillow gives RGB; PaddleOCR expects BGR
        crop_bgr = crop_np[:, :, ::-1].copy()
        start = time.perf_counter()
        ocr = self._get_ocr()
        text, conf = ocr.recognize_crop(crop_bgr)
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        low_conf = conf < self.min_conf
        if low_conf:
            text = ""
        return JobResult(
            rel_path=job.rel_path,
            text=text,
            confidence=conf,
            width=job.width,
            height=job.height,
            elapsed_ms=elapsed_ms,
            low_confidence=low_conf,
        )

    def run(self, jobs: Sequence[ImageJob], dry_run: bool) -> Tuple[List[JobResult], Dict[str, int]]:
        stats = {
            "attempted": len(jobs),
            "skipped_small": 0,
            "skipped_large": 0,
            "skipped_failed": 0,
            "low_confidence": 0,
        }
        results: List[JobResult] = []
        if dry_run:
            for job in jobs:
                if job.height < self.min_height:
                    stats["skipped_small"] += 1
                elif job.width > 512 or job.height > 512:
                    stats["skipped_large"] += 1
                else:
                    stats.setdefault("will_process", 0)
                    stats["will_process"] += 1
            return results, stats

        from concurrent.futures import ThreadPoolExecutor

        def task(job: ImageJob) -> Optional[JobResult]:
            if job.height < self.min_height:
                with stats_lock:
                    stats["skipped_small"] += 1
                return None
            if job.width > 512 or job.height > 512:
                with stats_lock:
                    stats["skipped_large"] += 1
                return None
            res = self._process(job)
            if res is None:
                with stats_lock:
                    if job.height < self.min_height:
                        stats["skipped_small"] += 1
                    elif job.width > 512 or job.height > 512:
                        stats["skipped_large"] += 1
                    else:
                        stats["skipped_failed"] += 1
            return res

        stats_lock = threading.Lock()
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = [executor.submit(task, job) for job in jobs]
            for fut in futures:
                res = fut.result()
                if res is not None:
                    results.append(res)
                    if res.low_confidence:
                        stats.setdefault("low_confidence", 0)
                        stats["low_confidence"] += 1
        stats["processed"] = len(results)
        return results, stats


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch OCR on plate crops")
    parser.add_argument("--input", required=True, help="plates/ directory to scan")
    parser.add_argument("--rec-model-dir", required=True, help="PaddleOCR rec model dir")
    parser.add_argument("--use-gpu", default="false", choices=["true", "false"], help="Use GPU if available")
    parser.add_argument("--min-height", type=int, default=64, help="Minimum crop height to OCR")
    parser.add_argument("--min-conf", type=float, default=0.2, help="Minimum confidence threshold")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--dry-run", default="false", choices=["true", "false"], help="Only count without OCR")
    return parser.parse_args(argv)


def scan_images(
    input_dir: Path, processed: Iterable[str]
) -> Tuple[List[ImageJob], int, int, int]:
    processed_set = set(processed)
    jobs: List[ImageJob] = []
    total_images = 0
    already_done = 0
    unreadable = 0
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTS:
            continue
        total_images += 1
        try:
            rel_path = str(path.relative_to(ROOT))
        except ValueError:
            rel_path = str(path.resolve())
        if rel_path in processed_set:
            already_done += 1
            continue
        try:
            with Image.open(path) as im:
                width, height = im.size
        except (OSError, UnidentifiedImageError):
            unreadable += 1
            continue
        jobs.append(
            ImageJob(
                abs_path=path,
                rel_path=rel_path,
                width=int(width),
                height=int(height),
            )
        )
    return jobs, total_images, already_done, unreadable


def load_existing(csv_path: Path) -> List[str]:
    if not csv_path.exists():
        return []
    processed: List[str] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = (row.get("image_path") or "").strip()
            if img_path:
                processed.append(img_path)
    return processed


def append_results(csv_path: Path, rows: Sequence[JobResult], used_gpu: bool) -> None:
    if not rows:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(CSV_HEADER)
        for row in sorted(rows, key=lambda r: r.rel_path):
            writer.writerow(
                [
                    row.rel_path,
                    row.text,
                    f"{row.confidence:.4f}",
                    row.width,
                    row.height,
                    "paddleocr",
                    "true" if used_gpu else "false",
                    row.elapsed_ms,
                ]
            )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        print(f"Input directory not found: {input_dir}")
        return 1
    rec_model_dir = Path(args.rec_model_dir).expanduser().resolve()
    use_gpu = args.use_gpu.lower() == "true"
    dry_run = args.dry_run.lower() == "true"

    csv_path = input_dir / "plate_ocr_results.csv"
    existing = load_existing(csv_path)
    jobs, total_images, already_done, unreadable = scan_images(input_dir, existing)

    runner = BatchRunner(
        input_dir=input_dir,
        rec_model_dir=rec_model_dir,
        use_gpu=use_gpu,
        min_height=int(args.min_height),
        min_conf=float(args.min_conf),
        num_workers=int(args.num_workers),
    )

    start = time.perf_counter()
    results, stats = runner.run(jobs, dry_run=dry_run)
    elapsed = time.perf_counter() - start

    if not dry_run:
        append_results(csv_path, results, used_gpu=use_gpu)

    total_scanned = total_images
    skipped_small = stats.get("skipped_small", 0)
    skipped_large = stats.get("skipped_large", 0)
    skipped_failed = stats.get("skipped_failed", 0) + unreadable
    low_conf = stats.get("low_confidence", 0)
    newly_processed = (
        len(results)
        if not dry_run
        else stats.get("will_process", 0)
    )
    skipped_existing = already_done

    print("\n==== Plate OCR Batch Summary ====")
    print(f"Input directory   : {input_dir}")
    print(f"Model directory   : {rec_model_dir}")
    print(f"Use GPU           : {use_gpu}")
    print(f"Dry run           : {dry_run}")
    print(f"Total images      : {total_scanned}")
    print(f"Already processed : {skipped_existing}")
    if skipped_small:
        print(f"Skipped (<{runner.min_height}px): {skipped_small}")
    if skipped_large:
        print(f"Skipped (>512px)  : {skipped_large}")
    if skipped_failed:
        print(f"Skipped (failed)  : {skipped_failed}")
    print(f"Newly processed   : {newly_processed}")
    if low_conf:
        print(
            "  其中低于置信度阈值的有 "
            f"{low_conf} 张，可通过 --min-conf 调整阈值。"
        )
    print(f"Elapsed           : {elapsed:.2f}s")
    print("================================\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
