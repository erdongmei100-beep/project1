"""Batch OCR utility for plate crops."""
from __future__ import annotations

import argparse
import csv
import inspect
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import cv2  # type: ignore
import numpy as np

try:  # pragma: no cover - optional dependency guard
    from paddleocr import PaddleOCR  # type: ignore
except Exception:  # pragma: no cover - allow fallback
    PaddleOCR = None

try:  # pragma: no cover - optional dependency guard
    from rapidocr_onnxruntime import RapidOCR  # type: ignore
except Exception:  # pragma: no cover - allow fallback
    RapidOCR = None

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "data" / "outputs"
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
class OCRResult:
    rel_path: str
    text: str
    confidence: float
    width: int
    height: int
    elapsed_ms: int


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch OCR for fine plate crops")
    parser.add_argument(
        "--input",
        default=None,
        help="Directory of plate crops. Defaults to data/outputs/*/plates_fine or fallback to plates/.",
    )
    parser.add_argument(
        "--engine",
        choices=["paddle", "rapid"],
        default="paddle",
        help="OCR engine to use.",
    )
    parser.add_argument(
        "--rec-model-dir",
        default="",
        help="PaddleOCR recognition model directory (required for paddle engine).",
    )
    parser.add_argument(
        "--use-gpu",
        default="false",
        help="Whether to enable GPU acceleration (engine dependent).",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=48,
        help="Minimum crop height for OCR processing.",
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.10,
        help="Minimum confidence threshold to keep OCR text.",
    )
    return parser.parse_args(argv)


def resolve_input_dirs(input_arg: Optional[str]) -> List[Path]:
    if input_arg:
        path = Path(input_arg).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            raise FileNotFoundError(f"Input directory not found: {path}")
        return [path]

    if not DEFAULT_OUTPUTS.exists():
        return []

    fine_dirs = sorted(p for p in DEFAULT_OUTPUTS.glob("*/plates_fine") if p.is_dir())
    if fine_dirs:
        return fine_dirs
    plate_dirs = sorted(p for p in DEFAULT_OUTPUTS.glob("*/plates") if p.is_dir())
    return plate_dirs


def load_existing(csv_path: Path) -> set[str]:
    existing: set[str] = set()
    if not csv_path.exists():
        return existing
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rel = (row.get("image_path") or "").strip()
            if rel:
                existing.add(rel)
    return existing


def enhance_for_ocr(image_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)
    blur = cv2.GaussianBlur(enhanced, (0, 0), 0.6)
    sharpened = cv2.addWeighted(enhanced, 1.6, blur, -0.6, 0)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


def build_paddle_runner(rec_model_dir: str, use_gpu: bool) -> Tuple[Callable[[np.ndarray], Tuple[str, float]], str]:
    if PaddleOCR is None:
        raise RuntimeError("PaddleOCR is not available. Install paddleocr to use --engine paddle.")
    if not rec_model_dir:
        raise ValueError("--rec-model-dir is required for paddle engine")
    model_path = Path(rec_model_dir).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"PaddleOCR model directory not found: {model_path}")
    kwargs = {
        "det": True,
        "rec": True,
        "cls": False,
        "use_angle_cls": False,
        "rec_model_dir": str(model_path),
        "use_gpu": use_gpu,
    }
    signature = inspect.signature(PaddleOCR.__init__)
    if "lang" in signature.parameters:
        kwargs["lang"] = "ch"
    if "show_log" in signature.parameters:
        kwargs["show_log"] = False
    ocr = PaddleOCR(**kwargs)

    def runner(image_bgr: np.ndarray) -> Tuple[str, float]:
        try:
            result = ocr.ocr(image_bgr, cls=False)
        except Exception:
            return "", 0.0
        if not result:
            return "", 0.0
        best_text = ""
        best_conf = 0.0
        for item in result:
            if not item or len(item) < 2:
                continue
            data = item[1]
            if not data or len(data) < 2:
                continue
            text = str(data[0]).strip()
            try:
                conf = float(data[1])
            except Exception:
                conf = 0.0
            if conf > best_conf:
                best_conf = conf
                best_text = text
        return best_text, best_conf

    return runner, "paddle"


def build_rapid_runner() -> Tuple[Callable[[np.ndarray], Tuple[str, float]], str]:
    if RapidOCR is None:
        raise RuntimeError("RapidOCR is not available. Install rapidocr-onnxruntime to use --engine rapid.")
    ocr = RapidOCR()

    def runner(image_bgr: np.ndarray) -> Tuple[str, float]:
        try:
            result, _ = ocr(image_bgr)
        except Exception:
            return "", 0.0
        if not result:
            return "", 0.0
        best_text = ""
        best_conf = 0.0
        for item in result:
            if not item or len(item) < 3:
                continue
            text = str(item[1]).strip()
            try:
                conf = float(item[2])
            except Exception:
                conf = 0.0
            if conf > best_conf:
                best_conf = conf
                best_text = text
        return best_text, best_conf

    return runner, "rapid"


def collect_images(input_dir: Path) -> List[Path]:
    images: List[Path] = []
    for path in sorted(input_dir.iterdir()):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTS:
            images.append(path)
    return images


def write_results(csv_path: Path, rows: Iterable[OCRResult], engine: str, use_gpu: bool) -> None:
    rows_list = list(rows)
    if not rows_list:
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(CSV_HEADER)
        for row in rows_list:
            writer.writerow(
                [
                    row.rel_path,
                    row.text,
                    f"{row.confidence:.4f}",
                    row.width,
                    row.height,
                    engine,
                    "true" if use_gpu else "false",
                    row.elapsed_ms,
                ]
            )


def process_directory(
    input_dir: Path,
    engine_name: str,
    rec_model_dir: str,
    use_gpu: bool,
    min_height: int,
    min_conf: float,
) -> Tuple[int, int, int, int, int]:
    run_dir = input_dir.parent
    csv_path = input_dir / "plate_ocr_results.csv"
    processed_before = load_existing(csv_path)
    images = collect_images(input_dir)

    if engine_name == "paddle":
        runner, engine_label = build_paddle_runner(rec_model_dir, use_gpu)
    else:
        runner, engine_label = build_rapid_runner()

    results: List[OCRResult] = []
    skipped_small = 0
    skipped_existing = 0
    skipped_unreadable = 0
    skipped_conf = 0

    for path in images:
        rel_path = str(path.relative_to(run_dir))
        if rel_path in processed_before:
            skipped_existing += 1
            continue
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            skipped_unreadable += 1
            continue
        height, width = image.shape[:2]
        if height < min_height:
            skipped_small += 1
            continue
        enhanced = enhance_for_ocr(image)
        start = time.perf_counter()
        text, conf = runner(enhanced)
        elapsed_ms = int(round((time.perf_counter() - start) * 1000))
        if conf < min_conf or not text:
            skipped_conf += 1
            text = ""
        results.append(
            OCRResult(
                rel_path=rel_path,
                text=text,
                confidence=conf,
                width=width,
                height=height,
                elapsed_ms=elapsed_ms,
            )
        )

    write_results(csv_path, results, engine_label, use_gpu)

    print("\n==== Plate OCR Batch Summary ====")
    print(f"Input directory   : {input_dir}")
    print(f"Model directory   : {rec_model_dir if rec_model_dir else '-'}")
    print(f"OCR engine        : {engine_label}")
    print(f"Use GPU           : {use_gpu}")
    print(f"Total images      : {len(images)}")
    print(f"Already processed : {skipped_existing}")
    if skipped_small:
        print(f"Skipped (<{min_height}px): {skipped_small}")
    if skipped_unreadable:
        print(f"Skipped (unreadable): {skipped_unreadable}")
    if skipped_conf:
        print(f"Below min-conf     : {skipped_conf}")
    newly_processed = len(results)
    print(f"Newly processed   : {newly_processed}")
    print("================================\n")

    return (
        len(images),
        skipped_existing,
        skipped_small,
        skipped_unreadable,
        newly_processed,
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        input_dirs = resolve_input_dirs(args.input)
    except Exception as exc:
        print(exc)
        return 1

    if not input_dirs:
        print("No plate directories found. Use --input to specify a directory explicitly.")
        return 1

    use_gpu = parse_bool(args.use_gpu)
    total_processed = 0
    total_new = 0
    for directory in input_dirs:
        try:
            stats = process_directory(
                directory,
                engine_name=args.engine,
                rec_model_dir=args.rec_model_dir,
                use_gpu=use_gpu,
                min_height=int(args.min_height),
                min_conf=float(args.min_conf),
            )
        except Exception as exc:
            print(f"Failed to process {directory}: {exc}")
            continue
        total_processed += stats[0]
        total_new += stats[-1]

    if total_new == 0:
        print("No new images were processed. Adjust thresholds or check input paths if needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
