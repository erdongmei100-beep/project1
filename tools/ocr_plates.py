#!/usr/bin/env python3
"""Batch OCR pipeline for fine plate crops."""
from __future__ import annotations

import argparse
import csv
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

try:  # pragma: no cover - optional dependency check
    import cv2
    _CV2_IMPORT_ERROR: Optional[Exception] = None
except Exception as exc:  # pragma: no cover - dependency may be missing
    cv2 = None  # type: ignore[assignment]
    _CV2_IMPORT_ERROR = exc

import numpy as np

if __package__ is None:  # pragma: no cover - direct execution support
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from src.plates.plate_ocr import PlateOCR

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
_REQUIRED_PADDLE_FILES = (
    "inference.pdmodel",
    "inference.pdiparams",
    "inference.pdiparams.info",
)

_EXAMPLE_POWERSHELL = textwrap.dedent(
    """\
    python -m tools.ocr_plates `
      --input data/outputs/ambulance/plates `
      --rec-model-dir weights/ppocr/ch_PP-OCRv4_rec_infer `
      --engine auto `
      --use-gpu false `
      --min-height 64 `
      --min-conf 0.20 `
      --num-workers 4 `
      --dry-run false
    """
).strip()

_EXAMPLE_BASH = textwrap.dedent(
    """\
    python -m tools.ocr_plates \
      --input data/outputs/ambulance/plates \
      --rec-model-dir weights/ppocr/ch_PP-OCRv4_rec_infer \
      --engine paddle \
      --min-height 64 \
      --min-conf 0.20 \
      --num-workers 4 \
      --use-gpu
    """
).strip()


class FriendlyArgumentParser(argparse.ArgumentParser):
    """ArgumentParser that prints friendly examples on error."""

    def error(self, message: str) -> None:  # pragma: no cover - argparse exits
        self.print_usage(sys.stderr)
        examples = textwrap.dedent(
            f"""
            {self.prog}: error: {message}

            PowerShell 示例:
              {_EXAMPLE_POWERSHELL}

            bash 示例:
              {_EXAMPLE_BASH}
            """
        )
        self.exit(2, examples)


def str2bool(value: str) -> bool:
    """Convert a string to boolean, accepting common variants."""

    true_set = {"1", "true", "t", "yes", "y", "on"}
    false_set = {"0", "false", "f", "no", "n", "off"}
    if isinstance(value, bool):
        return value
    value_lower = value.strip().lower()
    if value_lower in true_set:
        return True
    if value_lower in false_set:
        return False
    raise argparse.ArgumentTypeError(f"无效布尔值: {value}")


def _build_parser() -> FriendlyArgumentParser:
    parser = FriendlyArgumentParser(
        description="Recognise text on cropped plate images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            f"""
            示例 (PowerShell 布尔值风格):
              {_EXAMPLE_POWERSHELL}

            示例 (bash 开关风格):
              {_EXAMPLE_BASH}
            """
        ),
    )
    parser.add_argument("--input", help="Directory containing plate crops")
    parser.add_argument(
        "--rec-model-dir",
        default="weights/ppocr/ch_PP-OCRv4_rec_infer",
        help="PaddleOCR recognition model directory",
    )
    parser.add_argument(
        "--engine",
        choices=["paddle", "rapid", "auto"],
        default="paddle",
        help="OCR engine to use (auto 会在 Paddle 模型缺失时降级到 rapid)",
    )
    parser.add_argument(
        "--use-gpu",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{true,false}",
        help="Enable GPU for PaddleOCR",
    )
    parser.add_argument(
        "--no-use-gpu",
        dest="use_gpu",
        action="store_false",
        help="Disable GPU explicitly",
    )
    parser.add_argument(
        "--dry-run",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        metavar="{true,false}",
        help="Plan OCR without executing (不会写入结果)",
    )
    parser.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Disable dry-run mode",
    )
    parser.add_argument(
        "--min-height",
        type=int,
        default=64,
        help="Minimum side length (px) required to run OCR",
    )
    parser.add_argument("--min-conf", type=float, default=0.20, help="Minimum confidence threshold")
    parser.add_argument("--num-workers", type=int, default=4, help="Thread count for OCR")
    parser.add_argument("--csv", help="Optional CSV file to update with OCR results")
    parser.add_argument("--force", action="store_true", help="Re-run OCR even if results exist")
    parser.add_argument("--self-test", action="store_true", help="Run argument parsing self-test")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.self_test and not args.input:
        parser.error("--input 参数是必需的 (self-test 模式除外)")
    return args


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


def _validate_paddle_model_dir(model_dir: Path) -> Tuple[bool, List[str]]:
    missing: List[str] = []
    if not model_dir.exists():
        missing = list(_REQUIRED_PADDLE_FILES)
        return False, missing
    for filename in _REQUIRED_PADDLE_FILES:
        target = model_dir / filename
        if not target.is_file():
            missing.append(filename)
    return len(missing) == 0, missing


def _resolve_engine(requested_engine: str, model_dir: Path) -> Tuple[str, Optional[str], Optional[str]]:
    """Resolve OCR engine considering auto fallback and model availability."""

    engine = requested_engine.lower()
    fallback_reason: Optional[str] = None
    error_message: Optional[str] = None

    if engine == "auto":
        ok, missing = _validate_paddle_model_dir(model_dir)
        if ok:
            engine = "paddle"
        else:
            fallback_reason = ", ".join(missing)
            engine = "rapid"
    elif engine == "paddle":
        ok, missing = _validate_paddle_model_dir(model_dir)
        if not ok:
            missing_str = ", ".join(missing)
            error_message = (
                f"PaddleOCR 模型目录不完整: {model_dir}\n"
                f"缺少以下文件: {missing_str}\n"
                "请从 https://github.com/PaddlePaddle/PaddleOCR/releases 下载官方 inference 模型并解压至该目录，"
                "或使用 --engine rapid"
            )
    return engine, fallback_reason, error_message


def _print_param_overview(
    *,
    input_dir: Path,
    engine: str,
    requested_engine: str,
    use_gpu: bool,
    dry_run: bool,
    min_height: int,
    min_conf: float,
    num_workers: int,
    rec_model_dir: Path,
) -> None:
    print("参数设置:")
    if requested_engine != engine:
        print(f"  engine: {engine} (requested: {requested_engine})")
    else:
        print(f"  engine: {engine}")
    print(f"  use_gpu: {'true' if use_gpu else 'false'}")
    print(f"  dry_run: {'true' if dry_run else 'false'}")
    print(f"  min_height: {min_height}px")
    print(f"  min_conf: {min_conf:.2f}")
    print(f"  num_workers: {num_workers}")
    print(f"  input: {input_dir}")
    print(f"  rec_model_dir: {rec_model_dir}")


def _run_self_test() -> int:
    from tempfile import TemporaryDirectory

    parser = _build_parser()
    ns1 = parser.parse_args(
        [
            "--input",
            "dummy",
            "--use-gpu",
            "false",
            "--dry-run",
            "false",
            "--engine",
            "auto",
        ]
    )
    assert ns1.use_gpu is False and ns1.dry_run is False
    ns2 = parser.parse_args([
        "--input",
        "dummy",
        "--use-gpu",
        "--dry-run",
    ])
    assert ns2.use_gpu is True and ns2.dry_run is True
    with TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        ns3 = parser.parse_args([
            "--input",
            tmp_dir,
            "--rec-model-dir",
            tmp_dir,
            "--engine",
            "auto",
            "--dry-run",
        ])
        assert ns3.engine == "auto"
        ok, missing = _validate_paddle_model_dir(tmp_path)
        assert not ok and set(missing) == set(_REQUIRED_PADDLE_FILES)
        resolved, fallback_reason, error_msg = _resolve_engine("auto", tmp_path)
        assert resolved == "rapid" and fallback_reason
        assert error_msg is None
    print("self-test OK")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return _run_self_test()

    if cv2 is None and _CV2_IMPORT_ERROR is not None:
        raise SystemExit(
            "OpenCV (cv2) 导入失败: "
            f"{_CV2_IMPORT_ERROR}. 请先安装 OpenCV 并确保系统依赖完整。"
        )

    from src.plates.plate_ocr import PlateOCR, PlateOCRError

    input_dir = Path(args.input).expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"输入目录不存在: {input_dir}")

    rec_model_dir = Path(args.rec_model_dir).expanduser().resolve()

    requested_engine = args.engine.lower()
    engine, fallback_reason, error_message = _resolve_engine(requested_engine, rec_model_dir)
    if error_message:
        print(error_message)
        return 2
    if requested_engine == "auto" and engine == "rapid" and fallback_reason:
        print(
            "[WARN] PaddleOCR 模型缺失，自动降级到 RapidOCR。缺少: "
            f"{fallback_reason}"
        )

    base_root = input_dir.parent
    try:
        input_prefix = input_dir.relative_to(base_root)
    except ValueError:
        input_prefix = Path(input_dir.name)

    jobs = _scan_images(input_dir)
    total_jobs = len(jobs)
    if not jobs:
        print("未找到可处理的图片。")
        return 0

    _print_param_overview(
        input_dir=input_dir,
        engine=engine,
        requested_engine=requested_engine,
        use_gpu=args.use_gpu,
        dry_run=args.dry_run,
        min_height=int(args.min_height),
        min_conf=float(args.min_conf),
        num_workers=int(args.num_workers),
        rec_model_dir=rec_model_dir,
    )
    if args.dry_run:
        print("dry-run: 不会实际识别，只输出计划处理数量。")

    existing_results = {}
    results_path = input_dir / "plate_ocr_results.csv"
    if not args.force:
        existing_results = _load_existing_results(results_path)

    csv_rows: List[Dict[str, str]] = []
    csv_header: List[str] = []
    row_mapping: Dict[str, int] = {}
    csv_path: Optional[Path] = None
    if args.csv:
        csv_path = Path(args.csv).expanduser().resolve()
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
        csv_rows, csv_header = _load_csv(csv_path)
        row_mapping = _match_rows(csv_rows)

    stats = {
        "total": total_jobs,
        "skipped_existing": 0,
        "skipped_small": 0,
        "errors": 0,
        "processed": 0,
        "planned": 0,
        "new_results": 0,
        "low_conf": 0,
    }
    stats_lock = Lock()

    ocr: Optional["PlateOCR"] = None
    if not args.dry_run:
        try:
            ocr = PlateOCR(
                engine=engine,
                use_gpu=args.use_gpu,
                rec_model_dir=str(rec_model_dir) if engine == "paddle" else None,
                min_conf=args.min_conf,
            )
        except PlateOCRError as exc:
            raise SystemExit(str(exc))
        ocr_dir = input_dir / "ocr"
        ocr_dir.mkdir(parents=True, exist_ok=True)
    else:
        ocr_dir = input_dir / "ocr"

    processed_records: Dict[str, OCRRecord] = {}
    start_time = time.perf_counter()

    def process_job(job: OCRJob, perform_ocr: bool) -> Optional[OCRRecord]:
        image_key = _normalise_path(input_prefix / job.rel_path)
        if not args.force and image_key in existing_results:
            previous = existing_results[image_key]
            if (previous.get("plate_text") or "").strip():
                with stats_lock:
                    stats["skipped_existing"] += 1
                print(f"[SKIP] {job.rel_path} 已有结果")
                return None
        image = cv2.imread(str(job.path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            with stats_lock:
                stats["errors"] += 1
            print(f"[WARN] 无法读取图片: {job.path}")
            return None
        height, width = int(image.shape[0]), int(image.shape[1])
        if min(height, width) < int(args.min_height):
            with stats_lock:
                stats["skipped_small"] += 1
            print(
                f"[SKIP] {job.rel_path} 尺寸过小 {width}x{height}px (<{args.min_height})"
            )
            return None
        if not perform_ocr:
            with stats_lock:
                stats["planned"] += 1
            print(f"[PLAN] {job.rel_path} 将执行 OCR")
            return None

        assert ocr is not None
        result = ocr(image)
        text = result.get("text", "").strip()
        conf = float(result.get("conf", 0.0))
        points = result.get("points")
        elapsed = float(result.get("elapsed_ms", 0.0))
        engine_used = str(result.get("engine", engine))
        if conf < args.min_conf:
            with stats_lock:
                stats["low_conf"] += 1
            text = ""
        annotated = _draw_annotation(image, text, conf, points)
        ocr_rel = input_prefix / "ocr" / f"{job.path.stem}_ocr.jpg"
        ocr_path = ocr_dir / f"{job.path.stem}_ocr.jpg"
        cv2.imwrite(str(ocr_path), annotated)
        record = OCRRecord(
            image_path=image_key,
            plate_text=text,
            rec_confidence=conf,
            width=width,
            height=height,
            ocr_engine=engine_used,
            used_gpu=bool(args.use_gpu),
            elapsed_ms=elapsed,
            ocr_img=_normalise_path(ocr_rel),
        )
        with stats_lock:
            stats["processed"] += 1
            if image_key not in existing_results or args.force:
                stats["new_results"] += 1
        return record

    if args.dry_run:
        for job in jobs:
            process_job(job, perform_ocr=False)
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
            future_map = {executor.submit(process_job, job, True): job for job in jobs}
            for future in as_completed(future_map):
                job = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:  # pragma: no cover - runtime guard
                    with stats_lock:
                        stats["errors"] += 1
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

    elapsed_s = time.perf_counter() - start_time

    if not args.dry_run and processed_records:
        combined = dict(existing_results)
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

    if not args.dry_run and csv_rows and csv_path is not None:
        if csv_header:
            if "plate_text" not in csv_header:
                csv_header.append("plate_text")
            for extra in ["plate_ocr_conf", "plate_ocr_img"]:
                if extra not in csv_header:
                    csv_header.append(extra)
        else:
            csv_header = ["plate_text", "plate_ocr_conf", "plate_ocr_img"]
        _save_csv(csv_path, csv_rows, csv_header)

    print("\n处理摘要:")
    print(f"  OCR 引擎: {engine}")
    print(f"  使用 GPU: {'true' if args.use_gpu else 'false'}")
    print(f"  dry_run: {'true' if args.dry_run else 'false'}")
    print(f"  总计图片: {stats['total']}")
    print(f"  已有结果跳过: {stats['skipped_existing']}")
    print(f"  尺寸不足跳过: {stats['skipped_small']}")
    if args.dry_run:
        print(f"  计划执行: {stats['planned']}")
    else:
        print(f"  实际处理: {stats['processed']}")
        print(f"  新增结果: {stats['new_results']}")
    print(f"  低置信过滤: {stats['low_conf']}")
    print(f"  错误计数: {stats['errors']}")
    if fallback_reason:
        print(f"  引擎降级: rapid (缺少 {fallback_reason})")
    print(f"  总耗时: {elapsed_s:.2f}s")

    if args.dry_run:
        print("dry-run 完成。")
    else:
        print("完成 OCR。")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
