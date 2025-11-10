"""Fill main event CSV with plate OCR results."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUTS = ROOT / "data" / "outputs"
OUTPUTS_ROOT = DEFAULT_OUTPUTS
REQUIRED_COLUMNS = ["plate_text", "plate_ocr_conf", "plate_ocr_img"]


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge OCR results back into the main CSV")
    parser.add_argument("--csv", required=True, help="Path to the main event CSV (e.g. data/outputs/<run>/<run>.csv)")
    parser.add_argument(
        "--outputs-root",
        default=str(DEFAULT_OUTPUTS),
        help="Root directory for data/outputs (used for relative path normalisation)",
    )
    return parser.parse_args(argv)


def normalize_rel_path(raw: str) -> str:
    if not raw:
        return ""
    norm = raw.replace("\\", "/")
    for marker in ("plates_fine/", "plates/"):
        idx = norm.find(marker)
        if idx != -1:
            return str(Path(norm[idx:]))
    outputs_tag = "data/outputs/"
    if outputs_tag in norm:
        idx = norm.find(outputs_tag)
        rel = norm[idx + len(outputs_tag) :]
        return str(Path(rel))
    try:
        candidate = Path(norm)
        abs_candidate = candidate if candidate.is_absolute() else (OUTPUTS_ROOT / candidate)
        rel = abs_candidate.resolve(strict=False).relative_to(OUTPUTS_ROOT.resolve(strict=False))
        return str(Path(*rel.parts))
    except Exception:
        pass
    return str(Path(norm))


def normalize_key(path_str: str) -> Optional[str]:
    if not path_str:
        return None
    name = Path(path_str).name
    stem = Path(name).stem
    if stem.endswith("_plate"):
        stem = stem[: -len("_plate")]
    return stem or None


def read_rows(csv_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(row) for row in reader]
        fieldnames = list(reader.fieldnames or [])
    for column in REQUIRED_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    return rows, fieldnames


def write_rows(csv_path: Path, rows: List[Dict[str, str]], fieldnames: List[str]) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def ingest_results(csv_path: Path, prefer_fine: bool, store: Dict[str, Dict[str, str]]) -> None:
    if not csv_path.exists():
        return
    with open(csv_path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            image_path = normalize_rel_path((row.get("image_path") or "").strip())
            key = normalize_key(image_path)
            if not key:
                continue
            text = (row.get("plate_text") or "").strip()
            conf_str = (row.get("rec_confidence") or row.get("confidence") or "0").strip()
            try:
                conf = float(conf_str)
            except ValueError:
                conf = 0.0
            record = {
                "text": text,
                "conf": conf,
                "rel_path": image_path,
            }
            if prefer_fine:
                store[key] = record
            else:
                store.setdefault(key, record)


def determine_key_for_row(row: Dict[str, str]) -> Optional[str]:
    for column in ("plate_ocr_img", "plate_img", "tail_img"):
        candidate = normalize_rel_path((row.get(column) or "").strip())
        key = normalize_key(candidate)
        if key:
            return key
    return None


def update_rows(
    rows: List[Dict[str, str]],
    records: Dict[str, Dict[str, str]],
) -> tuple[int, int]:
    updated = 0
    total = len(rows)
    for row in rows:
        key = determine_key_for_row(row)
        if not key:
            continue
        record = records.get(key)
        if not record:
            continue
        text_value = record.get("text", "").strip()
        if not text_value:
            text_value = "null"
        conf_value = record.get("conf", 0.0)
        try:
            conf_float = float(conf_value)
        except (TypeError, ValueError):
            conf_float = 0.0
        rel_path = normalize_rel_path(record.get("rel_path", ""))
        row["plate_text"] = text_value
        row["plate_ocr_conf"] = f"{conf_float:.4f}"
        if rel_path:
            row["plate_ocr_img"] = rel_path
        updated += 1
    return updated, total


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    csv_path = Path(args.csv).expanduser().resolve()
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return 1
    outputs_root = Path(args.outputs_root).expanduser().resolve()
    globals()["OUTPUTS_ROOT"] = outputs_root

    rows, fieldnames = read_rows(csv_path)
    if not rows:
        print("No rows to update in the main CSV.")
        return 0

    records: Dict[str, Dict[str, str]] = {}
    fine_results = csv_path.parent / "plates_fine" / "plate_ocr_results.csv"
    base_results = csv_path.parent / "plates" / "plate_ocr_results.csv"
    ingest_results(fine_results, prefer_fine=True, store=records)
    ingest_results(base_results, prefer_fine=False, store=records)

    updated, total = update_rows(rows, records)
    write_rows(csv_path, rows, fieldnames)

    skipped = total - updated
    print(f"Updated rows: {updated}/{total}; skipped: {skipped}")
    if updated == 0:
        print("No matching OCR records were found. Ensure plate_ocr_results.csv is generated first.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
