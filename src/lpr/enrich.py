from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import pandas as pd

from .recognizer import HyperLPR3Recognizer

logger = logging.getLogger(__name__)


def _determine_input_field(df: pd.DataFrame, configured_field: str | None) -> str:
    if configured_field and configured_field in df.columns:
        return configured_field
    for candidate in ("best_frame_path", "best_frame"):
        if candidate in df.columns:
            return candidate
    tried = [configured_field] if configured_field else []
    tried.extend(["best_frame_path", "best_frame"])
    raise ValueError(f"CSV is missing a best-frame path column (tried: {', '.join(tried)})")


def enrich_events_csv_with_lpr(input_csv: str, output_csv: str | None, cfg: Dict) -> str:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    lpr_cfg = cfg or {}
    input_field = lpr_cfg.get("input_field") or None
    encoding = lpr_cfg.get("write_encoding", "utf-8-sig")
    progress_interval = int(lpr_cfg.get("progress_interval", 50) or 0)

    recognizer = HyperLPR3Recognizer(
        backend=lpr_cfg.get("backend", "onnxruntime-gpu"),
        model_dir=lpr_cfg.get("model_dir") or None,
        quality_filter=lpr_cfg.get("quality_filter") or {},
    )

    df = pd.read_csv(input_path)
    input_field = _determine_input_field(df, input_field)
    logger.info("Using column '%s' for LPR image paths", input_field)

    df["plate_text"] = ""
    df["plate_score"] = 0.0
    df["plate_bbox"] = ""
    df["lpr_status"] = ""

    status_counter: Dict[str, int] = {}

    for idx, row in df.iterrows():
        raw_value = row.get(input_field, "")
        image_value = "" if pd.isna(raw_value) else str(raw_value).strip()
        image_path = Path(image_value) if image_value else None

        if not image_value or image_path is None or not image_path.exists():
            status = "missing_file"
            plate_text = ""
            plate_score = 0.0
            plate_bbox = ""
        else:
            result = recognizer.recognize(str(image_path))
            plate_text = result.plate_text
            plate_score = result.plate_score
            plate_bbox = json.dumps(result.plate_bbox) if result.plate_bbox else ""
            status = result.status

        df.at[idx, "plate_text"] = plate_text
        df.at[idx, "plate_score"] = plate_score
        df.at[idx, "plate_bbox"] = plate_bbox
        df.at[idx, "lpr_status"] = status
        status_counter[status] = status_counter.get(status, 0) + 1

        if idx < 5:
            logger.info(
                "Row %d | column=%s | path=%s | exists=%s | status=%s",
                idx,
                input_field,
                image_value,
                image_path.exists() if image_path else False,
                status,
            )

        if progress_interval and (idx + 1) % progress_interval == 0:
            logger.info("Processed %d rows", idx + 1)

    if not output_csv or str(output_csv).lower() == "auto":
        output_path = input_path.with_name("events_with_plate.csv")
    else:
        output_path = Path(output_csv)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=encoding)

    total_rows = len(df)
    logger.info(
        "LPR enrichment completed: %d rows | ok=%d | empty=%d | missing=%d | fail=%d",
        total_rows,
        status_counter.get("ok", 0),
        status_counter.get("empty", 0),
        status_counter.get("missing_file", 0),
        status_counter.get("fail", 0),
    )

    return str(output_path)
