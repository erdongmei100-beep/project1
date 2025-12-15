from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .recognizer import HyperLPR3Recognizer

logger = logging.getLogger(__name__)


def _select_image_path(row: pd.Series, primary_field: str, fallback_field: str) -> Optional[Path]:
    primary_value = str(row.get(primary_field, "") or "").strip()
    if primary_value:
        primary_path = Path(primary_value)
        if primary_path.exists():
            return primary_path
    fallback_value = str(row.get(fallback_field, "") or "").strip()
    if fallback_value:
        fallback_path = Path(fallback_value)
        if fallback_path.exists():
            return fallback_path
    return None


def enrich_events_csv_with_lpr(input_csv: str, output_csv: str, cfg: Dict) -> str:
    input_path = Path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    lpr_cfg = cfg or {}
    input_field = lpr_cfg.get("input_field", "plate_crop_path")
    fallback_field = lpr_cfg.get("fallback_field", "best_frame_path")
    encoding = lpr_cfg.get("write_encoding", "utf-8-sig")
    progress_interval = int(lpr_cfg.get("progress_interval", 50) or 0)

    recognizer = HyperLPR3Recognizer(
        backend=lpr_cfg.get("backend", "onnxruntime-gpu"),
        model_dir=lpr_cfg.get("model_dir") or None,
        quality_filter=lpr_cfg.get("quality_filter") or {},
    )

    df = pd.read_csv(input_path)

    df["plate_text"] = ""
    df["plate_score"] = 0.0
    df["plate_bbox"] = ""
    df["lpr_status"] = ""

    status_counter: Dict[str, int] = {}

    for idx, row in df.iterrows():
        image_path = _select_image_path(row, input_field, fallback_field)
        if image_path is None:
            status = "missing_file"
            df.at[idx, "plate_text"] = ""
            df.at[idx, "plate_score"] = 0.0
            df.at[idx, "plate_bbox"] = ""
            df.at[idx, "lpr_status"] = status
            status_counter[status] = status_counter.get(status, 0) + 1
            continue

        result = recognizer.recognize(str(image_path))
        df.at[idx, "plate_text"] = result.plate_text
        df.at[idx, "plate_score"] = result.plate_score
        df.at[idx, "plate_bbox"] = result.plate_bbox if result.plate_bbox is not None else ""
        df.at[idx, "lpr_status"] = result.status
        status_counter[result.status] = status_counter.get(result.status, 0) + 1

        if progress_interval and (idx + 1) % progress_interval == 0:
            logger.info("Processed %d rows", idx + 1)

    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding=encoding)

    total_rows = len(df)
    ok = status_counter.get("ok", 0)
    logger.info(
        "LPR enrichment completed: %d rows | ok=%d | missing=%d | empty=%d | fail=%d | bad_quality=%d",
        total_rows,
        ok,
        status_counter.get("missing_file", 0),
        status_counter.get("empty", 0),
        status_counter.get("fail", 0),
        status_counter.get("bad_quality", 0),
    )

    return str(output_path)
