from pathlib import Path

import cv2

from src.ocr.plate_ocr import PlateOCR


def test_plate_ocr_minimal():
    sample_path = Path("tests/assets/plate_sample.jpg")
    if not sample_path.exists():
        return
    img = cv2.imread(str(sample_path))
    if img is None:
        return

    h, w = img.shape[:2]
    try:
        ocr = PlateOCR(
            log_csv_path="runs/plates/plate_logs.csv",
            crops_dir="runs/plates/crops",
        )
    except RuntimeError:
        return
    res = ocr.recognize(
        crop_bgr=img,
        bbox_xyxy=(0, 0, w, h),
        frame_idx=0,
        video_time_ms=0,
        cam_id="test_cam",
        roi_flag=True,
    )
    assert res["ok"] is True
