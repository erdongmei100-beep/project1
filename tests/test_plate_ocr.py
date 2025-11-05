import os

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for tests
    cv2 = None

from src.ocr.plate_ocr import PlateOCR


def test_plate_ocr_minimal():
    img_path = "tests/assets/plate_sample.jpg"
    if not os.path.exists(img_path):
        return

    if cv2 is None:
        return
    img = cv2.imread(img_path)
    assert img is not None
    h, w = img.shape[:2]
    try:
        ocr = PlateOCR(
            log_csv_path="runs/plates/plate_logs.csv",
            crops_dir="runs/plates/crops"
        )
    except RuntimeError:
        return
    res = ocr.recognize(
        crop_bgr=img,
        bbox_xyxy=(0, 0, w, h),
        frame_idx=0,
        video_time_ms=0,
        cam_id="test_cam",
        roi_flag=True
    )
    assert res["ok"] is True
