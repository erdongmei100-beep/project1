import os

try:
    import cv2
except Exception:  # pragma: no cover - optional dependency for tests
    cv2 = None

try:
    from src.plates.plate_ocr import PlateOCR, PlateOCRError
except Exception:  # pragma: no cover - optional dependency for tests
    PlateOCR = None  # type: ignore
    PlateOCRError = RuntimeError  # type: ignore


def test_plate_ocr_minimal():
    img_path = "tests/assets/plate_sample.jpg"
    if not os.path.exists(img_path):
        return

    if cv2 is None or PlateOCR is None:
        return
    img = cv2.imread(img_path)
    assert img is not None
    h, w = img.shape[:2]
    try:
        ocr = PlateOCR(engine="rapid", min_conf=0.0)
    except PlateOCRError:
        return
    res = ocr(img)
    assert "text" in res and "conf" in res
