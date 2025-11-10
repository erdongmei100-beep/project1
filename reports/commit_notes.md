# Commit Notes

This commit records the current state of the fine plate cropping and OCR workflow after prior refinements.

- Fine plate crops are saved in `plates_fine/` as tightly framed colour images.
- OCR utilities parse RapidOCR/Paddle outputs correctly and populate CSV summaries.
- Main CSV backfilling aligns `plate_text`, `plate_ocr_conf`, and `plate_ocr_img` with fine crops.

