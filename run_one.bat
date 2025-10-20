@echo off
setlocal
cd /d %~dp0

if exist .venv\Scripts\activate call .venv\Scripts\activate

python run.py ^
  --source "data/videos/ambulance.mp4" ^
  --roi "data/rois/ambulance.json" ^
  --save-video --save-csv --clip --plate

echo.
echo ==== RUN DONE ====
endlocal
