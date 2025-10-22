@echo off
setlocal ENABLEEXTENSIONS
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PYTHON=%ROOT%\.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
    echo [ERROR] 未找到虚拟环境，请先运行 setup_env.bat。
    exit /b 1
)
call "%ROOT%\.venv\Scripts\activate"
python "%ROOT%\run.py" --source "data\videos\ambulance.mp4" --roi "data\rois\ambulance.json" --save-video --clip --plate
endlocal
