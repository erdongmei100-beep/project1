@echo off
setlocal ENABLEEXTENSIONS
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PROJ=%ROOT%\project"
set "RUNPY=%PROJ%\run.py"
if not exist "%RUNPY%" (
    echo [ERROR] 未找到运行脚本: "%RUNPY%"
    echo 请确认 bat 文件放在项目根目录。
    pause
    exit /b 1
)
set "PREFERRED_PY=%ROOT%\.venv1\Scripts\python.exe"
if exist "%PREFERRED_PY%" (
    set "PYTHON_EXE=%PREFERRED_PY%"
) else (
    where python >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] 未找到 Python，请先安装或运行 setup_env.bat。
        pause
        exit /b 1
    )
    for /f "delims=" %%I in ('where python ^| findstr /i "python.exe"') do (
        if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
    )
    if not defined PYTHON_EXE (
        echo [ERROR] 无法定位系统 Python。
        pause
        exit /b 1
    )
)
set "VIDEO_DIR=%PROJ%\data\videos"
if not exist "%VIDEO_DIR%" set "VIDEO_DIR=%PROJ%"
for /f "usebackq delims=" %%I in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms; $dlg = New-Object System.Windows.Forms.OpenFileDialog; $dlg.InitialDirectory = '%VIDEO_DIR:'=''%'; $dlg.Filter = 'Video Files|*.mp4;*.mov;*.avi;*.mkv|All Files|*.*'; $dlg.Multiselect = $false; if ($dlg.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) { Write-Output $dlg.FileName }"`) do set "VIDEO_FILE=%%I"
if not defined VIDEO_FILE (
    echo 未选择任何视频，已取消。
    pause
    exit /b 0
)
for %%I in ("%VIDEO_FILE%") do (
    set "VIDEO_STEM=%%~nI"
)
set "ROI_FILE=%PROJ%\data\rois\%VIDEO_STEM%.json"
set "LOG_DIR=%ROOT%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
set "LOG_FILE=%LOG_DIR%\run_one_last.log"
pushd "%PROJ%" >nul 2>&1
set "CMD=\"%PYTHON_EXE%\" \"%RUNPY%\" --source \"%VIDEO_FILE%\" --config \"configs\default.yaml\" --save-video --save-csv"
if exist "%ROI_FILE%" (
    set "CMD=%CMD% --roi \"%ROI_FILE%\""
)
cmd /c "%CMD%" 1>"%LOG_FILE%" 2>&1
set "EXITCODE=%ERRORLEVEL%"
popd >nul 2>&1
echo.
echo 命令执行完成，退出码 %EXITCODE%。
echo 日志保存在: "%LOG_FILE%"
pause
exit /b %EXITCODE%
