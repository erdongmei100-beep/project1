@echo off
setlocal EnableDelayedExpansion

pushd "%~dp0"

for /f "usebackq delims=" %%F in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms;$dialog = New-Object System.Windows.Forms.OpenFileDialog;$dialog.Title = '选择待处理的视频';$dialog.Filter = 'Video Files|*.mp4;*.mov;*.avi;*.mkv|All Files|*.*';$videos = Join-Path (Resolve-Path '.\data\videos') '';if(Test-Path $videos){$dialog.InitialDirectory = $videos};if($dialog.ShowDialog() -eq 'OK'){Write-Output $dialog.FileName}"`) do set "VIDEO_PATH=%%F"

if not defined VIDEO_PATH (
    echo 未选择视频，流程已退出。
    goto :EOF
)

set "ROI_TARGET="
set "ROI_ARG="

for %%I in ("!VIDEO_PATH!") do set "VIDEO_STEM=%%~nI"
if exist "data\rois\!VIDEO_STEM!.json" (
    set "ROI_TARGET=%CD%\data\rois\!VIDEO_STEM!.json"
) else (
    echo.
    echo 未找到同名 ROI（data\rois\!VIDEO_STEM!.json）。
    choice /M "是否手动选择 ROI?"
    if errorlevel 2 (
        echo 将按自动回退规则继续。
    ) else (
        for /f "usebackq delims=" %%R in (`powershell -NoProfile -Command "Add-Type -AssemblyName System.Windows.Forms;$dialog = New-Object System.Windows.Forms.OpenFileDialog;$dialog.Title = '选择 ROI JSON 文件（可选）';$dialog.Filter = 'ROI Files|*.json|All Files|*.*';$rois = Join-Path (Resolve-Path '.\data\rois') '';if(Test-Path $rois){$dialog.InitialDirectory = $rois};if($dialog.ShowDialog() -eq 'OK'){Write-Output $dialog.FileName}"`) do set "ROI_TARGET=%%R"
    )
)

if defined ROI_TARGET (
    set "ROI_ARG=--roi ""!ROI_TARGET!"""
)

echo.
echo 运行参数：
echo   视频: !VIDEO_PATH!
if defined ROI_TARGET (
    echo   ROI:   !ROI_TARGET!
) else (
    echo   ROI:   （自动回退）
)
echo   开启功能: --clip --plate --save-video --save-csv
echo.

python run.py --source "!VIDEO_PATH!" !ROI_ARG! --clip --plate --save-video --save-csv
if errorlevel 1 (
    echo.
    echo 处理失败，请检查上方日志。
) else (
    echo.
    echo 处理完成，输出位于 %CD%\data\outputs。
)

popd

endlocal
