@echo off
setlocal ENABLEEXTENSIONS
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "PROJ=%ROOT%\project"
set "VENV_DIR=%ROOT%\.venv1"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
if exist "%VENV_PY%" (
    echo 已检测到现有虚拟环境: "%VENV_DIR%"
) else (
    echo 正在创建虚拟环境 .venv1 ...
    where py >nul 2>&1
    if not errorlevel 1 (
        echo 使用命令: py -3 -m venv .venv1
        pushd "%ROOT%" >nul 2>&1
        py -3 -m venv "%VENV_DIR%"
        set "CREATE_ERR=%ERRORLEVEL%"
        popd >nul 2>&1
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] 未找到可用的 Python 解释器。
            pause
            exit /b 1
        )
        for /f "delims=" %%I in ('where python ^| findstr /i "python.exe"') do (
            if not defined SYS_PY set "SYS_PY=%%I"
        )
        if not defined SYS_PY (
            echo [ERROR] 无法定位 python.exe。
            pause
            exit /b 1
        )
        echo 使用命令: "%SYS_PY%" -m venv .venv1
        pushd "%ROOT%" >nul 2>&1
        "%SYS_PY%" -m venv "%VENV_DIR%"
        set "CREATE_ERR=%ERRORLEVEL%"
        popd >nul 2>&1
    )
    if defined CREATE_ERR if not "%CREATE_ERR%"=="0" (
        echo [ERROR] 虚拟环境创建失败，退出码 %CREATE_ERR%。
        pause
        exit /b %CREATE_ERR%
    )
)
if not exist "%VENV_PY%" (
    echo [ERROR] 未能找到虚拟环境解释器: "%VENV_PY%"
    pause
    exit /b 1
)
echo 升级 pip 和 wheel...
"%VENV_PY%" -m pip install --upgrade pip wheel -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [WARNING] pip/wheel 升级时出现问题，请检查网络后重试。
)
if not exist "%PROJ%\requirements.txt" (
    echo [ERROR] 未找到依赖文件: "%PROJ%\requirements.txt"
    pause
    exit /b 1
)
echo 安装项目依赖...
"%VENV_PY%" -m pip install -r "%PROJ%\requirements.txt" -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [WARNING] 依赖安装过程中存在错误，请查看上方日志。
)
echo.
echo ===== 自检 =====
if exist "%PROJ%\yolov8n.pt" (
    echo [OK] 检测到 yolov8n.pt
) else if exist "%PROJ%\weights\yolov8n.pt" (
    echo [OK] 检测到 weights\yolov8n.pt
) else (
    echo [WARN] 未找到 yolov8n.pt，请确认已下载对应检测权重。
)
if exist "%PROJ%\configs\tracker\bytetrack.yaml" (
    echo [OK] 检测到 configs\tracker\bytetrack.yaml
) else (
    echo [WARN] 缺少 configs\tracker\bytetrack.yaml，请确认配置是否完整。
)
echo.
echo 环境准备完成，现在可运行 run_one.bat。
pause
exit /b 0
