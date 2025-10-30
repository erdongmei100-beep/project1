@echo off
setlocal ENABLEEXTENSIONS
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
set "VENV_DIR=%ROOT%\.venv"
if exist "%VENV_DIR%" (
    echo Removing existing virtual environment: "%VENV_DIR%"
    rmdir /s /q "%VENV_DIR%"
)

where python >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 Python。请先安装 Python 3.9+。
    exit /b 1
)

echo Creating virtual environment...
python -m venv "%VENV_DIR%"
if errorlevel 1 (
    echo [ERROR] 虚拟环境创建失败。
    exit /b 1
)

set "PIP=%VENV_DIR%\Scripts\pip.exe"
set "PYTHON=%VENV_DIR%\Scripts\python.exe"

"%PYTHON%" -m pip install --upgrade pip
if errorlevel 1 (
    echo [WARNING] pip 升级失败，可手动重试。
)

set "INDEX_OPT="
if defined PIP_INDEX_URL (
    echo Using existing PIP_INDEX_URL=%PIP_INDEX_URL%
) else (
    set "INDEX_OPT=-i https://pypi.tuna.tsinghua.edu.cn/simple"
)

echo Installing dependencies from requirements.txt...
"%PIP%" install -r "%ROOT%\requirements.txt" %INDEX_OPT%
if errorlevel 1 (
    echo [WARNING] 依赖安装过程中出现问题，请检查输出日志。
)

echo.
echo 虚拟环境已就绪。请运行:
echo     call .\.venv\Scripts\activate
exit /b 0
