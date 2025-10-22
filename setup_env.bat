@echo off
setlocal
cd /d %~dp0

set "PY_EXE="
if exist .venv\Scripts\python.exe set "PY_EXE=.venv\Scripts\python.exe"

if not defined PY_EXE (
    echo Creating virtual environment under %CD%\.venv ...
    where py >nul 2>&1
    if not errorlevel 1 (
        py -3 -m venv .venv || py -3.11 -m venv .venv
    ) else (
        where python >nul 2>&1
        if errorlevel 1 (
            echo [ERROR] Python is not available in PATH. Please install Python 3.11.
            exit /b 1
        )
        for /f "delims=" %%I in ('where python ^| findstr /i "python.exe"') do if not defined PY_EXE set "PY_EXE=%%I"
        if not defined PY_EXE (
            echo [ERROR] Unable to locate python.exe.
            exit /b 1
        )
        "%PY_EXE%" -m venv .venv
    )
)

if not exist .venv\Scripts\python.exe (
    echo [ERROR] Failed to create virtual environment at %CD%\.venv.
    exit /b 1
)

set "PY_EXE=.venv\Scripts\python.exe"
"%PY_EXE%" -m pip install --upgrade pip
"%PY_EXE%" -m pip install -r requirements.txt

echo.
echo ==== ENV READY ====
endlocal
