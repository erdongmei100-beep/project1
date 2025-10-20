@echo off
setlocal
cd /d %~dp0

git fetch origin
git switch -C work origin/work
git reset --hard origin/work

git lfs install
git lfs pull

echo.
echo ==== SYNC DONE (work) ====
endlocal
