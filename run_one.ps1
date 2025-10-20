# run_one.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path

$RunPy = Join-Path $Root "run.py"
if (!(Test-Path $RunPy)) {
  Write-Host "[ERROR] 找不到 run.py，请确认脚本位于仓库根目录。" -ForegroundColor Red
  Read-Host "按回车退出"
  exit 1
}

Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.InitialDirectory = Join-Path $Root "data\videos"
$dlg.Filter = "Video files|*.mp4;*.mov;*.avi;*.mkv|All files|*.*"
if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
  Write-Host "[INFO] 你取消了选择。"
  Read-Host "按回车退出"
  exit 0
}
$src = $dlg.FileName
$stem = [System.IO.Path]::GetFileNameWithoutExtension($src)
$roiRel = "data\rois\$stem.json"

$venvPy = Join-Path $Root ".venv\Scripts\python.exe"
$py = if (Test-Path $venvPy) { $venvPy } else { "python" }

Write-Host "[RUN] video=$src"
Write-Host "[RUN] roi  =$roiRel"
Write-Host "[RUN] exec =$py"

Push-Location $Root
try {
  & $py "run.py" --source "$src" --config "configs\default.yaml" --roi "$roiRel" --save-video --save-csv --clip --plate
  $rc = $LASTEXITCODE
} finally {
  Pop-Location
}

Write-Host ""
Write-Host "运行结束，退出码 $rc"
Read-Host "按回车关闭"
