# run_one.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Proj = Join-Path $Root "project"

if (!(Test-Path (Join-Path $Proj "run.py"))) {
  Write-Host "[ERROR] 找不到 project\run.py" -ForegroundColor Red
  Read-Host "按回车退出"
  exit 1
}

# 选视频
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.InitialDirectory = (Join-Path $Proj "data\videos")
$dlg.Filter = "Video files|*.mp4;*.mov;*.avi;*.mkv|All files|*.*"
if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
  Write-Host "[INFO] 你取消了选择。"
  Read-Host "按回车退出"
  exit 0
}
$src = $dlg.FileName
$stem = [System.IO.Path]::GetFileNameWithoutExtension($src)
$roiRel = "data\rois\$stem.json"

# 选 Python：优先 .venv1，其次系统 python
$py = Join-Path $Root ".venv1\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

# 打印摘要
Write-Host "[RUN] video=$src"
Write-Host "[RUN] roi  =$roiRel"
Write-Host "[RUN] exec =$py"

Push-Location $Proj
try {
  & $py "run.py" --source "$src" --config "configs\default.yaml" --save-video --save-csv --roi "$roiRel"
  $rc = $LASTEXITCODE
} finally {
  Pop-Location
}
Write-Host ""
Write-Host "运行结束，退出码 $rc"
Read-Host "按回车关闭"
