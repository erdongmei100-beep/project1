# run_one.ps1
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Repo = $Root

if (!(Test-Path (Join-Path $Repo "run.py"))) {
  Write-Host "[ERROR] 找不到 run.py" -ForegroundColor Red
  Read-Host "按回车退出"
  exit 1
}

# 选视频
Add-Type -AssemblyName System.Windows.Forms
$dlg = New-Object System.Windows.Forms.OpenFileDialog
$dlg.InitialDirectory = (Join-Path $Repo "data\videos")
$dlg.Filter = "Video files|*.mp4;*.mov;*.avi;*.mkv|All files|*.*"
if ($dlg.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) {
  Write-Host "[INFO] 你取消了选择。"
  Read-Host "按回车退出"
  exit 0
}
$src = $dlg.FileName
# 选 Python：优先仓库内 .venv，其次系统 python
$py = Join-Path $Repo ".venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

# 打印摘要
Write-Host "[RUN] video=$src"
Write-Host "[RUN] exec =$py"

Push-Location $Repo
try {
  & $py "run.py" --source "$src" --config "configs\default.yaml" --save-video --save-csv"
  $rc = $LASTEXITCODE
} finally {
  Pop-Location
}
Write-Host ""
Write-Host "运行结束，退出码 $rc"
Read-Host "按回车关闭"
