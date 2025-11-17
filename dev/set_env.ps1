# Usage: .\dev\set_env.ps1
$env:PYTHONPATH = (Get-Location).Path
Write-Host "PYTHONPATH set to $env:PYTHONPATH"
