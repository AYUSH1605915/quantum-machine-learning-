$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

Write-Host "Checking Python 3.12..." -ForegroundColor Cyan
$python = $null
try {
    & py -3.12 --version *> $null
    if ($LASTEXITCODE -eq 0) { $python = "py -3.12" }
} catch {}

if (-not $python) {
    Write-Host "Python 3.12 is not installed." -ForegroundColor Yellow
    Write-Host "Install it with:" -ForegroundColor Yellow
    Write-Host "  winget install -e --id Python.Python.3.12 --scope user" -ForegroundColor White
    Write-Host "Then reopen PowerShell and rerun this script." -ForegroundColor Yellow
    exit 1
}

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "Creating .venv..." -ForegroundColor Cyan
    & py -3.12 -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
Write-Host "`nEnvironment ready." -ForegroundColor Green
Write-Host "Run: .\scripts\run_demo.ps1" -ForegroundColor White
