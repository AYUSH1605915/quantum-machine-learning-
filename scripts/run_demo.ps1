$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    throw "Virtual environment missing. Run .\scripts\setup_demo.ps1 first."
}
if (-not (Test-Path "data\parkinsons.data")) {
    throw "Parkinson's dataset missing. Run .\scripts\prepare_parkinsons_data.ps1 first."
}
Write-Host "Starting QuantumMed Parkinson's demo at http://127.0.0.1:5000" -ForegroundColor Cyan
& .\.venv\Scripts\python.exe app.py
