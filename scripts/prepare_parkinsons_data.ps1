param(
    [string]$ZipPath = "$env:USERPROFILE\Downloads\ParkinsonsData.zip"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$DataDir = Join-Path $ProjectRoot "data"
New-Item -ItemType Directory -Force -Path $DataDir | Out-Null

if (-not (Test-Path $ZipPath)) {
    throw "Archive not found: $ZipPath"
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [System.IO.Compression.ZipFile]::OpenRead($ZipPath)

function Extract-ZipEntryBySuffix {
    param([string]$Suffix, [string]$Destination)
    $matches = @($zip.Entries | Where-Object { $_.FullName.Replace('\','/').EndsWith($Suffix) })
    if ($matches.Count -eq 0) { throw "Could not find $Suffix in archive" }
    $entry = $matches[0]
    $inStream = $entry.Open()
    try {
        $outStream = [System.IO.File]::Create($Destination)
        try { $inStream.CopyTo($outStream) } finally { $outStream.Dispose() }
    } finally { $inStream.Dispose() }
    Write-Host "Extracted $($entry.FullName) -> $Destination" -ForegroundColor Green
}

try {
    Extract-ZipEntryBySuffix "/Clinical/UCI_Clinical/parkinsons.data" (Join-Path $DataDir "parkinsons.data")
    Extract-ZipEntryBySuffix "/Clinical/UCI_Clinical/parkinsons.names" (Join-Path $DataDir "parkinsons.names")
    Extract-ZipEntryBySuffix "/Clinical/UCI_Clinical/telemonitoring/parkinsons_updrs.data" (Join-Path $DataDir "parkinsons_updrs.data")
    Extract-ZipEntryBySuffix "/Clinical/UCI_Clinical/telemonitoring/parkinsons_updrs.names" (Join-Path $DataDir "parkinsons_updrs.names")
}
finally {
    $zip.Dispose()
}

Write-Host "`nData preparation complete. The 6.97 GB archive was NOT fully extracted." -ForegroundColor Cyan
Get-ChildItem $DataDir | Select-Object Name, Length
