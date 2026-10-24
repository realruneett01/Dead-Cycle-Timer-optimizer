# DCTO installer for Windows: picks a compatible Python, builds .venv, auto-configures for this host.
#   .\install.ps1 [-Endpoint opc.tcp://plc:4840]
param([string]$Endpoint = "")
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

# requirements.txt pins numpy<2, which has wheels only up to Python 3.12.
$py = $null
foreach ($v in @("3.12", "3.11")) {
    & py "-$v" -c "import sys" 2>$null
    if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break }
}
if (-not $py) { throw "No Python 3.11/3.12 found. Install one, or use Docker: docker compose up -d" }

& $py[0] $py[1] -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --quiet --upgrade pip
& .\.venv\Scripts\python.exe -m pip install --quiet -r requirements.txt

$cfg = @("-m", "deploy", "configure")
if ($Endpoint) { $cfg += @("--endpoint", $Endpoint) }
& .\.venv\Scripts\python.exe @cfg
& .\.venv\Scripts\python.exe -m deploy doctor | Out-Null
Write-Host "Installed. Start with: .\.venv\Scripts\python.exe -m deploy run"
