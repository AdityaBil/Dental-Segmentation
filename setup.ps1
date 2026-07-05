# setup.ps1 — One-time environment setup for ToothFairy (Windows)
#
# Usage:
#   Set-ExecutionPolicy -Scope Process Bypass
#   .\setup.ps1

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ToothFairy project setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# ── 1. Python ───────────────────────────────────────────────────────────────
$python = Get-Command python -ErrorAction SilentlyContinue
if (-not $python -or (python --version 2>&1) -match "not found|Microsoft Store") {
    Write-Host "`nPython was not found." -ForegroundColor Yellow
    Write-Host "Install Python 3.10+ from https://www.python.org/downloads/"
    Write-Host "Enable 'Add Python to PATH', then re-run this script."
    Write-Host "Or run: winget install Python.Python.3.12"
    exit 1
}

python --version

# ── 2. Virtual environment ────────────────────────────────────────────────────
if (-not (Test-Path ".venv")) {
    Write-Host "`nCreating virtual environment…"
    python -m venv .venv
}

Write-Host "Activating virtual environment…"
& ".\.venv\Scripts\Activate.ps1"

Write-Host "Installing dependencies…"
python -m pip install --upgrade pip
pip install -r requirements.txt

# ── 3. Dataset ───────────────────────────────────────────────────────────────
$datasetRoot = Join-Path $ProjectRoot "data\toothfairy_dataset"
$imagesDir = Join-Path $datasetRoot "imagesTr"
$labelsDir = Join-Path $datasetRoot "labelsTr"

if ((Test-Path $imagesDir) -and (Test-Path $labelsDir)) {
    Write-Host "`nDataset already present at $datasetRoot" -ForegroundColor Green
} else {
    Write-Host "`nDataset not found locally." -ForegroundColor Yellow
    Write-Host "Drive folder: https://drive.google.com/drive/folders/154c_JD__HEN2j5R9ujYuDDIzom5hUbGi"
    $answer = Read-Host "Download automatically with gdown now? [y/N]"
    if ($answer -match "^[Yy]$") {
        python download_dataset.py --output $datasetRoot
    } else {
        Write-Host "Download manually and place files under:"
        Write-Host "  $datasetRoot\imagesTr"
        Write-Host "  $datasetRoot\labelsTr"
    }
}

# ── 4. Verify ─────────────────────────────────────────────────────────────────
Write-Host "`nRunning verification…"
python verify_setup.py

Write-Host "`nDone. To train:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python train.py"
