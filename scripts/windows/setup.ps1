# SnatchGuard setup on Windows 11 with an NVIDIA GPU (e.g. RTX 4060).
# Run from PowerShell in the repo folder:   powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
# Needs: Python 3.10-3.13 (python.org installer, "Add to PATH" or the py launcher), an up-to-date NVIDIA driver.
param([string]$PythonVersion = "3.12")
# Native tools (pip, nvidia-smi) write warnings to stderr; exit codes are checked explicitly instead.
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")

function Find-Python {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $null = & py "-$PythonVersion" -c "import sys" 2>&1
        if ($LASTEXITCODE -eq 0) { return @("py", "-$PythonVersion") }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) { return @("python") }
    throw "Python not found. Install Python $PythonVersion from python.org (tick 'Add python.exe to PATH')."
}

Write-Host "== NVIDIA driver"
if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv }
else { Write-Warning "nvidia-smi not found: install the latest NVIDIA Game Ready / Studio driver first." }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    $pyCmd = @(Find-Python)
    Write-Host "== creating .venv with $($pyCmd -join ' ')"
    $exe = $pyCmd[0]
    $rest = @()
    if ($pyCmd.Length -gt 1) { $rest = $pyCmd[1..($pyCmd.Length - 1)] }
    & $exe @rest -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "could not create the virtual environment" }
}
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "virtual environment missing: $py" }
& $py -m pip install --upgrade pip

Write-Host "== PyTorch with CUDA (trying the newest CUDA builds first)"
$ok = $false
foreach ($cu in @("cu130", "cu128", "cu126", "cu124")) {
    # remove any earlier attempt: pip would otherwise keep a same-version build for another CUDA
    & $py -m pip uninstall -y torch torchvision | Out-Null
    & $py -m pip install torch torchvision --index-url "https://download.pytorch.org/whl/$cu"
    if ($LASTEXITCODE -ne 0) { continue }
    & $py -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -eq 0) { $ok = $true; Write-Host "   CUDA build $cu works"; break }
}
if (-not $ok) { throw "No CUDA-enabled PyTorch could see the GPU. Update the NVIDIA driver and re-run this script." }

Write-Host "== SnatchGuard and its dependencies"
& $py -m pip install -e ".[edge,central,dev,train,learned]"
if ($LASTEXITCODE -ne 0) { throw "installing SnatchGuard failed (see the pip output above)" }
& $py -c "import torch, ultralytics; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__, '| ultralytics', ultralytics.__version__)"

Write-Host "== tests"
& $py -m pytest -q
Write-Host ""
Write-Host "Setup done. Next:"
Write-Host "  1. Snatch 1.0:  .\.venv\Scripts\python.exe scripts\snatch_dataset.py download --to D:\snatch_zips"
Write-Host "  2. Everything:  powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Zips D:\snatch_zips"
