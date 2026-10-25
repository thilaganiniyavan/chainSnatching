# After training: free the disk space the live system does not need. It lists what it will delete, with sizes,
# and asks first (add -Yes to skip the question).
#   powershell -ExecutionPolicy Bypass -File scripts\windows\free_space.ps1 -Data C:\verifier_data
# Deletes: the training data folder (research-use crops; delete them after the run anyway), the quick-check
# folders, the trained models inside the run folders (the live system uses the copies in models\verifier), the
# pretrained networks (a trained model contains all its weights; they are downloaded again if you train again)
# and pip's download cache.
# Keeps: the code, .venv, models\verifier (the live models), the reports, verifier_ft_results.zip, the demo clips.
param([string]$Data = "C:\verifier_data", [switch]$Yes)
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path "models\verifier\verifier_ft.pt")) { throw "models\verifier\verifier_ft.pt not found: run train_verifier.ps1 to the end first." }

function Get-Size([string]$Path) {
    $s = (Get-ChildItem -LiteralPath $Path -Recurse -Force -File -ErrorAction SilentlyContinue | Measure-Object -Property Length -Sum).Sum
    if ($s) { return [double]$s } else { return 0.0 }
}

$targets = @()
# the training data: only a folder that really holds the package (index.json + strips), never its parent
if (Test-Path (Join-Path $Data "index.json")) { $targets += (Resolve-Path $Data).Path }
elseif (Test-Path (Join-Path $Data "verifier_data\index.json")) { $targets += (Resolve-Path (Join-Path $Data "verifier_data")).Path }
foreach ($q in @("runtime\verifier_ft_quick", "runtime\verifier_ft_swin_quick", "models\pretrained")) {
    if (Test-Path $q) { $targets += (Resolve-Path $q).Path }
}
foreach ($f in (Get-ChildItem -Path "runtime\verifier_ft*\*.pt" -File -ErrorAction SilentlyContinue)) {
    if ($f.FullName -notmatch "_quick") { $targets += $f.FullName }     # quick folders go whole, above
}
$hub = Join-Path $env:USERPROFILE ".cache\torch\hub\checkpoints"
foreach ($n in @("r2plus1d_18-91a641e6.pth", "swin3d_t-7615ae03.pth")) {
    $p = Join-Path $hub $n
    if (Test-Path $p) { $targets += $p }
}

$total = 0.0
Write-Host "Will delete:"
foreach ($t in $targets) {
    $sz = Get-Size $t
    $total += $sz
    Write-Host ("  {0,8:N0} MB  {1}" -f ($sz / 1MB), $t)
}
Write-Host ("  {0,8:N0} MB  in total, plus pip's download cache" -f ($total / 1MB))
if ($targets.Count -eq 0) { Write-Host "Nothing to delete." }
if (-not $Yes) {
    $answer = Read-Host "Delete these? Type y and press Enter"
    if ($answer -ne "y") { Write-Host "Nothing deleted."; exit 0 }
}
foreach ($t in $targets) { Remove-Item -LiteralPath $t -Recurse -Force -ErrorAction SilentlyContinue }
if (Test-Path $py) { & $py -m pip cache purge | Out-Null }
$drive = (Get-Location).Drive.Name
Write-Host ("Done: about {0:N1} GB freed. Free space on {1}: {2:N1} GB." -f ($total / 1GB), $drive, ((Get-PSDrive $drive).Free / 1GB))
Write-Host "The live system still works: powershell -ExecutionPolicy Bypass -File scripts\windows\run_live.ps1"
