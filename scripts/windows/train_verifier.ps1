# Fine-tune the learned verifier on this GPU, evaluate it, and pick the model to deploy (snatchguard/verifier/finetune.py).
# Run from PowerShell in the repo folder, after scripts\windows\setup.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data C:\verifier_data -Quick   # ~10-min check
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data C:\verifier_data          # the real run
#   ... -Batch 8                 # GPUs with 4-6 GB (it also halves the batch by itself on "out of memory")
#   ... -SkipSwin                # R(2+1)D-18 only (about 40 % less time)
#   ... -Extra "--steps","3000"  # any extra arguments for finetune.py run
# Two networks are trained under the same protocols (R(2+1)D-18, then Swin3D-T); `finetune select` picks the one
# to deploy from the Snatch 1.0 cross-validation and copies it to models\verifier\verifier_ft.pt (+ selection.json).
param([Parameter(Mandatory = $true)][string]$Data, [switch]$Quick, [int]$Batch = 16, [switch]$SkipSwin,
      [string[]]$Extra = @())
$ErrorActionPreference = "Continue"   # python exit codes are passed through
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv: run scripts\windows\setup.ps1 first." }
if (-not (Test-Path (Join-Path $Data "index.json")) -and (Test-Path (Join-Path $Data "verifier_data\index.json"))) {
    $Data = Join-Path $Data "verifier_data"   # unzipped into a folder of the same name
}
if (-not (Test-Path (Join-Path $Data "index.json"))) { throw "$Data has no index.json: point -Data at the unzipped verifier_data folder." }
$sfx = ""
$common = @("--data", $Data, "--batch", "$Batch")
if ($Quick) { $sfx = "_quick"; $common += @("--quick", "--steps", "20") }
$runs = @(@{Arch = "r2plus1d_18"; Out = "runtime\verifier_ft$sfx"; Protocols = @("external", "curve", "s1-cv", "final")})
if (-not $SkipSwin) {
    $runs += @{Arch = "swin3d_t"; Out = "runtime\verifier_ft_swin$sfx"; Protocols = @("external", "s1-cv", "final")}
}
$deploy = "models\verifier\verifier_ft.pt"
if ($Quick) { $deploy = "runtime\verifier_ft_quick\deploy\verifier_ft.pt" }
# keep Windows awake while training
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$k32 = Add-Type -MemberDefinition $sig -Name "Power" -Namespace "Win32" -PassThru
[void]$k32::SetThreadExecutionState([uint32]"0x80000001")
$code = 0
$done = @()
try {
    foreach ($r in $runs) {
        Write-Host ""
        Write-Host "== $($r.Arch) -> $($r.Out)"
        $argsList = @("-m", "snatchguard.verifier.finetune", "run") + $common +
                    @("--arch", $r.Arch, "--out", $r.Out, "--protocols") + $r.Protocols + $Extra
        & $py @argsList
        if ($LASTEXITCODE -eq 0) { $done += $r }
        elseif ($r.Arch -eq "r2plus1d_18") { $code = $LASTEXITCODE; break }
        else { Write-Warning "$($r.Arch) failed (exit $LASTEXITCODE): continuing with the runs that finished" }
    }
    if ($code -eq 0) {
        $selArgs = @("-m", "snatchguard.verifier.finetune", "select", "--out", $deploy, "--runs") + ($done | ForEach-Object { $_.Out })
        & $py @selArgs
        $code = $LASTEXITCODE
    }
} finally { [void]$k32::SetThreadExecutionState([uint32]"0x80000000") }
if ($code -ne 0) { exit $code }
if ($Quick) {
    Write-Host ""
    Write-Host "Quick check passed. Now run the same command without -Quick."
    exit 0
}
# results to send back: reports, logs and test scores (a few MB); the trained models stay here for live use
$stage = "runtime\verifier_results"
if (Test-Path $stage) { Remove-Item -Recurse -Force $stage }
New-Item -ItemType Directory -Path $stage | Out-Null
foreach ($r in $done) {
    $d = Join-Path $stage $r.Arch
    New-Item -ItemType Directory -Path $d | Out-Null
    Get-ChildItem $r.Out -File | Where-Object { $_.Extension -ne ".pt" -and $_.Extension -ne ".tmp" } | Copy-Item -Destination $d
}
Copy-Item "models\verifier\selection.json" $stage
Compress-Archive -Path "$stage\*" -DestinationPath "verifier_ft_results.zip" -Force
Write-Host ""
Write-Host "Done. The model to use live is $deploy (see models\verifier\selection.json)."
Write-Host "Send back this one file:  $(Resolve-Path verifier_ft_results.zip)"
exit 0
