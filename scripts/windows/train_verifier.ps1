# Fine-tune the learned verifier on this GPU and evaluate it (snatchguard/verifier/finetune.py).
# Run from PowerShell in the repo folder, after scripts\windows\setup.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data D:\verifier_data -Quick   # 5-min check
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_verifier.ps1 -Data D:\verifier_data          # the real run
#   ... -Batch 8                 # GPUs with 4-6 GB (it also halves the batch by itself on "out of memory")
#   ... -Extra "--steps","3000"  # any extra arguments for finetune.py run
param([Parameter(Mandatory = $true)][string]$Data, [switch]$Quick, [int]$Batch = 16, [string[]]$Extra = @())
$ErrorActionPreference = "Continue"   # python exit codes are passed through
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv: run scripts\windows\setup.ps1 first." }
if (-not (Test-Path (Join-Path $Data "index.json")) -and (Test-Path (Join-Path $Data "verifier_data\index.json"))) {
    $Data = Join-Path $Data "verifier_data"   # unzipped into a folder of the same name
}
if (-not (Test-Path (Join-Path $Data "index.json"))) { throw "$Data has no index.json: point -Data at the unzipped verifier_data folder." }
$out = "runtime\verifier_ft"
$argsList = @("-m", "snatchguard.verifier.finetune", "run", "--data", $Data, "--batch", "$Batch")
if ($Quick) { $out = "runtime\verifier_ft_quick"; $argsList += @("--quick", "--steps", "20") }
$argsList += @("--out", $out) + $Extra
# keep Windows awake while training
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$k32 = Add-Type -MemberDefinition $sig -Name "Power" -Namespace "Win32" -PassThru
[void]$k32::SetThreadExecutionState([uint32]"0x80000001")
try { & $py @argsList } finally { [void]$k32::SetThreadExecutionState([uint32]"0x80000000") }
$code = $LASTEXITCODE
if ($code -eq 0 -and -not $Quick) {
    Compress-Archive -Path "$out\*" -DestinationPath "verifier_ft_results.zip" -Force
    Write-Host ""
    Write-Host "Done. Send back this one file:  $(Resolve-Path verifier_ft_results.zip)"
} elseif ($code -eq 0) {
    Write-Host ""
    Write-Host "Quick check passed. Now run the same command without -Quick."
}
exit $code
