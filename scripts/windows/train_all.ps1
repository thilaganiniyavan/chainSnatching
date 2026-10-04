# Full pipeline on the GPU: data -> datasets -> pseudo-labels -> training -> caches -> tuning -> acceptance report.
# Resumable: run it again after a crash or reboot and finished steps are skipped.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\train_all.ps1 -Zips D:\snatch_zips
#   ... -Profile quick           # smaller/faster first run (YOLO26n)
#   ... -Baseline                # also measure the pretrained models (before/after table)
#   ... -Extra "--from","cache"  # any extra arguments for training.run_all
param([string]$Zips = "", [ValidateSet("full", "quick")][string]$Profile = "full", [switch]$Baseline,
      [string[]]$Extra = @())
$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$py = ".\.venv\Scripts\python.exe"
$argsList = @("-m", "training.run_all", "--profile", $Profile)
if ($Zips) { $argsList += @("--zips", $Zips) }
if ($Baseline) { $argsList += "--baseline" }
$argsList += $Extra
# keep Windows awake while training
$sig = '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint esFlags);'
$k32 = Add-Type -MemberDefinition $sig -Name "Power" -Namespace "Win32" -PassThru
[void]$k32::SetThreadExecutionState([uint32]"0x80000001")
try { & $py @argsList } finally { [void]$k32::SetThreadExecutionState([uint32]"0x80000000") }
