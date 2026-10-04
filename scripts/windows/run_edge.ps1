# Run the edge node on this GPU machine and send alerts to the central server on the laptop.
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_edge.ps1 -Central http://192.168.1.20:8000
#   ... -Source rtsp://user:pass@camera-ip/stream      # a real camera instead of the demo playlist
#   ... -ApiKey my-edge-key                            # must match SNATCHGUARD_API_KEYS on the laptop
param([Parameter(Mandatory = $true)][string]$Central, [string]$Config = "configs\edge.gpu.yaml",
      [string]$Source = "", [string]$ApiKey = "")
$ErrorActionPreference = "Continue"   # native tools log to stderr; python exit codes are passed through
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
if (-not (Test-Path $Config)) { throw "$Config not found: run train_all.ps1 first (its 'install' step writes it)." }
$py = ".\.venv\Scripts\python.exe"
$argsList = @("-m", "snatchguard.edge", "--config", $Config, "--central", $Central)
if ($Source) { $argsList += @("--source", $Source) }
if ($ApiKey) { $env:SNATCHGUARD_API_KEY = $ApiKey }
& $py @argsList
