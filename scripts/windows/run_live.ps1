# Live SnatchGuard on this GPU laptop: the central dashboard (+ phone alerts) and one camera with the verifier.
# Run from PowerShell in the repo folder after train_verifier.ps1:
#   powershell -ExecutionPolicy Bypass -File scripts\windows\run_live.ps1                  # demo: Snatch-Web test clips
#   ... -Source 0                                       # the laptop's webcam
#   ... -Source rtsp://user:pass@camera-ip/stream       # a CCTV camera
#   ... -TelegramToken 123456:ABC... -TelegramChat 987654321   # phone alerts (see snatchguard\central\notify.py)
# The demo plays the Snatch-Web test set with models\verifier\verifier_ft_external.pt, the selected network trained
# without those clips; a real camera uses models\verifier\verifier_ft.pt, trained on everything.
param([string]$Source = "playlist:web_heldout", [string]$Model = "", [string]$TelegramToken = "",
      [string]$TelegramChat = "", [int]$Port = 8000, [string]$PublicUrl = "")
$ErrorActionPreference = "Continue"
Set-Location (Resolve-Path "$PSScriptRoot\..\..")
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "No .venv: run scripts\windows\setup.ps1 first." }
if (-not $Model) {
    if ($Source -eq "playlist:web_heldout") { $Model = "models\verifier\verifier_ft_external.pt" }
    else { $Model = "models\verifier\verifier_ft.pt" }
}
if (-not (Test-Path $Model)) { throw "$Model not found: run scripts\windows\train_verifier.ps1 first." }
foreach ($m in @("models\yolo26n.pt", "models\yolo26n-pose.pt")) {
    if (-not (Test-Path $m)) { throw "$m not found: it comes in snatchguard_code.zip (models folder)." }
}
if ($Source -eq "playlist:web_heldout" -and -not (Test-Path "data\snatch_web\clips\snatch")) {
    throw "data\snatch_web\clips not found: unzip snatch_web_test_clips.zip into $(Get-Location)."
}
if ($TelegramToken -and $TelegramChat) {
    $env:SNATCHGUARD_TELEGRAM_TOKEN = $TelegramToken
    $env:SNATCHGUARD_TELEGRAM_CHAT = $TelegramChat
    if (-not $PublicUrl) { $PublicUrl = "http://127.0.0.1:$Port" }
    $env:SNATCHGUARD_PUBLIC_URL = $PublicUrl
    Write-Host "== phone alerts: Telegram chat $TelegramChat"
} else {
    Write-Host "== no phone alerts (add -TelegramToken and -TelegramChat); the dashboard still alerts with a sound"
}
$env:SNATCHGUARD_API_KEYS = "dev-edge-key"
New-Item -ItemType Directory -Force -Path "runtime" | Out-Null
Write-Host "== central dashboard on http://127.0.0.1:$Port"
$central = Start-Process -FilePath $py -ArgumentList @("-m", "snatchguard.central", "--host", "0.0.0.0", "--port", "$Port") `
    -PassThru -NoNewWindow -RedirectStandardError "runtime\central.log"
$up = $false
foreach ($i in 1..30) {
    Start-Sleep -Seconds 1
    try { if ((Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:$Port/healthz" -TimeoutSec 2).StatusCode -eq 200) { $up = $true; break } }
    catch { }
}
if (-not $up) { Stop-Process -Id $central.Id -ErrorAction SilentlyContinue; throw "central did not start: see runtime\central.log" }
Start-Process "http://127.0.0.1:$Port"
Write-Host "== camera: $Source, verifier $Model (Ctrl+C to stop)"
try {
    & $py -m snatchguard.edge --config configs\edge.live_gpu.yaml --source $Source --name "Live camera" `
        --central "http://127.0.0.1:$Port" --api-key dev-edge-key --verifier-model $Model
} finally {
    Stop-Process -Id $central.Id -ErrorAction SilentlyContinue
}
