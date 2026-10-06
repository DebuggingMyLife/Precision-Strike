$ErrorActionPreference = 'Continue'
$watchedPid = 50752
$projectDir = 'C:\Users\Oat\OneDrive - UTS\Spring 2026\Reinforcement Learning\A3\Precision-Strike'
$python = Join-Path $projectDir '.venv\Scripts\python.exe'
$envFile = Join-Path $projectDir 'drone_soccer_env.py'
# Separate status log during the wait phase, same reasoning as
# watch_and_continue.ps1: train_v3.0.log is still being actively written by
# the process we're waiting on, so only start redirecting into the phase-2
# log once that process has actually exited.
$statusFile = Join-Path $projectDir 'watch_status.log'
$logFile = Join-Path $projectDir 'train_v3.1.log'

Set-Location $projectDir

"[$(Get-Date)] Waiting for v3.0 training (PID $watchedPid, wander-mode opponents, cage body) to finish..." | Out-File -FilePath $statusFile -Append -Encoding utf8

try {
    Wait-Process -Id $watchedPid -ErrorAction Stop
} catch {
    "[$(Get-Date)] PID $watchedPid not found (already finished) - proceeding." | Out-File -FilePath $statusFile -Append -Encoding utf8
}

"[$(Get-Date)] v3.0 finished. Flipping OPPONENTS_CHASE_DRONE False -> True for the v3.1 chase-mode stage..." | Out-File -FilePath $statusFile -Append -Encoding utf8

$content = Get-Content -Path $envFile -Raw
$flipped = $content -replace '(?m)^OPPONENTS_CHASE_DRONE = False$', 'OPPONENTS_CHASE_DRONE = True'
if ($flipped -eq $content) {
    "[$(Get-Date)] WARNING: OPPONENTS_CHASE_DRONE = False not found in $envFile - flag NOT flipped, v3.1 would train under wander mode instead of chase mode. Aborting auto-continue; flip it manually and run train_continue.py yourself." | Out-File -FilePath $statusFile -Append -Encoding utf8
    exit 1
}
Set-Content -Path $envFile -Value $flipped -NoNewline -Encoding utf8

"[$(Get-Date)] Flag flipped. Starting v3.0 -> v3.1, +4M steps (chase-mode opponents)..." | Out-File -FilePath $statusFile -Append -Encoding utf8
"[$(Get-Date)] Starting v3.0 -> v3.1, +4M steps." | Out-File -FilePath $logFile -Encoding utf8

& $python train_continue.py *>> $logFile

"[$(Get-Date)] v3.1 run finished." | Out-File -FilePath $statusFile -Append -Encoding utf8
