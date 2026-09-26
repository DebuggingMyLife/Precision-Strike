$ErrorActionPreference = 'Continue'
$watchedPid = 44944
$projectDir = 'C:\Users\Oat\OneDrive - UTS\Spring 2026\Reinforcement Learning\A3\Precision-Strike'
$python = Join-Path $projectDir '.venv\Scripts\python.exe'
# Separate status log during the wait phase — train_continue.log is still
# being actively written by the v2.0->v2.1 process we're waiting on, so
# writing to it now would race/collide with that. Only start redirecting
# INTO train_continue.log once v2.2 actually launches below, by which point
# Wait-Process guarantees the old process has exited.
$statusFile = Join-Path $projectDir 'watch_status.log'
$logFile = Join-Path $projectDir 'train_continue.log'

Set-Location $projectDir

"[$(Get-Date)] Waiting for v2.0 -> v2.1 training (PID $watchedPid) to finish..." | Out-File -FilePath $statusFile -Encoding utf8

try {
    Wait-Process -Id $watchedPid -ErrorAction Stop
} catch {
    "[$(Get-Date)] PID $watchedPid not found (already finished) - proceeding." | Out-File -FilePath $statusFile -Append -Encoding utf8
}

"[$(Get-Date)] v2.1 finished. Starting v2.1 -> v2.2, +4M steps..." | Out-File -FilePath $statusFile -Append -Encoding utf8
"[$(Get-Date)] Starting v2.1 -> v2.2, +4M steps." | Out-File -FilePath $logFile -Encoding utf8

& $python train_continue.py *>> $logFile

"[$(Get-Date)] v2.2 run finished." | Out-File -FilePath $statusFile -Append -Encoding utf8
