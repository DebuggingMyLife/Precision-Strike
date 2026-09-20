$ErrorActionPreference = 'Continue'
$watchedPid = 32376
$projectDir = 'C:\Users\Oat\OneDrive - UTS\Spring 2026\Reinforcement Learning\A3\Precision-Strike'
$python = Join-Path $projectDir '.venv\Scripts\python.exe'
$logFile = Join-Path $projectDir 'train_continue.log'

Set-Location $projectDir

"[$(Get-Date)] Waiting for train.py (PID $watchedPid) to finish..." | Out-File -FilePath $logFile -Encoding utf8

try {
    Wait-Process -Id $watchedPid -ErrorAction Stop
} catch {
    "[$(Get-Date)] PID $watchedPid not found (already finished) - proceeding." | Out-File -FilePath $logFile -Append -Encoding utf8
}

"[$(Get-Date)] Original training finished. Starting 2M-step continuation run..." | Out-File -FilePath $logFile -Append -Encoding utf8

& $python train_continue.py *>> $logFile

"[$(Get-Date)] Continuation run finished." | Out-File -FilePath $logFile -Append -Encoding utf8
