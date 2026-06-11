# ============================================================================
# Quick restart — just kill old experiments and relaunch.
# Run from YOUR PowerShell (not the AI's connection):
#   powershell -ExecutionPolicy Bypass -File quick_restart.ps1
# ============================================================================

param(
    [string]$EcsHost = "139.224.162.205",
    [string]$SshKey  = "C:\Users\13083\Downloads\geo.pem",
    [string]$User    = "root"
)

$ErrorActionPreference = "Stop"
$SshArgs = @("-i", $SshKey, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10")

Write-Host "Testing SSH..." -ForegroundColor Yellow
ssh @SshArgs "${User}@${EcsHost}" "echo OK"
if ($LASTEXITCODE -ne 0) { Write-Host "SSH failed!" -ForegroundColor Red; exit 1 }

Write-Host "Killing old experiments..." -ForegroundColor Yellow
ssh @SshArgs "${User}@${EcsHost}" "pkill -f run_all.py 2>/dev/null; echo done"

Write-Host "Launching experiments..." -ForegroundColor Green
$cmd = @'
source /root/geo_env/bin/activate
cd /root/geo_seq_inference
PYTHONUNBUFFERED=1 nohup python -u experiments/run_all.py --signal-scale 4.0 > /root/run.log 2>&1 &
PID=$!
sleep 5
if kill -0 $PID 2>/dev/null; then
    echo "RUNNING PID=$PID"
    echo "Python: $(python3 --version)"
    echo "CPUs: $(nproc)"
    tail -5 /root/run.log
else
    echo "FAILED"
    tail -20 /root/run.log
fi
'@

ssh @SshArgs "${User}@${EcsHost}" "bash -s" <<< $cmd

Write-Host ""
Write-Host "Monitor:  ssh -i $SshKey ${User}@${EcsHost} 'tail -f /root/run.log'" -ForegroundColor Cyan
Write-Host "Check:    ssh -i $SshKey ${User}@${EcsHost} 'pgrep -af run_all.py'" -ForegroundColor Cyan
