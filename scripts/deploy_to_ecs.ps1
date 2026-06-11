# ============================================================================
# One-click deployment script for Alibaba Cloud ECS.
#
# Usage (run from PowerShell on your local machine):
#   powershell -ExecutionPolicy Bypass -File deploy_to_ecs.ps1
#
# Or copy-paste into a PowerShell window.
# ============================================================================

param(
    [string]$EcsHost = "139.224.162.205",
    [string]$SshKey  = "C:\Users\13083\Downloads\geo.pem",
    [string]$User    = "root"
)

$ErrorActionPreference = "Stop"

# ── SSH base args ────────────────────────────────────────────────────────────
$SshArgs = @("-i", $SshKey, "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10")

function Invoke-SSH {
    param([string]$Cmd)
    ssh @SshArgs "${User}@${EcsHost}" $Cmd
}

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " geo_seq_inference — ECS Deployment" -ForegroundColor Cyan
Write-Host " Target: ${User}@${EcsHost}" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Test connection ──────────────────────────────────────────────────
Write-Host "[1/6] Testing SSH connection..." -ForegroundColor Yellow
try {
    $test = Invoke-SSH "echo OK"
    if ($test -ne "OK") {
        Write-Host "  ERROR: Cannot connect. Check ECS status in console." -ForegroundColor Red
        exit 1
    }
    Write-Host "  Connected." -ForegroundColor Green
} catch {
    Write-Host "  ERROR: $_" -ForegroundColor Red
    exit 1
}

# ── Step 2: Write setup script (piece by piece to avoid SSH timeout) ─────────
Write-Host "[2/6] Writing setup script to ECS..." -ForegroundColor Yellow

$script = @'
#!/bin/bash
set -e
exec > /root/setup.log 2>&1
echo "=== Setup started at $(date) ==="
echo "=== Phase 1: System packages ==="
apt-get update -qq
apt-get install -y -qq python3.10-venv
echo "Python: $(python3 --version)"

echo "=== Phase 2: Virtual env ==="
python3 -m venv /root/geo_env --clear
source /root/geo_env/bin/activate
pip install --upgrade pip -q
pip install numpy scipy pandas matplotlib joblib pyarrow pytest pyyaml -q
echo "venv ready"

echo "=== Phase 3: Clone repo ==="
cd /root
rm -rf geo_seq_inference
git clone https://github.com/wenchuoFu/Geo.git geo_seq_inference

echo "=== Phase 4: Install package ==="
cd /root/geo_seq_inference
pip install -e . -q

echo "=== Phase 5: Verify tests ==="
python -m pytest tests/test_sign_repr.py -q --tb=short 2>&1 || echo "TEST_WARNING: some tests failed"

echo ""
echo "=============================================="
echo " SETUP COMPLETE at $(date)"
echo " Python : $(python3 --version)"
echo " CPUs   : $(nproc)"
echo " Memory : $(free -h | grep Mem | awk '{print $2}')"
echo " Disk   : $(df -h / | tail -1 | awk '{print $4}') free"
echo "=============================================="
echo "READY_TO_RUN"
'@

# Write script in one shot (short command, just a cat redirect)
$scriptBytes = [System.Text.Encoding]::UTF8.GetBytes($script)
$scriptB64 = [Convert]::ToBase64String($scriptBytes)
Invoke-SSH "echo $scriptB64 | base64 -d > /root/setup.sh && chmod +x /root/setup.sh && echo SCRIPT_OK"
Write-Host "  Script uploaded." -ForegroundColor Green

# ── Step 3: Execute setup in background ──────────────────────────────────────
Write-Host "[3/6] Launching setup (background)..." -ForegroundColor Yellow
Invoke-SSH "nohup bash /root/setup.sh > /dev/null 2>&1 & echo PID=\$!"
Write-Host "  Setup launched. This takes 3-5 minutes." -ForegroundColor Green

# ── Step 4: Wait for setup to finish ─────────────────────────────────────────
Write-Host "[4/6] Waiting for setup to complete..." -ForegroundColor Yellow
$maxWait = 600  # 10 minutes
$waited = 0
do {
    Start-Sleep -Seconds 15
    $waited += 15
    try {
        $log = Invoke-SSH "tail -2 /root/setup.log 2>/dev/null"
        Write-Host "  [$waited s] $log"
        if ($log -match "READY_TO_RUN") {
            Write-Host "  Setup complete!" -ForegroundColor Green
            break
        }
        if ($log -match "error|failed|FAILED" -and $log -notmatch "TEST_WARNING") {
            Write-Host "  ERROR detected in setup log!" -ForegroundColor Red
            Invoke-SSH "cat /root/setup.log"
            exit 1
        }
    } catch {
        Write-Host "  Connection lost, retrying in 30s..." -ForegroundColor DarkYellow
        Start-Sleep -Seconds 30
        $waited += 30
    }
} while ($waited -lt $maxWait)

if ($waited -ge $maxWait) {
    Write-Host "  TIMEOUT: Setup did not complete in 10 min. Check manually:" -ForegroundColor Red
    Write-Host "  ssh -i $SshKey ${User}@${EcsHost} 'cat /root/setup.log'"
    exit 1
}

# ── Step 5: Run experiments ──────────────────────────────────────────────────
Write-Host "[5/6] Starting experiments..." -ForegroundColor Yellow
Invoke-SSH @"
source /root/geo_env/bin/activate
cd /root/geo_seq_inference
nohup python experiments/run_all.py --signal-scale 4.0 > /root/run.log 2>&1 &
echo "EXPERIMENT_PID=\$!"
"@
Write-Host "  Experiments launched!" -ForegroundColor Green

# ── Step 6: Print monitoring instructions ────────────────────────────────────
Write-Host "[6/6] All set!" -ForegroundColor Green
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host " MONITORING COMMANDS" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "  # Check experiment progress:"
Write-Host "  ssh -i $SshKey ${User}@${EcsHost} 'tail -30 /root/run.log'"
Write-Host ""
Write-Host "  # Check if still running:"
Write-Host "  ssh -i $SshKey ${User}@${EcsHost} 'pgrep -f run_all.py && echo RUNNING || echo DONE'"
Write-Host ""
Write-Host "  # Download results when done:"
Write-Host "  scp -i $SshKey -r ${User}@${EcsHost}:/root/geo_seq_inference/experiments/results/ .\experiments\results\"
Write-Host "  scp -i $SshKey -r ${User}@${EcsHost}:/root/geo_seq_inference/experiments/plots/ .\experiments\plots\"
Write-Host ""
Write-Host "  # DONT FORGET: Set auto-release time in ECS console!"
Write-Host "    控制台 -> 实例 -> 释放设置 -> 定时释放 -> 明天 08:00"
Write-Host ""
