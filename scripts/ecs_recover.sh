#!/bin/bash
# ============================================================================
# ECS recovery script — checks environment, reinstalls if needed, launches experiments.
# Run via: ssh root@host "bash -s" < ecs_recover.sh
# ============================================================================
set -e
exec > /root/recover.log 2>&1
echo "=== Recover started at $(date) ==="

# ── Kill any old processes ─────────────────────────────────────────────────
pkill -f run_all.py 2>/dev/null || true
sleep 1

# ── Check / setup virtual env ──────────────────────────────────────────────
if [ ! -f /root/geo_env/bin/python3 ]; then
    echo "=== Rebuilding venv ==="
    apt-get update -qq
    apt-get install -y -qq python3.10-venv
    python3 -m venv /root/geo_env --clear
    source /root/geo_env/bin/activate
    pip install --upgrade pip -q
    pip install numpy scipy pandas matplotlib joblib pyarrow pytest pyyaml -q
else
    echo "=== venv OK ==="
    source /root/geo_env/bin/activate
fi

# ── Check / update repo ────────────────────────────────────────────────────
if [ -d /root/geo_seq_inference/.git ]; then
    echo "=== Updating repo ==="
    cd /root/geo_seq_inference
    git pull origin main 2>/dev/null || echo "git pull failed, using existing"
else
    echo "=== Cloning repo ==="
    cd /root
    rm -rf geo_seq_inference
    git clone https://github.com/wenchuoFu/Geo.git geo_seq_inference
    cd /root/geo_seq_inference
fi

# ── Install package ────────────────────────────────────────────────────────
pip install -e . -q 2>&1 || echo "pip install warning (non-fatal)"

# ── Create results dirs ────────────────────────────────────────────────────
mkdir -p experiments/results experiments/plots

# ── Add @reboot cron for auto-restart ──────────────────────────────────────
(crontab -l 2>/dev/null | grep -v 'run_all.py'; echo "@reboot sleep 60 && source /root/geo_env/bin/activate && cd /root/geo_seq_inference && PYTHONUNBUFFERED=1 nohup python -u experiments/run_all.py --signal-scale 4.0 > /root/run.log 2>&1 &") | crontab -

# ── Launch experiments ─────────────────────────────────────────────────────
echo "=== Launching experiments at $(date) ==="
PYTHONUNBUFFERED=1 nohup python -u experiments/run_all.py --signal-scale 4.0 > /root/run.log 2>&1 &
PID=$!
echo "PID=$PID"

# ── Verify launch ──────────────────────────────────────────────────────────
sleep 5
if kill -0 $PID 2>/dev/null; then
    echo "=== RUNNING (PID=$PID) ==="
    echo "Python: $(python3 --version)"
    echo "CPUs: $(nproc)"
    echo "Memory: $(free -h | grep Mem | awk '{print $2}')"
    echo "Disk: $(df -h / | tail -1 | awk '{print $4}') free"
    echo "READY_TO_RUN"
else
    echo "=== FAILED TO LAUNCH ==="
    tail -20 /root/run.log
    exit 1
fi
