#!/bin/bash
# ============================================================================
# Alibaba Cloud ECS one-click setup for geo_seq_inference experiments.
#
# This script handles:
#   1. Python 3.12 + virtual env + all dependencies
#   2. Git clone of the project repo
#   3. NAS mount for persistent results (survives spot instance reclamation)
#
# Prerequisites (done once in Alibaba Cloud console):
#   - Create NAS file system (通用型容量型, NFSv3, ~0.35 元/GB/mo)
#   - Create mount target in the same VPC as your ECS
#   - Note the mount point address (e.g., 123456-nas.cn-beijing.nas.aliyuncs.com)
#
# Usage on fresh Ubuntu 22.04 ECS:
#   chmod +x setup_aliyun.sh
#   NAS_HOST=123456-nas.cn-beijing.nas.aliyuncs.com ./setup_aliyun.sh
# ============================================================================

set -euo pipefail

GIT_REPO="${GIT_REPO:-https://github.com/wenchuoFu/Geo.git}"
NAS_HOST="${NAS_HOST:-}"  # Set this: your NAS mount target address

echo "=============================================="
echo " geo_seq_inference — Alibaba Cloud Setup"
echo "=============================================="
echo ""

# ── System packages ──────────────────────────────────────────────────────
echo "=== Step 1: System packages ==="
sudo apt-get update -qq
sudo apt-get install -y -qq python3.12 python3.12-venv python3.12-dev build-essential nfs-common git

# ── Python environment ────────────────────────────────────────────────────
echo ""
echo "=== Step 2: Python virtual environment ==="
if [ ! -d ~/geo_env ]; then
    python3.12 -m venv ~/geo_env
fi
source ~/geo_env/bin/activate
pip install --upgrade pip -q
pip install numpy scipy pandas matplotlib joblib pyarrow pytest pyyaml -q

# ── Git clone ─────────────────────────────────────────────────────────────
echo ""
echo "=== Step 3: Git clone ==="
if [ ! -d ~/geo_seq_inference ]; then
    git clone "$GIT_REPO" ~/geo_seq_inference
    echo "  Cloned $GIT_REPO"
else
    cd ~/geo_seq_inference && git pull && cd ~
    echo "  Pulled latest from $GIT_REPO"
fi

# ── Install the package ───────────────────────────────────────────────────
echo ""
echo "=== Step 4: Install package ==="
cd ~/geo_seq_inference
pip install -e . -q

# ── NAS mount for persistent results ──────────────────────────────────────
echo ""
echo "=== Step 5: NAS persistent storage ==="
NAS_MOUNT="/mnt/nas_experiments"

if [ -n "$NAS_HOST" ]; then
    sudo mkdir -p "$NAS_MOUNT"
    # Check if already mounted
    if mountpoint -q "$NAS_MOUNT"; then
        echo "  NAS already mounted at $NAS_MOUNT"
    else
        echo "  Mounting $NAS_HOST to $NAS_MOUNT ..."
        sudo mount -t nfs -o vers=3,nolock,proto=tcp,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2 "$NAS_HOST:/" "$NAS_MOUNT"
        echo "  NAS mounted successfully"
    fi

    # Create results/plots dirs on NAS
    mkdir -p "$NAS_MOUNT/results" "$NAS_MOUNT/plots"
    sudo chown -R "$(whoami):$(whoami)" "$NAS_MOUNT"

    # Symlink experiments/results and plots -> NAS
    rm -rf ~/geo_seq_inference/experiments/results ~/geo_seq_inference/experiments/plots
    ln -s "$NAS_MOUNT/results" ~/geo_seq_inference/experiments/results
    ln -s "$NAS_MOUNT/plots" ~/geo_seq_inference/experiments/plots
    echo "  Results & plots symlinked to NAS (persistent!)"
else
    echo "  WARNING: NAS_HOST not set — results will be LOST if spot instance reclaimed!"
    echo "  Set with: NAS_HOST=xxx.nas.aliyuncs.com ./setup_aliyun.sh"
fi

# ── Verify ─────────────────────────────────────────────────────────────────
echo ""
echo "=== Step 6: Verify ==="
echo "  CPU cores : $(nproc)"
echo "  Memory    : $(free -h | grep Mem | awk '{print $2}')"
echo "  Python    : $(python3.12 --version)"
echo "  Disk      : $(df -h / | tail -1 | awk '{print $4}') free"

# ── Run instructions ──────────────────────────────────────────────────────
echo ""
echo "=============================================="
echo " SETUP COMPLETE"
echo "=============================================="
echo ""
echo "=== RUN EXPERIMENTS ==="
echo ""
echo "  cd ~/geo_seq_inference"
echo "  source ~/geo_env/bin/activate"
echo "  nohup python experiments/run_all.py --signal-scale 4.0 > run.log 2>&1 &"
echo ""
echo "=== MONITOR ==="
echo ""
echo "  tail -f ~/geo_seq_inference/run.log"
echo ""
echo "=== IF SPOT INSTANCE RECLAIMED ==="
echo ""
echo "  1. Create new ECS (same VPC, same security group)"
echo "  2. Re-run this script:"
echo "     NAS_HOST=$NAS_HOST ./setup_aliyun.sh"
echo "  3. Resume experiments (skipping completed ones):"
echo "     python experiments/run_all.py --signal-scale 4.0 --skip E2"
echo "  Results on NAS are intact — no data lost!"
echo ""
if [ -z "$NAS_HOST" ]; then
    echo "  !! NAS not configured. To add it:"
    echo "  1. Create NAS in Alibaba Cloud console (same region as ECS)"
    echo "  2. Note the mount target address"
    echo "  3. Re-run: NAS_HOST=<address> ./setup_aliyun.sh"
fi
