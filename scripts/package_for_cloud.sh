#!/bin/bash
# ============================================================================
# Package geo_seq_inference for Alibaba Cloud upload.
# Run from d:\python\geo\ directory.
#
# Usage (Git Bash / WSL):
#   cd /d/python/geo
#   bash geo_seq_inference/scripts/package_for_cloud.sh
#
# Then upload with:
#   scp geo_seq_inference_cloud.zip root@<ECS_IP>:~/
# ============================================================================

set -euo pipefail

PROJECT="geo_seq_inference"
ARCHIVE="${PROJECT}_cloud.zip"
SRC_DIR="./${PROJECT}"

echo "=== Packaging ${PROJECT} for cloud deployment ==="

# Clean up previous archive
rm -f "$ARCHIVE"

# Build zip excluding dev artifacts
zip -r "$ARCHIVE" "$SRC_DIR" \
    -x "${SRC_DIR}/__pycache__/*" \
    -x "${SRC_DIR}/.git/*" \
    -x "${SRC_DIR}/.mypy_cache/*" \
    -x "${SRC_DIR}/.pytest_cache/*" \
    -x "${SRC_DIR}/*.egg-info/*" \
    -x "${SRC_DIR}/experiments/results/*" \
    -x "${SRC_DIR}/experiments/plots/*" \
    -x "${SRC_DIR}/node_modules/*"

SIZE=$(du -h "$ARCHIVE" | cut -f1)
echo ""
echo "=== Created: ${ARCHIVE} (${SIZE}) ==="
echo ""
echo "=== UPLOAD INSTRUCTIONS ==="
echo ""
echo "1. Upload to ECS:"
echo "   scp ${ARCHIVE} root@<ECS_PUBLIC_IP>:~/"
echo ""
echo "2. SSH into ECS and extract:"
echo "   ssh root@<ECS_PUBLIC_IP>"
echo "   unzip ${ARCHIVE} -d ~/"
echo ""
echo "3. Install and run:"
echo "   cd ~/geo_seq_inference"
echo "   source ~/geo_env/bin/activate"
echo "   pip install -e ."
echo "   python experiments/run_all.py --signal-scale 4.0"
echo ""
echo "4. Download results:"
echo "   scp -r root@<ECS_IP>:~/geo_seq_inference/experiments/results/ ./"
echo "   scp -r root@<ECS_IP>:~/geo_seq_inference/experiments/plots/ ./"
