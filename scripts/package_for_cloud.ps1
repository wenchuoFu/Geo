# ============================================================================
# Package geo_seq_inference for cloud upload (Windows PowerShell)
#
# Usage:
#   cd d:\python\geo
#   powershell -ExecutionPolicy Bypass -File geo_seq_inference\scripts\package_for_cloud.ps1
# ============================================================================

param(
    [string]$EcsIp = "",
    [string]$SshKey = ""
)

$Project = "geo_seq_inference"
$Archive = "${Project}_cloud.zip"

Write-Host "=== Packaging ${Project} for cloud deployment ===" -ForegroundColor Cyan

# Remove old archive
if (Test-Path $Archive) { Remove-Item $Archive }

# Create zip (7z or Compress-Archive)
if (Get-Command "7z" -ErrorAction SilentlyContinue) {
    # 7z is faster and excludes better
    & 7z a -tzip $Archive "${Project}\" `
        -xr!"__pycache__" -xr!".git" -xr!"*.egg-info" `
        -xr!"experiments\results\*" -xr!"experiments\plots\*" `
        -xr!"node_modules"
} else {
    # Fallback: PowerShell Compress-Archive (slower, includes everything)
    Write-Host "  7z not found, using Compress-Archive (slower)..."
    Compress-Archive -Path "${Project}" -DestinationPath $Archive -Force
}

$size = [math]::Round((Get-Item $Archive).Length / 1KB, 1)
Write-Host "  Created: ${Archive} (${size} KB)" -ForegroundColor Green

Write-Host ""
Write-Host "=== UPLOAD INSTRUCTIONS ===" -ForegroundColor Yellow
Write-Host ""
Write-Host "1. Upload to ECS (SCP):"
if ($SshKey) {
    Write-Host "   scp -i ${SshKey} ${Archive} root@<ECS_IP>:~/" -ForegroundColor Gray
} else {
    Write-Host "   scp ${Archive} root@<ECS_IP>:~/" -ForegroundColor Gray
}
Write-Host ""
Write-Host "   Or use Alibaba Cloud web console: ECS -> Instance -> Remote Connection -> Upload File"
Write-Host ""
Write-Host "2. SSH in, extract & run:"
Write-Host "   ssh root@<ECS_IP>"
Write-Host "   unzip ${Archive} -d ~/"
Write-Host "   cd ~/geo_seq_inference"
Write-Host "   source ~/geo_env/bin/activate"
Write-Host "   pip install -e ."
Write-Host "   nohup python experiments/run_all.py --signal-scale 4.0 > run.log 2>&1 &"
Write-Host ""
Write-Host "3. Monitor progress:"
Write-Host "   tail -f ~/geo_seq_inference/run.log"
Write-Host ""
Write-Host "4. Download results when done:"
Write-Host "   scp -r root@<ECS_IP>:~/geo_seq_inference/experiments/results/ .\experiments\results\"
Write-Host "   scp -r root@<ECS_IP>:~/geo_seq_inference/experiments/plots/ .\experiments\plots\"

# If ECS IP provided, try direct upload
if ($EcsIp) {
    Write-Host ""
    Write-Host "=== Attempting upload to ${EcsIp} ... ===" -ForegroundColor Cyan
    if ($SshKey) {
        scp -i $SshKey $Archive "root@${EcsIp}:~/"
    } else {
        scp $Archive "root@${EcsIp}:~/"
    }
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  Upload successful!" -ForegroundColor Green
    } else {
        Write-Host "  Upload failed. Check ECS IP and SSH connectivity." -ForegroundColor Red
    }
}
