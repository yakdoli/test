#!/usr/bin/env bash
# Safe 프리셋으로 최대 스케일 DICOM 테스트 실행 스크립트
# 실행: bash tmp_rovodev_safe_run.sh

set -euo pipefail

# 1) Safe 프리셋 적용 (4x80GB 기준, 보수적 시작)
export GPU_MAX_MEMORY_FRACTION=${GPU_MAX_MEMORY_FRACTION:-0.9}
export WORKER_ESTIMATED_VRAM_GB=${WORKER_ESTIMATED_VRAM_GB:-18}
export MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-3}
export MAX_TOTAL_GPU_WORKERS=${MAX_TOTAL_GPU_WORKERS:-12}
export PER_WORKER_BATCH_SIZE=${PER_WORKER_BATCH_SIZE:-1}

echo "[ENV] Safe preset applied"
echo "GPU_MAX_MEMORY_FRACTION=$GPU_MAX_MEMORY_FRACTION"
echo "WORKER_ESTIMATED_VRAM_GB=$WORKER_ESTIMATED_VRAM_GB"
echo "MAX_WORKERS_PER_GPU=$MAX_WORKERS_PER_GPU"
echo "MAX_TOTAL_GPU_WORKERS=$MAX_TOTAL_GPU_WORKERS"
echo "PER_WORKER_BATCH_SIZE=$PER_WORKER_BATCH_SIZE"

# 2) Sanity 체크
python tmp_rovodev_gpu_sanity.py || true

# 3) 최대 스케일 테스트 실행
python test_max_scale_dicom.py || true
