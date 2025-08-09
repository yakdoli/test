#!/usr/bin/env bash
# Balanced 프리셋 + 내부동시성 4로 최대 스케일 DICOM 테스트 실행
# 실행: bash tmp_rovodev_balanced_run.sh

set -euo pipefail

# 1) Balanced 프리셋 적용 (4x80GB 권장)
export GPU_MAX_MEMORY_FRACTION=${GPU_MAX_MEMORY_FRACTION:-0.9}
export WORKER_ESTIMATED_VRAM_GB=${WORKER_ESTIMATED_VRAM_GB:-16}
export MAX_WORKERS_PER_GPU=${MAX_WORKERS_PER_GPU:-4}
export MAX_TOTAL_GPU_WORKERS=${MAX_TOTAL_GPU_WORKERS:-16}
export PER_WORKER_BATCH_SIZE=${PER_WORKER_BATCH_SIZE:-1}
export PER_WORKER_CONCURRENCY=${PER_WORKER_CONCURRENCY:-4}

echo "[ENV] Balanced preset applied (internal concurrency=4)"
echo "GPU_MAX_MEMORY_FRACTION=$GPU_MAX_MEMORY_FRACTION"
echo "WORKER_ESTIMATED_VRAM_GB=$WORKER_ESTIMATED_VRAM_GB"
echo "MAX_WORKERS_PER_GPU=$MAX_WORKERS_PER_GPU"
echo "MAX_TOTAL_GPU_WORKERS=$MAX_TOTAL_GPU_WORKERS"
echo "PER_WORKER_BATCH_SIZE=$PER_WORKER_BATCH_SIZE"
echo "PER_WORKER_CONCURRENCY=$PER_WORKER_CONCURRENCY"

# 2) Sanity 체크 (정보 출력용)
python tmp_rovodev_gpu_sanity.py || true

# 3) 최대 스케일 테스트 실행
python test_max_scale_dicom.py || true
