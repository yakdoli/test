#!/usr/bin/env bash
set -euo pipefail

# Xinference(vLLM) 서버 시작 스크립트 (A100 80G x4 / RAM 930G / CPU 64코어 최적화)

MODEL_NAME=${MODEL_NAME:-qwen2.5-vl-instruct}
MODEL_SIZE_B=${MODEL_SIZE_B:-7}
PORT=${PORT:-9997}

# vLLM 권장 동시성/메모리 설정 (필요시 환경변수로 오버라이드)
TP_SIZE=${TP_SIZE:-4}                    # 4×A100 사용
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.92}

echo "🚀 Xinference 서버 시작 중..."
echo "모델: ${MODEL_NAME} (${MODEL_SIZE_B}B)"
echo "엔진: vLLM"
echo "포트: ${PORT}"
echo "텐서 병렬: ${TP_SIZE}, 컨텍스트: ${MAX_MODEL_LEN}, 동시 시퀀스: ${MAX_NUM_SEQS}, GPU 메모리 비율: ${GPU_MEM_UTIL}"
echo

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export HF_HOME=${HF_HOME:-/workspace/.hf}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-8}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-32}
# vLLM 다중프로세스 방식(교착 회피용). 필요 시 조정 가능
export VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-forkserver}

# 사전 확인
if ! command -v xinference >/dev/null 2>&1; then
    echo "❌ xinference 명령을 찾을 수 없습니다. 'pip install xinference[all]' 후 재시도하세요." >&2
    exit 1
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    echo "📟 GPU 요약:"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

# Xinference(vLLM) 서버 시작
# 주의: 아래 vLLM 관련 플래그는 사용 중인 Xinference 버전에 따라 인자명이 다를 수 있습니다.
#       'xinference launch --help'로 지원 인자를 확인 후 필요 시 수정하세요.
xinference launch \
    --model-engine vLLM \
    --model-name "${MODEL_NAME}" \
    --size-in-billions "${MODEL_SIZE_B}" \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}"

echo
echo "✅ Xinference 서버가 http://localhost:${PORT} 에서 실행 중입니다."
echo "PDF 변환을 시작하려면 'python main.py' 또는 'python main_parallel.py'를 실행하세요."