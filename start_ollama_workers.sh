#!/bin/bash
# Ollama 모델 워커 실행 스크립트

# GPU 1, 2, 3에 대해 각각 다른 포트로 Ollama 서버를 실행합니다.
# 기존 서비스(GPU 0, 포트 11434)는 그대로 실행 중이라고 가정합니다.
export OLLAMA_MODELS="/workspace/ollama_models"
export OLLAMA_CONTEXT_LENGTH=126000
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_NEW_ENGINE=1
# 워커 1 (GPU 0)
export CUDA_VISIBLE_DEVICES=0
OLLAMA_HOST=0.0.0.0:11434 ollama serve &
WORKER1_PID=$!
echo "✅ 워커 1 시작 (GPU 0, 포트 11434, PID: $WORKER1_PID)"

echo "🚀 추가 Ollama 워커를 시작합니다..."

# 워커 2 (GPU 1)
export CUDA_VISIBLE_DEVICES=1
OLLAMA_HOST=0.0.0.0:11435 ollama serve &
WORKER2_PID=$!
echo "✅ 워커 2 시작 (GPU 1, 포트 11435, PID: $WORKER2_PID)"

# 워커 3 (GPU 2)
export CUDA_VISIBLE_DEVICES=2

OLLAMA_HOST=0.0.0.0:11436 ollama serve &
WORKER3_PID=$!
echo "✅ 워커 3 시작 (GPU 2, 포트 11436, PID: $WORKER3_PID)"

# 워커 4 (GPU 3)
export CUDA_VISIBLE_DEVICES=3

OLLAMA_HOST=0.0.0.0:11437 ollama serve &
WORKER4_PID=$!
echo "✅ 워커 4 시작 (GPU 3, 포트 11437, PID: $WORKER4_PID)"

echo "
모든 추가 워커가 백그라운드에서 실행되었습니다."
echo "프로세스를 종료하려면 다음 명령어를 사용하세요:"
echo "kill $WORKER1_PID $WORKER2_PID $WORKER3_PID $WORKER4_PID"
