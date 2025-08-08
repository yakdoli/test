#!/bin/bash

# Xinference 서버 시작 스크립트
# qwen2-vl-instruct 모델을 vLLM 엔진으로 실행

echo "🚀 Xinference 서버 시작 중..."
echo "모델: qwen2-vl-instruct (7B, GPTQ, Int8)"
echo "엔진: vLLM"
echo "포트: 9997"
echo ""

# Xinference 서버 시작
xinference launch \
    --model-engine vLLM \
    --model-name qwen2-vl-instruct \
    --size-in-billions 7 \
    --model-format gptq \
    --quantization Int8 \
    --port 9997

echo ""
echo "✅ Xinference 서버가 http://localhost:9997에서 실행 중입니다."
echo "PDF 변환을 시작하려면 'python main.py' 또는 'python main_parallel.py'를 실행하세요."