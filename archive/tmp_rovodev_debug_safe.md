# Safe 설정 디버깅 가이드 (4x80GB 기준)

## 1) 실행
```bash
bash tmp_rovodev_safe_run.sh | tee safe_run.log
```

## 2) 로그에서 확인할 지표
- "GPU당 워커 수용력", "총 병렬 인스턴스(워커)", "배치 크기(워커당)"
- "워커 i (GPU g) 초기화 완료" → 모든 워커가 기동되었는지
- 처리 완료 요약: 총 시간, 처리량(페이지/초), 성공/실패

## 3) 문제 발생 시
### OOM (Out of Memory)
- 증상: 모델 초기화 실패, generate 중 CUDA out of memory
- 조치:
  - MAX_WORKERS_PER_GPU를 3 → 2로 낮춤
  - 또는 WORKER_ESTIMATED_VRAM_GB를 18 → 20으로 상향
  - PER_WORKER_BATCH_SIZE는 1 유지 권장

### 활용률 저조(각 GPU util < 50%)
- 조치:
  - MAX_WORKERS_PER_GPU 3 → 4로 상승 (OOM이 없다면)
  - 또는 WORKER_ESTIMATED_VRAM_GB 18 → 16으로 하향해 GPU당 워커 수 증가

### 초기화가 매우 느림
- 첫 모델 다운로드(캐시) 여부 확인
- 디스크 속도 확인, 기타 프로세스가 GPU 점유 중인지 확인(nvidia-smi)

## 4) 단계적 튜닝 절차
1. Safe 프리셋으로 1회 실행 → 완료/안정성 확인
2. 활용률이 낮다면 Balanced 프리셋으로 변경 후 재실행
3. 여전히 낮다면 Aggressive 프리셋으로 시도하되, OOM 시 즉시 롤백

## 5) 모니터링 명령
```bash
nvidia-smi -l 2
```

## 6) 롤백/정리
- 실패 시 환경 변수 초기화 후 Safe 프리셋으로 재실행
- 장시간 처리 전에는 항상 경량 테스트(test_lightweight_max_scale.py)로 사전 검증 권장
