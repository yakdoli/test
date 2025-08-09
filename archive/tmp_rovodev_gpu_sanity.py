# -*- coding: utf-8 -*-
"""
GPU/환경 사전 점검 스크립트 (Safe 설정용)
- CUDA 가용성, 디바이스 정보, 간단한 텐서 연산 검증
- 스테이징(DICOM) 이미지 존재 여부 확인
"""
import os
from pathlib import Path

try:
    import torch
except Exception as e:
    print(f"❌ PyTorch 로드 실패: {e}")
    raise

import config

def check_cuda():
    print("[CUDA 체크]")
    print(f"  torch 버전: {torch.__version__}")
    print(f"  CUDA 가용: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return
    dev_count = torch.cuda.device_count()
    print(f"  GPU 개수: {dev_count}")
    for i in range(dev_count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024**3)
        print(f"   - GPU {i}: {props.name}, {total_gb:.1f}GB, CC {props.major}.{props.minor}")

    # 간단 텐서 연산 검증
    try:
        with torch.cuda.device(0):
            a = torch.randn((1024, 1024), device='cuda')
            b = torch.randn((1024, 1024), device='cuda')
            c = torch.mm(a, b)
            print(f"  ✔ CUDA 텐서 연산 OK (matrix mul, shape={tuple(c.shape)})")
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"  ⚠ CUDA 텐서 연산 실패: {e}")


def check_staging():
    print("\n[스테이징 이미지 체크]")
    dicom_dir = config.STAGING_DIR / "DICOM"
    if not dicom_dir.exists():
        print(f"  ❌ 디렉토리 없음: {dicom_dir}")
        return False
    images = sorted(list(dicom_dir.glob('*.jpeg')))
    print(f"  이미지 파일: {len(images)}개 발견")
    if images[:3]:
        for p in images[:3]:
            print(f"   - {p.name}")
    return len(images) > 0


def print_safe_env_snapshot():
    print("\n[Safe 환경 변수 스냅샷]")
    keys = [
        'GPU_MAX_MEMORY_FRACTION', 'WORKER_ESTIMATED_VRAM_GB',
        'MAX_WORKERS_PER_GPU', 'MAX_TOTAL_GPU_WORKERS', 'PER_WORKER_BATCH_SIZE',
        'PER_WORKER_CONCURRENCY'
    ]
    for k in keys:
        print(f"  {k} = {os.getenv(k)}")


def main():
    print("🚀 GPU Sanity & Staging Check (Safe)")
    print("="*60)
    print_safe_env_snapshot()
    check_cuda()
    ok = check_staging()
    print("\n결과: ")
    print(f"  스테이징 준비: {'OK' if ok else 'NG'}")

if __name__ == '__main__':
    main()
