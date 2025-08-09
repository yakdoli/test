#!/usr/bin/env python3
"""
시스템 리소스 분석 및 Ollama 처리 능력 테스트
CPU 코어 수 기반 최적 워커 설정 확인
"""
import os
import psutil
import time
import threading
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import sys
sys.path.append('.')

import config

def analyze_system_resources():
    """시스템 리소스 분석"""
    print("🔍 시스템 리소스 분석")
    print("=" * 50)
    
    # CPU 정보
    cpu_count = psutil.cpu_count(logical=False)  # 물리적 코어
    cpu_threads = psutil.cpu_count(logical=True)  # 논리적 코어 (하이퍼스레딩 포함)
    cpu_freq = psutil.cpu_freq()
    
    print(f"💻 CPU 정보:")
    print(f"  - 물리적 코어: {cpu_count}개")
    print(f"  - 논리적 코어: {cpu_threads}개 (하이퍼스레딩 포함)")
    print(f"  - 기본 주파수: {cpu_freq.current:.0f} MHz")
    print(f"  - 최대 주파수: {cpu_freq.max:.0f} MHz")
    
    # 메모리 정보
    memory = psutil.virtual_memory()
    print(f"\n💾 메모리 정보:")
    print(f"  - 총 메모리: {memory.total / (1024**3):.1f} GB")
    print(f"  - 사용 가능: {memory.available / (1024**3):.1f} GB")
    print(f"  - 사용률: {memory.percent:.1f}%")
    
    # 현재 CPU 사용률
    cpu_usage = psutil.cpu_percent(interval=1, percpu=True)
    avg_cpu = sum(cpu_usage) / len(cpu_usage)
    print(f"\n⚡ 현재 CPU 사용률:")
    print(f"  - 평균: {avg_cpu:.1f}%")
    print(f"  - 코어별: {[f'{usage:.1f}%' for usage in cpu_usage]}")
    
    # 권장 워커 수 계산
    recommended_workers = {
        "conservative": cpu_count,  # 물리적 코어 수
        "balanced": cpu_threads,    # 논리적 코어 수
        "aggressive": cpu_threads + 2  # 논리적 코어 + 2
    }
    
    print(f"\n🔧 권장 워커 수:")
    print(f"  - 보수적: {recommended_workers['conservative']}개 (물리적 코어)")
    print(f"  - 균형적: {recommended_workers['balanced']}개 (논리적 코어)")
    print(f"  - 적극적: {recommended_workers['aggressive']}개 (논리적 코어 + 2)")
    print(f"  - 현재 설정: {config.MAX_WORKERS}개")
    
    return recommended_workers

def test_ollama_concurrent_capacity():
    """Ollama 동시 처리 능력 테스트"""
    print(f"\n🧪 Ollama 동시 처리 능력 테스트")
    print("=" * 50)
    
    # 테스트용 간단한 이미지 (DICOM 첫 번째 페이지)
    test_image = Path("staging/DICOM/page_001.jpeg")
    if not test_image.exists():
        print("❌ 테스트 이미지가 없습니다.")
        return
    
    print(f"🖼️ 테스트 이미지: {test_image.name}")
    
    # 다양한 동시 요청 수로 테스트
    concurrent_levels = [1, 2, 4, 6, 8, 10, 12]
    results = {}
    
    for concurrent in concurrent_levels:
        print(f"\n🔧 동시 요청 수: {concurrent}개")
        
        # 간단한 프롬프트로 빠른 테스트
        simple_prompt = "Describe this image briefly in one sentence."
        
        start_time = time.time()
        success_count = 0
        error_count = 0
        
        def single_request():
            try:
                # 이미지를 base64로 인코딩
                import base64
                with open(test_image, "rb") as f:
                    image_base64 = base64.b64encode(f.read()).decode('utf-8')
                
                payload = {
                    "model": config.OLLAMA_MODEL,
                    "prompt": simple_prompt,
                    "images": [image_base64],
                    "stream": False
                }
                
                response = requests.post(
                    f"{config.OLLAMA_BASE_URL}/api/generate",
                    json=payload,
                    timeout=60
                )
                
                if response.status_code == 200:
                    return True
                else:
                    return False
                    
            except Exception as e:
                print(f"    ❌ 요청 오류: {str(e)}")
                return False
        
        # 병렬 요청 실행
        with ThreadPoolExecutor(max_workers=concurrent) as executor:
            futures = [executor.submit(single_request) for _ in range(concurrent)]
            
            for future in as_completed(futures):
                if future.result():
                    success_count += 1
                else:
                    error_count += 1
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 결과 저장
        results[concurrent] = {
            "total_time": total_time,
            "success_count": success_count,
            "error_count": error_count,
            "throughput": success_count / total_time if total_time > 0 else 0
        }
        
        print(f"  ⏱️ 시간: {total_time:.2f}초")
        print(f"  ✅ 성공: {success_count}/{concurrent}")
        print(f"  ❌ 실패: {error_count}/{concurrent}")
        print(f"  📊 처리량: {results[concurrent]['throughput']:.2f} req/sec")
        
        # 실패율이 높으면 중단
        if error_count / concurrent > 0.5:
            print(f"  ⚠️ 실패율 {error_count/concurrent*100:.1f}% - 테스트 중단")
            break
        
        # 다음 테스트 전 잠시 대기
        time.sleep(2)
    
    # 결과 분석
    print(f"\n📊 Ollama 처리 능력 분석:")
    print(f"{'동시 요청':<8} {'시간(초)':<8} {'성공률':<8} {'처리량':<12} {'권장도':<8}")
    print("-" * 50)
    
    best_throughput = 0
    optimal_concurrent = 1
    
    for concurrent, result in results.items():
        success_rate = result['success_count'] / concurrent * 100
        throughput = result['throughput']
        
        if success_rate >= 90 and throughput > best_throughput:
            best_throughput = throughput
            optimal_concurrent = concurrent
            recommendation = "✅ 최적"
        elif success_rate >= 80:
            recommendation = "🟡 양호"
        else:
            recommendation = "❌ 과부하"
        
        print(f"{concurrent:<8} {result['total_time']:<8.2f} {success_rate:<8.1f}% {throughput:<12.2f} {recommendation}")
    
    print(f"\n🎯 권장 설정:")
    print(f"  - 최적 동시 요청 수: {optimal_concurrent}개")
    print(f"  - 최대 처리량: {best_throughput:.2f} req/sec")
    
    return optimal_concurrent

def recommend_optimal_settings():
    """최적 설정 권장"""
    print(f"\n🔧 최적 설정 권장")
    print("=" * 50)
    
    # 시스템 리소스 분석
    cpu_threads = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # Ollama 처리 능력 기반 권장
    print(f"📋 현재 설정:")
    print(f"  - MAX_CONCURRENT_REQUESTS: {config.MAX_CONCURRENT_REQUESTS}")
    print(f"  - MAX_WORKERS: {config.MAX_WORKERS}")
    print(f"  - CHUNK_SIZE: {config.CHUNK_SIZE}")
    
    # 권장 설정 계산
    recommended_concurrent = min(6, cpu_threads // 2)  # CPU 부하 고려
    recommended_workers = cpu_threads  # 논리적 코어 수
    recommended_chunk = max(3, recommended_concurrent)  # 동시 요청 수와 비슷하게
    
    print(f"\n💡 권장 설정:")
    print(f"  - MAX_CONCURRENT_REQUESTS: {recommended_concurrent}개")
    print(f"  - MAX_WORKERS: {recommended_workers}개")
    print(f"  - CHUNK_SIZE: {recommended_chunk}개")
    
    # 설정 변경 코드 생성
    config_code = f"""
# 시스템 최적화된 병렬 처리 설정
ENABLE_PARALLEL_PROCESSING = True
MAX_CONCURRENT_REQUESTS = {recommended_concurrent}  # CPU 기반 최적화
MAX_WORKERS = {recommended_workers}  # 논리적 코어 수
CHUNK_SIZE = {recommended_chunk}  # 메모리 최적화
REQUEST_TIMEOUT = 180
RETRY_DELAY = 1
"""
    
    print(f"\n📝 config.py 권장 설정:")
    print(config_code)
    
    return {
        "concurrent": recommended_concurrent,
        "workers": recommended_workers,
        "chunk_size": recommended_chunk
    }

if __name__ == "__main__":
    # 시스템 리소스 분석
    system_info = analyze_system_resources()
    
    # Ollama 처리 능력 테스트
    optimal_concurrent = test_ollama_concurrent_capacity()
    
    # 최적 설정 권장
    optimal_settings = recommend_optimal_settings()
    
    print(f"\n🎉 분석 완료!")
    print(f"시스템에 최적화된 설정을 config.py에 적용하세요.")