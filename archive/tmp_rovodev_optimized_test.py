#!/usr/bin/env python3
"""
최적화된 설정으로 DICOM 성능 테스트
64코어 시스템 + 12개 동시 요청 성능 측정
"""
import time
import sys
from pathlib import Path
sys.path.append('.')

import config
from parallel_ollama_client import ParallelOllamaClient

def test_optimized_performance():
    """최적화된 설정으로 성능 테스트"""
    print("🚀 최적화된 병렬 처리 성능 테스트")
    print("=" * 60)
    
    # 시스템 정보 출력
    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    print(f"💻 시스템 정보:")
    print(f"  - CPU 코어: {cpu_count}개")
    print(f"  - 메모리: {memory_gb:.1f} GB")
    print(f"  - CPU 사용률: {psutil.cpu_percent(interval=1):.1f}%")
    
    print(f"\n⚙️ 최적화된 설정:")
    print(f"  - 동시 요청: {config.MAX_CONCURRENT_REQUESTS}개")
    print(f"  - 워커 스레드: {config.MAX_WORKERS}개")
    print(f"  - 청크 크기: {config.CHUNK_SIZE}개")
    
    # DICOM 이미지 확인
    staging_dir = config.STAGING_DIR / "DICOM"
    if not staging_dir.exists():
        print("❌ DICOM 스테이징 디렉토리가 없습니다.")
        return
    
    image_paths = sorted(list(staging_dir.glob("*.jpeg")))
    if not image_paths:
        print("❌ DICOM 이미지 파일이 없습니다.")
        return
    
    print(f"\n🖼️ 테스트 대상:")
    print(f"  - 파일: DICOM.pdf")
    print(f"  - 페이지 수: {len(image_paths)}개")
    
    # 병렬 클라이언트 초기화
    client = ParallelOllamaClient()
    
    # 연결 확인
    if not client.check_ollama_connection() or not client.check_model_availability():
        print("❌ Ollama 연결 실패")
        return
    
    print(f"✅ Ollama 연결 성공")
    
    # 성능 테스트 실행
    print(f"\n🚀 최적화된 병렬 처리 시작...")
    
    # CPU 모니터링 시작
    cpu_before = psutil.cpu_percent(interval=None)
    memory_before = psutil.virtual_memory()
    
    start_time = time.time()
    
    try:
        markdown_content = client.convert_images_to_markdown_parallel(image_paths)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # CPU 모니터링 종료
        cpu_after = psutil.cpu_percent(interval=1)
        memory_after = psutil.virtual_memory()
        
        # 결과 분석
        if markdown_content and markdown_content.strip():
            # 후처리 적용
            processed_content = client.post_process_syncfusion_content(markdown_content, "DICOM")
            
            # 임시 파일 저장
            output_file = Path("tmp_rovodev_dicom_optimized.md")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(processed_content)
            
            file_size = output_file.stat().st_size
            lines = len(processed_content.splitlines())
            words = len(processed_content.split())
            
            print(f"\n✅ 최적화된 병렬 처리 완료!")
            print(f"⏱️ 처리 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
            print(f"📊 처리량: {len(image_paths)/total_time:.2f} 페이지/초")
            print(f"📄 출력 크기: {file_size:,} bytes ({file_size/1024:.1f} KB)")
            print(f"📝 라인 수: {lines:,}")
            print(f"📖 단어 수: {words:,}")
            
            # 이전 결과들과 비교
            performance_history = {
                "순차 처리": {"time": 98.83, "throughput": 7.3},
                "병렬 처리 (6개)": {"time": 79.49, "throughput": 9.1},
                "최적화 (12개)": {"time": total_time, "throughput": len(image_paths)/total_time*60}
            }
            
            print(f"\n📈 성능 비교:")
            print(f"{'방식':<15} {'시간(초)':<10} {'처리량(p/분)':<12} {'속도 향상':<10}")
            print("-" * 50)
            
            baseline_time = performance_history["순차 처리"]["time"]
            
            for method, perf in performance_history.items():
                speedup = baseline_time / perf["time"]
                print(f"{method:<15} {perf['time']:<10.2f} {perf['throughput']:<12.1f} {speedup:<10.1f}배")
            
            # 시스템 리소스 사용량
            print(f"\n💻 시스템 리소스 사용량:")
            print(f"  - CPU 사용률: {cpu_after:.1f}% (처리 중 평균)")
            print(f"  - 메모리 사용: {(memory_after.used - memory_before.used) / (1024**2):.1f} MB 증가")
            print(f"  - 메모리 사용률: {memory_after.percent:.1f}%")
            
            # 통계 정보
            stats = client.stats
            print(f"\n📊 처리 통계:")
            print(f"  - 총 요청: {stats['total_requests']}")
            print(f"  - 성공: {stats['successful_requests']}")
            print(f"  - 실패: {stats['failed_requests']}")
            print(f"  - 성공률: {stats['successful_requests']/stats['total_requests']*100:.1f}%")
            if stats['successful_requests'] > 0:
                avg_time = stats['total_time'] / stats['successful_requests']
                print(f"  - 평균 응답 시간: {avg_time:.2f}초")
            
            # 최종 성능 향상 계산
            original_time = 98.83
            speedup = original_time / total_time
            time_saved = original_time - total_time
            efficiency = speedup / config.MAX_CONCURRENT_REQUESTS * 100
            
            print(f"\n🎯 최종 성과:")
            print(f"  ⚡ 총 속도 향상: {speedup:.1f}배")
            print(f"  ⏰ 시간 절약: {time_saved:.1f}초 ({time_saved/60:.1f}분)")
            print(f"  📊 병렬 효율성: {efficiency:.1f}%")
            print(f"  🚀 최대 처리량: {len(image_paths)/total_time*60:.1f} 페이지/분")
            
        else:
            print("❌ 최적화된 병렬 처리 실패")
            
    except Exception as e:
        print(f"❌ 테스트 중 오류: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimized_performance()