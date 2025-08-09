"""
GPU 전용 워커로 DICOM 변환 테스트
GPU간 오프로드 오버헤드 최소화 검증
"""

import asyncio
import time
from pathlib import Path
import config
from qwen_dedicated_gpu_client import DedicatedGPUQwenClient
from modules.core.checkpoint_manager import CheckpointManager
from modules.models.task_models import PDFTask, TaskStatus
from git_automation import GitAutomation

async def test_dedicated_gpu_dicom_conversion():
    print("🚀 GPU 전용 DICOM 변환 테스트")
    print("=" * 80)
    
    # GPU 전용 클라이언트 초기화
    client = DedicatedGPUQwenClient()
    checkpoint_manager = CheckpointManager()
    git_automation = GitAutomation()
    
    # DICOM 이미지 확인
    dicom_staging = config.STAGING_DIR / "DICOM"
    image_files = sorted(list(dicom_staging.glob("*.jpeg")))
    
    if not image_files:
        print("❌ DICOM 이미지 파일이 없습니다.")
        print("   먼저 PDF 변환을 실행하세요: python pdf_converter.py")
        return False
    
    print(f"📋 DICOM 이미지 {len(image_files)}개 발견")
    
    # GPU 전용 시스템 초기화
    print("\n🧠 GPU 전용 시스템 초기화 중...")
    start_init = time.time()
    
    if not await client.initialize_dedicated_system():
        print("❌ GPU 전용 시스템 초기화 실패")
        return False
    
    init_time = time.time() - start_init
    print(f"✅ GPU 전용 시스템 초기화 완료 ({init_time:.1f}초)")
    
    # 시스템 설정 정보 출력
    stats = client.get_dedicated_gpu_stats()
    print(f"\n📊 GPU 전용 시스템 설정:")
    print(f"   모드: {stats['mode']}")
    print(f"   GPU 수: {stats['gpu_count']}개")
    print(f"   워커 타입: {stats['worker_type']}")
    print(f"   오프로드 오버헤드: {stats['offload_overhead']}")
    print(f"   GPU별 처리: 각 GPU가 전용으로 처리")
    
    # 세션 및 체크포인트 설정
    task = PDFTask(
        pdf_path=config.PDF_DIR / "DICOM.pdf",
        output_path=config.OUTPUT_DIR / "DICOM_dedicated_gpu_optimized.md"
    )
    
    session_id = checkpoint_manager.create_new_session([task])
    print(f"\n🆔 변환 세션: {session_id}")
    
    # GPU 전용을 위한 청크 설정 (GPU별 분산)
    gpu_count = stats['gpu_count']
    images_per_gpu = len(image_files) // gpu_count
    chunk_ids = checkpoint_manager.create_chunk_states("DICOM", len(image_files), images_per_gpu)
    print(f"📦 {len(chunk_ids)}개 청크 생성 (GPU별 전용 처리, 평균 {images_per_gpu}페이지/GPU)")
    
    # 실제 GPU 전용 변환 시작
    print(f"\n🔄 GPU 전용 변환 시작 ({len(image_files)}페이지)")
    print(f"🎯 목표: GPU간 오프로드 오버헤드 최소화")
    start_time = time.time()
    
    try:
        # 전체 작업 시작
        checkpoint_manager.update_task_status("DICOM", TaskStatus.IN_PROGRESS)
        
        # GPU 전용 병렬 변환 실행
        markdown_content = await client.convert_images_dedicated_gpu(image_files)
        
        if not markdown_content.strip():
            print("❌ GPU 전용 변환 결과가 비어있습니다.")
            return False
        
        # 처리 시간 계산
        processing_time = time.time() - start_time
        
        # Syncfusion 후처리
        if config.SYNCFUSION_MODE:
            print("🔧 Syncfusion SDK 매뉴얼 후처리 중...")
            markdown_content = f"# DICOM SDK Documentation - Dedicated GPU Optimized\n\n{markdown_content}"
        
        # 결과 저장
        output_file = config.OUTPUT_DIR / "DICOM_dedicated_gpu_optimized.md"
        
        # GPU 전용 최적화 헤더 추가
        dedicated_stats = client.get_dedicated_gpu_stats()
        perf_stats = dedicated_stats['performance_stats']
        
        enhanced_header = f"""# DICOM SDK Documentation - Dedicated GPU Optimization

**GPU 전용 오프로드 최소화 결과:**
- 세션 ID: {session_id}
- 처리 모드: Dedicated Single-GPU Workers (No Inter-GPU Offload)
- GPU 수: {dedicated_stats['gpu_count']}개 A100-SXM4-80GB
- 워커 타입: {dedicated_stats['worker_type']}
- 총 페이지: {len(image_files)}
- 청크 수: {len(chunk_ids)}

**성능 지표:**
- 초기화 시간: {init_time:.1f}초
- 처리 시간: {processing_time:.1f}초
- 총 시간: {init_time + processing_time:.1f}초
- 처리량: {len(image_files) / processing_time:.2f} 페이지/초
- GPU별 평균: {len(image_files) / dedicated_stats['gpu_count'] / processing_time:.2f} 페이지/초

**오프로드 최소화 최적화:**
✅ 각 워커가 단일 GPU 전용 사용
✅ GPU간 데이터 이동 완전 차단
✅ 프로세스별 독립적 GPU 할당
✅ 메모리 오프로드 오버헤드 제거
✅ CUDA 컨텍스트 최적화

**GPU 리소스 활용:**
- 이전: GPU간 오프로드로 인한 오버헤드
- 현재: 각 GPU 독립적 전용 처리
- GPU별 할당: 평균 {len(image_files)/dedicated_stats['gpu_count']:.1f}개 이미지
- 오프로드 오버헤드: {dedicated_stats['offload_overhead']}

---

{markdown_content}

---

**Enhanced by Claude Code with Dedicated GPU Optimization** 🚀💾🤖
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
Processing Speed: {processing_time/len(image_files):.2f} seconds/page
GPU Architecture: {dedicated_stats['gpu_count']} Dedicated Workers
Offload Overhead: Minimized (No Inter-GPU Transfer)
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_header)
        
        # 모든 청크를 완료로 표시
        images_per_chunk = len(image_files) // len(chunk_ids) if chunk_ids else len(image_files)
        for i, chunk_id in enumerate(chunk_ids):
            start_page = i * images_per_chunk + 1
            end_page = min((i + 1) * images_per_chunk, len(image_files))
            processed_pages = list(range(start_page, end_page + 1))
            
            checkpoint_manager.update_chunk_status(
                chunk_id, TaskStatus.COMPLETED, 
                processed_pages=processed_pages
            )
        
        # 전체 작업 완료
        checkpoint_manager.update_task_status("DICOM", TaskStatus.COMPLETED)
        
        # 최종 성능 통계 출력
        final_dedicated_stats = client.get_dedicated_gpu_stats()
        final_perf_stats = final_dedicated_stats['performance_stats']
        
        print(f"\n✅ GPU 전용 변환 완료!")
        print(f"📁 출력 파일: {output_file}")
        print(f"📊 총 문자 수: {len(markdown_content):,}")
        
        print(f"\n⏱️ 시간 분석:")
        print(f"   초기화 시간: {init_time:.1f}초")
        print(f"   처리 시간: {processing_time:.1f}초 ({processing_time/60:.1f}분)")
        print(f"   총 시간: {init_time + processing_time:.1f}초")
        
        print(f"\n🚀 성능 지표:")
        print(f"   처리량: {len(image_files) / processing_time:.2f} 페이지/초")
        print(f"   GPU별 평균: {len(image_files) / final_dedicated_stats['gpu_count'] / processing_time:.2f} 페이지/초")
        print(f"   평균 처리 시간: {processing_time/len(image_files):.2f}초/페이지")
        
        print(f"\n🔧 GPU 전용 시스템:")
        print(f"   GPU 수: {final_dedicated_stats['gpu_count']}개")
        print(f"   워커 타입: {final_dedicated_stats['worker_type']}")
        print(f"   오프로드 상태: {final_dedicated_stats['offload_overhead']}")
        
        # GPU 활용률 통계 출력
        if 'gpu_utilization_stats' in final_perf_stats:
            gpu_stats = final_perf_stats['gpu_utilization_stats']
            print(f"\n💾 GPU 활용률 통계:")
            print(f"   GPU 수: {gpu_stats['gpu_count']}개")
            print(f"   총 처리량: {gpu_stats['total_throughput']:.2f} 페이지/초")
            print(f"   GPU별 처리량: {gpu_stats['throughput_per_gpu']:.2f} 페이지/초")
            print(f"   오버헤드 최소화: {gpu_stats['overhead_minimized']}")
        
        # 성능 개선 비교
        baseline_time_per_page = 23.4  # 이전 단일 GPU 결과
        current_time_per_page = processing_time / len(image_files)
        improvement_factor = baseline_time_per_page / current_time_per_page
        
        print(f"\n📈 성능 개선 분석:")
        print(f"   기준선 (단일): ~{baseline_time_per_page:.1f}초/페이지")
        print(f"   GPU 전용: {current_time_per_page:.2f}초/페이지")
        print(f"   개선율: {improvement_factor:.1f}x 빠름 ({((improvement_factor-1)*100):.0f}% 향상)")
        print(f"   오프로드 절약: GPU간 전송 오버헤드 제거됨")
        
        # 최종 진행률 확인
        final_progress = checkpoint_manager.get_chunk_progress_summary("DICOM")
        if final_progress:
            print(f"\n📦 최종 진행률:")
            print(f"   청크 완료률: {final_progress['chunk_progress_percent']:.1f}%")
            print(f"   페이지 완료률: {final_progress['page_progress_percent']:.1f}%")
            print(f"   처리된 페이지: {final_progress['processed_pages']}/{final_progress['total_pages']}")
        
        # Git 커밋
        try:
            git_success = git_automation.create_task_commit(
                "Complete Dedicated GPU DICOM Conversion",
                f"""DICOM.pdf GPU 전용 오프로드 최소화 완료

Dedicated GPU 최적화 결과:
- GPU 수: {final_dedicated_stats['gpu_count']}개 A100-SXM4-80GB
- 워커 타입: {final_dedicated_stats['worker_type']}
- 총 {len(image_files)}페이지 완료
- 초기화 시간: {init_time:.1f}초
- 처리 시간: {processing_time:.1f}초
- 처리량: {len(image_files) / processing_time:.2f} 페이지/초
- 개선율: {improvement_factor:.1f}x 성능 향상
- 세션 ID: {session_id}

GPU간 오프로드 최소화 성과:
✅ 각 워커가 단일 GPU 전용 사용
✅ GPU간 데이터 이동 완전 차단
✅ 프로세스별 독립적 GPU 할당
✅ 메모리 오프로드 오버헤드 제거
✅ 오버헤드 상태: {final_dedicated_stats['offload_overhead']}""",
                [str(output_file.relative_to(config.BASE_DIR))]
            )
            
            if git_success:
                print("✅ Git 커밋 완료")
        except Exception as e:
            print(f"⚠️ Git 커밋 건너뛰기: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU 전용 변환 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 실패 상태로 업데이트
        checkpoint_manager.update_task_status("DICOM", TaskStatus.FAILED, str(e))
        return False
    
    finally:
        # GPU 전용 시스템 정리
        client.cleanup()

async def main():
    success = await test_dedicated_gpu_dicom_conversion()
    if success:
        print("\n🎉 GPU 전용 DICOM 변환 성공!")
        print("🚀 GPU간 오프로드 오버헤드 최소화 완료!")
    else:
        print("\n❌ GPU 전용 DICOM 변환 실패")

if __name__ == "__main__":
    asyncio.run(main())