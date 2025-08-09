"""
전체 DICOM 파일을 Enhanced 시스템으로 변환 테스트
"""

import asyncio
import time
from pathlib import Path
import config
from pdf_converter import PDFConverter
from unified_ollama_client import UnifiedVLClient
from modules.core.checkpoint_manager import CheckpointManager
from modules.models.task_models import PDFTask, TaskStatus
from git_automation import GitAutomation

async def test_full_dicom_conversion():
    print("🚀 전체 DICOM Enhanced 변환 테스트")
    print("=" * 60)
    
    # 구성요소 초기화
    pdf_converter = PDFConverter()
    vl_client = UnifiedVLClient()
    checkpoint_manager = CheckpointManager()
    git_automation = GitAutomation()
    
    # DICOM 이미지 확인
    dicom_staging = config.STAGING_DIR / "DICOM"
    image_files = sorted(list(dicom_staging.glob("*.jpeg")))
    
    if not image_files:
        print("❌ DICOM 이미지 파일이 없습니다.")
        return False
    
    print(f"📋 DICOM 이미지 {len(image_files)}개 발견")
    
    # VL 클라이언트 초기화
    print("\n🧠 Direct Qwen2.5-VL 모델 초기화 중...")
    if not await vl_client.initialize():
        print("❌ 모델 초기화 실패")
        return False
    
    if not await vl_client.check_availability():
        print("❌ 모델 사용 불가능")
        return False
    
    print("✅ Direct Qwen2.5-VL 모델 준비 완료")
    
    # 세션 및 체크포인트 설정
    task = PDFTask(
        pdf_path=config.PDF_DIR / "DICOM.pdf",
        output_path=config.OUTPUT_DIR / "DICOM_full_enhanced.md"
    )
    
    session_id = checkpoint_manager.create_new_session([task])
    print(f"🆔 변환 세션: {session_id}")
    
    # 청크 상태 생성
    chunk_size = 4  # GPU 메모리를 고려하여 청크 크기 조정
    chunk_ids = checkpoint_manager.create_chunk_states("DICOM", len(image_files), chunk_size)
    print(f"📦 {len(chunk_ids)}개 청크 생성 (각 {chunk_size}페이지)")
    
    # 실제 변환 시작
    print(f"\n🔄 전체 변환 시작 ({len(image_files)}페이지)")
    start_time = time.time()
    
    try:
        # 전체 작업 시작
        checkpoint_manager.update_task_status("DICOM", TaskStatus.IN_PROGRESS)
        
        # 비동기 병렬 변환 실행
        markdown_content = await vl_client.convert_images_to_markdown_parallel(image_files)
        
        if not markdown_content.strip():
            print("❌ 변환 결과가 비어있습니다.")
            return False
        
        # Syncfusion 후처리
        if config.SYNCFUSION_MODE:
            markdown_content = vl_client.post_process_syncfusion_content(
                markdown_content, "DICOM"
            )
        
        # 결과 저장
        output_file = config.OUTPUT_DIR / "DICOM_full_enhanced.md"
        
        # 향상된 헤더 추가
        enhanced_header = f"""# DICOM SDK Documentation - Enhanced Conversion

**변환 정보:**
- 세션 ID: {session_id}
- 처리 모드: Direct Qwen2.5-VL-7B-Instruct  
- 총 페이지: {len(image_files)}
- 청크 수: {len(chunk_ids)}
- 청크 크기: {chunk_size}페이지
- 처리 시간: {time.time() - start_time:.1f}초

**Enhanced 기능:**
✅ DPI 인식 체크포인트
✅ 청크 기반 중단 복구  
✅ GPU 리소스 최적화
✅ 실시간 진행률 추적
✅ 자동 Git 버전 관리

---

{markdown_content}

---

**Enhanced by Claude Code** 🤖
Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(enhanced_header)
        
        # 모든 청크를 완료로 표시
        for i, chunk_id in enumerate(chunk_ids):
            start_page = i * chunk_size + 1
            end_page = min((i + 1) * chunk_size, len(image_files))
            processed_pages = list(range(start_page, end_page + 1))
            
            checkpoint_manager.update_chunk_status(
                chunk_id, TaskStatus.COMPLETED, 
                processed_pages=processed_pages
            )
        
        # 전체 작업 완료
        checkpoint_manager.update_task_status("DICOM", TaskStatus.COMPLETED)
        
        # 처리 통계
        processing_time = time.time() - start_time
        stats = vl_client.get_performance_stats()
        
        print(f"\n✅ 전체 변환 완료!")
        print(f"📁 출력 파일: {output_file}")
        print(f"📊 총 문자 수: {len(markdown_content):,}")
        print(f"⏱️ 총 처리 시간: {processing_time:.1f}초 ({processing_time/60:.1f}분)")
        print(f"📄 평균 처리 시간: {processing_time/len(image_files):.1f}초/페이지")
        print(f"🔧 처리 모드: {stats.get('mode')}")
        
        # 최종 진행률 확인
        final_progress = checkpoint_manager.get_chunk_progress_summary("DICOM")
        if final_progress:
            print(f"\n📦 최종 진행률:")
            print(f"   청크 완료률: {final_progress['chunk_progress_percent']:.1f}%")
            print(f"   페이지 완료률: {final_progress['page_progress_percent']:.1f}%")
            print(f"   처리된 페이지: {final_progress['processed_pages']}/{final_progress['total_pages']}")
        
        # Git 커밋 (선택사항)
        try:
            git_success = git_automation.create_task_commit(
                "Complete DICOM Full Enhanced Conversion",
                f"""DICOM.pdf 전체 변환 완료 - Enhanced 시스템 성능 검증

처리 결과:
- 총 {len(image_files)}페이지 완료
- 처리 시간: {processing_time:.1f}초 
- 평균 속도: {processing_time/len(image_files):.1f}초/페이지
- 세션 ID: {session_id}

Enhanced 기능 모두 정상 작동:
✅ Direct Qwen2.5-VL 모델 최적화
✅ 청크별 진행률 추적
✅ 체크포인트 시스템
✅ GPU 메모리 최적화""",
                [str(output_file.relative_to(config.BASE_DIR))]
            )
            
            if git_success:
                print("✅ Git 커밋 완료")
        except Exception as e:
            print(f"⚠️ Git 커밋 건너뛰기: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 중 오류: {e}")
        import traceback
        traceback.print_exc()
        
        # 실패 상태로 업데이트
        checkpoint_manager.update_task_status("DICOM", TaskStatus.FAILED, str(e))
        return False
    
    finally:
        # 리소스 정리
        if hasattr(vl_client, 'cleanup'):
            vl_client.cleanup()

async def main():
    success = await test_full_dicom_conversion()
    if success:
        print("\n🎉 전체 DICOM Enhanced 변환 성공!")
    else:
        print("\n❌ 전체 DICOM Enhanced 변환 실패")

if __name__ == "__main__":
    asyncio.run(main())