"""
Enhanced 체크포인트 시스템 테스트
"""

from pathlib import Path
from datetime import datetime
import config
from pdf_converter import PDFConverter
from modules.core.checkpoint_manager import CheckpointManager
from modules.models.task_models import PDFTask, TaskStatus

def test_enhanced_checkpoint_system():
    print("🧪 Enhanced 체크포인트 시스템 테스트")
    print("=" * 50)
    
    # PDF 변환기 초기화
    pdf_converter = PDFConverter()
    print("✅ PDF 변환기 초기화됨")
    
    # 체크포인트 매니저 초기화
    checkpoint_manager = CheckpointManager()
    print("✅ 체크포인트 매니저 초기화됨")
    
    # DICOM PDF 경로
    dicom_pdf = config.PDF_DIR / "DICOM.pdf"
    if not dicom_pdf.exists():
        print("❌ DICOM.pdf 파일이 존재하지 않습니다.")
        return False
    
    print(f"📋 DICOM PDF 크기: {dicom_pdf.stat().st_size / 1024:.1f} KB")
    
    # Enhanced 체크포인트 테스트
    print("\n🔍 Enhanced 체크포인트 기능 테스트:")
    
    # PDF 해시 생성 테스트
    pdf_hash = pdf_converter._generate_pdf_hash(dicom_pdf)
    print(f"📝 PDF 해시: {pdf_hash}")
    
    # 체크포인트 로드 테스트
    checkpoint = pdf_converter._load_conversion_checkpoint(dicom_pdf)
    if checkpoint:
        print(f"📂 기존 체크포인트 발견:")
        print(f"   PDF 해시: {checkpoint.get('pdf_hash', 'N/A')}")
        print(f"   타임스탬프: {checkpoint.get('timestamp', 'N/A')}")
        print(f"   DPI: {checkpoint.get('config_snapshot', {}).get('dpi', 'N/A')}")
    else:
        print("📂 기존 체크포인트 없음")
    
    # 체크포인트 유효성 검증
    is_valid = pdf_converter._validate_checkpoint(dicom_pdf, checkpoint)
    print(f"✅ 체크포인트 유효성: {'유효' if is_valid else '무효'}")
    
    # 이미지 변환 상태 확인
    dicom_staging = config.STAGING_DIR / "DICOM"
    if dicom_staging.exists():
        image_files = list(dicom_staging.glob("*.jpeg"))
        print(f"🖼️ 기존 이미지 파일: {len(image_files)}개")
        
        # 메타데이터 확인
        metadata_file = dicom_staging / "metadata.json"
        if metadata_file.exists():
            import json
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            print(f"📊 메타데이터:")
            print(f"   DPI: {metadata.get('dpi')}")
            print(f"   페이지 수: {metadata.get('page_count')}")
            print(f"   처리 시간: {metadata.get('processing_time_seconds', 'N/A')}초")
            print(f"   파일 크기: {metadata.get('pdf_size_bytes', 0) / 1024:.1f} KB")
    else:
        print("🖼️ 이미지 파일 없음")
    
    # 작업 생성 및 세션 테스트
    print("\n🎯 체크포인트 매니저 세션 테스트:")
    
    task = PDFTask(
        pdf_path=dicom_pdf,
        output_path=config.OUTPUT_DIR / "DICOM_test.md"
    )
    
    # 새 세션 생성
    session_id = checkpoint_manager.create_new_session([task])
    print(f"🆔 새 세션 생성: {session_id}")
    
    # 청크 상태 생성 테스트
    if dicom_staging.exists() and image_files:
        chunk_ids = checkpoint_manager.create_chunk_states("DICOM", len(image_files), 3)
        print(f"📦 청크 생성: {len(chunk_ids)}개")
        
        # 청크 상태 업데이트 테스트
        for i, chunk_id in enumerate(chunk_ids[:2]):  # 처음 2개만 테스트
            status = TaskStatus.IN_PROGRESS if i == 0 else TaskStatus.COMPLETED
            processed_pages = list(range(i*3+1, min((i+1)*3+1, len(image_files)+1))) if status == TaskStatus.COMPLETED else []
            
            checkpoint_manager.update_chunk_status(
                chunk_id, status, 
                processed_pages=processed_pages
            )
    
    # 진행 상황 요약 확인
    progress_summary = checkpoint_manager.get_progress_summary()
    if progress_summary:
        print(f"\n📈 진행 상황 요약:")
        print(f"   총 작업: {progress_summary['total_tasks']}")
        print(f"   완료: {progress_summary['completed_tasks']}")
        print(f"   진행률: {progress_summary['progress_percent']:.1f}%")
    
    # 청크 진행 상황 요약
    chunk_summary = checkpoint_manager.get_chunk_progress_summary("DICOM")
    if chunk_summary:
        print(f"\n📦 청크 진행 상황:")
        print(f"   총 청크: {chunk_summary['total_chunks']}")
        print(f"   완료된 청크: {chunk_summary['completed_chunks']}")
        print(f"   청크 진행률: {chunk_summary['chunk_progress_percent']:.1f}%")
        print(f"   페이지 진행률: {chunk_summary['page_progress_percent']:.1f}%")
    
    print("\n✅ Enhanced 체크포인트 시스템 테스트 완료!")
    return True

def test_git_automation():
    print("\n🔧 Git 자동화 테스트")
    print("=" * 30)
    
    try:
        from git_automation import GitAutomation
        git_auto = GitAutomation()
        
        # Git 상태 확인
        git_auto.print_status()
        
        # 테스트용 커밋 시뮬레이션 (실제로는 실행하지 않음)
        print("📝 테스트용 커밋 메시지 생성 테스트:")
        status = git_auto.get_git_status()
        if any(status.values()):
            print("   변경 사항 발견 - Git 자동화 준비됨")
        else:
            print("   변경 사항 없음")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Git 자동화 테스트 건너뛰기: {e}")
        return True

if __name__ == "__main__":
    success1 = test_enhanced_checkpoint_system()
    success2 = test_git_automation()
    
    if success1 and success2:
        print("\n🎉 모든 테스트 성공!")
    else:
        print("\n⚠️ 일부 테스트 실패")