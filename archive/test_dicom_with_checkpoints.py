"""
체크포인트 시스템을 사용한 DICOM 변환 테스트
기존 안정된 Xinference 시스템에 Enhanced 체크포인트 적용
"""

import asyncio
import time
from pathlib import Path
import config
from pdf_converter import PDFConverter
from modules.core.checkpoint_manager import CheckpointManager
from modules.models.task_models import PDFTask, TaskStatus

# Xinference 대신 mock 변환 사용 (시연 목적)
class MockVLClient:
    """Mock VL Client for demonstration"""
    
    def __init__(self):
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'processing_time': 0
        }
    
    async def initialize(self):
        return True
    
    async def check_availability(self):
        return True
    
    async def convert_images_to_markdown_parallel(self, image_paths):
        """Mock 변환 - 실제로는 각 이미지를 분석"""
        print(f"🔄 Mock 변환 시작: {len(image_paths)}개 이미지")
        
        # 시뮬레이션된 처리 시간
        await asyncio.sleep(2)  # 실제로는 모델 처리 시간
        
        self.stats['total_requests'] += len(image_paths)
        self.stats['successful_requests'] += len(image_paths)
        
        # Mock 마크다운 콘텐츠 생성
        mock_content = []
        for i, image_path in enumerate(image_paths, 1):
            mock_content.append(f"""
## 페이지 {i}

**파일**: {image_path.name}

### 내용 요약
이것은 DICOM SDK 문서의 페이지 {i}입니다. 

- **주요 기능**: DICOM 이미지 처리
- **API 클래스**: DicomProcessor
- **예제 코드**: C# 및 VB.NET 샘플

```csharp
// 예제 DICOM 처리 코드
DicomProcessor processor = new DicomProcessor();
var result = processor.LoadImage("sample.dcm");
```

**참고사항**: 실제 변환에서는 Qwen2.5-VL 모델이 이미지의 실제 내용을 분석합니다.

---
""")
        
        return "\n".join(mock_content)
    
    def get_performance_stats(self):
        return {
            'mode': 'mock_mode',
            'detailed_stats': self.stats
        }

async def test_dicom_with_enhanced_checkpoints():
    print("🧪 Enhanced 체크포인트를 사용한 DICOM 변환 테스트")
    print("=" * 60)
    
    # 구성 요소 초기화
    pdf_converter = PDFConverter()
    checkpoint_manager = CheckpointManager()
    vl_client = MockVLClient()
    
    print("✅ 모든 구성 요소 초기화됨")
    
    # DICOM 이미지 확인
    dicom_staging = config.STAGING_DIR / "DICOM"
    if not dicom_staging.exists():
        print("❌ DICOM 스테이징 디렉토리가 없습니다.")
        return False
    
    image_files = sorted(list(dicom_staging.glob("*.jpeg")))
    if not image_files:
        print("❌ DICOM 이미지 파일이 없습니다.")
        return False
    
    print(f"📋 DICOM 이미지 {len(image_files)}개 발견")
    
    # VL 클라이언트 초기화
    await vl_client.initialize()
    await vl_client.check_availability()
    print("✅ Mock VL 클라이언트 준비 완료")
    
    # 세션 생성
    task = PDFTask(
        pdf_path=config.PDF_DIR / "DICOM.pdf",
        output_path=config.OUTPUT_DIR / "DICOM_enhanced_checkpoint.md"
    )
    
    session_id = checkpoint_manager.create_new_session([task])
    print(f"🆔 새 변환 세션: {session_id}")
    
    # 청크 상태 생성 (3페이지씩)
    chunk_size = 3
    chunk_ids = checkpoint_manager.create_chunk_states("DICOM", len(image_files), chunk_size)
    print(f"📦 {len(chunk_ids)}개 청크 생성 (각 {chunk_size}페이지)")
    
    # 청크별 처리 시뮬레이션
    total_content = []
    start_time = time.time()
    
    for chunk_idx, chunk_id in enumerate(chunk_ids):
        print(f"\n🔄 청크 {chunk_idx + 1}/{len(chunk_ids)} 처리 중...")
        
        # 청크 시작
        checkpoint_manager.update_chunk_status(chunk_id, TaskStatus.IN_PROGRESS)
        
        # 해당 청크의 이미지들 가져오기
        start_page = chunk_idx * chunk_size
        end_page = min(start_page + chunk_size, len(image_files))
        chunk_images = image_files[start_page:end_page]
        
        try:
            # 실제 변환 (Mock)
            chunk_content = await vl_client.convert_images_to_markdown_parallel(chunk_images)
            
            if chunk_content.strip():
                total_content.append(chunk_content)
                
                # 처리된 페이지 번호 계산
                processed_pages = list(range(start_page + 1, end_page + 1))
                
                # 청크 완료
                checkpoint_manager.update_chunk_status(
                    chunk_id, TaskStatus.COMPLETED, 
                    processed_pages=processed_pages
                )
                
                print(f"✅ 청크 {chunk_idx + 1} 완료 ({len(processed_pages)}페이지)")
            else:
                # 청크 실패
                checkpoint_manager.update_chunk_status(
                    chunk_id, TaskStatus.FAILED, 
                    error_message="Empty content"
                )
                print(f"❌ 청크 {chunk_idx + 1} 실패")
                
        except Exception as e:
            # 청크 실패
            checkpoint_manager.update_chunk_status(
                chunk_id, TaskStatus.FAILED, 
                error_message=str(e)
            )
            print(f"❌ 청크 {chunk_idx + 1} 오류: {e}")
        
        # 중간 진행 상황 출력
        chunk_progress = checkpoint_manager.get_chunk_progress_summary("DICOM")
        if chunk_progress:
            print(f"   📊 진행률: {chunk_progress['chunk_progress_percent']:.1f}% "
                  f"({chunk_progress['completed_chunks']}/{chunk_progress['total_chunks']} 청크)")
    
    # 전체 작업 완료
    checkpoint_manager.update_task_status("DICOM", TaskStatus.COMPLETED)
    
    # 최종 결과 저장
    if total_content:
        final_content = f"""# DICOM Enhanced Checkpoint Test Results

**테스트 정보:**
- 세션 ID: {session_id}
- 처리 모드: Mock with Enhanced Checkpoints
- 총 페이지: {len(image_files)}
- 청크 수: {len(chunk_ids)}
- 청크 크기: {chunk_size}페이지

**처리 시간:** {time.time() - start_time:.1f}초

---

{"".join(total_content)}

---

## 체크포인트 시스템 특징

1. **중단 복구**: 각 청크별로 상태 저장하여 정확한 지점에서 재시작 가능
2. **진행 추적**: 실시간 청크 및 페이지 단위 진행률 추적
3. **오류 격리**: 개별 청크 실패가 전체 작업에 영향을 주지 않음
4. **상세 로깅**: 각 청크의 상태와 처리 결과 기록

**Enhanced by Claude Code**
"""
        
        output_file = config.OUTPUT_DIR / "DICOM_enhanced_checkpoint.md"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(final_content)
        
        print(f"\n✅ 최종 결과 저장: {output_file}")
    
    # 최종 통계 출력
    processing_time = time.time() - start_time
    stats = vl_client.get_performance_stats()
    final_progress = checkpoint_manager.get_progress_summary()
    chunk_progress = checkpoint_manager.get_chunk_progress_summary("DICOM")
    
    print(f"\n{'='*60}")
    print("🎯 Enhanced 체크포인트 변환 완료!")
    print(f"{'='*60}")
    print(f"⏱️ 총 처리 시간: {processing_time:.1f}초")
    print(f"📊 전체 진행률: {final_progress.get('progress_percent', 0):.1f}%")
    print(f"📦 청크 진행률: {chunk_progress.get('chunk_progress_percent', 0):.1f}%")
    print(f"📄 페이지 진행률: {chunk_progress.get('page_progress_percent', 0):.1f}%")
    print(f"🔧 처리 모드: {stats.get('mode')}")
    
    if chunk_progress:
        print(f"\n📈 상세 통계:")
        print(f"   총 청크: {chunk_progress['total_chunks']}")
        print(f"   완료된 청크: {chunk_progress['completed_chunks']}")
        print(f"   실패한 청크: {chunk_progress['failed_chunks']}")
        print(f"   처리된 페이지: {chunk_progress['processed_pages']}/{chunk_progress['total_pages']}")
    
    return True

async def main():
    success = await test_dicom_with_enhanced_checkpoints()
    if success:
        print("\n🎉 DICOM Enhanced 체크포인트 테스트 성공!")
    else:
        print("\n❌ DICOM Enhanced 체크포인트 테스트 실패")

if __name__ == "__main__":
    asyncio.run(main())