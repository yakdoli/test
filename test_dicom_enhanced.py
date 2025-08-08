"""
DICOM 파일 테스트를 위한 간단한 스크립트
"""

import asyncio
import sys
from pathlib import Path
import config
from pdf_converter import PDFConverter
from unified_ollama_client import UnifiedVLClient

async def test_dicom():
    print("🧪 DICOM 파일 Enhanced 테스트 시작")
    print("=" * 50)
    
    # PDF 변환기 초기화
    pdf_converter = PDFConverter()
    
    # DICOM 이미지 확인
    dicom_staging = config.STAGING_DIR / "DICOM"
    if not dicom_staging.exists():
        print("❌ DICOM 스테이징 디렉토리가 없습니다.")
        print("   먼저 PDF를 이미지로 변환하세요: python pdf_converter.py")
        return False
    
    image_files = sorted(list(dicom_staging.glob("*.jpeg")))
    if not image_files:
        print("❌ DICOM 이미지 파일이 없습니다.")
        return False
    
    print(f"📋 DICOM 이미지 {len(image_files)}개 발견")
    
    # VL 클라이언트 초기화
    print("\n🧠 비전-언어 모델 초기화 중...")
    vl_client = UnifiedVLClient()
    
    if not await vl_client.initialize():
        print("❌ 모델 초기화 실패")
        return False
    
    if not await vl_client.check_availability():
        print("❌ 모델 사용 불가능")
        return False
    
    mode = "Direct Qwen2.5-VL" if config.USE_DIRECT_QWEN else "Xinference API"
    print(f"✅ 모델 준비 완료: {mode}")
    
    # 테스트용 소수 페이지만 처리 (첫 3페이지)
    test_images = image_files[:3]
    print(f"\n🔄 테스트 변환 시작 (첫 {len(test_images)}페이지)")
    
    try:
        # 비동기 병렬 변환
        markdown_content = await vl_client.convert_images_to_markdown_parallel(test_images)
        
        if not markdown_content.strip():
            print("❌ 변환 결과가 비어있습니다.")
            return False
        
        # 결과 저장
        output_file = config.OUTPUT_DIR / "DICOM_enhanced_test.md"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# DICOM Enhanced Test Results\n\n")
            f.write(f"테스트 페이지: {len(test_images)}개\n")
            f.write(f"처리 모드: {mode}\n\n")
            f.write("---\n\n")
            f.write(markdown_content)
        
        print(f"✅ 테스트 변환 완료!")
        print(f"📁 출력 파일: {output_file}")
        print(f"📊 변환된 문자 수: {len(markdown_content)}")
        
        # 성능 통계
        stats = vl_client.get_performance_stats()
        print(f"\n📈 성능 통계:")
        print(f"   모드: {stats.get('mode', 'unknown')}")
        if 'detailed_stats' in stats:
            detailed = stats['detailed_stats']
            print(f"   총 요청: {detailed.get('total_requests', 0)}")
            print(f"   성공: {detailed.get('successful_requests', 0)}")
            print(f"   실패: {detailed.get('failed_requests', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ 변환 중 오류: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # 리소스 정리
        if hasattr(vl_client, 'cleanup'):
            vl_client.cleanup()

async def main():
    success = await test_dicom()
    if success:
        print("\n🎉 DICOM Enhanced 테스트 성공!")
    else:
        print("\n❌ DICOM Enhanced 테스트 실패")

if __name__ == "__main__":
    asyncio.run(main())