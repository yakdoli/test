#!/usr/bin/env python3
"""
모니터링이 포함된 common.pdf 변환 스크립트
"""
import time
import sys
from pathlib import Path
import config
from main import PDFToMarkdownConverter
import psutil
import os

class ConversionMonitor:
    def __init__(self):
        self.start_time = time.time()
        self.last_checkpoint = self.start_time
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
    def log_progress(self, message, page_num=None, total_pages=None):
        """진행 상황 로깅"""
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_step = current_time - self.last_checkpoint
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        
        if page_num and total_pages:
            progress = (page_num / total_pages) * 100
            remaining_pages = total_pages - page_num
            avg_time_per_page = elapsed_total / page_num if page_num > 0 else 0
            eta = remaining_pages * avg_time_per_page
            
            print(f"📊 {message}")
            print(f"   진행률: {progress:.1f}% ({page_num}/{total_pages})")
            print(f"   경과 시간: {elapsed_total/60:.1f}분")
            print(f"   예상 남은 시간: {eta/60:.1f}분")
            print(f"   메모리 사용량: {current_memory:.1f} MB")
            print(f"   페이지당 평균 시간: {avg_time_per_page:.1f}초")
        else:
            print(f"⏱️ {message}")
            print(f"   경과 시간: {elapsed_total:.1f}초 (단계: {elapsed_step:.1f}초)")
            print(f"   메모리 사용량: {current_memory:.1f} MB")
        
        self.last_checkpoint = current_time

def test_partial_conversion():
    """부분 변환 테스트 (첫 5페이지)"""
    print("🧪 부분 변환 테스트 (첫 5페이지)")
    print("-" * 50)
    
    monitor = ConversionMonitor()
    
    try:
        # PDF 변환기 초기화
        converter = PDFToMarkdownConverter()
        
        # PDF 파일 확인
        pdf_path = Path("pdfs/common.pdf")
        if not pdf_path.exists():
            print("❌ common.pdf 파일이 존재하지 않습니다.")
            return False
        
        monitor.log_progress("PDF 변환기 초기화 완료")
        
        # 환경 확인
        if not converter.check_prerequisites():
            print("❌ 실행 환경이 준비되지 않았습니다.")
            return False
        
        monitor.log_progress("실행 환경 확인 완료")
        
        # 첫 5페이지만 변환하기 위해 임시로 PDF 수정
        from pdf2image import convert_from_path
        import fitz
        
        # 원본 PDF에서 첫 5페이지만 추출
        doc = fitz.open(pdf_path)
        temp_pdf_path = Path("pdfs/common_test_5pages.pdf")
        
        # 새 PDF 생성 (첫 5페이지만)
        new_doc = fitz.open()
        for page_num in range(min(5, len(doc))):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        new_doc.save(temp_pdf_path)
        new_doc.close()
        doc.close()
        
        monitor.log_progress("테스트용 5페이지 PDF 생성 완료")
        
        # 5페이지 변환 실행
        print("\n🔄 5페이지 변환 시작...")
        
        # PDF 변환
        pdf_images = converter.pdf_converter.convert_all_pdfs()
        
        # common_test_5pages만 처리
        test_pdf_name = "common_test_5pages"
        if test_pdf_name in pdf_images:
            image_paths = pdf_images[test_pdf_name]
            monitor.log_progress(f"이미지 변환 완료", len(image_paths), len(image_paths))
            
            # 마크다운 변환
            success = converter.convert_single_pdf(test_pdf_name, image_paths)
            
            if success:
                monitor.log_progress("마크다운 변환 완료")
                
                # 결과 파일 확인
                output_file = config.OUTPUT_DIR / f"{test_pdf_name}.md"
                if output_file.exists():
                    file_size = output_file.stat().st_size
                    print(f"\n✅ 변환 성공!")
                    print(f"   출력 파일: {output_file}")
                    print(f"   파일 크기: {file_size:,} bytes")
                    
                    # 내용 미리보기
                    with open(output_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        preview = content[:500] + "..." if len(content) > 500 else content
                        print(f"   내용 미리보기:\n{preview}")
                    
                    return True
            else:
                print("❌ 마크다운 변환 실패")
                return False
        else:
            print("❌ 테스트 PDF 이미지 변환 실패")
            return False
        
    except Exception as e:
        print(f"❌ 부분 변환 테스트 실패: {e}")
        return False
    finally:
        # 임시 파일 정리
        temp_files = [
            Path("pdfs/common_test_5pages.pdf"),
            config.STAGING_DIR / "common_test_5pages"
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                if temp_file.is_dir():
                    import shutil
                    shutil.rmtree(temp_file)
                else:
                    temp_file.unlink()

def estimate_full_conversion():
    """전체 변환 시간 예측"""
    print("\n📊 전체 변환 시간 예측")
    print("-" * 50)
    
    # 5페이지 변환 결과 파일 확인
    test_output = config.OUTPUT_DIR / "common_test_5pages.md"
    
    if not test_output.exists():
        print("❌ 테스트 결과 파일이 없어 예측할 수 없습니다.")
        return
    
    # 전체 페이지 수 확인
    import fitz
    doc = fitz.open("pdfs/common.pdf")
    total_pages = len(doc)
    doc.close()
    
    # 5페이지 처리 시간 기반 예측
    test_file_size = test_output.stat().st_size
    estimated_full_size = (test_file_size / 5) * total_pages
    
    # 시간 예측 (5페이지 기준)
    pages_per_minute = 5 / 1  # 임시값, 실제 측정 필요
    estimated_time_minutes = total_pages / pages_per_minute
    
    print(f"   총 페이지 수: {total_pages}")
    print(f"   테스트 결과 크기: {test_file_size:,} bytes (5페이지)")
    print(f"   예상 전체 파일 크기: {estimated_full_size/1024/1024:.1f} MB")
    print(f"   예상 처리 시간: {estimated_time_minutes:.1f}분")
    
    if estimated_time_minutes > 30:
        print("   ⚠️ 처리 시간이 30분을 초과할 것으로 예상됩니다.")
        print("   💡 백그라운드 실행을 권장합니다: nohup python main.py common &")

def main():
    """메인 함수"""
    print("🚀 common.pdf 모니터링 변환 테스트")
    print("=" * 60)
    
    print(f"📋 현재 최적화 설정:")
    print(f"   DPI: {config.DPI}")
    print(f"   IMAGE_FORMAT: {config.IMAGE_FORMAT}")
    print(f"   SYNCFUSION_MODE: {config.SYNCFUSION_MODE}")
    print(f"   SEMANTIC_CHUNKING: {config.SEMANTIC_CHUNKING}")
    print(f"   EXTRACT_CODE_SNIPPETS: {config.EXTRACT_CODE_SNIPPETS}")
    
    # 1. 부분 변환 테스트
    if test_partial_conversion():
        print("\n✅ 부분 변환 테스트 성공")
        
        # 2. 전체 변환 예측
        estimate_full_conversion()
        
        # 3. 전체 변환 실행 여부 확인
        print("\n" + "=" * 60)
        print("🎯 전체 변환 실행 옵션:")
        print("   1. python main.py common")
        print("   2. nohup python main.py common > conversion.log 2>&1 &  # 백그라운드")
        print("   3. screen -S conversion python main.py common  # screen 세션")
        
    else:
        print("\n❌ 부분 변환 테스트 실패")
        print("   설정을 확인하고 다시 시도하세요.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())