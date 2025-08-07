#!/usr/bin/env python3
"""
common.pdf 성능 테스트 및 최적화 스크립트
"""
import time
import sys
from pathlib import Path
import config
from pdf_converter import PDFConverter
from ollama_client import OllamaClient
from main import PDFToMarkdownConverter
import psutil
import os

class PerformanceMonitor:
    def __init__(self):
        self.start_time = None
        self.start_memory = None
        
    def start(self):
        """성능 모니터링 시작"""
        self.start_time = time.time()
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024  # MB
        print(f"🚀 성능 모니터링 시작")
        print(f"   시작 메모리: {self.start_memory:.1f} MB")
        
    def checkpoint(self, description):
        """중간 체크포인트"""
        if self.start_time is None:
            return
            
        elapsed = time.time() - self.start_time
        current_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_diff = current_memory - self.start_memory
        
        print(f"⏱️ {description}")
        print(f"   경과 시간: {elapsed:.1f}초")
        print(f"   현재 메모리: {current_memory:.1f} MB (+{memory_diff:+.1f} MB)")
        
    def end(self):
        """성능 모니터링 종료"""
        if self.start_time is None:
            return
            
        total_time = time.time() - self.start_time
        final_memory = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        memory_diff = final_memory - self.start_memory
        
        print(f"🏁 성능 모니터링 완료")
        print(f"   총 소요 시간: {total_time:.1f}초")
        print(f"   최종 메모리: {final_memory:.1f} MB (+{memory_diff:+.1f} MB)")
        return total_time, memory_diff

def test_pdf_info():
    """PDF 기본 정보 확인"""
    print("📄 common.pdf 기본 정보 확인")
    print("-" * 50)
    
    pdf_path = Path("pdfs/common.pdf")
    if not pdf_path.exists():
        print("❌ common.pdf 파일이 존재하지 않습니다.")
        return False
    
    file_size = pdf_path.stat().st_size
    print(f"   파일 크기: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
    
    try:
        from pdf2image import convert_from_path
        # 첫 페이지만 변환하여 총 페이지 수 확인
        pages = convert_from_path(pdf_path, dpi=72, first_page=1, last_page=1)
        if pages:
            pages[0].close()
        
        # 전체 페이지 수 확인 (메모리 효율적 방법)
        import fitz  # PyMuPDF 사용 (더 빠름)
        doc = fitz.open(pdf_path)
        page_count = len(doc)
        doc.close()
        
        print(f"   총 페이지 수: {page_count} 페이지")
        print(f"   예상 처리 시간: {page_count * 10:.0f}초 (페이지당 10초 가정)")
        
        return page_count
        
    except ImportError:
        print("   ⚠️ PyMuPDF가 설치되지 않음, pdf2image로 대체")
        try:
            pages = convert_from_path(pdf_path, dpi=72)
            page_count = len(pages)
            for page in pages:
                page.close()
            print(f"   총 페이지 수: {page_count} 페이지")
            return page_count
        except Exception as e:
            print(f"   ❌ 페이지 수 확인 실패: {e}")
            return None
    except Exception as e:
        print(f"   ❌ PDF 정보 확인 실패: {e}")
        return None

def test_optimized_conversion():
    """최적화된 변환 테스트 (첫 3페이지만)"""
    print("\n🔧 최적화된 변환 테스트 (첫 3페이지)")
    print("-" * 50)
    
    monitor = PerformanceMonitor()
    monitor.start()
    
    try:
        # 임시로 DPI 낮추기
        original_dpi = config.DPI
        config.DPI = 150  # 성능 최적화
        
        # PDF 변환기 초기화
        pdf_converter = PDFConverter()
        pdf_path = Path("pdfs/common.pdf")
        
        monitor.checkpoint("PDF 변환기 초기화 완료")
        
        # 첫 3페이지만 변환
        print("   📸 첫 3페이지 이미지 변환 중...")
        from pdf2image import convert_from_path
        
        pages = convert_from_path(
            pdf_path, 
            dpi=config.DPI,
            first_page=1,
            last_page=3,
            fmt=config.IMAGE_FORMAT.lower()
        )
        
        monitor.checkpoint(f"첫 3페이지 이미지 변환 완료 ({len(pages)}페이지)")
        
        # 이미지 저장
        pdf_name = pdf_path.stem
        pdf_staging_dir = config.STAGING_DIR / f"{pdf_name}_test"
        pdf_staging_dir.mkdir(exist_ok=True)
        
        image_paths = []
        for i, page in enumerate(pages):
            image_filename = f"page_{i+1:03d}.{config.IMAGE_FORMAT.lower()}"
            image_path = pdf_staging_dir / image_filename
            page.save(image_path, config.IMAGE_FORMAT)
            image_paths.append(image_path)
            page.close()
        
        monitor.checkpoint("이미지 파일 저장 완료")
        
        # Ollama 연결 확인
        ollama_client = OllamaClient()
        if not ollama_client.check_ollama_connection():
            print("   ⚠️ Ollama 서버 연결 실패 - 이미지 변환만 테스트")
            config.DPI = original_dpi
            return True
        
        monitor.checkpoint("Ollama 연결 확인 완료")
        
        # 첫 번째 페이지만 마크다운 변환 테스트
        print("   🔄 첫 번째 페이지 마크다운 변환 테스트...")
        markdown_result = ollama_client.convert_image_to_markdown(image_paths[0])
        
        if markdown_result:
            print(f"   ✅ 마크다운 변환 성공 ({len(markdown_result)} 문자)")
            
            # 결과 미리보기
            preview = markdown_result[:200].replace('\n', ' ') + "..." if len(markdown_result) > 200 else markdown_result
            print(f"   📝 변환 결과 미리보기: {preview}")
            
            # Syncfusion 후처리 테스트
            if config.SYNCFUSION_MODE:
                processed = ollama_client.post_process_syncfusion_content(markdown_result, "common_test")
                code_snippets = ollama_client.extract_code_snippets(markdown_result, "common_test")
                
                total_snippets = sum(len(snippets) for snippets in code_snippets.values())
                print(f"   📝 추출된 코드 스니펫: {total_snippets}개")
                
                monitor.checkpoint("Syncfusion 후처리 완료")
        else:
            print("   ❌ 마크다운 변환 실패")
        
        # DPI 원복
        config.DPI = original_dpi
        
        total_time, memory_used = monitor.end()
        
        # 성능 예측
        if total_time > 0:
            page_count = test_pdf_info() or 50  # 기본값 50페이지
            estimated_total_time = (total_time / 3) * page_count
            estimated_memory = memory_used * (page_count / 3)
            
            print(f"\n📊 성능 예측 (전체 {page_count}페이지)")
            print(f"   예상 총 소요 시간: {estimated_total_time/60:.1f}분")
            print(f"   예상 메모리 사용량: {estimated_memory:.1f} MB")
            
            if estimated_total_time > 1800:  # 30분 초과
                print("   ⚠️ 처리 시간이 길 것으로 예상됩니다. DPI 조정을 권장합니다.")
            
            if estimated_memory > 2048:  # 2GB 초과
                print("   ⚠️ 메모리 사용량이 클 것으로 예상됩니다. 배치 처리를 권장합니다.")
        
        return True
        
    except Exception as e:
        print(f"   ❌ 최적화된 변환 테스트 실패: {e}")
        config.DPI = original_dpi
        return False

def suggest_optimizations():
    """성능 최적화 제안"""
    print("\n💡 성능 최적화 제안")
    print("-" * 50)
    
    current_dpi = config.DPI
    file_size = Path("pdfs/common.pdf").stat().st_size / 1024 / 1024  # MB
    
    print(f"   현재 DPI: {current_dpi}")
    print(f"   파일 크기: {file_size:.1f} MB")
    
    # DPI 최적화 제안
    if file_size > 10:  # 10MB 이상
        recommended_dpi = 120
        print(f"   🔧 권장 DPI: {recommended_dpi} (대용량 파일)")
    elif file_size > 5:  # 5MB 이상
        recommended_dpi = 150
        print(f"   🔧 권장 DPI: {recommended_dpi} (중간 크기 파일)")
    else:
        recommended_dpi = 200
        print(f"   🔧 권장 DPI: {recommended_dpi} (일반 크기 파일)")
    
    # 설정 최적화 제안
    optimizations = [
        ("IMAGE_FORMAT", "JPEG", "PNG보다 빠른 처리"),
        ("SEMANTIC_CHUNKING", "False", "후처리 시간 단축"),
        ("EXTRACT_CODE_SNIPPETS", "False", "코드 추출 생략으로 속도 향상")
    ]
    
    print("\n   📋 추가 최적화 옵션:")
    for setting, value, description in optimizations:
        current_value = getattr(config, setting, "N/A")
        print(f"      {setting}: {current_value} → {value} ({description})")
    
    # 배치 처리 제안
    print(f"\n   🔄 배치 처리 옵션:")
    print(f"      python main.py common  # 단일 파일 처리")
    print(f"      # config.py에서 DPI={recommended_dpi} 설정 후 실행")

def main():
    """메인 테스트 함수"""
    print("🚀 common.pdf 성능 테스트 및 최적화")
    print("=" * 60)
    
    # 1. PDF 기본 정보 확인
    page_count = test_pdf_info()
    
    if page_count is None:
        print("❌ PDF 정보를 확인할 수 없어 테스트를 중단합니다.")
        return 1
    
    # 2. 최적화된 변환 테스트
    if test_optimized_conversion():
        print("✅ 최적화된 변환 테스트 성공")
    else:
        print("❌ 최적화된 변환 테스트 실패")
    
    # 3. 최적화 제안
    suggest_optimizations()
    
    print("\n" + "=" * 60)
    print("🎉 성능 테스트 완료!")
    print("\n💡 다음 단계:")
    print("   1. config.py에서 권장 DPI로 설정 변경")
    print("   2. python main.py common 명령으로 전체 변환 실행")
    print("   3. 결과 품질 확인 후 필요시 DPI 재조정")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())