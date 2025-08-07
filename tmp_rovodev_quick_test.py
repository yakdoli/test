#!/usr/bin/env python3
"""
common.pdf 빠른 테스트 (첫 2페이지만)
"""
import time
import sys
from pathlib import Path
import config

def quick_test():
    """빠른 테스트 실행"""
    print("🚀 common.pdf 빠른 테스트 (첫 2페이지)")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # 1. 환경 확인
        print("1️⃣ 환경 확인...")
        from main import PDFToMarkdownConverter
        converter = PDFToMarkdownConverter()
        
        if not converter.check_prerequisites():
            print("❌ 환경 확인 실패")
            return False
        
        print("✅ 환경 확인 완료")
        
        # 2. 첫 2페이지만 변환
        print("\n2️⃣ 첫 2페이지 변환...")
        
        # PDF 경로 확인
        pdf_path = Path("pdfs/common.pdf")
        if not pdf_path.exists():
            print("❌ common.pdf 파일이 없습니다.")
            return False
        
        # 임시 2페이지 PDF 생성
        import fitz
        doc = fitz.open(pdf_path)
        temp_pdf_path = Path("pdfs/common_quick_test.pdf")
        
        new_doc = fitz.open()
        for page_num in range(min(2, len(doc))):
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        new_doc.save(temp_pdf_path)
        new_doc.close()
        doc.close()
        
        print(f"✅ 2페이지 테스트 PDF 생성: {temp_pdf_path}")
        
        # 3. 변환 실행
        print("\n3️⃣ 변환 실행...")
        converter.run("common_quick_test")
        
        # 4. 결과 확인
        print("\n4️⃣ 결과 확인...")
        output_file = config.OUTPUT_DIR / "common_quick_test.md"
        
        if output_file.exists():
            file_size = output_file.stat().st_size
            elapsed = time.time() - start_time
            
            print(f"✅ 변환 성공!")
            print(f"   출력 파일: {output_file}")
            print(f"   파일 크기: {file_size:,} bytes")
            print(f"   소요 시간: {elapsed:.1f}초")
            print(f"   페이지당 시간: {elapsed/2:.1f}초")
            
            # 전체 145페이지 예상 시간
            estimated_total = (elapsed / 2) * 145
            print(f"   전체 예상 시간: {estimated_total/60:.1f}분")
            
            # 내용 미리보기
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')[:10]  # 첫 10줄만
                print(f"\n📄 내용 미리보기:")
                for line in lines:
                    if line.strip():
                        print(f"   {line}")
            
            return True
        else:
            print("❌ 출력 파일이 생성되지 않았습니다.")
            return False
            
    except Exception as e:
        print(f"❌ 테스트 실패: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 임시 파일 정리
        temp_files = [
            Path("pdfs/common_quick_test.pdf"),
            config.STAGING_DIR / "common_quick_test"
        ]
        
        for temp_file in temp_files:
            if temp_file.exists():
                if temp_file.is_dir():
                    import shutil
                    shutil.rmtree(temp_file)
                else:
                    temp_file.unlink()
                print(f"🗑️ 임시 파일 정리: {temp_file}")

def main():
    """메인 함수"""
    print(f"📋 현재 최적화 설정:")
    print(f"   DPI: {config.DPI}")
    print(f"   IMAGE_FORMAT: {config.IMAGE_FORMAT}")
    print(f"   SEMANTIC_CHUNKING: {config.SEMANTIC_CHUNKING}")
    print(f"   EXTRACT_CODE_SNIPPETS: {config.EXTRACT_CODE_SNIPPETS}")
    print()
    
    if quick_test():
        print("\n🎉 빠른 테스트 성공!")
        print("\n💡 전체 변환 실행:")
        print("   python main.py common")
        return 0
    else:
        print("\n❌ 빠른 테스트 실패")
        return 1

if __name__ == "__main__":
    sys.exit(main())