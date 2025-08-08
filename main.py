"""
PDF to Markdown 변환 메인 프로그램
"""
import sys
from pathlib import Path
from tqdm import tqdm
import config
from pdf_converter import PDFConverter
from ollama_client import OllamaClient

class PDFToMarkdownConverter:
    def __init__(self):
        self.pdf_converter = PDFConverter()
        self.ollama_client = OllamaClient()
        self.output_dir = config.OUTPUT_DIR
        
    def check_prerequisites(self) -> bool:
        """
        실행 전 필수 조건 확인
        
        Returns:
            bool: 모든 조건이 충족되었는지 여부
        """
        print("🔍 실행 환경 확인 중...")
        
        # PDF 디렉토리 존재 확인
        if not config.PDF_DIR.exists():
            print(f"❌ PDF 디렉토리가 존재하지 않습니다: {config.PDF_DIR}")
            return False
        
        # PDF 파일 존재 확인
        pdf_files = list(config.PDF_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"❌ PDF 파일을 찾을 수 없습니다: {config.PDF_DIR}")
            return False
        
        print(f"✅ {len(pdf_files)}개의 PDF 파일 발견")
        
        # Xinference 서버 연결 확인
        if not self.ollama_client.check_xinference_connection():
            print("❌ Xinference 서버에 연결할 수 없습니다.")
            print("   다음 명령어로 Xinference를 시작하세요: xinference launch --model-engine vLLM --model-name qwen2-vl-instruct --size-in-billions 7 --model-format gptq --quantization Int8")
            return False
        
        print("✅ Xinference 서버 연결 성공")
        
        # 모델 사용 가능 여부 확인
        if not self.ollama_client.check_model_availability():
            print(f"❌ 모델 '{config.XINFERENCE_MODEL_NAME}'을 사용할 수 없습니다.")
            print("   Xinference에서 모델이 실행 중인지 확인하세요.")
            return False
        
        print(f"✅ 모델 '{config.XINFERENCE_MODEL_NAME}' 사용 가능")
        
        return True
    
    def convert_single_pdf(self, pdf_name: str, image_paths: list) -> bool:
        """
        단일 PDF의 이미지들을 마크다운으로 변환
        
        Args:
            pdf_name: PDF 파일명 (확장자 제외)
            image_paths: 이미지 파일 경로 리스트
            
        Returns:
            bool: 변환 성공 여부
        """
        print(f"\n📄 '{pdf_name}' 마크다운 변환 시작...")
        
        try:
            # 이미지들을 마크다운으로 변환
            markdown_content = self.ollama_client.convert_images_to_markdown(image_paths)
            
            if not markdown_content.strip():
                print(f"❌ '{pdf_name}' 변환 실패: 빈 내용")
                return False
            
            # Syncfusion 특화 후처리
            if config.SYNCFUSION_MODE:
                markdown_content = self.ollama_client.post_process_syncfusion_content(markdown_content, pdf_name)
                
                # 코드 스니펫 추출 및 저장
                if config.EXTRACT_CODE_SNIPPETS:
                    code_snippets = self.ollama_client.extract_code_snippets(markdown_content, pdf_name)
                    self.save_code_snippets(pdf_name, code_snippets)
            
            # 마크다운 파일 저장
            output_file = self.output_dir / f"{pdf_name}.md"
            
            # 파일 헤더 추가 (Syncfusion 모드가 아닌 경우에만)
            if not config.SYNCFUSION_MODE:
                header = f"""# {pdf_name}

> 원본 파일: {pdf_name}.pdf  
> 변환 일시: {Path(__file__).stat().st_mtime}  
> 총 페이지: {len(image_paths)}

---

"""
                full_content = header + markdown_content
            else:
                full_content = markdown_content
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(full_content)
            
            print(f"✅ '{pdf_name}' 변환 완료: {output_file}")
            return True
            
        except Exception as e:
            print(f"❌ '{pdf_name}' 변환 실패: {str(e)}")
            return False
    
    def save_code_snippets(self, pdf_name: str, code_snippets: dict):
        """
        추출된 코드 스니펫을 별도 파일로 저장
        
        Args:
            pdf_name: PDF 파일명
            code_snippets: 언어별 코드 스니펫 딕셔너리
        """
        code_dir = self.output_dir / "code_snippets" / pdf_name
        code_dir.mkdir(parents=True, exist_ok=True)
        
        for language, snippets in code_snippets.items():
            if snippets:
                file_extension = {
                    'csharp': 'cs',
                    'vb': 'vb',
                    'xml': 'xml',
                    'javascript': 'js',
                    'css': 'css',
                    'other': 'txt'
                }.get(language, 'txt')
                
                snippet_file = code_dir / f"{language}_snippets.{file_extension}"
                
                with open(snippet_file, 'w', encoding='utf-8') as f:
                    f.write(f"// {pdf_name} - {language.upper()} Code Snippets\n")
                    f.write(f"// Extracted on: {Path(__file__).stat().st_mtime}\n\n")
                    
                    for i, snippet in enumerate(snippets, 1):
                        f.write(f"// Snippet {i}\n")
                        f.write(f"{snippet}\n\n")
                        f.write("// " + "="*50 + "\n\n")
                
                print(f"📝 {language} 코드 스니펫 저장: {snippet_file} ({len(snippets)}개)")
    
    def run(self, specific_pdf: str = None):
        """
        전체 변환 프로세스 실행
        
        Args:
            specific_pdf: 특정 PDF만 변환할 경우 파일명 (확장자 제외)
        """
        print("🚀 PDF to Markdown 변환기 시작")
        print("=" * 50)
        
        if not self.check_prerequisites():
            print("\n❌ 실행 환경이 준비되지 않았습니다.")
            sys.exit(1)
        
        print("\n" + "=" * 50)
        
        # 1단계: PDF를 이미지로 변환
        print("📸 1단계: PDF를 이미지로 변환")
        pdf_images = self.pdf_converter.convert_pdfs(specific_pdf)
        
        if not pdf_images:
            print("❌ 처리할 PDF가 없습니다.")
            return
        
        print(f"\n📝 2단계: 마크다운으로 변환 ({len(pdf_images)}개 PDF)")
        
        # 2단계: 이미지를 마크다운으로 변환
        success_count = 0
        total_count = len(pdf_images)
        
        for pdf_name, image_paths in pdf_images.items():
            if self.convert_single_pdf(pdf_name, image_paths):
                success_count += 1
        
        # 결과 요약
        print("\n" + "=" * 50)
        print("🎉 변환 완료!")
        print(f"✅ 성공: {success_count}/{total_count}")
        
        if success_count < total_count:
            print(f"❌ 실패: {total_count - success_count}")
        
        print(f"📁 출력 디렉토리: {self.output_dir}")

def main():
    """메인 함수"""
    converter = PDFToMarkdownConverter()
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        specific_pdf = sys.argv[1]
        converter.run(specific_pdf)
    else:
        converter.run()

if __name__ == "__main__":
    main()