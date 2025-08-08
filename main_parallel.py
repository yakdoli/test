"""
비동기 병렬 처리가 적용된 PDF to Markdown 변환 메인 프로그램
"""
import sys
import time
import asyncio
from pathlib import Path
from tqdm import tqdm
import config
from pdf_converter import PDFConverter
from parallel_ollama_client import AsyncXinferenceClient

class AsyncParallelPDFToMarkdownConverter:
    def __init__(self):
        self.pdf_converter = PDFConverter()
        self.ollama_client = AsyncXinferenceClient()
        self.output_dir = config.OUTPUT_DIR
        
    async def check_prerequisites(self) -> bool:
        """실행 전 필수 조건 확인"""
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
        if not await self.ollama_client.check_xinference_connection():
            print("❌ Xinference 서버에 연결할 수 없습니다.")
            print("   다음 명령어로 Xinference를 시작하세요: ./start_xinference.sh")
            return False
        
        print("✅ Xinference 서버 연결 성공")
        
        # 모델 사용 가능 여부 확인
        if not await self.ollama_client.check_model_availability():
            print(f"❌ 모델 '{config.XINFERENCE_MODEL_NAME}'을 사용할 수 없습니다.")
            print("   Xinference에서 모델이 실행 중인지 확인하세요.")
            return False
        
        print(f"✅ 모델 '{config.XINFERENCE_MODEL_NAME}' 사용 가능 (UID: {self.ollama_client.model_uid})")
        print(f"🚀 비동기 처리 설정: 최대 {config.MAX_CONCURRENT_REQUESTS}개 동시 요청")
        
        return True
    
    async def convert_single_pdf_parallel(self, pdf_name: str, image_paths: list) -> bool:
        """
        단일 PDF의 이미지들을 비동기 병렬로 마크다운으로 변환
        
        Args:
            pdf_name: PDF 파일명 (확장자 제외)
            image_paths: 이미지 파일 경로 리스트
            
        Returns:
            bool: 변환 성공 여부
        """
        print(f"\n📄 '{pdf_name}' 비동기 병렬 마크다운 변환 시작...")
        print(f"🖼️ 총 {len(image_paths)}개 페이지")
        
        # 예상 시간 계산 (비동기 병렬 처리 고려)
        sequential_time = len(image_paths) * 8.24  # 순차 처리 시간
        async_efficiency = min(config.MAX_CONCURRENT_REQUESTS, len(image_paths)) / len(image_paths)
        estimated_time = sequential_time * (1 - async_efficiency * 0.8)  # 80% 효율성 가정 (비동기가 더 효율적)
        
        print(f"⏱️ 예상 시간: {estimated_time/60:.1f}분 (순차: {sequential_time/60:.1f}분)")
        
        try:
            start_time = time.time()
            
            # 비동기 병렬 이미지 변환
            markdown_content = await self.ollama_client.convert_images_to_markdown_parallel(image_paths)
            
            if not markdown_content.strip():
                print(f"❌ '{pdf_name}' 변환 실패: 빈 내용")
                return False
            
            # Syncfusion 특화 후처리
            if config.SYNCFUSION_MODE:
                markdown_content = self.ollama_client.post_process_syncfusion_content(markdown_content, pdf_name)
            
            # 마크다운 파일 저장
            output_file = self.output_dir / f"{pdf_name}_async_parallel.md"
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            
            end_time = time.time()
            total_time = end_time - start_time
            speedup = sequential_time / total_time if total_time > 0 else 1
            
            print(f"\n✅ '{pdf_name}' 비동기 병렬 변환 완료!")
            print(f"📁 출력 파일: {output_file}")
            print(f"⏱️ 실제 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
            print(f"🚀 속도 향상: {speedup:.1f}배")
            print(f"📊 처리량: {len(image_paths)/total_time:.2f} 페이지/초")
            
            return True
            
        except Exception as e:
            print(f"❌ '{pdf_name}' 변환 실패: {str(e)}")
            return False
    
    async def run(self, specific_pdf: str = None):
        """
        전체 변환 프로세스 실행 (비동기 병렬 처리)
        
        Args:
            specific_pdf: 특정 PDF만 변환할 경우 파일명 (확장자 제외)
        """
        print("🚀 PDF to Markdown 비동기 병렬 변환기 시작")
        print("=" * 60)
        
        # 실행 환경 확인
        if not await self.check_prerequisites():
            print("\n❌ 실행 환경이 준비되지 않았습니다.")
            sys.exit(1)
        
        print("\n" + "=" * 60)
        
        # 1단계: PDF를 이미지로 변환 (기존 이미지 사용)
        print("📸 1단계: 이미지 파일 확인")
        
        if specific_pdf:
            staging_dir = config.STAGING_DIR / specific_pdf
            if not staging_dir.exists():
                print(f"❌ {specific_pdf} 스테이징 디렉토리가 없습니다.")
                print("   먼저 PDF를 이미지로 변환하세요: python pdf_converter.py")
                return
            
            image_paths = sorted(list(staging_dir.glob("*.jpeg")))
            if not image_paths:
                print(f"❌ {specific_pdf} 이미지 파일이 없습니다.")
                return
            
            pdf_images = {specific_pdf: image_paths}
            print(f"🎯 특정 PDF 처리: {specific_pdf} ({len(image_paths)}개 이미지)")
        else:
            # 모든 PDF의 이미지 확인
            pdf_images = {}
            for pdf_dir in config.STAGING_DIR.iterdir():
                if pdf_dir.is_dir():
                    image_paths = sorted(list(pdf_dir.glob("*.jpeg")))
                    if image_paths:
                        pdf_images[pdf_dir.name] = image_paths
            
            if not pdf_images:
                print("❌ 변환할 이미지가 없습니다.")
                print("   먼저 PDF를 이미지로 변환하세요: python pdf_converter.py")
                return
        
        print(f"\n📝 2단계: 비동기 병렬 마크다운 변환 ({len(pdf_images)}개 PDF)")
        
        # 2단계: 이미지를 비동기 병렬로 마크다운으로 변환
        success_count = 0
        total_count = len(pdf_images)
        total_start_time = time.time()
        
        for pdf_name, image_paths in pdf_images.items():
            if await self.convert_single_pdf_parallel(pdf_name, image_paths):
                success_count += 1
        
        total_end_time = time.time()
        total_processing_time = total_end_time - total_start_time
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("🎉 비동기 병렬 변환 완료!")
        print(f"✅ 성공: {success_count}/{total_count}")
        print(f"⏱️ 전체 시간: {total_processing_time:.2f}초 ({total_processing_time/60:.1f}분)")
        
        if success_count < total_count:
            print(f"❌ 실패: {total_count - success_count}")
        
        print(f"📁 출력 디렉토리: {self.output_dir}")

async def main():
    """비동기 메인 함수"""
    converter = AsyncParallelPDFToMarkdownConverter()
    
    # 명령행 인수 처리
    if len(sys.argv) > 1:
        specific_pdf = sys.argv[1]
        await converter.run(specific_pdf)
    else:
        await converter.run()

if __name__ == "__main__":
    # 이벤트 루프 실행
    asyncio.run(main())