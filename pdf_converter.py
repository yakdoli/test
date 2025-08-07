"""
PDF를 개별 페이지 이미지로 변환하는 모듈
"""
import os
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
import config

class PDFConverter:
    def __init__(self):
        self.pdf_dir = config.PDF_DIR
        self.staging_dir = config.STAGING_DIR
        
    def convert_pdf_to_images(self, pdf_path: Path) -> list:
        """
        PDF 파일을 개별 페이지 이미지로 변환
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            list: 생성된 이미지 파일 경로 리스트
        """
        print(f"PDF 변환 시작: {pdf_path.name}")
        
        # PDF 파일명에서 확장자 제거
        pdf_name = pdf_path.stem
        
        # 해당 PDF용 디렉토리 생성
        pdf_staging_dir = self.staging_dir / pdf_name
        pdf_staging_dir.mkdir(exist_ok=True)
        
        try:
            # PDF를 이미지로 변환
            pages = convert_from_path(
                pdf_path, 
                dpi=config.DPI,
                fmt=config.IMAGE_FORMAT.lower()
            )
            
            image_paths = []
            
            # 각 페이지를 개별 이미지 파일로 저장
            for i, page in enumerate(tqdm(pages, desc=f"{pdf_name} 페이지 변환")):
                image_filename = f"page_{i+1:03d}.{config.IMAGE_FORMAT.lower()}"
                image_path = pdf_staging_dir / image_filename
                
                # 이미지 저장
                page.save(image_path, config.IMAGE_FORMAT)
                image_paths.append(image_path)
                
            print(f"✅ {pdf_name}: {len(pages)}페이지 변환 완료")
            return image_paths
            
        except Exception as e:
            print(f"❌ {pdf_name} 변환 실패: {str(e)}")
            return []
    
    def convert_all_pdfs(self) -> dict:
        """
        pdfs 디렉토리의 모든 PDF 파일을 이미지로 변환
        
        Returns:
            dict: {pdf_name: [image_paths]} 형태의 딕셔너리
        """
        pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ PDF 파일을 찾을 수 없습니다.")
            return {}
        
        print(f"📁 {len(pdf_files)}개의 PDF 파일 발견")
        
        results = {}
        
        for pdf_file in pdf_files:
            image_paths = self.convert_pdf_to_images(pdf_file)
            if image_paths:
                results[pdf_file.stem] = image_paths
                
        return results

if __name__ == "__main__":
    converter = PDFConverter()
    results = converter.convert_all_pdfs()
    
    print(f"\n🎉 변환 완료: {len(results)}개 PDF 처리됨")