"""
PDF를 개별 페이지 이미지로 변환하는 모듈
(이미 변환된 파일 건너뛰기 기능 포함)
"""
import os
import json
import hashlib
from pathlib import Path
from pdf2image import convert_from_path
from PIL import Image
from tqdm import tqdm
from datetime import datetime, timedelta
import config

class PDFConverter:
    def __init__(self):
        self.pdf_dir = config.PDF_DIR
        self.staging_dir = config.STAGING_DIR
        self.checkpoint_dir = config.BASE_DIR / ".pdf_checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
    
    def _generate_pdf_hash(self, pdf_path: Path) -> str:
        """PDF 파일의 해시값 생성 (내용 변경 감지용)"""
        hash_md5 = hashlib.md5()
        with open(pdf_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()[:12]
    
    def _save_conversion_checkpoint(self, pdf_path: Path, metadata: dict):
        """변환 체크포인트 저장"""
        checkpoint_file = self.checkpoint_dir / f"{pdf_path.stem}_checkpoint.json"
        checkpoint_data = {
            'pdf_path': str(pdf_path),
            'pdf_hash': self._generate_pdf_hash(pdf_path),
            'metadata': metadata,
            'timestamp': datetime.now().isoformat(),
            'config_snapshot': {
                'dpi': config.DPI,
                'image_format': config.IMAGE_FORMAT
            }
        }
        
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)
    
    def _load_conversion_checkpoint(self, pdf_path: Path) -> dict:
        """변환 체크포인트 로드"""
        checkpoint_file = self.checkpoint_dir / f"{pdf_path.stem}_checkpoint.json"
        if not checkpoint_file.exists():
            return {}
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 체크포인트 로드 실패: {e}")
            return {}
    
    def _validate_checkpoint(self, pdf_path: Path, checkpoint: dict) -> bool:
        """체크포인트 유효성 검증"""
        if not checkpoint:
            return False
        
        # 파일 해시 확인 (PDF 내용 변경 감지)
        current_hash = self._generate_pdf_hash(pdf_path)
        if checkpoint.get('pdf_hash') != current_hash:
            print(f"⚠️ PDF 파일이 변경됨: {pdf_path.name}")
            return False
        
        # DPI 설정 확인
        config_snapshot = checkpoint.get('config_snapshot', {})
        if (config_snapshot.get('dpi') != config.DPI or 
            config_snapshot.get('image_format') != config.IMAGE_FORMAT):
            print(f"⚠️ 변환 설정 변경됨: {pdf_path.name}")
            return False
        
        return True
        
    def convert_pdf_to_images(self, pdf_path: Path) -> list:
        """
        PDF 파일을 개별 페이지 이미지로 변환 (건너뛰기 기능 포함)
        
        Args:
            pdf_path: PDF 파일 경로
            
        Returns:
            list: 생성된 이미지 파일 경로 리스트
        """
        pdf_name = pdf_path.stem
        pdf_staging_dir = self.staging_dir / pdf_name
        metadata_path = pdf_staging_dir / "metadata.json"

        # 향상된 체크포인트 확인
        checkpoint = self._load_conversion_checkpoint(pdf_path)
        
        # 기존 변환 결과 확인
        if (pdf_staging_dir.exists() and metadata_path.exists() and 
            self._validate_checkpoint(pdf_path, checkpoint)):
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                
            if metadata.get('dpi') == config.DPI:
                image_files = sorted(list(pdf_staging_dir.glob(f"*.{config.IMAGE_FORMAT.lower()}")))
                if image_files and len(image_files) == metadata.get('page_count', 0):
                    print(f"⏭️ '{pdf_name}' 변환 건너뛰기 (DPI: {config.DPI}, 페이지: {len(image_files)}, 체크포인트 유효)")
                    return image_files
                else:
                    print(f"⚠️ '{pdf_name}' 이미지 파일 불완전 - 재변환 필요")

        print(f"⏳ '{pdf_name}' 이미지 변환 시작 (DPI: {config.DPI})...")
        pdf_staging_dir.mkdir(exist_ok=True)
        start_time = datetime.now()
        
        try:
            pages = convert_from_path(
                pdf_path, 
                dpi=config.DPI,
                fmt=config.IMAGE_FORMAT.lower(),
                poppler_path=r"C:\Program Files\poppler-24.08.0\Library\bin"
            )
            
            image_paths = []
            for i, page in enumerate(tqdm(pages, desc=f"{pdf_name} 페이지 변환")):
                image_filename = f"page_{i+1:03d}.{config.IMAGE_FORMAT.lower()}"
                image_path = pdf_staging_dir / image_filename
                page.save(image_path, config.IMAGE_FORMAT)
                image_paths.append(image_path)

            # 향상된 메타데이터 저장
            end_time = datetime.now()
            processing_time = end_time - start_time
            
            enhanced_metadata = {
                'dpi': config.DPI, 
                'page_count': len(pages),
                'image_format': config.IMAGE_FORMAT,
                'processing_time_seconds': processing_time.total_seconds(),
                'created_at': start_time.isoformat(),
                'completed_at': end_time.isoformat(),
                'pdf_size_bytes': pdf_path.stat().st_size
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(enhanced_metadata, f, indent=2, ensure_ascii=False)
            
            # 체크포인트 저장
            self._save_conversion_checkpoint(pdf_path, enhanced_metadata)
                
            print(f"✅ '{pdf_name}': {len(pages)}페이지 변환 완료 ({processing_time.total_seconds():.1f}초)")
            return image_paths
            
        except Exception as e:
            print(f"❌ '{pdf_name}' 변환 실패: {str(e)}")
            return []
    
    def convert_pdfs(self, specific_pdf_name: str = None) -> dict:
        """
        지정된 또는 모든 PDF 파일을 이미지로 변환
        
        Args:
            specific_pdf_name: 특정 PDF 파일명 (확장자 제외). None이면 모두 변환.
            
        Returns:
            dict: {pdf_name: [image_paths]} 형태의 딕셔너리
        """
        if specific_pdf_name:
            pdf_files = list(self.pdf_dir.glob(f"{specific_pdf_name}.pdf"))
            if not pdf_files:
                print(f"❌ 지정된 PDF 파일을 찾을 수 없습니다: {specific_pdf_name}.pdf")
                return {}
        else:
            pdf_files = list(self.pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print("❌ PDF 파일을 찾을 수 없습니다.")
            return {}
        
        print(f"📁 {len(pdf_files)}개의 PDF 파일을 처리합니다.")
        
        results = {}
        for pdf_file in pdf_files:
            image_paths = self.convert_pdf_to_images(pdf_file)
            if image_paths:
                results[pdf_file.stem] = image_paths
                
        return results

if __name__ == "__main__":
    import sys
    converter = PDFConverter()
    
    if len(sys.argv) > 1:
        # 특정 PDF만 변환
        specific_pdf = sys.argv[1]
        results = converter.convert_pdfs(specific_pdf)
    else:
        # 모든 PDF 변환
        results = converter.convert_pdfs()
    
    print(f"\n🎉 변환 완료: {len(results)}개 PDF 처리됨")
