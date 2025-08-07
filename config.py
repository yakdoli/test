"""
PDF to Markdown 변환 프로젝트 설정
"""
import os
from pathlib import Path

# 기본 경로 설정
BASE_DIR = Path(__file__).parent
PDF_DIR = BASE_DIR / "pdfs"
STAGING_DIR = BASE_DIR / "staging"
OUTPUT_DIR = BASE_DIR / "output"

# Ollama 설정
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5vl:latest"

# 이미지 변환 설정
DPI = 150  # PDF를 이미지로 변환할 때 해상도 (성능 최적화)
IMAGE_FORMAT = "JPEG"  # 빠른 처리를 위해 JPEG 사용

# 디렉토리 생성
STAGING_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 지원하는 파일 확장자
SUPPORTED_EXTENSIONS = ['.pdf']

# Syncfusion SDK 매뉴얼 특화 설정
SYNCFUSION_MODE = True  # Syncfusion SDK 매뉴얼 처리 모드
EXTRACT_CODE_SNIPPETS = False  # 코드 스니펫 별도 추출 (성능 최적화)
PRESERVE_API_STRUCTURE = True  # API 구조 보존
ENABLE_RAG_OPTIMIZATION = True  # RAG 최적화 활성화

# 출력 형식 설정
INCLUDE_METADATA = True  # 메타데이터 포함
SEMANTIC_CHUNKING = False  # 의미 단위 청킹 (성능 최적화)
CROSS_REFERENCE_LINKS = True  # 교차 참조 링크 생성