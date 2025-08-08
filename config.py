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

# Xinference 설정
XINFERENCE_BASE_URL = "http://localhost:9997"

# Xinference 모델 설정
XINFERENCE_MODEL_NAME = "qwen2-vl-instruct"
XINFERENCE_MODEL_UID = None  # 런타임에 자동으로 할당됨

# Qwen2.5-VL 직접 사용 설정 (GPU 최적화)
USE_DIRECT_QWEN = True  # False면 Xinference 사용, True면 직접 모델 로드
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # HuggingFace 모델 ID
QWEN_DEVICE = "auto"  # "auto", "cuda", "cpu"
QWEN_TORCH_DTYPE = "auto"  # "auto", "float16", "bfloat16"
QWEN_TRUST_REMOTE_CODE = True
QWEN_USE_FLASH_ATTENTION = False  # Flash Attention 2 사용 (메모리 최적화) - 패키지 없음으로 비활성화

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
EXTRACT_CODE_SNIPPETS = True  # 코드 스니펫 별도 추출 (성능 최적화)
PRESERVE_API_STRUCTURE = True  # API 구조 보존
ENABLE_RAG_OPTIMIZATION = True  # RAG 최적화 활성화

# 출력 형식 설정
INCLUDE_METADATA = True  # 메타데이터 포함
SEMANTIC_CHUNKING = True  # 의미 단위 청킹 (성능 최적화)
CROSS_REFERENCE_LINKS = True  # 교차 참조 링크 생성

# 병렬 처리 설정 (64코어 AMD EPYC 최적화)
ENABLE_PARALLEL_PROCESSING = True  # 병렬 처리 활성화
MAX_CONCURRENT_REQUESTS = 12  # Ollama 동시 요청 수 (테스트 결과 최적값)
MAX_WORKERS = os.cpu_count()  # 최대 워커 스레드 수 (CPU 코어 수)
CHUNK_SIZE = 3  # 페이지 청크 크기 (테스트용으로 줄임)
REQUEST_TIMEOUT = 200  # 요청 타임아웃 (초)
RETRY_DELAY = 1  # 재시도 지연 시간 (초)