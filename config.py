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
MD_STAGING_DIR = BASE_DIR / "md_staging"  # 페이지 단위 Markdown 원본 저장소



XINFERENCE_BASE_URL = "http://localhost:9997"

# Xinference 모델 설정
XINFERENCE_MODEL_NAME = "qwen2.5-vl-instruct"
XINFERENCE_MODEL_UID = None  # 런타임에 자동으로 할당됨

# Qwen2.5-VL 직접 사용 설정 (GPU 최적화) - 메인 처리 방식
USE_DIRECT_QWEN = False  # 직접 모델 로드 (Xinference 대신)
QWEN_MODEL_PATH = "Qwen/Qwen2.5-VL-7B-Instruct"  # HuggingFace 모델 ID
QWEN_DEVICE = "auto"  # "auto", "cuda", "cpu"
QWEN_TORCH_DTYPE = "auto"  # "auto", "float16", "bfloat16"
QWEN_TRUST_REMOTE_CODE = True
QWEN_USE_FLASH_ATTENTION = True  # Flash Attention 2 사용 (메모리 최적화)
HF_CACHE_DIR = os.getenv("HF_HOME", "/workspace/.hf") # Hugging Face 모델 캐시 디렉토리

# 스케일링 방식 설정
SCALING_APPROACH = "auto"  # "auto", "scale_up", "scale_out", "process_isolation"
# auto: 시스템 리소스에 따라 자동 선택
# scale_up: 단일 GPU 전용 최적화 (높은 성능)
# scale_out: 다중 GPU 분산 처리 (메모리 분산)
# process_isolation: 프로세스별 GPU 격리 (높은 안정성)

# 이미지 변환 설정
DPI = 150  # PDF를 이미지로 변환할 때 해상도 (성능 최적화)
IMAGE_FORMAT = "JPEG"  # 빠른 처리를 위해 JPEG 사용

# 디렉토리 생성
STAGING_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
MD_STAGING_DIR.mkdir(exist_ok=True)

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

# 메타데이터 확장 (제품/버전/로캘)
PRODUCT_NAME = "Syncfusion Winforms"
PRODUCT_VERSION = "11.4.0.26"
DEFAULT_LOCALE = "en"  # 'auto' | 'en' | 'ko' 등

# 병렬 처리 설정 (Qwen2.5-VL 직접 로드 최적화)
ENABLE_PARALLEL_PROCESSING = True  # 병렬 처리 활성화
MAX_CONCURRENT_REQUESTS = 4  # GPU 메모리 고려 동시 요청 수 (조정됨)
MAX_WORKERS = min(os.cpu_count(), 32)  # 최대 워커 수 (GPU 메모리 고려)
CHUNK_SIZE = 12  # GPU 메모리 최적화를 위한 청크 크기
REQUEST_TIMEOUT = 300  # 모델 로드 시간 고려 증가된 타임아웃
RETRY_DELAY = 2  # 재시도 지연 시간

# GPU 워커/리소스 관리 설정 (단일 GPU/멀티 워커 지원)
# 각 워커는 단일 GPU만 사용하며, 아래 한도 내에서 GPU당 다중 워커를 허용
GPU_MAX_MEMORY_FRACTION = float(os.getenv("GPU_MAX_MEMORY_FRACTION", 0.9))  # GPU 메모리의 최대 사용 비율
WORKER_ESTIMATED_VRAM_GB = float(os.getenv("WORKER_ESTIMATED_VRAM_GB", 16))  # 워커 1개가 필요로 하는 추정 VRAM(GB)
MAX_WORKERS_PER_GPU = 12
_tmp_max_wpg = os.getenv("MAX_WORKERS_PER_GPU")
# int(_tmp_max_wpg) if _tmp_max_wpg else None  # GPU당 최대 워커 수 (없으면 자동 계산)
MAX_TOTAL_GPU_WORKERS = 16
_tmp_max_total = os.getenv("MAX_TOTAL_GPU_WORKERS")
# int(_tmp_max_total) if _tmp_max_total else None  # 전체 최대 GPU 워커 수
PER_WORKER_BATCH_SIZE = int(os.getenv("PER_WORKER_BATCH_SIZE", 4))  # 워커 1개가 처리하는 배치 크기
PER_WORKER_CONCURRENCY = int(os.getenv("PER_WORKER_CONCURRENCY", 4))  # 워커 내부 동시 처리 개수

# MD 스테이징 설정
ENABLE_MD_STAGING = True  # 개별 이미지 변환 직후 Markdown/메타 저장
MD_STAGING_INCLUDE_PROMPT = True  # 사용 프롬프트를 메타에 포함
MD_STAGING_WITH_MODE_SUBDIR = True  # 모드별 하위 디렉토리 분리 저장 (필요 시 True)
