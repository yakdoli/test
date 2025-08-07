# PDF to Markdown 변환기

PDF 파일을 개별 페이지 이미지로 변환하고, Ollama의 로컬 LLM(Qwen2.5-VL:7b)을 사용하여 마크다운 파일로 변환하는 Python 프로젝트입니다.

## 🚀 주요 기능

- PDF 파일을 고해상도 이미지로 변환 (페이지별 분할)
- Ollama 로컬 LLM을 활용한 이미지-텍스트 변환
- 마크다운 형식으로 구조화된 출력
- 배치 처리 및 개별 파일 처리 지원
- 진행 상황 표시 및 에러 처리

## 📁 프로젝트 구조

```
.
├── pdfs/           # 원본 PDF 파일들
├── staging/        # 변환된 이미지 파일들 (임시)
├── output/         # 최종 마크다운 파일들
├── main.py         # 메인 실행 파일
├── pdf_converter.py # PDF → 이미지 변환 모듈
├── ollama_client.py # Ollama API 클라이언트
├── config.py       # 설정 파일
└── requirements.txt # 의존성 패키지
```

## 🛠️ 설치 및 설정

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. 시스템 의존성

**Ubuntu/Debian:**
```bash
sudo apt-get install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
- [Poppler for Windows](http://blog.alivate.com.au/poppler-windows/) 다운로드 및 설치

### 3. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 (Linux/macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Ollama 서버 시작
ollama serve

# Qwen2.5-VL 모델 다운로드 (새 터미널에서)
ollama pull qwen2.5-vl:7b
```

## 🎯 사용법

### 모든 PDF 파일 변환
```bash
python main.py
```

### 특정 PDF 파일만 변환
```bash
python main.py "파일명"
```
예: `python main.py "DICOM"`

### 개별 모듈 테스트

**PDF 변환 테스트:**
```bash
python pdf_converter.py
```

**Ollama 연결 테스트:**
```bash
python ollama_client.py
```

## ⚙️ 설정 옵션

`config.py`에서 다음 설정을 변경할 수 있습니다:

```python
# 이미지 변환 품질
DPI = 200  # 높을수록 고품질, 처리 시간 증가

# Ollama 설정
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5-vl:7b"

# 이미지 형식
IMAGE_FORMAT = "PNG"  # PNG 또는 JPEG
```

## 📋 처리 과정

1. **PDF 스캔**: `pdfs/` 디렉토리에서 모든 PDF 파일 검색
2. **이미지 변환**: 각 PDF를 페이지별 이미지로 변환하여 `staging/` 저장
3. **환경 확인**: Ollama 서버 연결 및 모델 사용 가능 여부 확인
4. **텍스트 추출**: 각 이미지를 Ollama LLM으로 마크다운 변환
5. **파일 생성**: 변환된 내용을 `output/` 디렉토리에 `.md` 파일로 저장

## 🔧 문제 해결

### Ollama 연결 오류
```bash
# Ollama 서버 상태 확인
ollama list

# 서버 재시작
ollama serve
```

### 모델 다운로드 오류
```bash
# 모델 재다운로드
ollama pull qwen2.5-vl:7b

# 사용 가능한 모델 확인
ollama list
```

### PDF 변환 오류
- Poppler 설치 확인
- PDF 파일 권한 및 손상 여부 확인
- 메모리 부족 시 DPI 값 낮추기

## 📊 성능 최적화

- **DPI 조정**: 품질과 속도의 균형점 찾기
- **배치 크기**: 메모리 사용량에 따라 조정
- **병렬 처리**: 향후 멀티스레딩 지원 예정

## 🤝 기여하기

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

MIT License - 자세한 내용은 LICENSE 파일을 참조하세요.