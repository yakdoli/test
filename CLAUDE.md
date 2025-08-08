# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Dependencies Installation:**
```bash
pip install -r requirements.txt
```

**System Dependencies (Ubuntu/Debian):**
```bash
sudo apt-get install poppler-utils
```

**Run the converter:**
```bash
# Convert all PDFs
python main.py

# Convert specific PDF (filename without extension)
python main.py "DICOM"

# Run parallel processing version (optimized)
python main_parallel.py
```

**Test individual modules:**
```bash
# Test PDF conversion
python pdf_converter.py

# Test Xinference connection
python ollama_client.py
```

**Xinference setup:**
```bash
# Install Xinference
pip install xinference[all]

# Start Xinference server with qwen2-vl-instruct model
./start_xinference.sh

# Or manually:
xinference launch --model-engine vLLM --model-name qwen2-vl-instruct --size-in-billions 7 --model-format gptq --quantization Int8

# Check models via API
curl http://localhost:9997/v1/models
```

## Architecture

This is a PDF to Markdown converter that uses Xinference with qwen2-vl-instruct model to extract text and structure from PDF documents.

**Core Flow:**
1. **PDF Processing** (`pdf_converter.py`) - Converts PDFs to page-by-page JPEG images using pdf2image
2. **LLM Processing** (`ollama_client.py`) - Sends images to Xinference for text extraction and markdown conversion
3. **Main Orchestration** (`main.py`) - Coordinates the entire conversion pipeline
4. **Async Processing** (`main_parallel.py`, `parallel_ollama_client.py`) - Optimized version using async requests

**Directory Structure:**
- `pdfs/` - Input PDF files
- `staging/` - Temporary converted images (organized by PDF name)
- `output/` - Final markdown files
- `modules/` - Modular architecture components for parallel processing

**Configuration** (`config.py`):
- Single Xinference server at localhost:9997
- Optimized for concurrent processing with configurable request limits
- Syncfusion SDK manual processing mode with code snippet extraction
- Performance tuning: DPI=150, JPEG format, async processing

**Special Features:**
- **Skip Processing**: Automatically skips already converted files based on DPI settings
- **Syncfusion Mode**: Special handling for SDK documentation with code snippet extraction
- **Parallel Processing**: Utilizes multiple Ollama instances for high-throughput processing
- **Progress Tracking**: Uses tqdm for conversion progress visualization
- **Error Handling**: Robust error handling with retry mechanisms

The async version (`main_parallel.py`) is recommended for production use as it provides efficient concurrent processing with a single Xinference instance.