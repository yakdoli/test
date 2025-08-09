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
# Direct Qwen2.5-VL loading (RECOMMENDED)
python main_direct_qwen.py

# Convert specific PDF (filename without extension)
python main_direct_qwen.py "DICOM"

# Legacy main (basic single-threaded)
python main.py
```

**Test individual scaling approaches:**
```bash
# Test scale-up approach (single GPU optimization)
python qwen_direct_client.py

# Test scale-out approach (multi-GPU distribution)
python qwen_multi_gpu_client.py

# Test process isolation approach (complete isolation)
python qwen_process_isolation_client.py

# Test PDF to image conversion
python pdf_converter.py
```

**Legacy Xinference setup (archived):**
```bash
# NOTE: Xinference approach has been archived
# Direct Qwen2.5-VL loading is now the primary method

# Archived files in archive/ directory:
# - ollama_client.py
# - parallel_ollama_client.py  
# - unified_ollama_client.py
# - main_parallel.py
```

## Architecture

This is a PDF to Markdown converter that uses **direct Qwen2.5-VL-7B-Instruct model loading** for high-performance text extraction and structure conversion from PDF documents.

**Core Flow:**
1. **PDF Processing** (`pdf_converter.py`) - Converts PDFs to page-by-page JPEG images using pdf2image
2. **Direct LLM Processing** - Three scaling approaches available:
   - **Scale-up** (`qwen_direct_client.py`) - Single GPU optimization with dedicated loading
   - **Scale-out** (`qwen_multi_gpu_client.py`) - Multi-GPU distribution for higher throughput  
   - **Process Isolation** (`qwen_process_isolation_client.py`) - Complete process isolation for stability
3. **Unified Orchestration** (`main_direct_qwen.py`) - Interactive mode selection and conversion coordination

**Directory Structure:**
- `pdfs/` - Input PDF files
- `staging/` - Temporary converted images (organized by PDF name)
- `output/` - Final markdown files with scaling approach suffix
- `archive/` - Archived Xinference-based implementations

**Configuration** (`config.py`):
- Direct Qwen2.5-VL model loading (primary approach)
- Three scaling approaches: scale-up, scale-out, process-isolation
- GPU memory optimization and Flash Attention 2 support
- Syncfusion SDK manual processing mode with code snippet extraction
- Performance tuning: DPI=150, JPEG format, optimized chunk sizes

**Three Scaling Approaches:**

1. **Scale-up** (전용 GPU 로드):
   - Single GPU dedicated model loading
   - Highest performance per GPU
   - Best for high-end single GPU systems (24GB+ VRAM)

2. **Scale-out** (멀티 GPU 분산):
   - Multi-GPU model distribution 
   - Memory load balancing across GPUs
   - Best for multi-GPU systems (2+ GPUs)

3. **Process Isolation** (프로세스 격리):
   - Complete process separation per GPU
   - Maximum stability and memory isolation
   - Best for mission-critical processing

**Special Features:**
- **Interactive Mode Selection**: Automatic system analysis and scaling recommendation
- **Syncfusion Mode**: Specialized SDK documentation processing with code extraction
- **Flash Attention 2**: Memory-optimized attention mechanism
- **GPU Memory Management**: Intelligent memory allocation and cleanup
- **Progress Tracking**: Real-time processing statistics and throughput monitoring

The unified entry point (`main_direct_qwen.py`) is recommended for production use as it provides interactive scaling approach selection with optimal performance.