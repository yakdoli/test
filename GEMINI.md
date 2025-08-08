# Gemini Code Assistant Context

## Project Overview

This project is a Python-based tool for converting PDF files into Markdown documents. It works by first converting each page of a PDF into a high-resolution image. Then, it uses a locally running Ollama Large Language Model (specifically, `qwen2.5-vl:7b`, a vision-language model) to perform Optical Character Recognition (OCR) and generate structured Markdown content from each image.

The primary goal is to automate the extraction and formatting of text from PDF documents, making the content easily editable and searchable. The project supports batch processing of all PDFs in a directory or converting a single specified file.

**Key Technologies:**

*   **Python:** The core programming language.
*   **Ollama:** Used for running the local LLM for image-to-text conversion.
*   **Poppler:** A system dependency required for the PDF-to-image conversion process.
*   **Pillow:** Python imaging library used for handling images.
*   **tqdm:** Used for displaying progress bars during processing.

## Building and Running

### 1. Setup

**A. Install Python Dependencies:**

The required Python packages are listed in `requirements.txt`.

```bash
pip install -r requirements.txt
```

**B. Install System Dependencies:**

The Poppler library is required for PDF manipulation.

*   **Ubuntu/Debian:**
    ```bash
    sudo apt-get install poppler-utils
    ```
*   **macOS:**
    ```bash
    brew install poppler
    ```

**C. Set up Ollama:**

A local Ollama instance with the `qwen2.5-vl:7b` model is necessary.

1.  **Install Ollama:**
    ```bash
    curl -fsSL https://ollama.ai/install.sh | sh
    ```
2.  **Start the Ollama server:**
    ```bash
    ollama serve
    ```
3.  **Pull the required model:**
    ```bash
    ollama pull qwen2.5-vl:7b
    ```

### 2. Execution

The main script for running the conversion is `main.py`.

*   **To convert all PDF files** located in the `pdfs/` directory:
    ```bash
    python main.py
    ```
*   **To convert a specific PDF file** (provide the name without the extension):
    ```bash
    python main.py "DICOM"
    ```
*   **To run the conversion in parallel** (for improved performance on multi-core systems):
    ```bash
    python main_parallel.py
    ```

### 3. Configuration

Key parameters can be adjusted in `config.py`:

*   `DPI`: Resolution for the PDF-to-image conversion (e.g., `200`). Higher values improve quality but increase processing time.
*   `OLLAMA_BASE_URL`: The URL for the Ollama API endpoint (defaults to `http://localhost:11434`).
*   `OLLAMA_MODEL`: The name of the Ollama model to use (e.g., `qwen2.5-vl:7b`).
*   `IMAGE_FORMAT`: The intermediate image format (`PNG` or `JPEG`).

## Development Conventions

*   **Directory Structure:**
    *   `pdfs/`: Input PDF files.
    *   `staging/`: Temporary directory for intermediate image files.
    *   `output/`: Destination for the final Markdown files.
    *   `modules/`: Contains modularized parts of the application logic.
*   **Modular Code:** The project is structured into several modules:
    *   `pdf_converter.py`: Handles the PDF-to-image conversion logic.
    *   `ollama_client.py`: A client for interacting with the Ollama API.
    *   `parallel_ollama_client.py`: A parallel version of the Ollama client.
    *   `main.py` / `main_parallel.py`: Orchestrates the end-to-end conversion process.
*   **Error Handling:** The scripts include basic error handling and progress indicators.
*   **Contribution:** The project follows a standard GitHub contribution model (Fork -> Feature Branch -> Pull Request).
