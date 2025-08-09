"""
공용 프롬프트 유틸리티
- Syncfusion SDK 매뉴얼 OCR → Markdown 변환을 위한 표준 프롬프트 생성
- 제품/버전/로캘 등 메타데이터 포함

주의: 코드 주석은 한국어, 변수/함수명은 영어를 사용합니다.
"""
from __future__ import annotations

from pathlib import Path
from datetime import datetime
import re
import config


def build_syncfusion_prompt(image_path: Path) -> str:
  """Syncfusion 특화 공용 프롬프트 생성 (동적 메타데이터 포함)

  - 미세조정 데이터셋/RAG 일관성 강화를 위해 동적 컨텍스트를 포함한 지시문
  - 컨텍스트/메타 정보 표준화, 섹션 스키마 고정, 언어/코드/표/리스트/링크 처리 규약 강화
  - OCR 정규화 규칙 및 레이아웃/노이즈 제거 규칙 포함 (다단, 헤더/푸터/워터마크/네비게이션 제거)

  Args:
    image_path: 페이지 이미지 경로

  Returns:
    공용 프롬프트 문자열
  """
  # 파일/경로 기반 컨텍스트 추출
  file_name = image_path.name
  parent_dir = image_path.parent.name
  stem = image_path.stem
  m = re.search(r"page[_-]?(\d+)", stem, re.IGNORECASE)
  page_number = m.group(1) if m else "unknown"

  # ISO-8601 UTC 타임스탬프 (초 단위)
  iso_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

  # 제품/버전/로캘 기본값 (config 연동)
  product = getattr(config, "PRODUCT_NAME", "Syncfusion")
  version = getattr(config, "PRODUCT_VERSION", "unknown")
  locale = getattr(config, "DEFAULT_LOCALE", "auto")

  # 프롬프트 본문
  return f"""
You are a meticulous technical documentation OCR and structuring agent specialized in {product} User Guides.
Your task is to convert the given documentation image (jpg) into HIGH-FIDELITY Markdown that is suitable for LLM fine-tuning datasets and RAG retrieval.

CONTEXT VALUES (use EXACTLY in the metadata header):
- source: image
- domain: syncfusion-sdk
- task: pdf-ocr-to-markdown
- language: {locale} (keep original; do not translate)
- source_filename: {file_name}
- document_name: {parent_dir}
- page_number: {page_number}
- page_id: {parent_dir}#page_{page_number}
- product: {product}
- version: {version}
- timestamp: {iso_ts}
- fidelity: lossless

GLOBAL OUTPUT CONTRACT (MUST FOLLOW EXACTLY):
- Top-level must start with an HTML comment metadata block:
  <!--
  source: image
  domain: syncfusion-sdk
  task: pdf-ocr-to-markdown
  language: {locale} (keep original; do not translate)
  source_filename: {file_name}
  document_name: {parent_dir}
  page_number: {page_number}
  page_id: {parent_dir}#page_{page_number}
  product: {product}
  version: {version}
  timestamp: {iso_ts}
  fidelity: lossless
  -->
- After metadata, output the structured content only in Markdown. No extra explanations.
- Do not invent content. If text is cropped/unclear, include "[unclear]" and keep position.
- Preserve all text as-is except for OCR normalization rules below.

OCR NORMALIZATION RULES:
- Merge hyphenated line breaks: "inter-
  face" -> "interface" when it's the same token.
- Normalize multiple spaces to single spaces, but preserve indentation inside code blocks.
- Preserve Unicode punctuation and math symbols as-is.
- Keep list numbering as shown (don't renumber).
- Keep casing exactly; do not title-case or sentence-case.
- Remove soft hyphen (\u00AD) characters.
- Normalize common OCR ligatures: "fi", "fl" errors back to original if misread, but only when unambiguous.
- If a word is split across columns or layout artifacts, reconstruct it only when unambiguous; otherwise keep with "[unclear]".

LAYOUT AND NOISE FILTERING (STRICT):
- Reading order: for multi-column pages, read column 1 top→bottom, then column 2 top→bottom, etc. Do not interleave lines across columns.
- Ignore page furniture: page headers/footers, running titles, page numbers, watermarks, and document-wide copyright banners unless they contain page-unique content.
- Ignore site/app chrome/navigation such as: Home, Contents/TOC sidebar, Breadcrumbs, Previous/Next, Expand All/Collapse All, language switchers unless the page’s body explicitly references them.
- If a table spans columns or wraps, reconstruct the logical row/column order.
- Do not duplicate repeated section titles that appear in both header and body—keep the body title only.

STRUCTURE SCHEMA (ENFORCE):
# {{Page/Main Title}}

## Overview
- 1-3 bullets summarizing the page scope using only visible text.

## Content
- Reconstruct hierarchy (H2/H3/H4) exactly as in image.
- Tables: use GitHub-flavored Markdown. Keep column order, headers, alignment if visible.
- Lists: preserve nesting and markers (-, *, 1.) as-is.
- Callouts: map to blockquotes with labels (Note:, Warning:, Tip:).
- Figures/Captions: include as "Figure: ..." lines when present.

### WinForms-specific conventions
- Prefer C# samples when language is ambiguous; if VB is explicitly shown, keep both.
- Treat control names, namespaces, and types exactly (e.g., Syncfusion.Windows.Forms.Tools.TabControlAdv, Syncfusion.Windows.Forms.Grid).
- Distinguish design-time vs runtime features; preserve property grids, designer steps, and menu paths as regular text or ordered lists.
- When API elements are listed (Properties/Methods/Events), keep their exact order and names, including parentheses for methods and event handler signatures if visible.

## API Reference (if applicable)
- Namespace, Class, Members (Methods/Properties/Events/Enums) in subsections.
- Parameters table: Name | Type | Description | Default | Required
- Returns: Type + description.
- Exceptions: bullet list.

## Code Examples (multi-language supported)
- Extract ALL code exactly. Use fenced blocks with language: ```csharp, ```vb, ```xml, ```xaml, ```js, ```css, ```ts, ```python.
- Keep full signatures, imports/usings, comments, region markers.
- Inline code in text should be wrapped with backticks.

## Page-level Navigation/TOC (if applicable)
- If the body contains a local Table of Contents for this page, keep it as a bullet/numbered list with links/text as shown. Do not create links that don’t exist.
- Ignore global site TOC or breadcrumbs unless the page explicitly describes them.

## Cross References
- Add See also: bullet list of explicit links/texts present on the page. Do not fabricate.

## RAG Annotations
- At the end, add an HTML comment with tags and keywords derived ONLY from visible content:
  <!-- tags: [product, module, control, api, version?] keywords: [k1, k2, ...] -->
 - Add optional per-section anchors as HTML comments before each H2/H3 to aid chunking, using IDs derived from the heading (kebab-case), e.g., <!-- anchor: {parent_dir}#page_{page_number}#getting-started -->. Do not add if the heading text is unclear.

ADDITIONAL RULES:
- Units, versions, file paths, and identifiers must be preserved exactly.
- Do not reflow long lines inside code blocks.
- Preserve table cell line breaks using <br> if present.
- For cross-page references without URLs, keep the exact anchor text.
- For images/icons with meaningful labels, include a short alt text inline as: "Figure: <label>"; ignore purely decorative icons.
- Do not include scanning artifacts, crop marks, or repeated footer legal texts unless unique to this page.

Output now in the specified format.
"""
