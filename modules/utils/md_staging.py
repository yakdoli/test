from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import json
import re
from typing import Optional, Dict, Any

import config


@dataclass
class MDPageRecord:
    pdf_name: str
    page_number: int
    file_name: str
    image_rel_path: str
    created_at: str
    mode: Optional[str] = None
    prompt: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None


def _safe_pdf_name(name: str) -> str:
    # Keep simple ascii/ko, replace others with '-'
    return re.sub(r"[^\w\-\.\u3131-\u318E\uAC00-\uD7A3]", "-", name)


def _detect_pdf_name_from_image(image_path: Path) -> str:
    # parent dir name is our pdf_name convention
    return image_path.parent.name


def _detect_page_number(image_path: Path, default: int = 0) -> int:
    m = re.search(r"page[_-]?(\d+)", image_path.stem, re.IGNORECASE)
    try:
        return int(m.group(1)) if m else default
    except Exception:
        return default


def save_page_markdown(image_path: Path, markdown: str, *, mode: Optional[str] = None, prompt: Optional[str] = None, extra_meta: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save per-image markdown and metadata into md_staging/{pdf_name}/
    - Writes: page_XXX.md and page_XXX.json
    Returns path to the saved markdown file.
    """
    if not config.ENABLE_MD_STAGING:
        return Path()

    pdf_name = _detect_pdf_name_from_image(image_path)
    page_num = _detect_page_number(image_path, default=0)

    base_dir = config.MD_STAGING_DIR
    if config.MD_STAGING_WITH_MODE_SUBDIR and mode:
        base_dir = base_dir / mode

    target_dir = base_dir / _safe_pdf_name(pdf_name)
    target_dir.mkdir(parents=True, exist_ok=True)

    md_path = target_dir / f"page_{page_num:03d}.md"
    meta_path = target_dir / f"page_{page_num:03d}.json"

    # Write markdown
    md_path.write_text(markdown or "", encoding="utf-8")

    # Write metadata
    record = MDPageRecord(
        pdf_name=pdf_name,
        page_number=page_num,
        file_name=image_path.name,
        image_rel_path=str(image_path.relative_to(image_path.parents[1])) if image_path.exists() else str(image_path),
        created_at=datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        mode=mode,
        prompt=prompt if (prompt and config.MD_STAGING_INCLUDE_PROMPT) else None,
        extra=extra_meta or None,
    )
    meta_path.write_text(json.dumps(asdict(record), ensure_ascii=False, indent=2), encoding="utf-8")

    return md_path
