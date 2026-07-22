from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import fitz

try:
    from .geometry import BBox, bbox_intersects, bbox_to_list
    from .normalization import normalize_for_search, normalize_text, stable_id
except ImportError:
    from geometry import BBox, bbox_intersects, bbox_to_list
    from normalization import normalize_for_search, normalize_text, stable_id


HEADING_RE = re.compile(
    r"^\s*((phần|chương|bài|mục)\s+[\w\.\-]+|[IVXLCDM]+\s*\.|[0-9]+(\.[0-9]+){0,4}\.?)\s+",
    flags=re.IGNORECASE | re.UNICODE,
)


LIST_RE = re.compile(r"^\s*(?:[-*+]|\(?[0-9]{1,3}\)?[.)]|[a-zA-Z][.)])\s+")


SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?;:])\s+(?=[A-ZÀ-Ỵ0-9])", flags=re.UNICODE)


WORD_RE = re.compile(r"\S+", flags=re.UNICODE)


STRUCTURAL_HEADING_RE = re.compile(
    r"^\s*(?:[0-9]+(?:\.[0-9]+){0,5}\.?|[IVXLCDM]+\.?|[A-Z]\.)\s+\S",
    flags=re.IGNORECASE | re.UNICODE,
)


VALID_ROMAN_RE = re.compile(r"M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})")


MEDICAL_SECTION_KEYWORDS = {
    "symptoms": ("trieu chung", "lam sang", "dau hieu", "bieu hien"),
    "diagnosis": ("chan doan", "tieu chuan chan doan", "chan doan phan biet"),
    "tests": (
        "can lam sang",
        "xet nghiem",
        "cong thuc mau",
        "sinh hoa",
        "dien tim",
        "x quang",
        "sieu am",
        "ct",
        "mri",
    ),
    "treatment": ("dieu tri", "xu tri", "phac do", "cap cuu", "thuoc", "lieu dung"),
    "complications": ("bien chung", "di chung", "tien luong"),
}


HEADING_KEYWORDS = tuple(keyword for values in MEDICAL_SECTION_KEYWORDS.values() for keyword in values)


def remove_page_footer(text: str) -> str:
    """Remove page number footers from text block.
    
    Detects patterns like standalone page numbers at bottom of pages:
    - Single digits: 1, 2, 9
    - Multi-digit: 123, 456
    - With separators: 1/, /1/, -1-, etc.
    """
    if not text:
        return text
    
    lines = text.split("\n")
    filtered_lines = []
    
    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            filtered_lines.append(line)
            continue
        
        if re.fullmatch(r"^\d+$", clean_line) and len(clean_line) <= 3:
            continue
        
        if re.fullmatch(r"^[\-/]?\d+[\-/]?$", clean_line):
            continue
        
        if re.fullmatch(r"^page\s*\d+$", clean_line, flags=re.IGNORECASE):
            continue
        
        filtered_lines.append(line)
    
    result = "\n".join(filtered_lines)
    return normalize_text(result, preserve_lines=True)


def extract_text_blocks(page: fitz.Page) -> List[Dict[str, Any]]:
    blocks: List[Dict[str, Any]] = []
    page_dict = page.get_text("dict", sort=True)

    for block_index, block in enumerate(page_dict.get("blocks", []), start=1):
        if block.get("type") != 0:
            continue
        lines: List[str] = []
        fonts: List[str] = []
        sizes: List[float] = []
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            line_text = "".join(span.get("text", "") for span in spans)
            line_text = normalize_text(line_text, preserve_lines=False)
            if line_text:
                lines.append(line_text)
            for span in spans:
                font = span.get("font")
                if font:
                    fonts.append(font)
                size = span.get("size")
                if isinstance(size, (int, float)):
                    sizes.append(float(size))

        text = normalize_text("\n".join(lines), preserve_lines=True)
        text = remove_page_footer(text)
        if not text:
            continue
        bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))  # type: ignore[assignment]
        blocks.append(
            {
                "block_index": block_index,
                "bbox": bbox_to_list(bbox),
                "text": text,
                "font_names": sorted(set(fonts))[:5],
                "avg_font_size": round(sum(sizes) / len(sizes), 2) if sizes else None,
            }
        )

    return blocks


def page_text_from_blocks(blocks: Sequence[Dict[str, Any]]) -> str:
    paragraphs: List[str] = []
    previous_bbox: Optional[List[float]] = None

    for block in blocks:
        bbox = block.get("bbox")
        gap = None
        if previous_bbox and bbox:
            gap = bbox[1] - previous_bbox[3]
        previous_bbox = bbox
        if gap is not None and gap > 10:
            paragraphs.append("")
        paragraphs.append(block["text"])

    return normalize_text("\n".join(paragraphs), preserve_lines=True)


def block_bbox_tuple(block: Dict[str, Any]) -> Optional[BBox]:
    bbox = block.get("bbox")
    if not bbox or len(bbox) != 4:
        return None
    return tuple(float(value) for value in bbox)


def exclude_text_blocks_by_bboxes(
    blocks: Sequence[Dict[str, Any]],
    excluded_bboxes: Sequence[BBox],
) -> List[Dict[str, Any]]:
    if not excluded_bboxes:
        return list(blocks)
    kept: List[Dict[str, Any]] = []
    for block in blocks:
        bbox = block_bbox_tuple(block)
        if bbox is not None and any(bbox_intersects(bbox, excluded) for excluded in excluded_bboxes):
            continue
        kept.append(block)
    return kept


def median_float(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def semantic_bucket_for_text(text: str, headings: Sequence[str] = ()) -> str:
    folded = normalize_for_search(" ".join([*headings, text]))
    best_bucket = "general"
    best_pos: Optional[int] = None
    for bucket, keywords in MEDICAL_SECTION_KEYWORDS.items():
        for keyword in keywords:
            if len(keyword) <= 3:
                match = re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", folded)
                pos = match.start() if match else -1
            else:
                pos = folded.find(keyword)
            if pos >= 0 and (best_pos is None or pos < best_pos):
                best_bucket = bucket
                best_pos = pos
    return best_bucket


def looks_like_heading_line(
    line: str,
    *,
    avg_font_size: Optional[float] = None,
    body_font_size: Optional[float] = None,
) -> bool:
    clean = normalize_text(line, preserve_lines=False)
    if not clean or len(clean) > 160:
        return False

    folded = normalize_for_search(clean).strip(" .:-")
    if not folded:
        return False
    if any(folded == keyword or folded.startswith(f"{keyword}:") for keyword in HEADING_KEYWORDS):
        return True
    structural_match = STRUCTURAL_HEADING_RE.match(clean)
    if structural_match:
        first_token = clean.split()[0].strip(".")
        is_roman_token = bool(re.fullmatch(r"[IVXLCDM]+", first_token, flags=re.IGNORECASE))
        if is_roman_token and not VALID_ROMAN_RE.fullmatch(first_token.upper()):
            structural_match = None
    if structural_match:
        return len(WORD_RE.findall(clean)) <= 16
    if clean.endswith(":") and any(folded.startswith(keyword) for keyword in HEADING_KEYWORDS):
        return True
    if clean.isupper() and 5 <= len(clean) <= 120 and len(WORD_RE.findall(clean)) <= 14:
        return True
    if (
        avg_font_size is not None
        and body_font_size is not None
        and avg_font_size >= body_font_size + 1.5
        and len(WORD_RE.findall(clean)) <= 14
    ):
        return True
    return False


def heading_level(line: str) -> int:
    clean = normalize_text(line, preserve_lines=False)
    number_match = re.match(r"^\s*([0-9]+(?:\.[0-9]+){0,5})\.?\s+", clean)
    if number_match:
        return min(number_match.group(1).count(".") + 1, 5)
    roman_match = re.match(r"^\s*([IVXLCDM]+)\.?\s+", clean, flags=re.IGNORECASE)
    if roman_match and VALID_ROMAN_RE.fullmatch(roman_match.group(1).upper()):
        return 1
    if re.match(r"^\s*[A-Z]\.\s+", clean):
        return 2
    folded = normalize_for_search(clean).strip(" .:-")
    if any(folded.startswith(keyword) for keyword in HEADING_KEYWORDS):
        return 3
    return 2


def update_heading_path(path: Sequence[str], heading: str, level: int) -> List[str]:
    clean = normalize_text(heading, preserve_lines=False)
    level = max(1, min(level, 5))
    updated = list(path[: level - 1])
    updated.append(clean)
    return updated[-5:]


def split_section_text_into_units(text: str, max_chars: int) -> List[str]:
    normalized = normalize_text(text, preserve_lines=True)
    raw_paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    units: List[str] = []
    for paragraph in raw_paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and all(LIST_RE.match(line) for line in lines):
            candidates = lines
        else:
            candidates = [paragraph]
        for candidate in candidates:
            if len(candidate) <= max_chars:
                units.append(candidate)
            else:
                units.extend(split_long_unit(candidate, max_chars))
    return units


def last_n_words(text: str, word_count: int) -> str:
    """Extract last N words from text for overlap context."""
    if word_count <= 0:
        return ""
    words = WORD_RE.findall(text)
    if not words:
        return ""
    return " ".join(words[-word_count:])


def _merge_short_chunks(
    chunks: List[Dict[str, Any]],
    min_chars: int = 150,
    max_chars: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Merge chunks shorter than min_chars with adjacent chunks.
    
    Enforces minimum chunk size while preserving all content. Short chunks
    are merged forward into the next chunk to maintain reading order.
    """
    if not chunks or min_chars <= 0:
        return chunks
    
    merged = []
    i = 0
    while i < len(chunks):
        chunk = chunks[i].copy()
        content = chunk.get("content", "")
        
        if len(content) < min_chars and i < len(chunks) - 1:
            next_chunk = chunks[i + 1].copy()
            next_content = next_chunk.get("content", "")
            combined = f"{content}\n\n{next_content}"
            if max_chars is None or len(combined) <= max_chars:
                chunk["content"] = combined
                chunk["paragraph_index"] = next_chunk.get("paragraph_index", i + 1)
                merged.append(chunk)
                i += 2
                continue
        merged.append(chunk)
        i += 1
    
    return merged


def _split_section_into_chunks(
    text: str,
    max_chars: int = 1400,
    overlap_words: int = 50,
) -> List[str]:
    """Split section text into chunks with overlap between them.
    
    Args:
        text: Section text to split
        max_chars: Maximum characters per chunk
        overlap_words: Number of words to overlap between chunks
    
    Returns:
        List of chunks with overlap context at the beginning of each
    """
    units = split_section_text_into_units(text, max_chars)
    if not units:
        return []

    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = f"{current}\n\n{unit}" if current else unit
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
        overlap = last_n_words(current, overlap_words)
        overlap_tokens = overlap.split()
        while overlap_tokens and len(" ".join(overlap_tokens)) + len(unit) + 2 > max_chars:
            overlap_tokens.pop(0)
        overlap = " ".join(overlap_tokens)
        current = f"{overlap}\n\n{unit}" if overlap else unit

    if current:
        chunks.append(current)
    return chunks


def split_text_blocks_into_chunks(
    blocks: Sequence[Dict[str, Any]],
    *,
    max_chars: int,
    overlap_words: int = 45,
) -> List[Dict[str, Any]]:
    font_sizes = [
        float(block["avg_font_size"])
        for block in blocks
        if isinstance(block.get("avg_font_size"), (int, float))
    ]
    body_font_size = median_float(font_sizes)
    sections: List[Dict[str, Any]] = []
    heading_path: List[str] = []
    current_lines: List[str] = []
    section_headings: List[str] = []
    previous_bbox: Optional[BBox] = None

    def flush_section() -> None:
        nonlocal current_lines, section_headings
        section_text = normalize_text("\n".join(current_lines), preserve_lines=True)
        if section_text:
            sections.append(
                {
                    "section_index": len(sections) + 1,
                    "headings": list(section_headings),
                    "text": section_text,
                }
            )
        current_lines = []

    for block in blocks:
        bbox = block_bbox_tuple(block)
        if previous_bbox is not None and bbox is not None and current_lines:
            gap = bbox[1] - previous_bbox[3]
            if gap > 10 and current_lines[-1] != "":
                current_lines.append("")
        previous_bbox = bbox
        avg_font_size = block.get("avg_font_size")
        block_font_size = float(avg_font_size) if isinstance(avg_font_size, (int, float)) else None
        lines = [line.strip() for line in str(block.get("text") or "").splitlines() if line.strip()]
        for line in lines:
            if looks_like_heading_line(line, avg_font_size=block_font_size, body_font_size=body_font_size):
                flush_section()
                heading_path = update_heading_path(heading_path, line, heading_level(line))
                section_headings = list(heading_path)
            else:
                current_lines.append(line)
    flush_section()

    chunks: List[Dict[str, Any]] = []
    for section in sections:
        headings = section["headings"]
        heading_context = " > ".join(headings)
        section_text = section["text"]
        min_chunk_size = 150
        
        if len(section_text) > max_chars:
            chunk_texts = _split_section_into_chunks(
                section_text,
                max_chars=max_chars,
                overlap_words=overlap_words,
            )
        else:
            chunk_texts = [section_text]
        
        section_chunks = []
        for paragraph_index, unit in enumerate(chunk_texts, start=1):
            content = f"{heading_context}\n\n{unit}" if heading_context else unit
            section_chunks.append(
                {
                    "content": content,
                    "headings": headings,
                    "section_index": section["section_index"],
                    "paragraph_index": paragraph_index,
                    "semantic_type": semantic_bucket_for_text(unit, headings),
                }
            )
        
        section_chunks = _merge_short_chunks(
            section_chunks,
            min_chars=min_chunk_size,
            max_chars=max_chars + len(heading_context) + 2,
        )
        chunks.extend(section_chunks)

    if chunks:
        return chunks

    fallback_text = page_text_from_blocks(blocks)
    fallback_chunks = [
        {
            "content": chunk,
            "headings": [],
            "section_index": index,
            "paragraph_index": 1,
            "semantic_type": semantic_bucket_for_text(chunk),
        }
        for index, chunk in enumerate(
            _split_text_into_chunks(
                fallback_text,
                max_chars=max_chars,
                overlap_words=overlap_words,
            ),
            start=1,
        )
    ]
    return _merge_short_chunks(fallback_chunks, min_chars=150, max_chars=max_chars)


def split_long_unit(unit: str, max_chars: int) -> List[str]:
    sentences = [part.strip() for part in SENTENCE_SPLIT_RE.split(unit) if part.strip()]
    if len(sentences) > 1 and all(len(sentence) <= max_chars for sentence in sentences):
        return sentences

    words = WORD_RE.findall(unit)
    if not words:
        return []

    pieces: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        if len(word) > max_chars:
            if current:
                pieces.append(" ".join(current))
                current = []
                current_len = 0
            pieces.extend(word[index : index + max_chars] for index in range(0, len(word), max_chars))
            continue
        extra = len(word) + (1 if current else 0)
        if current and current_len + extra > max_chars:
            pieces.append(" ".join(current))
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len += extra
    if current:
        pieces.append(" ".join(current))
    return pieces


def paragraph_units(text: str, max_chars: int) -> List[str]:
    normalized = normalize_text(text, preserve_lines=True)
    raw_paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]
    units: List[str] = []

    for paragraph in raw_paragraphs:
        lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
        if len(lines) > 1 and all(LIST_RE.match(line) for line in lines):
            units.extend(lines)
            continue
        if len(paragraph) <= max_chars:
            units.append(paragraph)
            continue
        units.extend(split_long_unit(paragraph, max_chars))

    return units


def last_words(text: str, word_count: int) -> str:
    if word_count <= 0:
        return ""
    words = WORD_RE.findall(text)
    return " ".join(words[-word_count:])


def _split_text_into_chunks(text: str, *, max_chars: int = 1400, overlap_words: int = 45) -> List[str]:
    units = paragraph_units(text, max_chars=max_chars)
    chunks: List[str] = []
    current = ""

    for unit in units:
        separator = "\n\n" if current else ""
        if not current:
            current = unit
            continue

        if len(current) + len(separator) + len(unit) <= max_chars:
            current = f"{current}{separator}{unit}"
        else:
            chunks.append(current.strip())
            overlap = last_words(current, overlap_words)
            current = f"{overlap}\n\n{unit}" if overlap else unit

    if current.strip():
        chunks.append(current.strip())

    deduped: List[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        signature = normalize_for_search(chunk)
        if signature and signature not in seen:
            seen.add(signature)
            deduped.append(chunk)
    return deduped


def current_headings(text: str, limit: int = 3) -> List[str]:
    headings: List[str] = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if looks_like_heading_line(clean):
            headings.append(normalize_text(clean, preserve_lines=False))
    return headings[-limit:]


def make_chunk(
    *,
    doc_name: str,
    page_number: int,
    chunk_type: str,
    chunk_index: int,
    content: str,
    metadata: Dict[str, Any],
    reconstruction: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "id": stable_id(doc_name, page_number, chunk_type, chunk_index, content[:120]),
        "doc_name": doc_name,
        "page_number": page_number,
        "chunk_type": chunk_type,
        "chunk_index": chunk_index,
        "content": content,
        "search_text": normalize_for_search(content),
        "metadata": metadata,
        "reconstruction": reconstruction or {},
    }


def nearest_text_for_bbox(text_blocks: Sequence[Dict[str, Any]], bbox: BBox, limit: int = 600) -> str:
    target = fitz.Rect(bbox)
    scored: List[Tuple[float, str]] = []
    for block in text_blocks:
        block_bbox = block.get("bbox")
        if not block_bbox:
            continue
        rect = fitz.Rect(block_bbox)
        if bbox_intersects((rect.x0, rect.y0, rect.x1, rect.y1), bbox):
            distance = 0.0
        else:
            distance = abs(rect.y0 - target.y0) + abs(rect.x0 - target.x0)
        scored.append((distance, block["text"]))
    scored.sort(key=lambda item: item[0])
    text = "\n".join(value for _, value in scored[:4])
    return text[:limit]
