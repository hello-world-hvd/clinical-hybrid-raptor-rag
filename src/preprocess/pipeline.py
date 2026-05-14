from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import logging
import re
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber

try:
    from ftfy import fix_text as ftfy_fix_text
except Exception:  # optional dependency
    ftfy_fix_text = None


LOGGER = logging.getLogger("preprocess")
BBox = Tuple[float, float, float, float]

MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Æ", "áº", "á»", "�")
VIETNAMESE_CHARS = set(
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    "ÈÉẸẺẼÊỀẾỆỂỄ"
    "ÌÍỊỈĨ"
    "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "ÙÚỤỦŨƯỪỨỰỬỮ"
    "ỲÝỴỶỸĐ"
)

CHAR_TRANSLATION = str.maketrans(
    {
        "\ufeff": "",
        "\u00ad": "",
        "\u00a0": " ",
        "\u200b": "",
        "\u200c": "",
        "\u200d": "",
        "\u2060": "",
        "ﬀ": "ff",
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
        "−": "-",
        "–": "-",
        "—": "-",
        "‐": "-",
        "“": '"',
        "”": '"',
        "„": '"',
        "’": "'",
        "‘": "'",
        "′": "'",
        "″": '"',
        "×": " x ",
        "÷": " / ",
        "≤": " <= ",
        "≥": " >= ",
        "≦": " <= ",
        "≧": " >= ",
        "±": " +/- ",
        "≈": " ~= ",
        "≠": " != ",
        "→": " -> ",
        "←": " <- ",
        "↔": " <-> ",
        "⇒": " => ",
        "↑": " tang ",
        "↓": " giam ",
        "•": "- ",
        "●": "- ",
        "▪": "- ",
        "◦": "- ",
        "Ƣ": "Ư",
        "ƣ": "ư",
        "µ": "micro",
        "μ": "micro",
        "°": " degree ",
        "℃": " degree C ",
        "℉": " degree F ",
        "‰": " per mille ",
    }
)

CID_REPLACEMENTS = {
    "(cid:54)": ">=",
    "(cid:110)": "↑",
    "(cid:112)": "↓",
    "(cid:113)": "°",
    "(cid:114)": "↔",
}

SUPERSCRIPT_TRANSLATION = str.maketrans(
    {
        "⁰": "^0",
        "¹": "^1",
        "²": "^2",
        "³": "^3",
        "⁴": "^4",
        "⁵": "^5",
        "⁶": "^6",
        "⁷": "^7",
        "⁸": "^8",
        "⁹": "^9",
        "⁺": "^+",
        "⁻": "^-",
        "⁽": "^(",
        "⁾": "^)",
        "ⁿ": "^n",
    }
)

SUBSCRIPT_TRANSLATION = str.maketrans(
    {
        "₀": "0",
        "₁": "1",
        "₂": "2",
        "₃": "3",
        "₄": "4",
        "₅": "5",
        "₆": "6",
        "₇": "7",
        "₈": "8",
        "₉": "9",
        "₊": "+",
        "₋": "-",
        "₍": "(",
        "₎": ")",
    }
)

MEDICAL_UNIT_PATTERNS = (
    (re.compile(r"\bmicro\s*g\b", flags=re.IGNORECASE), "mcg"),
    (re.compile(r"\bmicro\s*l\b", flags=re.IGNORECASE), "uL"),
    (re.compile(r"\bmicro\s*mol\b", flags=re.IGNORECASE), "umol"),
)

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


def repair_mojibake(text: str) -> str:
    """Repair common UTF-8 text decoded as cp1252/latin1."""
    if not text:
        return ""

    candidates = [text]
    current = text
    for _ in range(2):
        for encoding in ("cp1252", "latin1"):
            try:
                repaired = current.encode(encoding).decode("utf-8")
            except UnicodeError:
                continue
            if repaired not in candidates:
                candidates.append(repaired)
        current = candidates[-1]

    if ftfy_fix_text is not None:
        try:
            fixed = ftfy_fix_text(text)
            if fixed not in candidates:
                candidates.append(fixed)
        except Exception:
            pass

    def score(value: str) -> int:
        marker_penalty = sum(value.count(marker) for marker in MOJIBAKE_MARKERS) * 80
        replacement_penalty = value.count("\ufffd") * 200
        vietnamese_bonus = sum(1 for ch in value if ch in VIETNAMESE_CHARS) * 3
        ascii_bonus = sum(1 for ch in value if ch.isascii()) // 20
        return vietnamese_bonus + ascii_bonus - marker_penalty - replacement_penalty

    return max(candidates, key=score)


def normalize_text(text: str, *, preserve_lines: bool = True) -> str:
    """Normalize PDF text while preserving medical/scientific meaning."""
    if not text:
        return ""

    text = repair_mojibake(text)
    text = unicodedata.normalize("NFC", text)
    for source, replacement in CID_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = text.translate(CHAR_TRANSLATION)
    text = text.translate(SUPERSCRIPT_TRANSLATION)
    text = text.translate(SUBSCRIPT_TRANSLATION)

    for pattern, replacement in MEDICAL_UNIT_PATTERNS:
        text = pattern.sub(replacement, text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?<=\w)-\n(?=\w)", "", text, flags=re.UNICODE)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)

    if not preserve_lines:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def normalize_for_search(text: str) -> str:
    text = normalize_text(text, preserve_lines=False).lower()
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def stable_id(*parts: Any) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def safe_filename(value: str) -> str:
    value = normalize_for_search(value)
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("._") or "document"


def bbox_to_list(bbox: Optional[BBox]) -> Optional[List[float]]:
    if bbox is None:
        return None
    return [round(float(v), 2) for v in bbox]


def bbox_intersects(a: Optional[BBox], b: Optional[BBox]) -> bool:
    if a is None or b is None:
        return False
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    return ax0 < bx1 and ax1 > bx0 and ay0 < by1 and ay1 > by0


def rect_area(rect: fitz.Rect) -> float:
    return max(0.0, rect.width) * max(0.0, rect.height)


def clamp_bbox(bbox: BBox, page_rect: fitz.Rect, margin: float = 4.0) -> fitz.Rect:
    x0, y0, x1, y1 = bbox
    return fitz.Rect(
        max(page_rect.x0, x0 - margin),
        max(page_rect.y0, y0 - margin),
        min(page_rect.x1, x1 + margin),
        min(page_rect.y1, y1 + margin),
    )


def render_clip(page: fitz.Page, bbox: BBox, output_path: Path, *, dpi: int = 144) -> str:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    clip = clamp_bbox(bbox, page.rect)
    matrix = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=matrix, clip=clip, alpha=False)
    pix.save(str(output_path))
    del pix
    return output_path.as_posix()


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
    return tuple(float(value) for value in bbox)  # type: ignore[return-value]


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


def split_text_blocks_into_semantic_chunks(
    blocks: Sequence[Dict[str, Any]],
    *,
    max_chars: int,
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
    current_headings: List[str] = []
    previous_bbox: Optional[BBox] = None

    def flush_section() -> None:
        nonlocal current_lines, current_headings
        section_text = normalize_text("\n".join(current_lines), preserve_lines=True)
        if section_text:
            sections.append(
                {
                    "section_index": len(sections) + 1,
                    "headings": list(current_headings),
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
                current_headings = list(heading_path)
            else:
                current_lines.append(line)
    flush_section()

    chunks: List[Dict[str, Any]] = []
    for section in sections:
        headings = section["headings"]
        heading_context = " > ".join(headings)
        for paragraph_index, unit in enumerate(split_section_text_into_units(section["text"], max_chars), start=1):
            content = f"{heading_context}\n\n{unit}" if heading_context else unit
            chunks.append(
                {
                    "content": content,
                    "headings": headings,
                    "section_index": section["section_index"],
                    "paragraph_index": paragraph_index,
                    "semantic_type": semantic_bucket_for_text(unit, headings),
                }
            )

    if chunks:
        return chunks

    fallback_text = page_text_from_blocks(blocks)
    return [
        {
            "content": chunk,
            "headings": [],
            "section_index": index,
            "paragraph_index": 1,
            "semantic_type": semantic_bucket_for_text(chunk),
        }
        for index, chunk in enumerate(split_text_into_chunks(fallback_text, max_chars=max_chars, overlap_words=0), start=1)
    ]


def clean_table_rows(rows: Sequence[Sequence[Any]]) -> List[List[str]]:
    cleaned: List[List[str]] = []
    max_cols = max((len(row) for row in rows), default=0)
    for row in rows:
        cells = [normalize_text("" if cell is None else str(cell), preserve_lines=False) for cell in row]
        cells.extend([""] * (max_cols - len(cells)))
        if any(cell for cell in cells):
            cleaned.append(cells)

    if not cleaned:
        return []

    keep_cols = [
        index
        for index in range(max(len(row) for row in cleaned))
        if any(index < len(row) and row[index].strip() for row in cleaned)
    ]
    return [[row[index] if index < len(row) else "" for index in keep_cols] for row in cleaned]


def looks_like_table(
    rows: Sequence[Sequence[str]],
    *,
    strategy: str = "unknown",
    bbox: Optional[BBox] = None,
    page_width: Optional[float] = None,
) -> bool:
    if len(rows) < 2:
        return False
    col_count = max((len(row) for row in rows), default=0)
    if col_count < 2:
        return False
    cell_count = len(rows) * col_count
    non_empty = sum(1 for row in rows for cell in row if str(cell).strip())
    text_chars = sum(len(str(cell).strip()) for row in rows for cell in row)
    if non_empty / max(cell_count, 1) < 0.20 or text_chars < 12:
        return False
    if non_empty < 4:
        return False

    non_empty_cells = [str(cell).strip() for row in rows for cell in row if str(cell).strip()]
    short_alpha_cells = [
        cell
        for cell in non_empty_cells
        if len(cell) <= 4 and re.fullmatch(r"[A-Za-zÀ-ỴƯưĐđ.]+", cell, flags=re.UNICODE)
    ]
    fragmented_rows = 0
    for row in rows:
        row_cells = [str(cell).strip() for cell in row if str(cell).strip()]
        if not row_cells:
            continue
        row_short = [
            cell
            for cell in row_cells
            if len(cell) <= 4 and re.fullmatch(r"[A-Za-zÀ-ỴƯưĐđ.]+", cell, flags=re.UNICODE)
        ]
        if len(row_short) / len(row_cells) >= 0.60:
            fragmented_rows += 1

    fragmented_cell_ratio = len(short_alpha_cells) / max(len(non_empty_cells), 1)
    fragmented_row_ratio = fragmented_rows / max(len(rows), 1)
    if strategy == "text" and col_count >= 3 and fragmented_cell_ratio >= 0.50 and fragmented_row_ratio >= 0.35:
        return False

    if strategy == "text":
        lower_start_cells = [
            cell
            for cell in non_empty_cells
            if cell
            and cell.lower() not in {"tang", "giam", "degree"}
            and cell[0].islower()
            and any(ch.isalpha() for ch in cell)
        ]
        lower_start_ratio = len(lower_start_cells) / max(len(non_empty_cells), 1)
        if lower_start_ratio >= 0.22:
            return False
        if bbox is not None and page_width:
            table_width = max(0.0, bbox[2] - bbox[0])
            if col_count >= 3 and table_width < page_width * 0.35:
                return False
        joined = " ".join(non_empty_cells)
        dot_count = joined.count(".")
        if joined.count("...") >= 3 or dot_count / max(len(joined), 1) >= 0.08:
            return False
        rows_with_multiple_columns = sum(
            1 for row in rows if sum(1 for cell in row if str(cell).strip()) >= 2
        )
        if rows_with_multiple_columns < 3:
            return False
        if fragmented_cell_ratio >= 0.40 and fragmented_row_ratio >= 0.25:
            return False

    return True


def table_signature(rows: Sequence[Sequence[str]]) -> str:
    normalized = "\n".join("\t".join(normalize_for_search(cell) for cell in row) for row in rows)
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def rows_to_markdown(rows: Sequence[Sequence[str]]) -> str:
    if not rows:
        return ""

    width = max(len(row) for row in rows)
    padded = [list(row) + [""] * (width - len(row)) for row in rows]
    header = padded[0]
    body = padded[1:] or [[""] * width]

    def escape(cell: str) -> str:
        return str(cell).replace("\n", "<br>").replace("|", "\\|").strip()

    lines = [
        "| " + " | ".join(escape(cell) for cell in header) + " |",
        "| " + " | ".join("---" for _ in range(width)) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(escape(cell) for cell in row) + " |")
    return "\n".join(lines)


def rows_to_csv(rows: Sequence[Sequence[str]]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerows(rows)
    return buffer.getvalue()


def write_text(path: Path, content: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path.as_posix()


def extract_tables(page: pdfplumber.page.Page) -> List[Dict[str, Any]]:
    table_settings = [
        (
            "lines",
            {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "snap_tolerance": 3,
            "join_tolerance": 3,
            "edge_min_length": 3,
            "intersection_tolerance": 3,
            "text_tolerance": 2,
            },
        ),
        (
            "text",
            {
            "vertical_strategy": "text",
            "horizontal_strategy": "text",
            "snap_tolerance": 4,
            "join_tolerance": 4,
            "intersection_tolerance": 5,
            "text_tolerance": 2,
            "min_words_vertical": 2,
            "min_words_horizontal": 1,
            },
        ),
    ]

    tables: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for strategy, settings in table_settings:
        try:
            found = page.find_tables(table_settings=settings) or []
        except Exception as exc:
            LOGGER.debug("Table detection failed on page %s: %s", page.page_number, exc)
            continue

        for table in found:
            try:
                rows = clean_table_rows(table.extract() or [])
            except Exception:
                continue
            bbox = tuple(float(v) for v in table.bbox)
            if not looks_like_table(rows, strategy=strategy, bbox=bbox, page_width=float(page.width)):
                continue
            signature = table_signature(rows)
            if signature in seen:
                continue
            seen.add(signature)
            tables.append(
                {
                    "rows": rows,
                    "bbox": bbox,
                    "signature": signature,
                    "extractor": f"pdfplumber:{strategy}",
                }
            )

    return tables


def detect_visual_regions(page: fitz.Page, table_bboxes: Sequence[BBox]) -> List[Dict[str, Any]]:
    visual_regions: List[Dict[str, Any]] = []
    page_area = rect_area(page.rect)
    image_blocks = [block for block in page.get_text("dict").get("blocks", []) if block.get("type") == 1]

    for block in image_blocks:
        bbox = tuple(float(v) for v in block.get("bbox", (0, 0, 0, 0)))  # type: ignore[assignment]
        if rect_area(fitz.Rect(bbox)) < page_area * 0.01:
            continue
        visual_regions.append(
            {
                "visual_index": len(visual_regions) + 1,
                "kind": "image",
                "bbox": bbox,
                "reason": "embedded_image",
            }
        )

    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []

    drawing_rects = [
        fitz.Rect(drawing.get("rect"))
        for drawing in drawings
        if drawing.get("rect") is not None and rect_area(fitz.Rect(drawing.get("rect"))) > 4
    ]
    drawing_rects = [
        rect
        for rect in drawing_rects
        if not any(bbox_intersects((rect.x0, rect.y0, rect.x1, rect.y1), table_bbox) for table_bbox in table_bboxes)
    ]

    if len(drawing_rects) >= 5:
        union = fitz.Rect(drawing_rects[0])
        for rect in drawing_rects[1:]:
            union.include_rect(rect)
        if rect_area(union) >= page_area * 0.02:
            visual_regions.append(
                {
                    "visual_index": len(visual_regions) + 1,
                    "kind": "diagram",
                    "bbox": (union.x0, union.y0, union.x1, union.y1),
                    "reason": f"{len(drawing_rects)} vector drawing elements",
                }
            )

    return visual_regions


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


def split_text_into_chunks(text: str, *, max_chars: int = 1400, overlap_words: int = 45) -> List[str]:
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


def build_page_markdown(
    *,
    doc_name: str,
    page_number: int,
    text: str,
    tables: Sequence[Dict[str, Any]],
    visual_regions: Sequence[Dict[str, Any]],
) -> str:
    parts = [f"# {doc_name} - page {page_number}", ""]
    if text:
        parts.extend(["## Text", "", text, ""])
    for table in tables:
        parts.extend(
            [
                f"## Table {table['table_index']}",
                "",
                table["markdown"],
                "",
                f"- csv: {table.get('csv_path', '')}",
                f"- image: {table.get('asset_path', '')}",
                "",
            ]
        )
    for visual in visual_regions:
        parts.extend(
            [
                f"## Visual {visual['visual_index']}",
                "",
                f"- kind: {visual['kind']}",
                f"- reason: {visual['reason']}",
                f"- image: {visual.get('asset_path', '')}",
                f"- bbox: {bbox_to_list(visual['bbox'])}",
                "",
            ]
        )
    return "\n".join(parts).strip() + "\n"


def iter_input_pdfs(input_path: Path) -> Iterator[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() == ".pdf":
            yield input_path
        return
    yield from sorted(input_path.rglob("*.pdf"))


def process_pdf(
    pdf_path: Path,
    *,
    output_dir: Path,
    max_chars: int = 1400,
    overlap_words: int = 45,
    render_assets: bool = True,
    dpi: int = 144,
    max_pages: Optional[int] = None,
) -> Dict[str, Any]:
    started_at = time.time()
    doc_key = safe_filename(pdf_path.stem)
    doc_output_dir = output_dir / doc_key
    markdown_dir = doc_output_dir / "markdown_pages"
    table_dir = doc_output_dir / "tables"
    asset_dir = doc_output_dir / "assets"
    jsonl_path = doc_output_dir / "chunks.jsonl"
    manifest_path = doc_output_dir / "manifest.json"

    doc_output_dir.mkdir(parents=True, exist_ok=True)
    markdown_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    asset_dir.mkdir(parents=True, exist_ok=True)

    chunk_count = 0
    table_count = 0
    visual_count = 0
    pages_processed = 0
    page_summaries: List[Dict[str, Any]] = []

    LOGGER.info("Processing %s", pdf_path)
    with fitz.open(str(pdf_path)) as fitz_doc, pdfplumber.open(str(pdf_path)) as plumber_doc:
        total_pages = len(fitz_doc)
        limit = min(total_pages, max_pages) if max_pages else total_pages

        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            for page_index in range(limit):
                page_number = page_index + 1
                fitz_page = fitz_doc[page_index]
                plumber_page = plumber_doc.pages[page_index]

                text_blocks = extract_text_blocks(fitz_page)
                tables = extract_tables(plumber_page)

                table_bboxes: List[BBox] = []
                for table_index, table in enumerate(tables, start=1):
                    table["table_index"] = table_index
                    table_count += 1
                    table_bboxes.append(table["bbox"])
                    table["markdown"] = rows_to_markdown(table["rows"])
                    table["csv"] = rows_to_csv(table["rows"])

                    base_name = f"page_{page_number:03d}_table_{table_index:02d}"
                    table["csv_path"] = write_text(table_dir / f"{base_name}.csv", table["csv"])
                    table["json_path"] = write_text(
                        table_dir / f"{base_name}.json",
                        json.dumps(
                            {
                                "doc_name": pdf_path.name,
                                "page_number": page_number,
                                "table_index": table_index,
                                "bbox": bbox_to_list(table["bbox"]),
                                "rows": table["rows"],
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                    )
                    if render_assets:
                        table["asset_path"] = render_clip(
                            fitz_page,
                            table["bbox"],
                            asset_dir / f"{base_name}.png",
                            dpi=dpi,
                        )

                text_blocks_for_chunks = exclude_text_blocks_by_bboxes(text_blocks, table_bboxes)
                page_text = page_text_from_blocks(text_blocks_for_chunks)
                headings = current_headings(page_text)
                visual_regions = detect_visual_regions(fitz_page, table_bboxes)
                for visual in visual_regions:
                    visual_count += 1
                    if render_assets:
                        visual["asset_path"] = render_clip(
                            fitz_page,
                            visual["bbox"],
                            asset_dir / f"page_{page_number:03d}_visual_{visual['visual_index']:02d}.png",
                            dpi=dpi,
                        )

                page_markdown = build_page_markdown(
                    doc_name=pdf_path.name,
                    page_number=page_number,
                    text=page_text,
                    tables=tables,
                    visual_regions=visual_regions,
                )
                page_markdown_path = write_text(markdown_dir / f"page_{page_number:03d}.md", page_markdown)

                page_chunks = split_text_blocks_into_semantic_chunks(
                    text_blocks_for_chunks,
                    max_chars=max_chars,
                )
                for chunk_index, chunk_payload in enumerate(page_chunks, start=1):
                    chunk = make_chunk(
                        doc_name=pdf_path.name,
                        page_number=page_number,
                        chunk_type="text",
                        chunk_index=chunk_index,
                        content=chunk_payload["content"],
                        metadata={
                            "headings": chunk_payload["headings"],
                            "page_headings": headings,
                            "section_index": chunk_payload["section_index"],
                            "paragraph_index": chunk_payload["paragraph_index"],
                            "semantic_type": chunk_payload["semantic_type"],
                            "max_chars": max_chars,
                            "overlap_words": 0,
                            "text_block_count": len(text_blocks),
                            "text_blocks_after_table_filter": len(text_blocks_for_chunks),
                            "page_markdown_path": page_markdown_path,
                        },
                        reconstruction={
                            "page_markdown_path": page_markdown_path,
                            "text_blocks": text_blocks_for_chunks,
                        },
                    )
                    jsonl_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1

                for table in tables:
                    chunk = make_chunk(
                        doc_name=pdf_path.name,
                        page_number=page_number,
                        chunk_type="table",
                        chunk_index=table["table_index"],
                        content=table["markdown"],
                        metadata={
                            "rows": len(table["rows"]),
                            "cols": max((len(row) for row in table["rows"]), default=0),
                            "bbox": bbox_to_list(table["bbox"]),
                            "extractor": table["extractor"],
                            "headings": headings,
                        },
                        reconstruction={
                            "rows": table["rows"],
                            "csv_path": table["csv_path"],
                            "json_path": table["json_path"],
                            "asset_path": table.get("asset_path"),
                            "page_markdown_path": page_markdown_path,
                            "bbox": bbox_to_list(table["bbox"]),
                        },
                    )
                    jsonl_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1

                for visual in visual_regions:
                    nearby_text = nearest_text_for_bbox(text_blocks, visual["bbox"])
                    content = normalize_text(
                        f"[{visual['kind'].upper()}] {visual['reason']}\n{nearby_text}",
                        preserve_lines=True,
                    )
                    chunk = make_chunk(
                        doc_name=pdf_path.name,
                        page_number=page_number,
                        chunk_type="visual",
                        chunk_index=visual["visual_index"],
                        content=content,
                        metadata={
                            "kind": visual["kind"],
                            "reason": visual["reason"],
                            "bbox": bbox_to_list(visual["bbox"]),
                            "headings": headings,
                        },
                        reconstruction={
                            "asset_path": visual.get("asset_path"),
                            "page_markdown_path": page_markdown_path,
                            "bbox": bbox_to_list(visual["bbox"]),
                        },
                    )
                    jsonl_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")
                    chunk_count += 1

                pages_processed += 1
                page_summaries.append(
                    {
                        "page_number": page_number,
                        "text_chars": len(page_text),
                        "tables": len(tables),
                        "visual_regions": len(visual_regions),
                        "chunks_so_far": chunk_count,
                    }
                )

                if page_number % 10 == 0 or page_number == limit:
                    LOGGER.info(
                        "Processed page %s/%s: chunks=%s tables=%s visuals=%s",
                        page_number,
                        limit,
                        chunk_count,
                        table_count,
                        visual_count,
                    )
                if page_number % 25 == 0:
                    gc.collect()

    summary = {
        "doc_name": pdf_path.name,
        "pdf_path": pdf_path.as_posix(),
        "output_dir": doc_output_dir.as_posix(),
        "jsonl_path": jsonl_path.as_posix(),
        "pages_total": pages_processed,
        "chunks": chunk_count,
        "tables": table_count,
        "visual_regions": visual_count,
        "max_chars": max_chars,
        "overlap_words": overlap_words,
        "render_assets": render_assets,
        "elapsed_seconds": round(time.time() - started_at, 2),
        "pages": page_summaries,
    }
    write_text(manifest_path, json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


def process_input(
    input_path: Path,
    *,
    output_dir: Path,
    max_chars: int,
    overlap_words: int,
    render_assets: bool,
    dpi: int,
    max_pages: Optional[int],
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    pdfs = list(iter_input_pdfs(input_path))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found at {input_path}")

    for pdf_path in pdfs:
        summaries.append(
            process_pdf(
                pdf_path,
                output_dir=output_dir,
                max_chars=max_chars,
                overlap_words=overlap_words,
                render_assets=render_assets,
                dpi=dpi,
                max_pages=max_pages,
            )
        )
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fast PDF preprocessing for Vietnamese medical IR datasets.")
    parser.add_argument("--input", type=Path, default=Path("data/raw"), help="PDF file or directory. Default: data/raw")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/preprocess_output"),
        help="Output directory. Default: data/processed/preprocess_output",
    )
    parser.add_argument("--max-chars", type=int, default=1400, help="Max characters per text chunk.")
    parser.add_argument("--overlap-words", type=int, default=45, help="Word overlap between text chunks.")
    parser.add_argument("--max-pages", type=int, default=None, help="Process only the first N pages for testing.")
    parser.add_argument("--dpi", type=int, default=144, help="DPI for table/visual region images.")
    parser.add_argument("--no-render-assets", action="store_true", help="Skip PNG rendering for tables/visuals.")
    parser.add_argument("--verbose", action="store_true", help="Show debug logs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    summaries = process_input(
        args.input,
        output_dir=args.output,
        max_chars=args.max_chars,
        overlap_words=args.overlap_words,
        render_assets=not args.no_render_assets,
        dpi=args.dpi,
        max_pages=args.max_pages,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
