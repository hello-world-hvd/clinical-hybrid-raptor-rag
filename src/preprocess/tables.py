from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pdfplumber

try:
    from .geometry import BBox
    from .normalization import normalize_for_search, normalize_text
except ImportError:
    from geometry import BBox
    from normalization import normalize_for_search, normalize_text

LOGGER = logging.getLogger("preprocess")


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
