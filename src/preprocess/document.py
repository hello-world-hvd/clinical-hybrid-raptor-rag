from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import fitz
import pdfplumber
from tqdm.auto import tqdm

try:
    from ..openrouter_client import (
        DEFAULT_MAX_REQUESTS_PER_RUN,
        DEFAULT_MIN_INTERVAL_SECONDS,
        DEFAULT_OPENROUTER_MODEL,
    )
    from .contextual_chunking import (
        DEFAULT_CACHE_PATH,
        DEFAULT_CONTEXT_BATCH_SIZE,
        ContextualChunker,
    )
    from .geometry import BBox, bbox_to_list, render_clip
    from .normalization import normalize_text, safe_filename
    from .tables import extract_tables, rows_to_csv, rows_to_markdown, write_text
    from .text import (
        current_headings,
        exclude_text_blocks_by_bboxes,
        extract_text_blocks,
        make_chunk,
        nearest_text_for_bbox,
        page_text_from_blocks,
        split_text_blocks_into_chunks,
    )
    from .visuals import detect_visual_regions
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openrouter_client import (
        DEFAULT_MAX_REQUESTS_PER_RUN,
        DEFAULT_MIN_INTERVAL_SECONDS,
        DEFAULT_OPENROUTER_MODEL,
    )
    from contextual_chunking import (
        DEFAULT_CACHE_PATH,
        DEFAULT_CONTEXT_BATCH_SIZE,
        ContextualChunker,
    )
    from geometry import BBox, bbox_to_list, render_clip
    from normalization import normalize_text, safe_filename
    from tables import extract_tables, rows_to_csv, rows_to_markdown, write_text
    from text import (
        current_headings,
        exclude_text_blocks_by_bboxes,
        extract_text_blocks,
        make_chunk,
        nearest_text_for_bbox,
        page_text_from_blocks,
        split_text_blocks_into_chunks,
    )
    from visuals import detect_visual_regions

LOGGER = logging.getLogger("preprocess")


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


def _write_contextual_text_batch(
    *,
    pending: List[Dict[str, Any]],
    batch_size: int,
    contextual_chunker: ContextualChunker,
    jsonl_file: Any,
) -> int:
    batch = pending[:batch_size]
    if not batch:
        return 0

    enriched_chunks = contextual_chunker.enrich_batch(
        [item["context_input"] for item in batch],
        batch_size=batch_size,
    )
    for item, enriched in zip(batch, enriched_chunks):
        metadata = dict(item["metadata"])
        metadata["context"] = enriched["context"]
        chunk = make_chunk(
            doc_name=item["doc_name"],
            page_number=item["page_number"],
            chunk_type="text",
            chunk_index=item["chunk_index"],
            content=enriched["content"],
            metadata=metadata,
            reconstruction=item["reconstruction"],
        )
        jsonl_file.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    del pending[: len(batch)]
    contextual_chunker.flush()
    return len(batch)


def process_pdf(
    pdf_path: Path,
    *,
    output_dir: Path,
    max_chars: int = 1400,
    overlap_words: int = 45,
    render_assets: bool = True,
    dpi: int = 144,
    max_pages: Optional[int] = None,
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    context_cache_path: Path = DEFAULT_CACHE_PATH,
    context_timeout: float = 90.0,
    openrouter_min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
    openrouter_max_requests: int | None = DEFAULT_MAX_REQUESTS_PER_RUN,
    context_batch_size: int = DEFAULT_CONTEXT_BATCH_SIZE,
    context_fail_open: bool = True,
    show_progress: bool = True,
) -> Dict[str, Any]:
    if context_batch_size <= 0:
        raise ValueError("context_batch_size must be positive")

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
    contextual_chunker = ContextualChunker(
        model=openrouter_model,
        cache_path=context_cache_path,
        timeout=context_timeout,
        min_interval_seconds=openrouter_min_interval,
        max_requests_per_run=openrouter_max_requests,
        autosave=False,
        fail_open=context_fail_open,
    )

    LOGGER.info("Processing %s", pdf_path)
    with fitz.open(str(pdf_path)) as fitz_doc, pdfplumber.open(str(pdf_path)) as plumber_doc:
        total_pages = len(fitz_doc)
        limit = min(total_pages, max_pages) if max_pages else total_pages

        with jsonl_path.open("w", encoding="utf-8") as jsonl_file:
            pending_text_chunks: List[Dict[str, Any]] = []
            context_progress = tqdm(
                total=0,
                desc=f"{pdf_path.name} contexts",
                unit="chunk",
                disable=not show_progress,
            )
            page_progress = tqdm(
                range(limit),
                desc=pdf_path.name,
                unit="page",
                disable=not show_progress,
            )
            for page_index in page_progress:
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

                page_chunks = split_text_blocks_into_chunks(
                    text_blocks_for_chunks,
                    max_chars=max_chars,
                    overlap_words=overlap_words,
                )
                context_progress.total += len(page_chunks)
                context_progress.refresh()
                for chunk_index, chunk_payload in enumerate(page_chunks, start=1):
                    pending_text_chunks.append(
                        {
                            "doc_name": pdf_path.name,
                            "page_number": page_number,
                            "chunk_index": chunk_index,
                            "context_input": {
                                "content": chunk_payload["content"],
                                "doc_name": pdf_path.name,
                                "page_number": page_number,
                                "headings": chunk_payload["headings"],
                                "document_context": page_text,
                            },
                            "metadata": {
                                "headings": chunk_payload["headings"],
                                "page_headings": headings,
                                "section_index": chunk_payload["section_index"],
                                "paragraph_index": chunk_payload["paragraph_index"],
                                "semantic_type": chunk_payload["semantic_type"],
                                "chunking_strategy": "contextual",
                                "context_model": openrouter_model,
                                "context_batch_size": context_batch_size,
                                "max_chars": max_chars,
                                "overlap_words": overlap_words,
                                "text_block_count": len(text_blocks),
                                "text_blocks_after_table_filter": len(
                                    text_blocks_for_chunks
                                ),
                                "page_markdown_path": page_markdown_path,
                            },
                            "reconstruction": {
                                "page_markdown_path": page_markdown_path,
                                "text_blocks": text_blocks_for_chunks,
                            },
                        }
                    )

                    if len(pending_text_chunks) >= context_batch_size:
                        written = _write_contextual_text_batch(
                            pending=pending_text_chunks,
                            batch_size=context_batch_size,
                            contextual_chunker=contextual_chunker,
                            jsonl_file=jsonl_file,
                        )
                        chunk_count += written
                        context_progress.update(written)
                    context_stats = contextual_chunker.stats
                    context_progress.set_postfix(
                        batch=context_batch_size,
                        pending=len(pending_text_chunks),
                        api=context_stats["api_requests"],
                        cache=context_stats["cache_hits"],
                        fallback=context_stats["fallback_contexts"],
                    )

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
                context_stats = contextual_chunker.stats
                page_progress.set_postfix(
                    chunks=chunk_count + len(pending_text_chunks),
                    pending=len(pending_text_chunks),
                    tables=table_count,
                    visuals=visual_count,
                    api=context_stats["api_requests"],
                    cache=context_stats["cache_hits"],
                    fallback=context_stats["fallback_contexts"],
                )
                page_summaries.append(
                    {
                        "page_number": page_number,
                        "text_chars": len(page_text),
                        "tables": len(tables),
                        "visual_regions": len(visual_regions),
                        "chunks_so_far": chunk_count + len(pending_text_chunks),
                    }
                )

                if page_number % 10 == 0 or page_number == limit:
                    LOGGER.info(
                        "Processed page %s/%s: chunks=%s tables=%s visuals=%s",
                        page_number,
                        limit,
                        chunk_count + len(pending_text_chunks),
                        table_count,
                        visual_count,
                    )
                if page_number % 25 == 0:
                    gc.collect()

            while pending_text_chunks:
                written = _write_contextual_text_batch(
                    pending=pending_text_chunks,
                    batch_size=context_batch_size,
                    contextual_chunker=contextual_chunker,
                    jsonl_file=jsonl_file,
                )
                chunk_count += written
                context_progress.update(written)
                context_stats = contextual_chunker.stats
                context_progress.set_postfix(
                    batch=context_batch_size,
                    pending=len(pending_text_chunks),
                    api=context_stats["api_requests"],
                    cache=context_stats["cache_hits"],
                    fallback=context_stats["fallback_contexts"],
                )
            context_progress.close()
            contextual_chunker.flush()

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
        "chunking_strategy": "contextual",
        "context_model": openrouter_model,
        "context_cache_path": context_cache_path.as_posix(),
        "context_stats": contextual_chunker.stats,
        "context_batch_size": context_batch_size,
        "openrouter_min_interval_seconds": openrouter_min_interval,
        "openrouter_max_requests_per_run": openrouter_max_requests,
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
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL,
    context_cache_path: Path = DEFAULT_CACHE_PATH,
    context_timeout: float = 90.0,
    openrouter_min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS,
    openrouter_max_requests: int | None = DEFAULT_MAX_REQUESTS_PER_RUN,
    context_batch_size: int = DEFAULT_CONTEXT_BATCH_SIZE,
    context_fail_open: bool = True,
    show_progress: bool = True,
) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    pdfs = list(iter_input_pdfs(input_path))
    if not pdfs:
        raise FileNotFoundError(f"No PDF files found at {input_path}")

    pdf_progress = tqdm(
        pdfs,
        desc="Preprocessing PDFs",
        unit="pdf",
        disable=not show_progress,
    )
    for pdf_path in pdf_progress:
        pdf_progress.set_postfix(file=pdf_path.name)
        summaries.append(
            process_pdf(
                pdf_path,
                output_dir=output_dir,
                max_chars=max_chars,
                overlap_words=overlap_words,
                render_assets=render_assets,
                dpi=dpi,
                max_pages=max_pages,
                openrouter_model=openrouter_model,
                context_cache_path=context_cache_path,
                context_timeout=context_timeout,
                openrouter_min_interval=openrouter_min_interval,
                openrouter_max_requests=openrouter_max_requests,
                context_batch_size=context_batch_size,
                context_fail_open=context_fail_open,
                show_progress=show_progress,
            )
        )
    return summaries
