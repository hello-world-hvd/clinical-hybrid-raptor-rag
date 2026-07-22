from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    from ..openrouter_client import (
        DEFAULT_MAX_REQUESTS_PER_RUN,
        DEFAULT_MIN_INTERVAL_SECONDS,
        DEFAULT_OPENROUTER_MODEL,
    )
    from .contextual_chunking import (
        DEFAULT_CACHE_PATH,
        DEFAULT_CONTEXT_BATCH_SIZE,
    )
    from .document import process_input
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
    )
    from document import process_input


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess Vietnamese medical PDFs with contextual chunking."
    )
    parser.add_argument("--input", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/preprocess_output"),
    )
    parser.add_argument("--max-chars", type=int, default=1200)
    parser.add_argument("--overlap-words", type=int, default=60)
    parser.add_argument("--openrouter-model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--context-cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--context-timeout", type=float, default=90.0)
    parser.add_argument(
        "--context-batch-size",
        type=int,
        default=DEFAULT_CONTEXT_BATCH_SIZE,
        help="Maximum uncached chunks sent in one OpenRouter request.",
    )
    parser.add_argument(
        "--openrouter-min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL_SECONDS,
        help="Minimum seconds between OpenRouter requests.",
    )
    parser.add_argument(
        "--openrouter-max-requests",
        type=int,
        default=DEFAULT_MAX_REQUESTS_PER_RUN,
        help="Maximum OpenRouter HTTP attempts for this run; use 0 for unlimited.",
    )
    parser.add_argument(
        "--strict-context",
        action="store_true",
        help="Stop preprocessing instead of using deterministic context fallback.",
    )
    parser.add_argument("--max-pages", type=int, default=None)
    parser.add_argument("--dpi", type=int, default=144)
    parser.add_argument("--no-render-assets", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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
        openrouter_model=args.openrouter_model,
        context_cache_path=args.context_cache,
        context_timeout=args.context_timeout,
        context_batch_size=args.context_batch_size,
        openrouter_min_interval=max(0.0, args.openrouter_min_interval),
        openrouter_max_requests=args.openrouter_max_requests,
        context_fail_open=not args.strict_context,
        show_progress=not args.no_progress,
    )
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
