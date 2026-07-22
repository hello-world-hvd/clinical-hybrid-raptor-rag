from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from tqdm import tqdm

from src.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterClient,
    OpenRouterRequestBudgetExceeded,
)
from src.preprocess.normalization import normalize_text, stable_id


LOGGER = logging.getLogger("graph_rag.entity_extractor")
DEFAULT_CHUNKS_PATH = Path(
    "data/processed/preprocess_output/ngo-doc/chunks.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/processed/graph_rag/ngo-doc")
MEDICAL_ENTITY_TYPES = (
    "POISON",
    "DISEASE",
    "SYMPTOM",
    "DRUG",
    "TOXIN",
    "LAB_TEST",
    "PROCEDURE",
    "DOSAGE",
    "THRESHOLD",
    "ORGAN_SYSTEM",
    "CONTRAINDICATION",
    "RISK_FACTOR",
    "TIME",
    "OTHER",
)


@dataclass(frozen=True)
class ExtractionConfig:
    chunks_path: Path = DEFAULT_CHUNKS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    model: str = DEFAULT_OPENROUTER_MODEL
    batch_size: int = 20
    max_chunk_chars: int = 1800
    max_tokens: int = 8000
    parse_retries: int = 2
    request_timeout_seconds: float = 90.0
    http_retries: int = 2
    min_interval_seconds: float = 3.0
    max_requests_per_run: int | None = 0
    include_visuals: bool = True
    force: bool = False

    @property
    def cache_path(self) -> Path:
        return self.output_dir / "entity_extraction_cache.json"

    @property
    def output_path(self) -> Path:
        return self.output_dir / "extractions.jsonl"

    @property
    def manifest_path(self) -> Path:
        return self.output_dir / "extraction_manifest.json"


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _strip_failed_context_prefix(text: str) -> str:
    marker = "\n\n"
    if not text.startswith("["):
        return text
    prefix, separator, remainder = text.partition(marker)
    if separator and (
        "trang " in prefix.casefold()
        or "tài liệu" in prefix.casefold()
        or "tai lieu" in prefix.casefold()
    ):
        return remainder
    return text


def load_source_chunks(config: ExtractionConfig) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for raw in read_jsonl(config.chunks_path):
        chunk_type = str(raw.get("chunk_type") or "text")
        if chunk_type == "visual" and not config.include_visuals:
            continue
        content = normalize_text(
            _strip_failed_context_prefix(str(raw.get("content") or "")),
            preserve_lines=True,
        )
        if len(content.strip()) < 12:
            continue
        chunks.append(
            {
                "chunk_id": str(raw.get("id") or stable_id(content)),
                "doc_name": str(raw.get("doc_name") or ""),
                "page_number": int(raw.get("page_number") or 0),
                "chunk_type": chunk_type,
                "content": content[: config.max_chunk_chars],
            }
        )
    return chunks


def _extract_balanced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("LLM response does not contain a JSON object")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise ValueError("LLM response contains an incomplete JSON object")


def parse_extraction_response(text: str) -> dict[str, Any]:
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = json.loads(_extract_balanced_object(candidate))
    if not isinstance(payload, dict) or not isinstance(payload.get("chunks"), list):
        raise ValueError("Expected a JSON object with a 'chunks' array")
    return payload


def _clean_entity(raw: Mapping[str, Any]) -> dict[str, str] | None:
    name = normalize_text(str(raw.get("name") or ""), preserve_lines=False)
    if not name:
        return None
    entity_type = str(raw.get("type") or "OTHER").strip().upper()
    if entity_type not in MEDICAL_ENTITY_TYPES:
        entity_type = "OTHER"
    return {
        "name": name,
        "type": entity_type,
        "description": normalize_text(
            str(raw.get("description") or ""), preserve_lines=False
        ),
    }


def _clean_relation(raw: Mapping[str, Any]) -> dict[str, str] | None:
    source = normalize_text(str(raw.get("source") or ""), preserve_lines=False)
    target = normalize_text(str(raw.get("target") or ""), preserve_lines=False)
    if not source or not target or source.casefold() == target.casefold():
        return None
    relation_type = re.sub(
        r"[^A-Z0-9_]+",
        "_",
        str(raw.get("type") or "RELATED_TO").strip().upper(),
    ).strip("_")
    return {
        "source": source,
        "target": target,
        "type": relation_type or "RELATED_TO",
        "description": normalize_text(
            str(raw.get("description") or ""), preserve_lines=False
        ),
    }


def normalize_batch_result(
    payload: Mapping[str, Any],
    batch: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    sources = {str(item["chunk_id"]): item for item in batch}
    normalized: dict[str, dict[str, Any]] = {}
    for raw in payload.get("chunks", []):
        if not isinstance(raw, Mapping):
            continue
        chunk_id = str(raw.get("chunk_id") or "")
        source = sources.get(chunk_id)
        if source is None:
            continue
        entities = [
            cleaned
            for entity in raw.get("entities", [])
            if isinstance(entity, Mapping)
            and (cleaned := _clean_entity(entity)) is not None
        ]
        relations = [
            cleaned
            for relation in raw.get("relations", [])
            if isinstance(relation, Mapping)
            and (cleaned := _clean_relation(relation)) is not None
        ]
        normalized[chunk_id] = {
            **source,
            "entities": entities,
            "relations": relations,
        }
    for chunk_id, source in sources.items():
        normalized.setdefault(
            chunk_id,
            {**source, "entities": [], "relations": []},
        )
    return normalized


class EntityExtractor:
    def __init__(
        self,
        config: ExtractionConfig,
        *,
        client: OpenRouterClient | None = None,
    ) -> None:
        self.config = config
        self.client = client or OpenRouterClient(
            model=config.model,
            timeout=config.request_timeout_seconds,
            max_retries=config.http_retries,
            min_interval_seconds=config.min_interval_seconds,
            max_requests_per_run=config.max_requests_per_run,
        )

    @staticmethod
    def _system_prompt() -> str:
        entity_types = ", ".join(MEDICAL_ENTITY_TYPES)
        return (
            "Bạn trích xuất knowledge graph từ hướng dẫn xử trí ngộ độc tiếng Việt. "
            "Chỉ trả về một JSON object hợp lệ, không markdown, không giải thích. "
            "Không suy diễn ngoài văn bản. Giữ nguyên tên thuốc, liều, đơn vị và "
            "ngưỡng xét nghiệm. Mỗi quan hệ phải nối hai entity xuất hiện trong "
            "cùng chunk. Entity type chỉ thuộc: "
            f"{entity_types}."
        )

    @staticmethod
    def _user_prompt(batch: Sequence[Mapping[str, Any]]) -> str:
        compact = [
            {
                "chunk_id": item["chunk_id"],
                "page": item["page_number"],
                "type": item["chunk_type"],
                "text": item["content"],
            }
            for item in batch
        ]
        schema = {
            "chunks": [
                {
                    "chunk_id": "id chính xác từ input",
                    "entities": [
                        {
                            "name": "tên entity",
                            "type": "một entity type được phép",
                            "description": "mô tả ngắn dựa trên chunk",
                        }
                    ],
                    "relations": [
                        {
                            "source": "tên entity nguồn",
                            "target": "tên entity đích",
                            "type": "QUAN_HE_VIET_HOA_KHONG_DAU",
                            "description": "mệnh đề quan hệ có bằng chứng",
                        }
                    ],
                }
            ]
        }
        return (
            "Trích xuất độc lập cho từng chunk. Phải trả lại đủ mọi chunk_id, kể "
            "cả khi entities/relations là mảng rỗng.\n\n"
            f"OUTPUT_SCHEMA:\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"INPUT_CHUNKS:\n{json.dumps(compact, ensure_ascii=False)}"
        )

    def extract_batch(
        self,
        batch: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(self.config.parse_retries + 1):
            try:
                raw = self.client.complete(
                    system_prompt=self._system_prompt(),
                    user_prompt=self._user_prompt(batch),
                    max_tokens=self.config.max_tokens,
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                return normalize_batch_result(parse_extraction_response(raw), batch)
            except OpenRouterRequestBudgetExceeded:
                raise
            except (TypeError, ValueError) as exc:
                last_error = exc
                LOGGER.warning(
                    "Invalid extraction JSON for batch of %s chunks (%s/%s): %s",
                    len(batch),
                    attempt + 1,
                    self.config.parse_retries + 1,
                    exc,
                )
        raise ValueError("Could not parse entity extraction response") from last_error


def _load_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _write_cache(path: Path, cache: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(cache, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _batches(items: Sequence[dict[str, Any]], size: int) -> Iterable[list[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield list(items[start : start + size])


def run_extraction(
    config: ExtractionConfig,
    *,
    extractor: EntityExtractor | None = None,
) -> dict[str, Any]:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    chunks = load_source_chunks(config)
    cache = {} if config.force else _load_cache(config.cache_path)
    pending = [item for item in chunks if item["chunk_id"] not in cache]
    extractor = extractor or EntityExtractor(config)
    started_at = time.monotonic()
    failed_batches = 0
    disabled_reason: str | None = None

    progress = tqdm(
        total=len(pending),
        desc="Extracting graph entities",
        unit="chunk",
    )
    try:
        for batch in _batches(pending, config.batch_size):
            try:
                extracted = extractor.extract_batch(batch)
            except OpenRouterRequestBudgetExceeded as exc:
                disabled_reason = str(exc)
                LOGGER.warning("Stopping cleanly: %s", exc)
                break
            except Exception as exc:
                failed_batches += 1
                LOGGER.error(
                    "Skipping failed batch (%s chunks); it remains uncached: %s",
                    len(batch),
                    exc,
                )
                progress.update(len(batch))
                continue
            cache.update(extracted)
            _write_cache(config.cache_path, cache)
            progress.update(len(batch))
            progress.set_postfix(
                api=getattr(extractor.client, "request_count", "?"),
                cached=len(cache),
                failed=failed_batches,
            )
    finally:
        progress.close()

    ordered = [cache[item["chunk_id"]] for item in chunks if item["chunk_id"] in cache]
    with config.output_path.open("w", encoding="utf-8") as handle:
        for item in ordered:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "chunks_path": str(config.chunks_path),
        "output_path": str(config.output_path),
        "source_chunks": len(chunks),
        "completed_chunks": len(ordered),
        "pending_chunks": len(chunks) - len(ordered),
        "failed_batches": failed_batches,
        "api_requests": getattr(extractor.client, "request_count", None),
        "stopped_reason": disabled_reason,
        "elapsed_seconds": round(time.monotonic() - started_at, 2),
        "config": {
            **asdict(config),
            "chunks_path": str(config.chunks_path),
            "output_dir": str(config.output_dir),
        },
    }
    config.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> ExtractionConfig:
    parser = argparse.ArgumentParser(
        description="Extract medical entities and relations in batches for Graph RAG."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--max-chunk-chars", type=int, default=1800)
    parser.add_argument("--max-tokens", type=int, default=8000)
    parser.add_argument("--parse-retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=90.0)
    parser.add_argument("--http-retries", type=int, default=2)
    parser.add_argument("--min-interval", type=float, default=3.0)
    parser.add_argument(
        "--max-requests",
        type=int,
        default=0,
        help="0 means unlimited for this run.",
    )
    parser.add_argument("--exclude-visuals", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    return ExtractionConfig(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        model=args.model,
        batch_size=args.batch_size,
        max_chunk_chars=args.max_chunk_chars,
        max_tokens=args.max_tokens,
        parse_retries=max(0, args.parse_retries),
        request_timeout_seconds=max(10.0, args.request_timeout),
        http_retries=max(1, args.http_retries),
        min_interval_seconds=max(0.0, args.min_interval),
        max_requests_per_run=0 if args.max_requests <= 0 else args.max_requests,
        include_visuals=not args.exclude_visuals,
        force=args.force,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    print(json.dumps(run_extraction(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
