from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import requests

try:
    from ..openrouter_client import (
        DEFAULT_MAX_REQUESTS_PER_RUN,
        DEFAULT_MIN_INTERVAL_SECONDS,
        DEFAULT_OPENROUTER_MODEL,
        OpenRouterClient,
        OpenRouterRequestBudgetExceeded,
    )
    from .normalization import normalize_text
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openrouter_client import (
        DEFAULT_MAX_REQUESTS_PER_RUN,
        DEFAULT_MIN_INTERVAL_SECONDS,
        DEFAULT_OPENROUTER_MODEL,
        OpenRouterClient,
        OpenRouterRequestBudgetExceeded,
    )
    from normalization import normalize_text


DEFAULT_CACHE_PATH = Path("data/cache/contextual_chunking/openrouter_contexts.json")
DEFAULT_CONTEXT_BATCH_SIZE = 20
LOGGER = logging.getLogger("preprocess.contextual_chunking")


class ContextualChunker:
    """Add a cached OpenRouter-generated context sentence to a text chunk."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        api_key: Optional[str] = None,
        cache_path: Path = DEFAULT_CACHE_PATH,
        timeout: float = 90.0,
        max_retries: int = 4,
        min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
        max_requests_per_run: int | None = DEFAULT_MAX_REQUESTS_PER_RUN,
        max_context_chars: int = 12000,
        client: Optional[OpenRouterClient] = None,
        session: Optional[requests.Session] = None,
        autosave: bool = True,
        fail_open: bool = True,
    ) -> None:
        self.model = model
        self.cache_path = cache_path
        self.max_context_chars = max_context_chars
        self.autosave = autosave
        self.fail_open = fail_open
        self.client = client or OpenRouterClient(
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            min_interval_seconds=min_interval_seconds,
            max_requests_per_run=max_requests_per_run,
            session=session,
        )
        self._cache = self._load_cache()
        self._cache_dirty = False
        self._api_disabled_reason: str | None = None
        self._cache_hits = 0
        self._generated_contexts = 0
        self._fallback_contexts = 0

    @property
    def stats(self) -> Dict[str, int | str | None]:
        return {
            "cache_hits": self._cache_hits,
            "generated_contexts": self._generated_contexts,
            "fallback_contexts": self._fallback_contexts,
            "api_requests": getattr(self.client, "request_count", 0),
            "api_disabled_reason": self._api_disabled_reason,
        }

    def _load_cache(self) -> Dict[str, str]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_cache(self) -> None:
        if not self._cache_dirty:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        temporary_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.cache_path)
        self._cache_dirty = False

    def flush(self) -> None:
        """Persist newly generated contextualization results to disk."""
        self._save_cache()

    def _cache_key(
        self,
        *,
        content: str,
        doc_name: str,
        page_number: int,
        headings: Sequence[str],
        document_context: str,
    ) -> str:
        payload = {
            "model": self.model,
            "content": content,
            "doc_name": doc_name,
            "page_number": page_number,
            "headings": list(headings),
            "document_context": document_context,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _build_prompt(
        self,
        *,
        content: str,
        doc_name: str,
        page_number: int,
        headings: Sequence[str],
        document_context: str,
    ) -> str:
        heading_text = " > ".join(headings) if headings else "Không xác định"
        clipped_context = document_context[: self.max_context_chars]
        return (
            f"Tài liệu y khoa: {doc_name}\n"
            f"Trang: {page_number}\n"
            f"Đường dẫn đề mục: {heading_text}\n\n"
            f"<document_context>\n{clipped_context}\n</document_context>\n\n"
            f"<chunk>\n{content}\n</chunk>\n\n"
            "Hãy viết đúng một câu tiếng Việt có dấu, ngắn gọn, mô tả đoạn này "
            "nằm ở đâu và nói về chủ đề gì trong tài liệu. Không lặp lại chi tiết "
            "của chunk, không suy diễn và không thêm kiến thức ngoài văn bản. "
            "Chỉ trả về câu ngữ cảnh, không thêm nhãn hoặc giải thích."
        )

    def _build_batch_prompt(
        self,
        chunks: Sequence[Mapping[str, Any]],
    ) -> str:
        items = []
        for item_id, chunk in enumerate(chunks):
            headings = [str(value) for value in chunk.get("headings") or []]
            items.append(
                {
                    "id": item_id,
                    "doc_name": str(chunk["doc_name"]),
                    "page_number": int(chunk["page_number"]),
                    "headings": headings,
                    "document_context": str(chunk.get("document_context") or "")[
                        : self.max_context_chars
                    ],
                    "chunk": str(chunk["content"]),
                }
            )
        return (
            "Tạo câu ngữ cảnh riêng cho từng chunk tài liệu y khoa bên dưới.\n"
            "Với mỗi item, viết đúng một câu tiếng Việt có dấu, ngắn gọn, mô tả "
            "chunk nằm ở đâu và nói về chủ đề gì. Không suy diễn, không thêm kiến "
            "thức ngoài nguồn và không bỏ sót item nào.\n\n"
            "Trả về duy nhất JSON hợp lệ theo dạng:\n"
            '{"contexts":[{"id":0,"context":"câu ngữ cảnh"}]}\n\n'
            f"<items>\n{json.dumps(items, ensure_ascii=False)}\n</items>"
        )

    @staticmethod
    def _parse_batch_response(content: str, expected_count: int) -> list[str]:
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content.strip(),
            flags=re.IGNORECASE,
        )
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Batch contextualization did not return a JSON object")
        payload = json.loads(stripped[start : end + 1])
        contexts = payload.get("contexts") if isinstance(payload, dict) else None
        if not isinstance(contexts, list):
            raise ValueError("Batch contextualization response has no contexts list")

        by_id: dict[int, str] = {}
        for item in contexts:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            context = item.get("context")
            if isinstance(item_id, int) and isinstance(context, str) and context.strip():
                by_id[item_id] = context.strip()

        missing = [item_id for item_id in range(expected_count) if item_id not in by_id]
        if missing:
            raise ValueError(
                f"Batch contextualization response is missing ids: {missing}"
            )
        return [by_id[item_id] for item_id in range(expected_count)]

    @staticmethod
    def _fallback_context(
        *,
        doc_name: str,
        page_number: int,
        headings: Sequence[str],
    ) -> str:
        if headings:
            return (
                f"Đoạn này thuộc mục {' > '.join(headings)} của tài liệu "
                f"{doc_name}, trang {page_number}."
            )
        return f"Đoạn này thuộc tài liệu {doc_name}, trang {page_number}."

    def enrich(
        self,
        *,
        content: str,
        doc_name: str,
        page_number: int,
        headings: Sequence[str] = (),
        document_context: str = "",
    ) -> Dict[str, str]:
        cache_key = self._cache_key(
            content=content,
            doc_name=doc_name,
            page_number=page_number,
            headings=headings,
            document_context=document_context,
        )
        context = self._cache.get(cache_key)
        if context is not None:
            self._cache_hits += 1
        elif self._api_disabled_reason is not None:
            context = self._fallback_context(
                doc_name=doc_name,
                page_number=page_number,
                headings=headings,
            )
            self._fallback_contexts += 1
        else:
            try:
                context = self.client.complete(
                    system_prompt=(
                        "Bạn tạo câu ngữ cảnh cho chunk tài liệu y khoa nhằm cải thiện "
                        "truy xuất. Tuân thủ tuyệt đối nội dung nguồn."
                    ),
                    user_prompt=self._build_prompt(
                        content=content,
                        doc_name=doc_name,
                        page_number=page_number,
                        headings=headings,
                        document_context=document_context,
                    ),
                    max_tokens=96,
                )
                context = normalize_text(context, preserve_lines=False)
                self._cache[cache_key] = context
                self._cache_dirty = True
                self._generated_contexts += 1
                if self.autosave:
                    self._save_cache()
            except (
                OpenRouterRequestBudgetExceeded,
                RuntimeError,
                ValueError,
            ) as exc:
                if not self.fail_open:
                    raise
                self._api_disabled_reason = str(exc)
                self._fallback_contexts += 1
                LOGGER.warning(
                    "OpenRouter contextualization disabled for the rest of this run: %s",
                    exc,
                )
                context = self._fallback_context(
                    doc_name=doc_name,
                    page_number=page_number,
                    headings=headings,
                )

        return {
            "context": context,
            "content": f"[Ngữ cảnh: {context}]\n{content}",
        }

    def enrich_batch(
        self,
        chunks: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = DEFAULT_CONTEXT_BATCH_SIZE,
    ) -> list[Dict[str, str]]:
        """Contextualize uncached chunks in API batches while preserving order."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not chunks:
            return []

        results: list[Dict[str, str] | None] = [None] * len(chunks)
        pending: list[tuple[int, Mapping[str, Any], str]] = []

        for index, chunk in enumerate(chunks):
            content = str(chunk["content"])
            doc_name = str(chunk["doc_name"])
            page_number = int(chunk["page_number"])
            headings = [str(value) for value in chunk.get("headings") or []]
            document_context = str(chunk.get("document_context") or "")
            cache_key = self._cache_key(
                content=content,
                doc_name=doc_name,
                page_number=page_number,
                headings=headings,
                document_context=document_context,
            )
            context = self._cache.get(cache_key)
            if context is not None:
                self._cache_hits += 1
                results[index] = {
                    "context": context,
                    "content": f"[Ngữ cảnh: {context}]\n{content}",
                }
            else:
                pending.append((index, chunk, cache_key))

        for start in range(0, len(pending), batch_size):
            group = pending[start : start + batch_size]
            if self._api_disabled_reason is not None:
                contexts = [
                    self._fallback_context(
                        doc_name=str(chunk["doc_name"]),
                        page_number=int(chunk["page_number"]),
                        headings=[
                            str(value) for value in chunk.get("headings") or []
                        ],
                    )
                    for _, chunk, _ in group
                ]
                self._fallback_contexts += len(group)
            else:
                group_chunks = [chunk for _, chunk, _ in group]
                try:
                    response = self.client.complete(
                        system_prompt=(
                            "Bạn tạo câu ngữ cảnh theo batch cho các chunk tài liệu "
                            "y khoa. Tuân thủ nội dung nguồn và trả về JSON hợp lệ."
                        ),
                        user_prompt=self._build_batch_prompt(group_chunks),
                        max_tokens=min(4096, max(256, len(group) * 128)),
                    )
                    contexts = self._parse_batch_response(response, len(group))
                    contexts = [
                        normalize_text(context, preserve_lines=False)
                        for context in contexts
                    ]
                    self._generated_contexts += len(group)
                    for (_, _, cache_key), context in zip(group, contexts):
                        self._cache[cache_key] = context
                    self._cache_dirty = True
                    if self.autosave:
                        self._save_cache()
                except (
                    OpenRouterRequestBudgetExceeded,
                    RuntimeError,
                    ValueError,
                ) as exc:
                    if not self.fail_open:
                        raise
                    self._api_disabled_reason = str(exc)
                    self._fallback_contexts += len(group)
                    LOGGER.warning(
                        "OpenRouter batch contextualization disabled for the rest "
                        "of this run: %s",
                        exc,
                    )
                    contexts = [
                        self._fallback_context(
                            doc_name=str(chunk["doc_name"]),
                            page_number=int(chunk["page_number"]),
                            headings=[
                                str(value) for value in chunk.get("headings") or []
                            ],
                        )
                        for chunk in group_chunks
                    ]

            for (index, chunk, _), context in zip(group, contexts):
                content = str(chunk["content"])
                results[index] = {
                    "context": context,
                    "content": f"[Ngữ cảnh: {context}]\n{content}",
                }

        if any(result is None for result in results):
            raise RuntimeError("Contextualization did not produce every batch result")
        return [result for result in results if result is not None]
