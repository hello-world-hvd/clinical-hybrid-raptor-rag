from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from ..openrouter_client import DEFAULT_OPENROUTER_MODEL, OpenRouterClient
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openrouter_client import DEFAULT_OPENROUTER_MODEL, OpenRouterClient


DEFAULT_QUERY_CACHE = Path("data/cache/query_optimization/gemma_queries.json")


@dataclass
class OptimizedQuery:
    original: str
    hyde_document: str
    multi_queries: list[str]
    sub_questions: list[str]
    step_back_query: str

    @classmethod
    def original_only(cls, query: str) -> "OptimizedQuery":
        return cls(
            original=query,
            hyde_document=query,
            multi_queries=[],
            sub_questions=[],
            step_back_query=query,
        )


class QueryOptimizer:
    """Generate HyDE and query expansions with one cached LLM request."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OPENROUTER_MODEL,
        multi_query_count: int = 3,
        cache_path: Path = DEFAULT_QUERY_CACHE,
        timeout: float = 90.0,
        client: Optional[OpenRouterClient] = None,
    ) -> None:
        self.model = model
        self.multi_query_count = max(1, multi_query_count)
        self.cache_path = cache_path
        self.client = client or OpenRouterClient(model=model, timeout=timeout)
        self._cache = self._load_cache()

    def _load_cache(self) -> Dict[str, Dict[str, Any]]:
        if not self.cache_path.exists():
            return {}
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return payload if isinstance(payload, dict) else {}

    def _save_cache(self) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.cache_path.with_suffix(f"{self.cache_path.suffix}.tmp")
        temporary.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.cache_path)

    def _cache_key(self, query: str) -> str:
        raw = f"{self.model}\n{self.multi_query_count}\n{query.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _prompt(self, query: str) -> str:
        return (
            "Tối ưu câu hỏi sau để truy xuất tài liệu y khoa tiếng Việt.\n\n"
            f"<query>{query}</query>\n\n"
            "Trả về duy nhất một JSON object hợp lệ với cấu trúc:\n"
            "{\n"
            '  "hyde_document": "một đoạn tài liệu giả định trả lời câu hỏi, '
            'giàu thuật ngữ truy xuất nhưng không bịa số liệu cụ thể",\n'
            f'  "multi_queries": ["đúng {self.multi_query_count} cách diễn đạt khác"],\n'
            '  "sub_questions": ["các câu hỏi con nếu câu hỏi phức tạp, nếu đơn giản trả []"],\n'
            '  "step_back_query": "một câu hỏi tổng quát hơn bao quát kiến thức nền"\n'
            "}\n"
            "Giữ nguyên tên bệnh, thuốc, xét nghiệm, đơn vị và thuật ngữ quan trọng. "
            "Không thêm markdown hoặc giải thích ngoài JSON."
        )

    @staticmethod
    def _parse_json(content: str) -> Dict[str, Any]:
        stripped = re.sub(
            r"^```(?:json)?\s*|\s*```$",
            "",
            content.strip(),
            flags=re.IGNORECASE,
        )
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Query optimizer did not return a JSON object")
        payload = json.loads(stripped[start : end + 1])
        if not isinstance(payload, dict):
            raise ValueError("Query optimization response must be an object")
        return payload

    @staticmethod
    def _clean_list(value: Any, *, exclude: str) -> list[str]:
        if not isinstance(value, list):
            return []
        cleaned: list[str] = []
        seen = {exclude.strip().casefold()}
        for item in value:
            text = str(item).strip()
            key = text.casefold()
            if text and key not in seen:
                seen.add(key)
                cleaned.append(text)
        return cleaned

    def optimize(self, query: str) -> OptimizedQuery:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        cache_key = self._cache_key(query)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return OptimizedQuery(**cached)

        content = self.client.complete(
            system_prompt=(
                "Bạn là bộ tối ưu truy vấn cho hệ thống RAG y khoa. "
                "Ưu tiên recall nhưng không làm sai ý định câu hỏi."
            ),
            user_prompt=self._prompt(query),
            max_tokens=900,
            temperature=0,
        )
        payload = self._parse_json(content)
        optimized = OptimizedQuery(
            original=query,
            hyde_document=str(payload.get("hyde_document") or query).strip(),
            multi_queries=self._clean_list(
                payload.get("multi_queries"),
                exclude=query,
            )[: self.multi_query_count],
            sub_questions=self._clean_list(
                payload.get("sub_questions"),
                exclude=query,
            ),
            step_back_query=str(payload.get("step_back_query") or query).strip(),
        )
        self._cache[cache_key] = asdict(optimized)
        self._save_cache()
        return optimized

    def hyde(self, query: str) -> str:
        """Return a hypothetical document to use for dense retrieval."""
        return self.optimize(query).hyde_document

    def multi_query(self, query: str, n: int | None = None) -> list[str]:
        """Return up to ``n`` semantically equivalent query variants."""
        queries = self.optimize(query).multi_queries
        if n is None:
            return queries
        if n <= 0:
            return []
        return queries[:n]

    def decompose(self, query: str) -> list[str]:
        """Split a complex query into independently retrievable questions."""
        return self.optimize(query).sub_questions

    def step_back(self, query: str) -> str:
        """Return a broader background question."""
        return self.optimize(query).step_back_query
