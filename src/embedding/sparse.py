from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence


DEFAULT_BM25_MODEL = "Qdrant/bm25"


class BM25SparseEmbedder:
    """FastEmbed BM25 encoder configured for language-neutral Vietnamese text."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_BM25_MODEL,
        cache_dir: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model: Optional[Any] = None

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import SparseTextEmbedding

            self._model = SparseTextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_files_only=self.local_files_only,
                language="english",
                disable_stemmer=True,
            )
        return self._model

    @staticmethod
    def _serialize(embedding: Any) -> dict[str, list]:
        return {
            "indices": embedding.indices.tolist(),
            "values": embedding.values.tolist(),
        }

    def encode_documents(self, texts: Sequence[str]) -> list[dict[str, list]]:
        return [self._serialize(item) for item in self._get_model().embed(list(texts))]

    def encode_query(self, query: str) -> dict[str, list]:
        return self._serialize(next(self._get_model().query_embed([query])))
