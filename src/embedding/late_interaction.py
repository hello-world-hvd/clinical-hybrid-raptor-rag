from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np


DEFAULT_COLBERT_MODEL = "colbert-ir/colbertv2.0"


class ColBERTEmbedder:
    """FastEmbed ColBERT encoder producing token-level multivectors."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_COLBERT_MODEL,
        cache_dir: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.local_files_only = local_files_only
        self._model: Optional[Any] = None

    def _get_model(self) -> Any:
        if self._model is None:
            from fastembed import LateInteractionTextEmbedding

            self._model = LateInteractionTextEmbedding(
                model_name=self.model_name,
                cache_dir=str(self.cache_dir) if self.cache_dir else None,
                local_files_only=self.local_files_only,
            )
        return self._model

    @staticmethod
    def _serialize(vector: Any) -> list[list[float]]:
        return np.asarray(vector, dtype="float32").tolist()

    def encode_documents(self, texts: Sequence[str]) -> list[list[list[float]]]:
        return [self._serialize(item) for item in self._get_model().embed(list(texts))]

    def encode_query(self, query: str) -> list[list[float]]:
        return self._serialize(next(self._get_model().query_embed([query])))
