from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm


DEFAULT_DENSE_MODEL = "BAAI/bge-m3"
MODAL_APP_NAME = "raptor-bge-m3-embedder"
MODAL_FUNCTION_NAME = "embed_batch"
_LOCAL_MODELS: Dict[Tuple[str, str, Optional[str]], Any] = {}


def _cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


class DenseEmbedder:
    """Reusable dense embedder with local and Modal backends."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_DENSE_MODEL,
        backend: str = "local",
        batch_size: int = 32,
        remote_batch_size: int = 96,
        device: Optional[str] = None,
        cache_folder: Optional[Path] = None,
        local_files_only: bool = False,
    ) -> None:
        if backend not in {"auto", "local", "modal"}:
            raise ValueError(f"Unsupported embedding backend: {backend}")
        self.model_name = model_name
        self.backend = backend
        self.batch_size = batch_size
        self.remote_batch_size = remote_batch_size
        self.device = device
        self.cache_folder = cache_folder
        self.local_files_only = local_files_only

    def _local_model(self) -> Any:
        from sentence_transformers import SentenceTransformer

        device = self.device or ("cuda" if _cuda_available() else "cpu")
        cache_key = (
            self.model_name,
            device,
            str(self.cache_folder) if self.cache_folder else None,
        )
        model = _LOCAL_MODELS.get(cache_key)
        if model is None:
            kwargs: Dict[str, Any] = {"device": device}
            if self.cache_folder is not None:
                kwargs["cache_folder"] = str(self.cache_folder)
            if self.local_files_only:
                kwargs["local_files_only"] = True
            model = SentenceTransformer(self.model_name, **kwargs)
            _LOCAL_MODELS[cache_key] = model
        return model

    def _encode_local(self, texts: Sequence[str], show_progress: bool) -> np.ndarray:
        vectors = self._local_model().encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(vectors, dtype="float32")

    def _encode_modal(self, texts: Sequence[str]) -> np.ndarray:
        import modal

        function = modal.Function.from_name(MODAL_APP_NAME, MODAL_FUNCTION_NAME)
        batches: list[np.ndarray] = []
        for start in tqdm(
            range(0, len(texts), self.remote_batch_size),
            desc="Modal embeddings",
        ):
            batch = list(texts[start : start + self.remote_batch_size])
            vectors = function.remote(
                batch,
                model_name=self.model_name,
                batch_size=self.batch_size,
            )
            batches.append(np.asarray(vectors, dtype="float32"))
        return np.vstack(batches) if batches else np.empty((0, 0), dtype="float32")

    def encode(
        self,
        texts: Sequence[str],
        *,
        show_progress: bool = False,
    ) -> Tuple[np.ndarray, str]:
        if not texts:
            return np.empty((0, 0), dtype="float32"), self.backend

        if self.backend in {"auto", "modal"}:
            try:
                return self._encode_modal(texts), "modal"
            except Exception:
                if self.backend == "modal":
                    raise

        return self._encode_local(texts, show_progress), "local"

    def encode_query(self, text: str) -> np.ndarray:
        vectors, _ = self.encode([text], show_progress=False)
        return vectors[0]


def load_or_create_dense_embeddings(
    *,
    texts: Sequence[str],
    identities: Sequence[str],
    cache_path: Path,
    embedder: DenseEmbedder,
) -> np.ndarray:
    if len(texts) != len(identities):
        raise ValueError("texts and identities must have the same length")

    metadata_path = cache_path.with_suffix(".meta.json")
    text_hash = hashlib.sha1(
        "\n".join(f"{identity}\t{text}" for identity, text in zip(identities, texts)).encode("utf-8")
    ).hexdigest()

    if cache_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("model_name") == embedder.model_name
            and metadata.get("text_hash") == text_hash
        ):
            return np.load(cache_path)

    vectors, backend_used = embedder.encode(texts, show_progress=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, vectors)
    metadata_path.write_text(
        json.dumps(
            {
                "model_name": embedder.model_name,
                "backend": backend_used,
                "text_hash": text_hash,
                "shape": list(vectors.shape),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return vectors
