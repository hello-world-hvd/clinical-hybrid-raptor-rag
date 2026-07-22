from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from tqdm import tqdm

from src.embedding import (
    DEFAULT_BM25_MODEL,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_DENSE_MODEL,
    BM25SparseEmbedder,
    ColBERTEmbedder,
    DenseEmbedder,
)
from src.retrieval.index_raptor import DEFAULT_TREE_PATH, load_nodes
from src.vectorstore import QdrantVectorStore


DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION = "clinical_raptor"
DEFAULT_MANIFEST_PATH = Path("data/processed/qdrant_index/manifest.json")


@dataclass(frozen=True)
class QdrantIndexConfig:
    nodes_path: Path = DEFAULT_TREE_PATH
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection_name: str = DEFAULT_COLLECTION
    dense_model: str = DEFAULT_DENSE_MODEL
    bm25_model: str = DEFAULT_BM25_MODEL
    colbert_model: str = DEFAULT_COLBERT_MODEL
    embedding_backend: str = "local"
    batch_size: int = 16
    device: str | None = None
    cache_dir: Path | None = Path("data/cache/embeddings")
    local_files_only: bool = False
    recreate: bool = False
    manifest_path: Path = DEFAULT_MANIFEST_PATH


def _batches(
    items: Sequence[dict[str, Any]],
    size: int,
) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _colbert_size(vectors: Sequence[Sequence[Sequence[float]]]) -> int:
    for document in vectors:
        if document:
            return len(document[0])
    raise ValueError("ColBERT returned no token vectors")


def build_qdrant_index(
    config: QdrantIndexConfig,
    *,
    store: QdrantVectorStore | None = None,
    dense_embedder: DenseEmbedder | None = None,
    sparse_embedder: BM25SparseEmbedder | None = None,
    colbert_embedder: ColBERTEmbedder | None = None,
) -> dict[str, Any]:
    if config.batch_size <= 0:
        raise ValueError("batch_size must be positive")

    nodes = load_nodes(config.nodes_path)
    store = store or QdrantVectorStore(
        url=config.qdrant_url,
        collection_name=config.collection_name,
    )
    dense_embedder = dense_embedder or DenseEmbedder(
        model_name=config.dense_model,
        backend=config.embedding_backend,
        batch_size=config.batch_size,
        device=config.device,
        cache_folder=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    sparse_embedder = sparse_embedder or BM25SparseEmbedder(
        model_name=config.bm25_model,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )
    colbert_embedder = colbert_embedder or ColBERTEmbedder(
        model_name=config.colbert_model,
        cache_dir=config.cache_dir,
        local_files_only=config.local_files_only,
    )

    collection_ready = False
    backend_used = config.embedding_backend
    indexed_count = 0
    batch_count = math.ceil(len(nodes) / config.batch_size)
    for batch in tqdm(
        _batches(nodes, config.batch_size),
        total=batch_count,
        desc="Indexing Qdrant",
    ):
        texts = [str(node.get("text") or "") for node in batch]
        dense_vectors, backend_used = dense_embedder.encode(texts, show_progress=False)
        sparse_vectors = sparse_embedder.encode_documents(texts)
        colbert_vectors = colbert_embedder.encode_documents(texts)

        if not collection_ready:
            dense_size = int(dense_vectors.shape[1])
            colbert_size = _colbert_size(colbert_vectors)
            if config.recreate:
                store.recreate_collection(
                    dense_size=dense_size,
                    colbert_size=colbert_size,
                )
            else:
                store.ensure_collection(
                    dense_size=dense_size,
                    colbert_size=colbert_size,
                )
            collection_ready = True

        store.upsert(
            nodes=batch,
            dense_vectors=dense_vectors,
            bm25_vectors=sparse_vectors,
            colbert_vectors=colbert_vectors,
            batch_size=config.batch_size,
        )
        indexed_count += len(batch)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nodes_path": str(config.nodes_path),
        "node_count": indexed_count,
        "qdrant_url": config.qdrant_url,
        "collection_name": config.collection_name,
        "models": {
            "dense": config.dense_model,
            "bm25": config.bm25_model,
            "colbert": config.colbert_model,
        },
        "embedding_backend": backend_used,
        "vector_names": {
            "dense": "dense",
            "bm25": "bm25",
            "colbert": "colbert",
        },
        "config": {
            **asdict(config),
            "nodes_path": str(config.nodes_path),
            "cache_dir": str(config.cache_dir) if config.cache_dir else None,
            "manifest_path": str(config.manifest_path),
        },
    }
    config.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    config.manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> QdrantIndexConfig:
    parser = argparse.ArgumentParser(
        description="Index RAPTOR nodes into Qdrant with BM25, dense, and ColBERT vectors."
    )
    parser.add_argument("--nodes", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--bm25-model", default=DEFAULT_BM25_MODEL)
    parser.add_argument("--colbert-model", default=DEFAULT_COLBERT_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "local", "modal"],
        default="local",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/embeddings"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    args = parser.parse_args()
    return QdrantIndexConfig(
        nodes_path=args.nodes,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        dense_model=args.dense_model,
        bm25_model=args.bm25_model,
        colbert_model=args.colbert_model,
        embedding_backend=args.embedding_backend,
        batch_size=args.batch_size,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        recreate=args.recreate,
        manifest_path=args.manifest,
    )


def main() -> None:
    config = parse_args()
    manifest = build_qdrant_index(config)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
