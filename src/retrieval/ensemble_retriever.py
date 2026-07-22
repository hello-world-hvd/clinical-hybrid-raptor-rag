from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from src.embedding import (
    DEFAULT_BM25_MODEL,
    DEFAULT_COLBERT_MODEL,
    DEFAULT_DENSE_MODEL,
    BM25SparseEmbedder,
    ColBERTEmbedder,
    DenseEmbedder,
)
from src.openrouter_client import (
    DEFAULT_OPENROUTER_MODEL,
    DEFAULT_OPENROUTER_RERANK_MODEL,
    OpenRouterRerankClient,
)
from src.retrieval.qdrant_indexer import DEFAULT_COLLECTION, DEFAULT_QDRANT_URL
from src.retrieval.query_optimizer import OptimizedQuery, QueryOptimizer
from src.vectorstore import QdrantVectorStore


@dataclass(frozen=True)
class QueryVariant:
    name: str
    text: str
    weight: float


@dataclass(frozen=True)
class EnsembleConfig:
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection_name: str = DEFAULT_COLLECTION
    dense_model: str = DEFAULT_DENSE_MODEL
    bm25_model: str = DEFAULT_BM25_MODEL
    colbert_model: str = DEFAULT_COLBERT_MODEL
    embedding_backend: str = "local"
    cache_dir: Path | None = Path("data/cache/embeddings")
    device: str | None = None
    local_files_only: bool = False
    bm25_limit: int = 30
    dense_limit: int = 40
    colbert_limit: int = 30
    candidate_limit: int = 30
    final_top_k: int = 10
    rrf_k: int = 60
    bm25_weight: float = 2.0
    dense_weight: float = 5.0
    colbert_weight: float = 3.0
    optimize_queries: bool = True
    query_model: str = DEFAULT_OPENROUTER_MODEL
    multi_query_count: int = 3
    optimization_fail_open: bool = True
    rerank: bool = True
    rerank_model: str = DEFAULT_OPENROUTER_RERANK_MODEL
    rerank_fail_open: bool = True


class EnsembleRetriever:
    """Retrieve from Qdrant with BM25, dense, and ColBERT, then rerank."""

    def __init__(
        self,
        config: EnsembleConfig | None = None,
        *,
        store: QdrantVectorStore | None = None,
        dense_embedder: DenseEmbedder | None = None,
        sparse_embedder: BM25SparseEmbedder | None = None,
        colbert_embedder: ColBERTEmbedder | None = None,
        optimizer: QueryOptimizer | None = None,
        reranker: OpenRouterRerankClient | None = None,
    ) -> None:
        self.config = config or EnsembleConfig()
        self.store = store or QdrantVectorStore(
            url=self.config.qdrant_url,
            collection_name=self.config.collection_name,
        )
        self.dense_embedder = dense_embedder or DenseEmbedder(
            model_name=self.config.dense_model,
            backend=self.config.embedding_backend,
            device=self.config.device,
            cache_folder=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        self.sparse_embedder = sparse_embedder or BM25SparseEmbedder(
            model_name=self.config.bm25_model,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        self.colbert_embedder = colbert_embedder or ColBERTEmbedder(
            model_name=self.config.colbert_model,
            cache_dir=self.config.cache_dir,
            local_files_only=self.config.local_files_only,
        )
        self._optimizer = optimizer
        self._reranker = reranker

    def _get_optimizer(self) -> QueryOptimizer:
        if self._optimizer is None:
            self._optimizer = QueryOptimizer(
                model=self.config.query_model,
                multi_query_count=self.config.multi_query_count,
            )
        return self._optimizer

    def _get_reranker(self) -> OpenRouterRerankClient:
        if self._reranker is None:
            self._reranker = OpenRouterRerankClient(model=self.config.rerank_model)
        return self._reranker

    def optimize(self, query: str) -> OptimizedQuery:
        if not self.config.optimize_queries:
            return OptimizedQuery.original_only(query)
        try:
            return self._get_optimizer().optimize(query)
        except Exception:
            if not self.config.optimization_fail_open:
                raise
            return OptimizedQuery.original_only(query)

    @staticmethod
    def _deduplicate(variants: Iterable[QueryVariant]) -> list[QueryVariant]:
        unique: list[QueryVariant] = []
        seen: set[str] = set()
        for variant in variants:
            text = " ".join(variant.text.split()).strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            unique.append(QueryVariant(variant.name, text, variant.weight))
        return unique

    def _lexical_variants(self, optimized: OptimizedQuery) -> list[QueryVariant]:
        variants = [QueryVariant("original", optimized.original, 1.0)]
        variants.extend(
            QueryVariant(f"multi_{index}", query, 0.85)
            for index, query in enumerate(optimized.multi_queries, start=1)
        )
        variants.extend(
            QueryVariant(f"sub_{index}", query, 0.9)
            for index, query in enumerate(optimized.sub_questions, start=1)
        )
        variants.append(QueryVariant("step_back", optimized.step_back_query, 0.65))
        return self._deduplicate(variants)

    def _dense_variants(self, optimized: OptimizedQuery) -> list[QueryVariant]:
        lexical = self._lexical_variants(optimized)
        hyde = " ".join(optimized.hyde_document.split()).strip()
        if hyde and hyde.casefold() != optimized.original.casefold():
            return self._deduplicate(
                [
                    QueryVariant("hyde", hyde, 1.1),
                    *[variant for variant in lexical if variant.name != "original"],
                ]
            )
        return lexical

    def _retrieve(
        self,
        optimized: OptimizedQuery,
    ) -> list[tuple[str, QueryVariant, float, list[dict[str, Any]]]]:
        ranked_lists: list[
            tuple[str, QueryVariant, float, list[dict[str, Any]]]
        ] = []

        for variant in self._lexical_variants(optimized):
            bm25_vector = self.sparse_embedder.encode_query(variant.text)
            ranked_lists.append(
                (
                    "bm25",
                    variant,
                    self.config.bm25_weight,
                    self.store.search_bm25(
                        bm25_vector,
                        limit=self.config.bm25_limit,
                        leaf_only=True,
                    ),
                )
            )

            colbert_vector = self.colbert_embedder.encode_query(variant.text)
            ranked_lists.append(
                (
                    "colbert",
                    variant,
                    self.config.colbert_weight,
                    self.store.search_colbert(
                        colbert_vector,
                        limit=self.config.colbert_limit,
                    ),
                )
            )

        for variant in self._dense_variants(optimized):
            dense_vector = self.dense_embedder.encode_query(variant.text)
            ranked_lists.append(
                (
                    "dense",
                    variant,
                    self.config.dense_weight,
                    self.store.search_dense(
                        dense_vector,
                        limit=self.config.dense_limit,
                    ),
                )
            )
        return ranked_lists

    def _fuse(
        self,
        ranked_lists: Sequence[
            tuple[str, QueryVariant, float, list[dict[str, Any]]]
        ],
    ) -> list[dict[str, Any]]:
        fused: dict[str, dict[str, Any]] = {}
        for channel, variant, channel_weight, results in ranked_lists:
            source = f"{channel}:{variant.name}"
            for rank, result in enumerate(results, start=1):
                node_id = str(result.get("node_id") or "")
                if not node_id:
                    continue
                contribution = (
                    channel_weight * variant.weight / (self.config.rrf_k + rank)
                )
                item = fused.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "score": 0.0,
                        "payload": dict(result.get("payload") or {}),
                        "matches": {},
                    },
                )
                item["score"] += contribution
                item["matches"][source] = {
                    "rank": rank,
                    "raw_score": float(result.get("score") or 0.0),
                    "rrf_contribution": contribution,
                }
                if not item["payload"] and result.get("payload"):
                    item["payload"] = dict(result["payload"])
        return sorted(fused.values(), key=lambda item: item["score"], reverse=True)

    def _rerank(
        self,
        query: str,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not candidates:
            return []
        documents = [
            str(candidate.get("payload", {}).get("text") or "")
            for candidate in candidates
        ]
        try:
            results = self._get_reranker().rerank(
                query=query,
                documents=documents,
                top_n=len(documents),
            )
        except Exception:
            if not self.config.rerank_fail_open:
                raise
            return list(candidates)

        reranked: list[dict[str, Any]] = []
        selected: set[int] = set()
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or not 0 <= index < len(candidates):
                continue
            if index in selected:
                continue
            updated = dict(candidates[index])
            updated["rerank_score"] = float(
                result.get("relevance_score", result.get("score", 0.0))
            )
            reranked.append(updated)
            selected.add(index)

        reranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        reranked.extend(
            candidate
            for index, candidate in enumerate(candidates)
            if index not in selected
        )
        return reranked

    @staticmethod
    def _format_result(item: Mapping[str, Any]) -> dict[str, Any]:
        payload = dict(item.get("payload") or {})
        text = str(payload.get("text") or "")
        return {
            **payload,
            "node_id": item["node_id"],
            "score": float(item.get("score") or 0.0),
            "rerank_score": item.get("rerank_score"),
            "matches": item.get("matches") or {},
            "snippet": text if len(text) <= 450 else f"{text[:447].rstrip()}...",
        }

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        optimize: bool | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        query = query.strip()
        if not query:
            raise ValueError("Query must not be empty")

        use_optimization = self.config.optimize_queries if optimize is None else optimize
        optimized = (
            self.optimize(query)
            if use_optimization
            else OptimizedQuery.original_only(query)
        )
        candidates = self._fuse(self._retrieve(optimized))[: self.config.candidate_limit]
        use_rerank = self.config.rerank if rerank is None else rerank
        if use_rerank:
            candidates = self._rerank(query, candidates)

        limit = top_k if top_k is not None else self.config.final_top_k
        return [self._format_result(item) for item in candidates[:limit]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query Qdrant with optimized BM25, dense, and ColBERT retrieval."
    )
    parser.add_argument("query")
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--bm25-limit", type=int, default=30)
    parser.add_argument("--dense-limit", type=int, default=40)
    parser.add_argument("--colbert-limit", type=int, default=30)
    parser.add_argument("--bm25-weight", type=float, default=2.0)
    parser.add_argument("--dense-weight", type=float, default=5.0)
    parser.add_argument("--colbert-weight", type=float, default=3.0)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--bm25-model", default=DEFAULT_BM25_MODEL)
    parser.add_argument("--colbert-model", default=DEFAULT_COLBERT_MODEL)
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "local", "modal"],
        default="local",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/embeddings"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--query-model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--multi-query-count", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--strict-optimization", action="store_true")
    parser.add_argument("--strict-rerank", action="store_true")
    parser.add_argument("--rerank-model", default=DEFAULT_OPENROUTER_RERANK_MODEL)
    return parser.parse_args()


def build_retriever(args: argparse.Namespace) -> EnsembleRetriever:
    return EnsembleRetriever(
        EnsembleConfig(
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            dense_model=args.dense_model,
            bm25_model=args.bm25_model,
            colbert_model=args.colbert_model,
            embedding_backend=args.embedding_backend,
            cache_dir=args.cache_dir,
            device=args.device,
            local_files_only=args.local_files_only,
            bm25_limit=args.bm25_limit,
            dense_limit=args.dense_limit,
            colbert_limit=args.colbert_limit,
            candidate_limit=args.candidate_limit,
            final_top_k=args.top_k,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            colbert_weight=args.colbert_weight,
            optimize_queries=not args.no_optimize,
            query_model=args.query_model,
            multi_query_count=max(1, args.multi_query_count),
            optimization_fail_open=not args.strict_optimization,
            rerank=not args.no_rerank,
            rerank_model=args.rerank_model,
            rerank_fail_open=not args.strict_rerank,
        )
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    results = build_retriever(args).search(args.query)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
