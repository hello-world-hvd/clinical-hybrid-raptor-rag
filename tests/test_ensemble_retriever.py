from src.retrieval.ensemble_retriever import EnsembleConfig, EnsembleRetriever
from src.retrieval.query_optimizer import OptimizedQuery


class FakeDenseEmbedder:
    def __init__(self) -> None:
        self.queries = []

    def encode_query(self, query):
        self.queries.append(query)
        return [float(len(self.queries))]


class FakeSparseEmbedder:
    def __init__(self) -> None:
        self.queries = []

    def encode_query(self, query):
        self.queries.append(query)
        return {"indices": [len(self.queries)], "values": [1.0]}


class FakeColBERTEmbedder:
    def __init__(self) -> None:
        self.queries = []

    def encode_query(self, query):
        self.queries.append(query)
        return [[float(len(self.queries))]]


class FakeStore:
    def __init__(self) -> None:
        self.bm25_calls = []
        self.dense_calls = []
        self.colbert_calls = []

    @staticmethod
    def _results(channel):
        if channel == "dense":
            order = ["dense-only", "shared"]
        else:
            order = ["shared", f"{channel}-only"]
        return [
            {
                "node_id": node_id,
                "score": 1.0 / rank,
                "payload": {"node_id": node_id, "text": f"text for {node_id}"},
            }
            for rank, node_id in enumerate(order, start=1)
        ]

    def search_bm25(self, vector, *, limit, leaf_only):
        self.bm25_calls.append((vector, limit, leaf_only))
        return self._results("bm25")

    def search_dense(self, vector, *, limit):
        self.dense_calls.append((vector, limit))
        return self._results("dense")

    def search_colbert(self, vector, *, limit):
        self.colbert_calls.append((vector, limit))
        return self._results("colbert")


class FakeOptimizer:
    def optimize(self, query):
        return OptimizedQuery(
            original=query,
            hyde_document="hypothetical answer",
            multi_queries=["rewritten query"],
            sub_questions=["sub question"],
            step_back_query="broader question",
        )


class FakeReranker:
    def __init__(self) -> None:
        self.query = None
        self.documents = []

    def rerank(self, *, query, documents, top_n):
        self.query = query
        self.documents = documents
        return [
            {"index": 1, "relevance_score": 0.99},
            {"index": 0, "relevance_score": 0.20},
        ]


def build_fake_retriever(*, rerank=True):
    store = FakeStore()
    dense = FakeDenseEmbedder()
    sparse = FakeSparseEmbedder()
    colbert = FakeColBERTEmbedder()
    reranker = FakeReranker()
    retriever = EnsembleRetriever(
        EnsembleConfig(
            rrf_k=10,
            candidate_limit=5,
            final_top_k=3,
            rerank=rerank,
        ),
        store=store,
        dense_embedder=dense,
        sparse_embedder=sparse,
        colbert_embedder=colbert,
        optimizer=FakeOptimizer(),
        reranker=reranker,
    )
    return retriever, store, dense, sparse, colbert, reranker


def test_optimized_queries_reach_expected_retrieval_channels():
    retriever, store, dense, sparse, colbert, _ = build_fake_retriever(rerank=False)

    results = retriever.search("original query")

    assert sparse.queries == [
        "original query",
        "rewritten query",
        "sub question",
        "broader question",
    ]
    assert colbert.queries == sparse.queries
    assert dense.queries == [
        "hypothetical answer",
        "rewritten query",
        "sub question",
        "broader question",
    ]
    assert all(call[2] is True for call in store.bm25_calls)
    assert results[0]["node_id"] == "shared"
    assert {"bm25:original", "colbert:original", "dense:hyde"} <= set(
        results[0]["matches"]
    )


def test_openrouter_rerank_indices_reorder_fused_candidates():
    retriever, _, _, _, _, reranker = build_fake_retriever(rerank=True)

    without_rerank = retriever.search("original query", rerank=False)
    with_rerank = retriever.search("original query", rerank=True)

    assert with_rerank[0]["node_id"] == without_rerank[1]["node_id"]
    assert with_rerank[0]["rerank_score"] == 0.99
    assert reranker.query == "original query"
    assert len(reranker.documents) >= 2


def test_retriever_falls_back_to_original_query_when_optimizer_fails():
    class BrokenOptimizer:
        def optimize(self, query):
            raise RuntimeError("service unavailable")

    retriever, _, dense, sparse, colbert, _ = build_fake_retriever(rerank=False)
    retriever._optimizer = BrokenOptimizer()

    retriever.search("original query")

    assert dense.queries == ["original query"]
    assert sparse.queries == ["original query"]
    assert colbert.queries == ["original query"]
