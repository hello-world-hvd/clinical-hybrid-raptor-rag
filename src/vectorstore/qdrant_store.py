from __future__ import annotations

import uuid
from typing import Any, Dict, Mapping, Optional, Sequence


DENSE_VECTOR_NAME = "dense"
BM25_VECTOR_NAME = "bm25"
COLBERT_VECTOR_NAME = "colbert"


class QdrantVectorStore:
    """Qdrant collection with dense, sparse BM25, and ColBERT vectors."""

    def __init__(
        self,
        *,
        url: str = "http://localhost:6333",
        collection_name: str = "clinical_raptor",
        timeout: float = 60.0,
        client: Optional[Any] = None,
    ) -> None:
        if client is None:
            from qdrant_client import QdrantClient

            client = QdrantClient(url=url, timeout=timeout)
        self.client = client
        self.collection_name = collection_name

    @staticmethod
    def point_id(node_id: str) -> str:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, node_id))

    def recreate_collection(
        self,
        *,
        dense_size: int,
        colbert_size: int,
    ) -> None:
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        self.create_collection(dense_size=dense_size, colbert_size=colbert_size)

    def ensure_collection(
        self,
        *,
        dense_size: int,
        colbert_size: int,
    ) -> bool:
        """Create the collection when missing and return whether it was created."""
        if self.client.collection_exists(self.collection_name):
            return False
        self.create_collection(dense_size=dense_size, colbert_size=colbert_size)
        return True

    def create_collection(
        self,
        *,
        dense_size: int,
        colbert_size: int,
    ) -> None:
        from qdrant_client import models

        if dense_size <= 0 or colbert_size <= 0:
            raise ValueError("dense_size and colbert_size must be positive")
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=dense_size,
                    distance=models.Distance.COSINE,
                ),
                COLBERT_VECTOR_NAME: models.VectorParams(
                    size=colbert_size,
                    distance=models.Distance.COSINE,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM,
                    ),
                    hnsw_config=models.HnswConfigDiff(m=0),
                ),
            },
            sparse_vectors_config={
                BM25_VECTOR_NAME: models.SparseVectorParams(
                    modifier=models.Modifier.IDF,
                )
            },
        )

    def upsert(
        self,
        *,
        nodes: Sequence[Dict[str, Any]],
        dense_vectors: Sequence[Sequence[float]],
        bm25_vectors: Sequence[Mapping[str, Sequence]],
        colbert_vectors: Sequence[Sequence[Sequence[float]]],
        batch_size: int = 32,
    ) -> None:
        from qdrant_client import models

        lengths = {
            len(nodes),
            len(dense_vectors),
            len(bm25_vectors),
            len(colbert_vectors),
        }
        if len(lengths) != 1:
            raise ValueError("Nodes and all vector collections must have equal length")

        for start in range(0, len(nodes), batch_size):
            points = []
            stop = min(start + batch_size, len(nodes))
            for index in range(start, stop):
                node = nodes[index]
                sparse = bm25_vectors[index]
                points.append(
                    models.PointStruct(
                        id=self.point_id(node["node_id"]),
                        payload=node,
                        vector={
                            DENSE_VECTOR_NAME: list(dense_vectors[index]),
                            BM25_VECTOR_NAME: models.SparseVector(
                                indices=list(sparse["indices"]),
                                values=list(sparse["values"]),
                            ),
                            COLBERT_VECTOR_NAME: [
                                list(token_vector)
                                for token_vector in colbert_vectors[index]
                            ],
                        },
                    )
                )
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    @staticmethod
    def _format_points(points: Sequence[Any]) -> list[Dict[str, Any]]:
        results: list[Dict[str, Any]] = []
        for point in points:
            payload = dict(point.payload or {})
            results.append(
                {
                    "node_id": payload.get("node_id"),
                    "score": float(point.score),
                    "payload": payload,
                }
            )
        return results

    def _query(
        self,
        *,
        vector_name: str,
        query: Any,
        limit: int,
        query_filter: Optional[Any] = None,
    ) -> list[Dict[str, Any]]:
        if limit <= 0:
            return []
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query,
            using=vector_name,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return self._format_points(response.points)

    def search_dense(
        self,
        query_vector: Sequence[float],
        *,
        limit: int,
        query_filter: Optional[Any] = None,
    ) -> list[Dict[str, Any]]:
        return self._query(
            vector_name=DENSE_VECTOR_NAME,
            query=list(query_vector),
            limit=limit,
            query_filter=query_filter,
        )

    def search_bm25(
        self,
        query_vector: Dict[str, Sequence],
        *,
        limit: int,
        leaf_only: bool = True,
        query_filter: Optional[Any] = None,
    ) -> list[Dict[str, Any]]:
        from qdrant_client import models

        if query_filter is None and leaf_only:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="is_leaf",
                        match=models.MatchValue(value=True),
                    )
                ]
            )
        return self._query(
            vector_name=BM25_VECTOR_NAME,
            query=models.SparseVector(
                indices=list(query_vector["indices"]),
                values=list(query_vector["values"]),
            ),
            limit=limit,
            query_filter=query_filter,
        )

    def search_colbert(
        self,
        query_vector: Sequence[Sequence[float]],
        *,
        limit: int,
        query_filter: Optional[Any] = None,
    ) -> list[Dict[str, Any]]:
        return self._query(
            vector_name=COLBERT_VECTOR_NAME,
            query=[list(token_vector) for token_vector in query_vector],
            limit=limit,
            query_filter=query_filter,
        )
