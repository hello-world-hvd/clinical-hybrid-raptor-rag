import json

import numpy as np

from src.retrieval.qdrant_indexer import QdrantIndexConfig, build_qdrant_index


class FakeDenseEmbedder:
    def encode(self, texts, *, show_progress):
        return np.asarray([[1.0, 0.0] for _ in texts], dtype="float32"), "fake"


class FakeSparseEmbedder:
    def encode_documents(self, texts):
        return [{"indices": [1], "values": [1.0]} for _ in texts]


class FakeColBERTEmbedder:
    def encode_documents(self, texts):
        return [[[1.0, 0.0], [0.0, 1.0]] for _ in texts]


class FakeStore:
    def __init__(self):
        self.ensure_calls = []
        self.recreate_calls = []
        self.upserts = []

    def ensure_collection(self, **kwargs):
        self.ensure_calls.append(kwargs)
        return True

    def recreate_collection(self, **kwargs):
        self.recreate_calls.append(kwargs)

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)


def write_nodes(path):
    rows = [
        {
            "node_id": "leaf-1",
            "is_leaf": True,
            "depth": 0,
            "text": "leaf text",
            "source_docs": ["guide.pdf"],
            "source_pages": [1],
        },
        {
            "node_id": "summary-1",
            "is_leaf": False,
            "depth": 1,
            "text": "summary text",
            "child_node_ids": ["leaf-1"],
            "source_docs": ["guide.pdf"],
            "source_pages": [1],
        },
    ]
    path.write_text(
        "\n".join(json.dumps(row) for row in rows),
        encoding="utf-8",
    )


def test_qdrant_indexer_creates_schema_and_upserts_all_vectors(tmp_path):
    nodes_path = tmp_path / "nodes.jsonl"
    manifest_path = tmp_path / "manifest.json"
    write_nodes(nodes_path)
    store = FakeStore()

    manifest = build_qdrant_index(
        QdrantIndexConfig(
            nodes_path=nodes_path,
            batch_size=1,
            manifest_path=manifest_path,
        ),
        store=store,
        dense_embedder=FakeDenseEmbedder(),
        sparse_embedder=FakeSparseEmbedder(),
        colbert_embedder=FakeColBERTEmbedder(),
    )

    assert store.ensure_calls == [{"dense_size": 2, "colbert_size": 2}]
    assert len(store.upserts) == 2
    assert manifest["node_count"] == 2
    assert manifest["embedding_backend"] == "fake"
    assert manifest_path.exists()
