from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.graph_rag.graph_builder import DEFAULT_GRAPH_PATH
from src.retrieval.qdrant_indexer import (
    DEFAULT_QDRANT_URL,
    QdrantIndexConfig,
    build_qdrant_index,
)


DEFAULT_GRAPH_NODES_PATH = DEFAULT_GRAPH_PATH.with_name("graph_nodes.jsonl")
DEFAULT_GRAPH_COLLECTION = "clinical_graph_rag"
DEFAULT_GRAPH_INDEX_MANIFEST = DEFAULT_GRAPH_PATH.with_name(
    "qdrant_index_manifest.json"
)


@dataclass(frozen=True)
class GraphIndexConfig:
    graph_path: Path = DEFAULT_GRAPH_PATH
    nodes_path: Path = DEFAULT_GRAPH_NODES_PATH
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection_name: str = DEFAULT_GRAPH_COLLECTION
    batch_size: int = 16
    embedding_backend: str = "local"
    device: str | None = None
    cache_dir: Path | None = Path("data/cache/embeddings")
    local_files_only: bool = False
    recreate: bool = False
    prepare_only: bool = False
    manifest_path: Path = DEFAULT_GRAPH_INDEX_MANIFEST


def _source_fields(evidence: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    rows = list(evidence)
    return {
        "source_docs": sorted(
            {str(item.get("doc_name") or "") for item in rows if item.get("doc_name")}
        ),
        "source_pages": sorted(
            {int(item.get("page_number") or 0) for item in rows if item.get("page_number")}
        ),
        "source_chunk_ids": sorted(
            {str(item.get("chunk_id") or "") for item in rows if item.get("chunk_id")}
        ),
        "source_types": sorted(
            {str(item.get("chunk_type") or "") for item in rows if item.get("chunk_type")}
        ),
    }


def graph_to_documents(graph: Mapping[str, Any]) -> list[dict[str, Any]]:
    entities = {
        str(item["node_id"]): item for item in graph.get("entities", [])
    }
    relation_by_entity: dict[str, list[Mapping[str, Any]]] = {
        entity_id: [] for entity_id in entities
    }
    for relation in graph.get("relations", []):
        relation_by_entity.setdefault(str(relation["source"]), []).append(relation)
        relation_by_entity.setdefault(str(relation["target"]), []).append(relation)

    documents: list[dict[str, Any]] = []
    for entity_id, entity in entities.items():
        relation_lines = []
        all_evidence = list(entity.get("evidence", []))
        for relation in relation_by_entity.get(entity_id, [])[:20]:
            source = entities.get(str(relation["source"]), {}).get(
                "name", relation["source"]
            )
            target = entities.get(str(relation["target"]), {}).get(
                "name", relation["target"]
            )
            relation_lines.append(
                f"{source} --{relation['type']}--> {target}: "
                f"{relation.get('description', '')}".strip()
            )
            all_evidence.extend(relation.get("evidence", []))
        evidence_lines = [
            f"[Trang {item.get('page_number')}] {item.get('snippet', '')}"
            for item in entity.get("evidence", [])[:5]
        ]
        text = "\n".join(
            part
            for part in (
                f"Thực thể: {entity['name']} ({entity['type']})",
                f"Mô tả: {entity.get('description', '')}",
                "Quan hệ:\n" + "\n".join(relation_lines) if relation_lines else "",
                "Bằng chứng:\n" + "\n".join(evidence_lines) if evidence_lines else "",
            )
            if part
        )
        documents.append(
            {
                "node_id": entity_id,
                "graph_kind": "entity",
                "entity_ids": [entity_id],
                "community_id": entity.get("community_id"),
                "is_leaf": True,
                "depth": 0,
                "text": text,
                **_source_fields(all_evidence),
            }
        )

    for relation in graph.get("relations", []):
        source = entities[str(relation["source"])]
        target = entities[str(relation["target"])]
        evidence_lines = [
            f"[Trang {item.get('page_number')}] {item.get('snippet', '')}"
            for item in relation.get("evidence", [])[:5]
        ]
        documents.append(
            {
                "node_id": str(relation["relation_id"]),
                "graph_kind": "relation",
                "entity_ids": [relation["source"], relation["target"]],
                "community_id": source.get("community_id"),
                "relation_type": relation["type"],
                "is_leaf": True,
                "depth": 0,
                "text": (
                    f"Quan hệ: {source['name']} --{relation['type']}--> "
                    f"{target['name']}.\n{relation.get('description', '')}\n"
                    + "\n".join(evidence_lines)
                ).strip(),
                **_source_fields(relation.get("evidence", [])),
            }
        )

    for community in graph.get("communities", []):
        evidence = []
        for entity_id in community.get("entity_ids", []):
            evidence.extend(entities.get(str(entity_id), {}).get("evidence", []))
        documents.append(
            {
                "node_id": str(community["community_id"]),
                "graph_kind": "community",
                "entity_ids": list(community.get("entity_ids", [])),
                "community_id": community["community_id"],
                "is_leaf": True,
                "depth": 1,
                "text": (
                    f"Chủ đề cộng đồng: {community.get('title', '')}\n"
                    f"{community.get('summary', '')}"
                ).strip(),
                **_source_fields(evidence),
            }
        )
    return documents


def write_graph_documents(
    graph_path: Path = DEFAULT_GRAPH_PATH,
    nodes_path: Path = DEFAULT_GRAPH_NODES_PATH,
) -> list[dict[str, Any]]:
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    documents = graph_to_documents(graph)
    nodes_path.parent.mkdir(parents=True, exist_ok=True)
    with nodes_path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    return documents


def build_graph_index(config: GraphIndexConfig) -> dict[str, Any]:
    documents = write_graph_documents(config.graph_path, config.nodes_path)
    if config.prepare_only:
        return {
            "nodes_path": str(config.nodes_path),
            "node_count": len(documents),
            "collection_name": config.collection_name,
            "indexed": False,
        }
    return build_qdrant_index(
        QdrantIndexConfig(
            nodes_path=config.nodes_path,
            qdrant_url=config.qdrant_url,
            collection_name=config.collection_name,
            embedding_backend=config.embedding_backend,
            batch_size=config.batch_size,
            device=config.device,
            cache_dir=config.cache_dir,
            local_files_only=config.local_files_only,
            recreate=config.recreate,
            manifest_path=config.manifest_path,
        )
    )


def parse_args() -> GraphIndexConfig:
    parser = argparse.ArgumentParser(
        description="Prepare Graph RAG documents and index them into Qdrant."
    )
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_GRAPH_NODES_PATH)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_GRAPH_COLLECTION)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--embedding-backend",
        choices=["auto", "local", "modal"],
        default="local",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/embeddings"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_GRAPH_INDEX_MANIFEST)
    args = parser.parse_args()
    return GraphIndexConfig(
        graph_path=args.graph,
        nodes_path=args.nodes,
        qdrant_url=args.qdrant_url,
        collection_name=args.collection,
        batch_size=max(1, args.batch_size),
        embedding_backend=args.embedding_backend,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
        recreate=args.recreate,
        prepare_only=args.prepare_only,
        manifest_path=args.manifest,
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(build_graph_index(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
