from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.graph_rag.graph_builder import DEFAULT_GRAPH_PATH
from src.graph_rag.graph_indexer import DEFAULT_GRAPH_COLLECTION
from src.retrieval.ensemble_retriever import EnsembleConfig, EnsembleRetriever
from src.retrieval.qdrant_indexer import DEFAULT_QDRANT_URL


@dataclass(frozen=True)
class GraphRetrieverConfig:
    graph_path: Path = DEFAULT_GRAPH_PATH
    qdrant_url: str = DEFAULT_QDRANT_URL
    collection_name: str = DEFAULT_GRAPH_COLLECTION
    seed_top_k: int = 10
    final_top_k: int = 12
    graph_hops: int = 2
    max_expanded_entities: int = 24
    hop_decay: float = 0.7
    optimize_queries: bool = True
    rerank: bool = True
    embedding_backend: str = "local"
    cache_dir: Path | None = Path("data/cache/embeddings")
    device: str | None = None
    local_files_only: bool = False


class GraphRetriever:
    """Hybrid vector retrieval followed by evidence-preserving graph expansion."""

    def __init__(
        self,
        config: GraphRetrieverConfig | None = None,
        *,
        ensemble: EnsembleRetriever | None = None,
        graph: Mapping[str, Any] | None = None,
    ) -> None:
        self.config = config or GraphRetrieverConfig()
        self.graph = (
            dict(graph)
            if graph is not None
            else json.loads(self.config.graph_path.read_text(encoding="utf-8"))
        )
        self.entities = {
            str(item["node_id"]): item for item in self.graph.get("entities", [])
        }
        self.communities = {
            str(item["community_id"]): item
            for item in self.graph.get("communities", [])
        }
        self.adjacency: dict[str, list[tuple[str, Mapping[str, Any]]]] = defaultdict(list)
        for relation in self.graph.get("relations", []):
            source = str(relation["source"])
            target = str(relation["target"])
            self.adjacency[source].append((target, relation))
            self.adjacency[target].append((source, relation))
        self.ensemble = ensemble or EnsembleRetriever(
            EnsembleConfig(
                qdrant_url=self.config.qdrant_url,
                collection_name=self.config.collection_name,
                final_top_k=self.config.seed_top_k,
                optimize_queries=self.config.optimize_queries,
                rerank=self.config.rerank,
                embedding_backend=self.config.embedding_backend,
                cache_dir=self.config.cache_dir,
                device=self.config.device,
                local_files_only=self.config.local_files_only,
            )
        )

    def _seed_entities(
        self,
        results: Sequence[Mapping[str, Any]],
    ) -> tuple[list[str], dict[str, float]]:
        ordered: list[str] = []
        scores: dict[str, float] = {}
        for rank, result in enumerate(results, start=1):
            seed_score = float(
                result.get("rerank_score")
                if result.get("rerank_score") is not None
                else result.get("score") or 1.0 / rank
            )
            entity_ids = list(result.get("entity_ids") or [])
            if result.get("graph_kind") == "entity" and result.get("node_id"):
                entity_ids.insert(0, result["node_id"])
            if result.get("graph_kind") == "community":
                community = self.communities.get(str(result.get("community_id")), {})
                entity_ids.extend(community.get("entity_ids", []))
            for entity_id in entity_ids:
                entity_id = str(entity_id)
                if entity_id not in self.entities:
                    continue
                if entity_id not in scores:
                    ordered.append(entity_id)
                scores[entity_id] = max(scores.get(entity_id, 0.0), seed_score)
        return ordered, scores

    def _expand(
        self,
        seed_ids: Sequence[str],
        seed_scores: Mapping[str, float],
    ) -> list[dict[str, Any]]:
        queue = deque((entity_id, 0, seed_scores[entity_id]) for entity_id in seed_ids)
        best: dict[str, tuple[int, float, Mapping[str, Any] | None]] = {}
        while queue and len(best) < self.config.max_expanded_entities:
            entity_id, hop, score = queue.popleft()
            previous = best.get(entity_id)
            if previous is not None and previous[1] >= score:
                continue
            best[entity_id] = (hop, score, None)
            if hop >= self.config.graph_hops:
                continue
            for neighbor_id, relation in self.adjacency.get(entity_id, []):
                next_score = (
                    score
                    * self.config.hop_decay
                    * min(2.0, 1.0 + 0.1 * float(relation.get("weight") or 1))
                )
                previous_neighbor = best.get(neighbor_id)
                if previous_neighbor is None or next_score > previous_neighbor[1]:
                    queue.append((neighbor_id, hop + 1, next_score))

        expanded = []
        for entity_id, (hop, score, _) in best.items():
            entity = self.entities[entity_id]
            relation_lines = []
            for neighbor_id, relation in self.adjacency.get(entity_id, [])[:12]:
                neighbor = self.entities.get(neighbor_id, {})
                relation_lines.append(
                    f"{entity['name']} --{relation['type']}--> "
                    f"{neighbor.get('name', neighbor_id)}: "
                    f"{relation.get('description', '')}".strip()
                )
            evidence = list(entity.get("evidence", []))
            text = "\n".join(
                [
                    f"Thực thể: {entity['name']} ({entity['type']})",
                    str(entity.get("description") or ""),
                    *relation_lines,
                    *[
                        f"[Trang {item.get('page_number')}] {item.get('snippet', '')}"
                        for item in evidence[:5]
                    ],
                ]
            ).strip()
            expanded.append(
                {
                    "node_id": entity_id,
                    "graph_kind": "expanded_entity",
                    "entity_ids": [entity_id],
                    "community_id": entity.get("community_id"),
                    "graph_hop": hop,
                    "graph_score": score,
                    "score": score,
                    "rerank_score": None,
                    "text": text,
                    "snippet": text[:450],
                    "source_docs": sorted(
                        {
                            str(item.get("doc_name"))
                            for item in evidence
                            if item.get("doc_name")
                        }
                    ),
                    "source_pages": sorted(
                        {
                            int(item.get("page_number"))
                            for item in evidence
                            if item.get("page_number")
                        }
                    ),
                    "source_chunk_ids": sorted(
                        {
                            str(item.get("chunk_id"))
                            for item in evidence
                            if item.get("chunk_id")
                        }
                    ),
                    "matches": {"graph_expansion": {"hop": hop, "score": score}},
                }
            )
        return sorted(
            expanded,
            key=lambda item: (item["graph_hop"], -item["graph_score"]),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        optimize: bool | None = None,
        rerank: bool | None = None,
    ) -> list[dict[str, Any]]:
        seeds = self.ensemble.search(
            query,
            top_k=self.config.seed_top_k,
            optimize=optimize,
            rerank=rerank,
        )
        seed_ids, scores = self._seed_entities(seeds)
        expanded = self._expand(seed_ids, scores)
        seed_node_ids = {str(item.get("node_id")) for item in seeds}
        combined = list(seeds)
        combined.extend(
            item for item in expanded if item["node_id"] not in seed_node_ids
        )
        limit = top_k if top_k is not None else self.config.final_top_k
        return combined[:limit]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query the Graph RAG Qdrant collection and expand graph evidence."
    )
    parser.add_argument("query")
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_GRAPH_COLLECTION)
    parser.add_argument("--seed-top-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--graph-hops", type=int, default=2)
    parser.add_argument("--max-expanded-entities", type=int, default=24)
    parser.add_argument("--hop-decay", type=float, default=0.7)
    parser.add_argument("--embedding-backend", choices=["auto", "local", "modal"], default="local")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache/embeddings"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--no-optimize", action="store_true")
    parser.add_argument("--no-rerank", action="store_true")
    return parser.parse_args()


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    retriever = GraphRetriever(
        GraphRetrieverConfig(
            graph_path=args.graph,
            qdrant_url=args.qdrant_url,
            collection_name=args.collection,
            seed_top_k=max(1, args.seed_top_k),
            final_top_k=max(1, args.top_k),
            graph_hops=max(0, args.graph_hops),
            max_expanded_entities=max(1, args.max_expanded_entities),
            hop_decay=max(0.0, args.hop_decay),
            optimize_queries=not args.no_optimize,
            rerank=not args.no_rerank,
            embedding_backend=args.embedding_backend,
            cache_dir=args.cache_dir,
            device=args.device,
            local_files_only=args.local_files_only,
        )
    )
    print(
        json.dumps(
            retriever.search(args.query, top_k=args.top_k),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
