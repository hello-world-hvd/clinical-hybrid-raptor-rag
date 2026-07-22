from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx

from src.graph_rag.community_summarizer import summarize_community
from src.preprocess.normalization import normalize_for_search, normalize_text, stable_id


DEFAULT_OUTPUT_DIR = Path("data/processed/graph_rag/ngo-doc")
DEFAULT_EXTRACTIONS_PATH = DEFAULT_OUTPUT_DIR / "extractions.jsonl"
DEFAULT_GRAPH_PATH = DEFAULT_OUTPUT_DIR / "knowledge_graph.json"


@dataclass(frozen=True)
class GraphBuildConfig:
    extractions_path: Path = DEFAULT_EXTRACTIONS_PATH
    graph_path: Path = DEFAULT_GRAPH_PATH
    min_entity_mentions: int = 1
    min_relation_mentions: int = 1
    evidence_chars: int = 420
    community_resolution: float = 1.0
    community_seed: int = 42


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def canonical_name(name: str) -> str:
    value = normalize_for_search(name)
    value = re.sub(r"[^\w%+./-]+", " ", value, flags=re.UNICODE)
    return " ".join(value.split()).casefold()


def _entity_id(name: str) -> str:
    return f"entity:{stable_id(canonical_name(name))}"


def _evidence(row: Mapping[str, Any], max_chars: int) -> dict[str, Any]:
    content = normalize_text(str(row.get("content") or ""), preserve_lines=False)
    return {
        "chunk_id": str(row.get("chunk_id") or ""),
        "doc_name": str(row.get("doc_name") or ""),
        "page_number": int(row.get("page_number") or 0),
        "chunk_type": str(row.get("chunk_type") or "text"),
        "snippet": content[:max_chars],
    }


def _detect_communities(
    entity_ids: list[str],
    relations: list[dict[str, Any]],
    *,
    resolution: float,
    seed: int,
) -> list[set[str]]:
    graph = nx.Graph()
    graph.add_nodes_from(entity_ids)
    for relation in relations:
        graph.add_edge(
            relation["source"],
            relation["target"],
            weight=float(relation.get("weight") or 1),
        )
    communities: list[set[str]] = []
    for component_ids in nx.connected_components(graph):
        component = graph.subgraph(component_ids)
        if component.number_of_nodes() < 3 or component.number_of_edges() == 0:
            communities.append(set(component_ids))
            continue
        detected = nx.community.louvain_communities(
            component,
            weight="weight",
            resolution=resolution,
            seed=seed,
        )
        communities.extend(set(items) for items in detected)
    return communities


def build_knowledge_graph(config: GraphBuildConfig) -> dict[str, Any]:
    entity_accumulator: dict[str, dict[str, Any]] = {}
    relation_accumulator: dict[tuple[str, str, str], dict[str, Any]] = {}

    for row in read_jsonl(config.extractions_path):
        evidence = _evidence(row, config.evidence_chars)
        names_in_chunk: dict[str, str] = {}
        for raw in row.get("entities", []):
            name = normalize_text(str(raw.get("name") or ""), preserve_lines=False)
            canonical = canonical_name(name)
            if not canonical:
                continue
            entity_id = _entity_id(name)
            names_in_chunk[canonical] = entity_id
            current = entity_accumulator.setdefault(
                entity_id,
                {
                    "node_id": entity_id,
                    "name": name,
                    "aliases": set(),
                    "type_counts": Counter(),
                    "descriptions": [],
                    "evidence": {},
                },
            )
            current["aliases"].add(name)
            current["type_counts"][str(raw.get("type") or "OTHER")] += 1
            description = normalize_text(
                str(raw.get("description") or ""), preserve_lines=False
            )
            if description and description not in current["descriptions"]:
                current["descriptions"].append(description)
            current["evidence"][evidence["chunk_id"]] = evidence

        for raw in row.get("relations", []):
            source_name = normalize_text(
                str(raw.get("source") or ""), preserve_lines=False
            )
            target_name = normalize_text(
                str(raw.get("target") or ""), preserve_lines=False
            )
            source_key = canonical_name(source_name)
            target_key = canonical_name(target_name)
            if not source_key or not target_key or source_key == target_key:
                continue
            source_id = names_in_chunk.get(source_key) or _entity_id(source_name)
            target_id = names_in_chunk.get(target_key) or _entity_id(target_name)
            for entity_id, name in ((source_id, source_name), (target_id, target_name)):
                entity_accumulator.setdefault(
                    entity_id,
                    {
                        "node_id": entity_id,
                        "name": name,
                        "aliases": {name},
                        "type_counts": Counter({"OTHER": 1}),
                        "descriptions": [],
                        "evidence": {evidence["chunk_id"]: evidence},
                    },
                )
            relation_type = str(raw.get("type") or "RELATED_TO").upper()
            key = (source_id, target_id, relation_type)
            current_relation = relation_accumulator.setdefault(
                key,
                {
                    "source": source_id,
                    "target": target_id,
                    "type": relation_type,
                    "descriptions": [],
                    "evidence": {},
                    "weight": 0,
                },
            )
            description = normalize_text(
                str(raw.get("description") or ""), preserve_lines=False
            )
            if description and description not in current_relation["descriptions"]:
                current_relation["descriptions"].append(description)
            current_relation["evidence"][evidence["chunk_id"]] = evidence
            current_relation["weight"] += 1

    entities = []
    retained_ids = set()
    for entity_id, item in entity_accumulator.items():
        mention_count = len(item["evidence"])
        if mention_count < config.min_entity_mentions:
            continue
        retained_ids.add(entity_id)
        type_counts: Counter = item["type_counts"]
        entity_type = type_counts.most_common(1)[0][0] if type_counts else "OTHER"
        entities.append(
            {
                "node_id": entity_id,
                "name": item["name"],
                "aliases": sorted(item["aliases"]),
                "type": entity_type,
                "description": " ".join(item["descriptions"][:3]),
                "mention_count": mention_count,
                "evidence": list(item["evidence"].values()),
            }
        )

    relations = []
    for item in relation_accumulator.values():
        if (
            item["source"] not in retained_ids
            or item["target"] not in retained_ids
            or item["weight"] < config.min_relation_mentions
        ):
            continue
        relation_id = "relation:" + stable_id(
            item["source"], item["type"], item["target"]
        )
        relations.append(
            {
                "relation_id": relation_id,
                "source": item["source"],
                "target": item["target"],
                "type": item["type"],
                "description": " ".join(item["descriptions"][:3]),
                "weight": item["weight"],
                "evidence": list(item["evidence"].values()),
            }
        )

    entity_map = {entity["node_id"]: entity for entity in entities}
    detected = _detect_communities(
        list(entity_map),
        relations,
        resolution=config.community_resolution,
        seed=config.community_seed,
    )
    communities = []
    for index, member_ids in enumerate(
        sorted(detected, key=lambda items: (-len(items), sorted(items))),
        start=1,
    ):
        community_id = f"community:{index:04d}"
        summary = summarize_community(
            community_id=community_id,
            entity_ids=sorted(member_ids),
            entities=entity_map,
            relations=relations,
        )
        communities.append(summary)
        for entity_id in member_ids:
            entity_map[entity_id]["community_id"] = community_id

    graph = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source": str(config.extractions_path),
        "entities": sorted(entities, key=lambda item: item["node_id"]),
        "relations": sorted(relations, key=lambda item: item["relation_id"]),
        "communities": communities,
        "stats": {
            "entities": len(entities),
            "relations": len(relations),
            "communities": len(communities),
            "entity_types": dict(
                sorted(Counter(item["type"] for item in entities).items())
            ),
        },
    }
    config.graph_path.parent.mkdir(parents=True, exist_ok=True)
    config.graph_path.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return graph


def parse_args() -> GraphBuildConfig:
    parser = argparse.ArgumentParser(
        description="Merge cached entity extraction into a Graph RAG knowledge graph."
    )
    parser.add_argument("--extractions", type=Path, default=DEFAULT_EXTRACTIONS_PATH)
    parser.add_argument("--graph", type=Path, default=DEFAULT_GRAPH_PATH)
    parser.add_argument("--min-entity-mentions", type=int, default=1)
    parser.add_argument("--min-relation-mentions", type=int, default=1)
    parser.add_argument("--evidence-chars", type=int, default=420)
    parser.add_argument("--community-resolution", type=float, default=1.0)
    args = parser.parse_args()
    return GraphBuildConfig(
        extractions_path=args.extractions,
        graph_path=args.graph,
        min_entity_mentions=max(1, args.min_entity_mentions),
        min_relation_mentions=max(1, args.min_relation_mentions),
        evidence_chars=max(80, args.evidence_chars),
        community_resolution=max(0.1, args.community_resolution),
    )


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    graph = build_knowledge_graph(parse_args())
    print(json.dumps(graph["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
