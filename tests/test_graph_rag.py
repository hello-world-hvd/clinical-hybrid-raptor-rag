from __future__ import annotations

import json

from src.graph_rag.entity_extractor import (
    EntityExtractor,
    ExtractionConfig,
    parse_extraction_response,
    run_extraction,
)
from src.graph_rag.graph_builder import GraphBuildConfig, build_knowledge_graph
from src.graph_rag.graph_indexer import graph_to_documents
from src.graph_rag.graph_retriever import GraphRetriever, GraphRetrieverConfig


class FakeExtractionClient:
    def __init__(self):
        self.request_count = 0
        self.batch_sizes = []

    def complete(self, **kwargs):
        self.request_count += 1
        prompt = kwargs["user_prompt"]
        payload = json.loads(prompt.split("INPUT_CHUNKS:\n", 1)[1])
        self.batch_sizes.append(len(payload))
        return json.dumps(
            {
                "chunks": [
                    {
                        "chunk_id": item["chunk_id"],
                        "entities": [
                            {
                                "name": "Paracetamol",
                                "type": "DRUG",
                                "description": "Thuốc có thể gây ngộ độc.",
                            },
                            {
                                "name": "N-acetylcystein",
                                "type": "DRUG",
                                "description": "Thuốc giải độc.",
                            },
                        ],
                        "relations": [
                            {
                                "source": "N-acetylcystein",
                                "target": "Paracetamol",
                                "type": "DIEU_TRI_NGO_DOC",
                                "description": "Được dùng điều trị ngộ độc.",
                            }
                        ],
                    }
                    for item in payload
                ]
            },
            ensure_ascii=False,
        )


def test_parse_extraction_response_accepts_fenced_json():
    payload = parse_extraction_response('```json\n{"chunks": []}\n```')
    assert payload == {"chunks": []}


def test_extraction_uses_batches_of_twenty_and_cache(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"chunk-{index}",
                    "doc_name": "Ngo-doc.pdf",
                    "page_number": index + 1,
                    "chunk_type": "text",
                    "content": "Nội dung về ngộ độc paracetamol và điều trị NAC.",
                },
                ensure_ascii=False,
            )
            for index in range(21)
        ),
        encoding="utf-8",
    )
    config = ExtractionConfig(
        chunks_path=chunks_path,
        output_dir=tmp_path / "graph",
        batch_size=20,
    )
    client = FakeExtractionClient()
    extractor = EntityExtractor(config, client=client)

    first = run_extraction(config, extractor=extractor)
    second = run_extraction(config, extractor=extractor)

    assert client.batch_sizes == [20, 1]
    assert first["completed_chunks"] == 21
    assert second["pending_chunks"] == 0


def test_builder_merges_entities_and_keeps_provenance(tmp_path):
    extractions = tmp_path / "extractions.jsonl"
    rows = [
        {
            "chunk_id": f"chunk-{page}",
            "doc_name": "Ngo-doc.pdf",
            "page_number": page,
            "chunk_type": "text",
            "content": f"Bằng chứng trang {page}",
            "entities": [
                {"name": "Paracetamol", "type": "DRUG", "description": "Thuốc"},
                {"name": "NAC", "type": "DRUG", "description": "Giải độc"},
            ],
            "relations": [
                {
                    "source": "NAC",
                    "target": "Paracetamol",
                    "type": "DIEU_TRI_NGO_DOC",
                    "description": "NAC điều trị ngộ độc paracetamol",
                }
            ],
        }
        for page in (10, 11)
    ]
    extractions.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
        encoding="utf-8",
    )
    graph = build_knowledge_graph(
        GraphBuildConfig(
            extractions_path=extractions,
            graph_path=tmp_path / "graph.json",
        )
    )
    documents = graph_to_documents(graph)

    assert graph["stats"]["entities"] == 2
    assert graph["stats"]["relations"] == 1
    assert graph["relations"][0]["weight"] == 2
    assert {10, 11} == {
        item["page_number"] for item in graph["relations"][0]["evidence"]
    }
    assert {"entity", "relation", "community"} <= {
        item["graph_kind"] for item in documents
    }


class FakeEnsemble:
    def search(self, query, **kwargs):
        return [
            {
                "node_id": "entity:a",
                "graph_kind": "entity",
                "entity_ids": ["entity:a"],
                "score": 1.0,
                "text": "A",
            }
        ]


def test_graph_retriever_expands_neighbor_entities():
    graph = {
        "entities": [
            {
                "node_id": "entity:a",
                "name": "A",
                "type": "DRUG",
                "evidence": [],
                "community_id": "community:1",
            },
            {
                "node_id": "entity:b",
                "name": "B",
                "type": "SYMPTOM",
                "evidence": [
                    {
                        "chunk_id": "chunk-b",
                        "doc_name": "Ngo-doc.pdf",
                        "page_number": 7,
                        "snippet": "Bằng chứng B",
                    }
                ],
                "community_id": "community:1",
            },
        ],
        "relations": [
            {
                "source": "entity:a",
                "target": "entity:b",
                "type": "GAY_RA",
                "description": "A gây B",
                "weight": 1,
            }
        ],
        "communities": [
            {
                "community_id": "community:1",
                "entity_ids": ["entity:a", "entity:b"],
            }
        ],
    }
    retriever = GraphRetriever(
        GraphRetrieverConfig(graph_hops=1, final_top_k=5),
        ensemble=FakeEnsemble(),
        graph=graph,
    )

    results = retriever.search("A gây gì?")

    assert [item["node_id"] for item in results] == ["entity:a", "entity:b"]
    assert results[1]["graph_hop"] == 1
    assert results[1]["source_pages"] == [7]
