from __future__ import annotations

import json
import unittest

from src.evaluation.ragas_testset_generator import (
    SourceDocument,
    build_overview_documents,
    build_page_documents,
    build_table_visual_documents,
    normalize_ragas_records,
    validate_category,
)


def source_context(*, pages, chunk_ids, source_types):
    metadata = {
        "doc_name": "Ngo-doc.pdf",
        "source_pages": pages,
        "source_chunk_ids": chunk_ids,
        "source_types": source_types,
    }
    return (
        "<source_metadata>"
        f"{json.dumps(metadata, ensure_ascii=False)}"
        "</source_metadata>\nNội dung tham chiếu."
    )


class RagasSourcePreparationTests(unittest.TestCase):
    def setUp(self):
        self.chunks = []
        for page in range(1, 13):
            self.chunks.append(
                {
                    "id": f"text-{page}",
                    "doc_name": "Ngo-doc.pdf",
                    "page_number": page,
                    "chunk_type": "text",
                    "chunk_index": 1,
                    "content": (
                        "[Ngữ cảnh: fallback]\n"
                        f"Nội dung ngộ độc tại trang {page}."
                    ),
                }
            )
        self.chunks.extend(
            [
                {
                    "id": "table-4",
                    "doc_name": "Ngo-doc.pdf",
                    "page_number": 4,
                    "chunk_type": "table",
                    "chunk_index": 1,
                    "content": "| Thuốc | Liều |\n|---|---|\n| NAC | 140 mg/kg |",
                    "reconstruction": {"asset_path": "table.png"},
                },
                {
                    "id": "visual-7",
                    "doc_name": "Ngo-doc.pdf",
                    "page_number": 7,
                    "chunk_type": "visual",
                    "chunk_index": 1,
                    "content": "[DIAGRAM] Sơ đồ xử trí",
                    "reconstruction": {"asset_path": "visual.png"},
                },
            ]
        )

    def test_builds_page_and_broad_overview_documents(self):
        page_documents = build_page_documents(self.chunks)
        overview_documents = build_overview_documents(
            page_documents,
            window_pages=4,
            stride_pages=3,
        )

        self.assertEqual(12, len(page_documents))
        self.assertTrue(
            all(len(document.metadata["source_pages"]) == 1 for document in page_documents)
        )
        self.assertTrue(
            all(
                len(document.metadata["source_pages"]) >= 3
                for document in overview_documents
            )
        )

    def test_table_visual_documents_keep_modality_provenance(self):
        page_documents = build_page_documents(self.chunks)
        documents = build_table_visual_documents(self.chunks, page_documents)

        self.assertEqual(2, len(documents))
        self.assertEqual(
            {"table", "visual"},
            {
                document.metadata["source_types"][0]
                for document in documents
            },
        )
        self.assertEqual(
            {"table-4", "visual-7"},
            {
                document.metadata["source_chunk_ids"][0]
                for document in documents
            },
        )


class RagasOutputValidationTests(unittest.TestCase):
    def test_normalizes_ragas_records_and_extracts_provenance(self):
        records = normalize_ragas_records(
            [
                {
                    "user_input": "Câu hỏi thử nghiệm?",
                    "reference": "Câu trả lời.",
                    "reference_contexts": [
                        source_context(
                            pages=[4, 7],
                            chunk_ids=["text-4", "text-7"],
                            source_types=["text"],
                        )
                    ],
                    "synthesizer_name": "multi_hop_specific",
                }
            ],
            category="multi_hop",
        )

        self.assertEqual([4, 7], records[0]["source_pages"])
        self.assertEqual(["text-4", "text-7"], records[0]["source_chunk_ids"])
        validate_category(records, category="multi_hop", expected_count=1)

    def test_validates_requested_category_constraints(self):
        overview = [
            {
                "id": "overview-1",
                "source_pages": [1, 2, 3, 4],
                "source_types": ["text"],
            }
        ]
        table_visual = [
            {
                "id": "table-1",
                "source_pages": [11],
                "source_types": ["table"],
            }
        ]

        validate_category(overview, category="overview", expected_count=1)
        validate_category(
            table_visual,
            category="table_visual",
            expected_count=1,
        )

        with self.assertRaises(ValueError):
            validate_category(
                [
                    {
                        "id": "bad-multi-hop",
                        "source_pages": [1],
                        "source_types": ["text"],
                    }
                ],
                category="multi_hop",
                expected_count=1,
            )


if __name__ == "__main__":
    unittest.main()
