from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from src.openrouter_client import DEFAULT_OPENROUTER_MODEL
from src.preprocess.contextual_chunking import ContextualChunker
from src.preprocess.text import split_text_blocks_into_chunks


class FakeResponse:
    status_code = 200
    headers = {}

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            "Đoạn này thuộc mục điều trị nhịp tim chậm "
                            "trong tài liệu."
                        )
                    }
                }
            ]
        }


class FakeSession:
    def __init__(self):
        self.calls = 0
        self.last_request = None

    def post(self, url, **kwargs):
        self.calls += 1
        self.last_request = {"url": url, **kwargs}
        return FakeResponse()


class FakeBatchClient:
    def __init__(self):
        self.batch_sizes = []
        self.request_count = 0

    def complete(self, **kwargs):
        match = re.search(
            r"<items>\s*(\[.*\])\s*</items>",
            kwargs["user_prompt"],
            flags=re.DOTALL,
        )
        items = json.loads(match.group(1))
        self.batch_sizes.append(len(items))
        self.request_count += 1
        return json.dumps(
            {
                "contexts": [
                    {
                        "id": item["id"],
                        "context": f"Ngữ cảnh cho chunk {item['id']}.",
                    }
                    for item in items
                ]
            },
            ensure_ascii=False,
        )


class ContextualChunkerTests(unittest.TestCase):
    def test_calls_openrouter_and_prefixes_context(self):
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            chunker = ContextualChunker(
                api_key="test-key",
                cache_path=Path(directory) / "contexts.json",
                session=session,
                min_interval_seconds=0,
                max_requests_per_run=10,
            )
            result = chunker.enrich(
                content="Atropine 0,5 mg tĩnh mạch.",
                doc_name="clinical.pdf",
                page_number=12,
                headings=["Điều trị", "Nhịp tim chậm"],
                document_context="Hướng dẫn xử trí cấp cứu.",
            )

        self.assertEqual(1, session.calls)
        self.assertEqual(
            DEFAULT_OPENROUTER_MODEL,
            session.last_request["json"]["model"],
        )
        self.assertTrue(result["content"].startswith("[Ngữ cảnh:"))
        self.assertTrue(result["content"].endswith("Atropine 0,5 mg tĩnh mạch."))

    def test_batches_twenty_uncached_chunks_per_request(self):
        client = FakeBatchClient()
        chunks = [
            {
                "content": f"Chunk {index}",
                "doc_name": "clinical.pdf",
                "page_number": index // 5 + 1,
                "headings": ["Điều trị"],
                "document_context": "Nội dung trang.",
            }
            for index in range(45)
        ]
        with tempfile.TemporaryDirectory() as directory:
            chunker = ContextualChunker(
                client=client,
                cache_path=Path(directory) / "contexts.json",
                autosave=False,
            )
            first_results = chunker.enrich_batch(chunks, batch_size=20)
            second_results = chunker.enrich_batch(chunks, batch_size=20)

        self.assertEqual([20, 20, 5], client.batch_sizes)
        self.assertEqual(45, len(first_results))
        self.assertEqual(first_results, second_results)
        self.assertEqual(3, client.request_count)
        self.assertEqual(45, chunker.stats["cache_hits"])
        self.assertEqual(45, chunker.stats["generated_contexts"])

    def test_uses_cache_without_second_api_call(self):
        session = FakeSession()
        with tempfile.TemporaryDirectory() as directory:
            cache_path = Path(directory) / "contexts.json"
            first = ContextualChunker(
                api_key="test-key",
                cache_path=cache_path,
                session=session,
                min_interval_seconds=0,
                max_requests_per_run=10,
            )
            kwargs = {
                "content": "Nội dung cần contextualize.",
                "doc_name": "clinical.pdf",
                "page_number": 1,
                "headings": ["Chẩn đoán"],
                "document_context": "Ngữ cảnh trang.",
            }
            first.enrich(**kwargs)

            second = ContextualChunker(
                api_key="test-key",
                cache_path=cache_path,
                session=session,
                min_interval_seconds=0,
                max_requests_per_run=10,
            )
            second.enrich(**kwargs)

        self.assertEqual(1, session.calls)


class BaseChunkingTests(unittest.TestCase):
    def test_heading_aware_chunks_are_created(self):
        blocks = [
            {
                "block_index": 1,
                "bbox": [0, 0, 100, 20],
                "text": "1. Chẩn đoán",
                "avg_font_size": 16,
            },
            {
                "block_index": 2,
                "bbox": [0, 30, 100, 80],
                "text": "Khám lâm sàng và thực hiện xét nghiệm cần thiết.",
                "avg_font_size": 11,
            },
        ]

        chunks = split_text_blocks_into_chunks(
            blocks,
            max_chars=200,
            overlap_words=4,
        )

        self.assertEqual(1, len(chunks))
        self.assertIn("1. Chẩn đoán", chunks[0]["content"])

    def test_oversized_text_is_split_within_size_limit(self):
        blocks = [
            {
                "block_index": 1,
                "bbox": [0, 0, 100, 200],
                "text": " ".join(f"word{index}" for index in range(80)),
                "avg_font_size": 11,
            }
        ]
        chunks = split_text_blocks_into_chunks(
            blocks,
            max_chars=90,
            overlap_words=8,
        )

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk["content"]) <= 90 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
