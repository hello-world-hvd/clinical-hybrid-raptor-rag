from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.openrouter_client import DEFAULT_OPENROUTER_MODEL
from src.raptor.build_raptor_tree import (
    BuildConfig,
    build_summary_prompt,
    openrouter_summarize_clusters,
)


class FakeOpenRouterClient:
    instances = []

    def __init__(self, *, model, timeout):
        self.model = model
        self.timeout = timeout
        self.calls = []
        self.__class__.instances.append(self)

    def complete(self, **kwargs):
        self.calls.append(kwargs)
        return "Tóm tắt cụm tài liệu y khoa."


class RaptorSummarizationTests(unittest.TestCase):
    def setUp(self):
        FakeOpenRouterClient.instances.clear()

    def test_prompt_keeps_medical_summary_constraints(self):
        prompt = build_summary_prompt(
            ["Điều trị bằng atropine.", "Theo dõi nhịp tim."],
            max_input_chars=1000,
        )

        self.assertIn("Không thêm thông tin ngoài nguồn", prompt)
        self.assertIn("[1] Điều trị bằng atropine.", prompt)
        self.assertIn("[2] Theo dõi nhịp tim.", prompt)

    @patch(
        "src.raptor.build_raptor_tree.OpenRouterClient",
        FakeOpenRouterClient,
    )
    def test_uses_gemma_on_openrouter_for_each_cluster(self):
        config = BuildConfig(
            input_path=Path("input"),
            output_dir=Path("output"),
        )
        summaries = openrouter_summarize_clusters(
            [["Đoạn một."], ["Đoạn hai."]],
            config,
        )

        client = FakeOpenRouterClient.instances[0]
        self.assertEqual(DEFAULT_OPENROUTER_MODEL, client.model)
        self.assertEqual(2, len(client.calls))
        self.assertEqual(384, client.calls[0]["max_tokens"])
        self.assertEqual(
            ["Tóm tắt cụm tài liệu y khoa.", "Tóm tắt cụm tài liệu y khoa."],
            summaries,
        )


if __name__ == "__main__":
    unittest.main()
