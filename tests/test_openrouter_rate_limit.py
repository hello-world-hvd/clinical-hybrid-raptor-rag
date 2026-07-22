from __future__ import annotations

import unittest

from src.openrouter_client import (
    OpenRouterClient,
    OpenRouterRateLimiter,
    OpenRouterRequestBudgetExceeded,
)
from src.preprocess.contextual_chunking import ContextualChunker


class FakeResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload or {}
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


class SequenceSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def post(self, *args, **kwargs):
        response = self.responses[self.calls]
        self.calls += 1
        return response


class OpenRouterRateLimitTests(unittest.TestCase):
    def test_request_budget_stops_before_an_extra_http_call(self):
        limiter = OpenRouterRateLimiter(
            min_interval_seconds=0,
            max_requests=1,
        )
        limiter.acquire()

        with self.assertRaises(OpenRouterRequestBudgetExceeded):
            limiter.acquire()

        self.assertEqual(1, limiter.request_count)

    def test_retries_429_and_respects_retry_after(self):
        session = SequenceSession(
            [
                FakeResponse(429, headers={"Retry-After": "0"}),
                FakeResponse(
                    200,
                    payload={
                        "choices": [{"message": {"content": "completed"}}],
                    },
                ),
            ]
        )
        client = OpenRouterClient(
            api_key="test-key",
            session=session,
            min_interval_seconds=0,
            max_requests_per_run=5,
            max_retries=2,
        )

        result = client.complete(
            system_prompt="system",
            user_prompt="user",
            max_tokens=10,
        )

        self.assertEqual("completed", result)
        self.assertEqual(2, session.calls)
        self.assertEqual(2, client.request_count)

    def test_contextual_chunker_uses_fallback_after_budget_exhaustion(self):
        session = SequenceSession(
            [
                FakeResponse(
                    200,
                    payload={
                        "choices": [{"message": {"content": "Ngữ cảnh đầu tiên."}}],
                    },
                )
            ]
        )
        chunker = ContextualChunker(
            api_key="test-key-fallback",
            session=session,
            min_interval_seconds=0,
            max_requests_per_run=1,
            autosave=False,
        )

        first = chunker.enrich(
            content="Chunk one",
            doc_name="guide.pdf",
            page_number=1,
        )
        second = chunker.enrich(
            content="Chunk two",
            doc_name="guide.pdf",
            page_number=2,
            headings=["Điều trị"],
        )
        third = chunker.enrich(
            content="Chunk three",
            doc_name="guide.pdf",
            page_number=3,
        )

        self.assertIn("Ngữ cảnh đầu tiên", first["content"])
        self.assertIn("Điều trị", second["context"])
        self.assertIn("trang 3", third["context"])
        self.assertEqual(1, session.calls)
        self.assertEqual(2, chunker.stats["fallback_contexts"])
        self.assertIsNotNone(chunker.stats["api_disabled_reason"])


if __name__ == "__main__":
    unittest.main()
