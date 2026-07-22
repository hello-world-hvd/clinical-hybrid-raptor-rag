import json

from src.retrieval.query_optimizer import QueryOptimizer


class FakeChatClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(self, **_: object) -> str:
        self.calls += 1
        return json.dumps(
            {
                "hyde_document": "Tài liệu giả định về xử trí ngộ độc.",
                "multi_queries": [
                    "Điều trị ngộ độc như thế nào?",
                    "Phác đồ xử trí nhiễm độc",
                    "Câu hỏi gốc",
                ],
                "sub_questions": ["Chẩn đoán ngộ độc?", "Điều trị ngộ độc?"],
                "step_back_query": "Nguyên tắc chung khi xử trí ngộ độc?",
            },
            ensure_ascii=False,
        )


def test_query_optimizer_generates_and_caches_all_strategies(tmp_path):
    client = FakeChatClient()
    optimizer = QueryOptimizer(
        client=client,
        multi_query_count=2,
        cache_path=tmp_path / "queries.json",
    )

    optimized = optimizer.optimize("Câu hỏi gốc")
    cached = optimizer.optimize("Câu hỏi gốc")

    assert optimized.hyde_document.startswith("Tài liệu giả định")
    assert optimized.multi_queries == [
        "Điều trị ngộ độc như thế nào?",
        "Phác đồ xử trí nhiễm độc",
    ]
    assert optimized.sub_questions == ["Chẩn đoán ngộ độc?", "Điều trị ngộ độc?"]
    assert optimized.step_back_query == "Nguyên tắc chung khi xử trí ngộ độc?"
    assert cached == optimized
    assert client.calls == 1


def test_public_strategy_methods_share_the_cached_optimization(tmp_path):
    client = FakeChatClient()
    optimizer = QueryOptimizer(client=client, cache_path=tmp_path / "queries.json")

    assert optimizer.hyde("Câu hỏi gốc")
    assert optimizer.multi_query("Câu hỏi gốc")
    assert optimizer.decompose("Câu hỏi gốc")
    assert optimizer.step_back("Câu hỏi gốc")
    assert client.calls == 1
