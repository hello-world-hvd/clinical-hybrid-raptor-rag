from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import types
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from dotenv import load_dotenv
from tqdm.auto import tqdm

from src.embedding import DEFAULT_DENSE_MODEL, DenseEmbedder
from src.openrouter_client import (
    DEFAULT_MIN_INTERVAL_SECONDS,
    DEFAULT_OPENROUTER_MODEL,
    OpenRouterClient,
)
from src.preprocess.normalization import normalize_text


DEFAULT_CHUNKS_PATH = Path(
    "data/processed/preprocess_output/ngo-doc/chunks.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("data/test")
SOURCE_PATTERN = re.compile(
    r"<source_metadata>(.*?)</source_metadata>",
    flags=re.DOTALL,
)
CONTEXT_PREFIX_PATTERN = re.compile(r"^\[Ngữ cảnh:.*?\]\s*", flags=re.DOTALL)


@dataclass(frozen=True)
class SourceDocument:
    page_content: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class RagasTestsetConfig:
    chunks_path: Path = DEFAULT_CHUNKS_PATH
    output_dir: Path = DEFAULT_OUTPUT_DIR
    llm_model: str = DEFAULT_OPENROUTER_MODEL
    embedding_model: str = DEFAULT_DENSE_MODEL
    multi_hop_count: int = 30
    overview_count: int = 20
    table_visual_count: int = 15
    overview_window_pages: int = 8
    overview_stride_pages: int = 6
    max_page_chars: int = 12000
    max_workers: int = 1
    openrouter_min_interval: float = DEFAULT_MIN_INTERVAL_SECONDS
    openrouter_max_requests: int = 0


CATEGORY_FILES = {
    "multi_hop": "data_test_multi_hop.json",
    "overview": "data_test_summary.json",
    "table_visual": "data_test_table_visual.json",
}


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_preprocessed_chunks(path: Path) -> list[dict[str, Any]]:
    chunks = list(read_jsonl(path))
    if not chunks:
        raise ValueError(f"No chunks found at {path}")
    return chunks


def _clean_content(value: Any) -> str:
    text = normalize_text(str(value or ""), preserve_lines=True)
    return CONTEXT_PREFIX_PATTERN.sub("", text, count=1).strip()


def _source_marker(metadata: Mapping[str, Any]) -> str:
    return (
        "<source_metadata>"
        + json.dumps(dict(metadata), ensure_ascii=False, sort_keys=True)
        + "</source_metadata>"
    )


def _document_content(text: str, metadata: Mapping[str, Any]) -> str:
    return f"{_source_marker(metadata)}\n{text.strip()}"


def build_page_documents(
    chunks: Sequence[Mapping[str, Any]],
    *,
    max_page_chars: int = 12000,
) -> list[SourceDocument]:
    by_page: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        if chunk.get("chunk_type") == "text":
            by_page[int(chunk["page_number"])].append(chunk)

    documents: list[SourceDocument] = []
    for page_number in sorted(by_page):
        page_chunks = sorted(
            by_page[page_number],
            key=lambda item: int(item.get("chunk_index") or 0),
        )
        text = "\n\n".join(
            content
            for content in (_clean_content(item.get("content")) for item in page_chunks)
            if content
        )[:max_page_chars]
        if not text:
            continue
        metadata = {
            "doc_name": str(page_chunks[0].get("doc_name") or "Ngo-doc.pdf"),
            "source_pages": [page_number],
            "source_chunk_ids": [str(item["id"]) for item in page_chunks],
            "source_types": ["text"],
        }
        documents.append(
            SourceDocument(
                page_content=_document_content(text, metadata),
                metadata=metadata,
            )
        )
    return documents


def build_overview_documents(
    page_documents: Sequence[SourceDocument],
    *,
    window_pages: int = 8,
    stride_pages: int = 6,
) -> list[SourceDocument]:
    if window_pages < 3:
        raise ValueError("overview window_pages must be at least 3")
    if stride_pages <= 0:
        raise ValueError("overview stride_pages must be positive")

    documents: list[SourceDocument] = []
    for start in range(0, len(page_documents), stride_pages):
        window = list(page_documents[start : start + window_pages])
        if len(window) < 3:
            break
        pages = sorted(
            {
                int(page)
                for document in window
                for page in document.metadata["source_pages"]
            }
        )
        chunk_ids = [
            str(chunk_id)
            for document in window
            for chunk_id in document.metadata["source_chunk_ids"]
        ]
        metadata = {
            "doc_name": "Ngo-doc.pdf",
            "source_pages": pages,
            "source_chunk_ids": chunk_ids,
            "source_types": ["text"],
            "context_scope": "overview_window",
        }
        body = "\n\n".join(
            f"## Trang {document.metadata['source_pages'][0]}\n"
            f"{SOURCE_PATTERN.sub('', document.page_content).strip()}"
            for document in window
        )
        documents.append(
            SourceDocument(
                page_content=_document_content(body, metadata),
                metadata=metadata,
            )
        )
    return documents


def build_table_visual_documents(
    chunks: Sequence[Mapping[str, Any]],
    page_documents: Sequence[SourceDocument],
    *,
    neighbor_chars: int = 4000,
) -> list[SourceDocument]:
    page_text = {
        int(document.metadata["source_pages"][0]): SOURCE_PATTERN.sub(
            "",
            document.page_content,
        ).strip()
        for document in page_documents
    }
    documents: list[SourceDocument] = []
    for chunk in chunks:
        source_type = str(chunk.get("chunk_type") or "")
        if source_type not in {"table", "visual"}:
            continue
        page_number = int(chunk["page_number"])
        content = _clean_content(chunk.get("content"))
        surrounding = page_text.get(page_number, "")[:neighbor_chars]
        metadata = {
            "doc_name": str(chunk.get("doc_name") or "Ngo-doc.pdf"),
            "source_pages": [page_number],
            "source_chunk_ids": [str(chunk["id"])],
            "source_types": [source_type],
            "asset_path": (chunk.get("reconstruction") or {}).get("asset_path"),
        }
        body = (
            f"## {source_type.upper()} - trang {page_number}\n{content}\n\n"
            f"## Văn bản cùng trang\n{surrounding}"
        )
        documents.append(
            SourceDocument(
                page_content=_document_content(body, metadata),
                metadata=metadata,
            )
        )
    return documents


def extract_provenance(contexts: Sequence[str]) -> dict[str, list[Any]]:
    pages: set[int] = set()
    chunk_ids: set[str] = set()
    source_types: set[str] = set()
    for context in contexts:
        for match in SOURCE_PATTERN.findall(str(context)):
            try:
                metadata = json.loads(match)
            except json.JSONDecodeError:
                continue
            pages.update(int(page) for page in metadata.get("source_pages") or [])
            chunk_ids.update(
                str(chunk_id)
                for chunk_id in metadata.get("source_chunk_ids") or []
            )
            source_types.update(
                str(source_type)
                for source_type in metadata.get("source_types") or []
            )
    return {
        "source_pages": sorted(pages),
        "source_chunk_ids": sorted(chunk_ids),
        "source_types": sorted(source_types),
    }


def _as_context_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [value]
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        return [value]
    return []


def normalize_ragas_records(
    records: Sequence[Mapping[str, Any]],
    *,
    category: str,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for index, record in enumerate(records, start=1):
        question = (
            record.get("user_input")
            or record.get("question")
            or record.get("query")
        )
        reference = (
            record.get("reference")
            or record.get("ground_truth")
            or record.get("response")
        )
        contexts = _as_context_list(
            record.get("reference_contexts")
            or record.get("contexts")
            or record.get("context")
        )
        if not question or not reference or not contexts:
            continue
        provenance = extract_provenance(contexts)
        normalized.append(
            {
                "id": f"ngo_doc_{category}_{index:03d}",
                "category": category,
                "question": str(question).strip(),
                "ground_truth": str(reference).strip(),
                "reference_contexts": contexts,
                **provenance,
                "synthesizer": str(
                    record.get("synthesizer_name")
                    or record.get("evolution_type")
                    or ""
                ),
            }
        )
    return normalized


def validate_category(
    records: Sequence[Mapping[str, Any]],
    *,
    category: str,
    expected_count: int,
) -> None:
    if len(records) != expected_count:
        raise ValueError(
            f"{category} produced {len(records)} records; expected {expected_count}"
        )
    for record in records:
        pages = record.get("source_pages") or []
        source_types = set(record.get("source_types") or [])
        if category == "multi_hop" and len(pages) < 2:
            raise ValueError(
                f"Multi-hop record {record.get('id')} does not reference multiple pages"
            )
        if category == "overview" and len(pages) < 3:
            raise ValueError(
                f"Overview record {record.get('id')} has insufficient page coverage"
            )
        if category == "table_visual" and not (
            {"table", "visual"} & source_types
        ):
            raise ValueError(
                f"Table/visual record {record.get('id')} has no table or visual source"
            )


def _to_langchain_documents(
    documents: Sequence[SourceDocument],
) -> list[Any]:
    try:
        from langchain_core.documents import Document
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required. Install project requirements first."
        ) from exc
    return [
        Document(
            page_content=document.page_content,
            metadata=document.metadata,
        )
        for document in documents
    ]


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") if isinstance(item, dict) else item)
            for item in content
        )
    return str(content)


def build_rate_limited_langchain_llm(config: RagasTestsetConfig) -> Any:
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
        from langchain_core.messages import AIMessage, SystemMessage
        from langchain_core.outputs import ChatGeneration, ChatResult
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required. Install project requirements first."
        ) from exc

    client = OpenRouterClient(
        model=config.llm_model,
        min_interval_seconds=config.openrouter_min_interval,
        max_requests_per_run=config.openrouter_max_requests,
        timeout=180.0,
        max_retries=6,
    )

    class RateLimitedOpenRouterChatModel(BaseChatModel):
        client: Any
        model_name: str

        @property
        def _llm_type(self) -> str:
            return "rate-limited-openrouter"

        @property
        def _identifying_params(self) -> dict[str, Any]:
            return {"model_name": self.model_name}

        def _generate(
            self,
            messages: Sequence[Any],
            stop: Sequence[str] | None = None,
            run_manager: Any = None,
            **kwargs: Any,
        ) -> Any:
            system_parts: list[str] = []
            user_parts: list[str] = []
            for message in messages:
                text = _message_text(message.content)
                if isinstance(message, SystemMessage):
                    system_parts.append(text)
                else:
                    user_parts.append(text)
            text = self.client.complete(
                system_prompt="\n\n".join(system_parts)
                or "You generate grounded evaluation data.",
                user_prompt="\n\n".join(user_parts),
                max_tokens=int(kwargs.get("max_tokens") or 2048),
                temperature=float(kwargs.get("temperature") or 0),
            )
            return ChatResult(
                generations=[ChatGeneration(message=AIMessage(content=text))]
            )

    return RateLimitedOpenRouterChatModel(
        client=client,
        model_name=config.llm_model,
    )


def build_langchain_embeddings(config: RagasTestsetConfig) -> Any:
    try:
        from langchain_core.embeddings import Embeddings
    except ImportError as exc:
        raise RuntimeError(
            "langchain-core is required. Install project requirements first."
        ) from exc

    embedder = DenseEmbedder(
        model_name=config.embedding_model,
        backend="local",
    )

    class ProjectEmbeddings(Embeddings):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            vectors, _ = embedder.encode(texts, show_progress=True)
            return vectors.tolist()

        def embed_query(self, text: str) -> list[float]:
            return embedder.encode_query(text).tolist()

    return ProjectEmbeddings()


def _import_ragas() -> tuple[Any, Any, Any]:
    _install_ragas_langchain_compatibility()
    try:
        from ragas.testset import TestsetGenerator
    except ImportError:
        try:
            from ragas.testset.generator import TestsetGenerator
        except ImportError as exc:
            raise RuntimeError(
                "Ragas is required. Install project requirements first."
            ) from exc

    try:
        from ragas.run_config import RunConfig
        from ragas.testset.synthesizers import (
            MultiHopAbstractQuerySynthesizer,
            MultiHopSpecificQuerySynthesizer,
            SingleHopSpecificQuerySynthesizer,
        )
    except ImportError as exc:
        raise RuntimeError(
            "This generator requires a modern Ragas testset API."
        ) from exc
    synthesizers = {
        "multi_hop_specific": MultiHopSpecificQuerySynthesizer,
        "multi_hop_abstract": MultiHopAbstractQuerySynthesizer,
        "single_hop_specific": SingleHopSpecificQuerySynthesizer,
    }
    return TestsetGenerator, RunConfig, synthesizers


def _install_ragas_langchain_compatibility() -> None:
    """Shim optional Vertex AI imports removed from LangChain Community 0.4."""
    module_name = "langchain_community.chat_models.vertexai"
    try:
        __import__(module_name)
    except ModuleNotFoundError:
        module = types.ModuleType(module_name)

        class ChatVertexAI:
            pass

        module.ChatVertexAI = ChatVertexAI
        sys.modules[module_name] = module

    try:
        import langchain_community.llms as community_llms

        try:
            getattr(community_llms, "VertexAI")
        except (AttributeError, ModuleNotFoundError):
            class VertexAI:
                pass

            community_llms.VertexAI = VertexAI
    except ImportError:
        return


def _make_synthesizer(factory: Any, llm: Any) -> Any:
    kwargs: dict[str, Any] = {"llm": llm}
    signature = inspect.signature(factory)
    if "llm_context" in signature.parameters:
        kwargs["llm_context"] = (
            "Generate the question and reference answer in Vietnamese. "
            "Use only the supplied Vietnamese medical source and preserve "
            "drug names, units, thresholds, table values, and page evidence."
        )
    return factory(**kwargs)


def _query_distribution(
    category: str,
    *,
    llm: Any,
    synthesizers: Mapping[str, Any],
) -> list[tuple[Any, float]]:
    if category == "multi_hop":
        return [
            (
                _make_synthesizer(
                    synthesizers["multi_hop_specific"],
                    llm,
                ),
                0.7,
            ),
            (
                _make_synthesizer(
                    synthesizers["multi_hop_abstract"],
                    llm,
                ),
                0.3,
            ),
        ]
    if category == "overview":
        return [
            (
                _make_synthesizer(
                    synthesizers["multi_hop_abstract"],
                    llm,
                ),
                1.0,
            )
        ]
    return [
        (
            _make_synthesizer(
                synthesizers["single_hop_specific"],
                llm,
            ),
            1.0,
        )
    ]


def _testset_records(testset: Any) -> list[dict[str, Any]]:
    if hasattr(testset, "to_pandas"):
        return testset.to_pandas().to_dict(orient="records")
    if hasattr(testset, "to_list"):
        return list(testset.to_list())
    if isinstance(testset, list):
        return [dict(item) for item in testset]
    raise TypeError("Unsupported Ragas testset result")


def generate_category(
    *,
    category: str,
    documents: Sequence[SourceDocument],
    count: int,
    config: RagasTestsetConfig,
    llm: Any,
    embeddings: Any,
) -> list[dict[str, Any]]:
    TestsetGenerator, RunConfig, synthesizers = _import_ragas()
    generator = TestsetGenerator.from_langchain(llm, embeddings)
    distribution = _query_distribution(
        category,
        llm=generator.llm,
        synthesizers=synthesizers,
    )
    run_config = RunConfig(
        max_workers=config.max_workers,
        timeout=240,
        max_retries=6,
        max_wait=120,
    )
    method = generator.generate_with_langchain_docs
    kwargs: dict[str, Any] = {
        "testset_size": count,
        "query_distribution": distribution,
        "run_config": run_config,
        "raise_exceptions": False,
    }
    supported = inspect.signature(method).parameters
    kwargs = {key: value for key, value in kwargs.items() if key in supported}
    testset = method(_to_langchain_documents(documents), **kwargs)
    records = normalize_ragas_records(
        _testset_records(testset),
        category=category,
    )
    validate_category(records, category=category, expected_count=count)
    return records


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def _category_inputs(
    chunks: Sequence[Mapping[str, Any]],
    config: RagasTestsetConfig,
) -> dict[str, list[SourceDocument]]:
    page_documents = build_page_documents(
        chunks,
        max_page_chars=config.max_page_chars,
    )
    return {
        "multi_hop": page_documents,
        "overview": build_overview_documents(
            page_documents,
            window_pages=config.overview_window_pages,
            stride_pages=config.overview_stride_pages,
        ),
        "table_visual": build_table_visual_documents(
            chunks,
            page_documents,
        ),
    }


def generate_testset(
    config: RagasTestsetConfig,
    *,
    categories: Sequence[str] = tuple(CATEGORY_FILES),
    force: bool = False,
) -> dict[str, list[dict[str, Any]]]:
    load_dotenv()
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")

    chunks = load_preprocessed_chunks(config.chunks_path)
    inputs = _category_inputs(chunks, config)
    counts = {
        "multi_hop": config.multi_hop_count,
        "overview": config.overview_count,
        "table_visual": config.table_visual_count,
    }
    llm = build_rate_limited_langchain_llm(config)
    embeddings = build_langchain_embeddings(config)
    generated: dict[str, list[dict[str, Any]]] = {}

    for category in tqdm(categories, desc="Ragas testsets", unit="category"):
        output_path = config.output_dir / CATEGORY_FILES[category]
        if output_path.exists() and not force:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            validate_category(
                existing,
                category=category,
                expected_count=counts[category],
            )
            generated[category] = existing
            continue

        records = generate_category(
            category=category,
            documents=inputs[category],
            count=counts[category],
            config=config,
            llm=llm,
            embeddings=embeddings,
        )
        _write_json(output_path, records)
        generated[category] = records

    combined = [
        record
        for category in CATEGORY_FILES
        for record in generated.get(category, [])
    ]
    if set(categories) == set(CATEGORY_FILES):
        expected_total = sum(counts.values())
        if len(combined) != expected_total:
            raise ValueError(
                f"Combined testset has {len(combined)} records; "
                f"expected {expected_total}"
            )
        _write_json(
            config.output_dir / "ngo_doc_ragas_testset.json",
            combined,
        )
        _write_json(
            config.output_dir / "ngo_doc_ragas_manifest.json",
            {
                "config": {
                    **asdict(config),
                    "chunks_path": str(config.chunks_path),
                    "output_dir": str(config.output_dir),
                },
                "counts": dict(Counter(item["category"] for item in combined)),
                "total": len(combined),
            },
        )
    return generated


def dry_run_summary(config: RagasTestsetConfig) -> dict[str, Any]:
    chunks = load_preprocessed_chunks(config.chunks_path)
    inputs = _category_inputs(chunks, config)
    return {
        "chunks_path": str(config.chunks_path),
        "chunk_types": dict(Counter(str(item.get("chunk_type")) for item in chunks)),
        "source_documents": {
            category: len(documents)
            for category, documents in inputs.items()
        },
        "requested_questions": {
            "multi_hop": config.multi_hop_count,
            "overview": config.overview_count,
            "table_visual": config.table_visual_count,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 65-question Ragas testset from Ngo-doc chunks."
    )
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--embedding-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--multi-hop-count", type=int, default=30)
    parser.add_argument("--overview-count", type=int, default=20)
    parser.add_argument("--table-visual-count", type=int, default=15)
    parser.add_argument("--overview-window-pages", type=int, default=8)
    parser.add_argument("--overview-stride-pages", type=int, default=6)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument(
        "--openrouter-min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL_SECONDS,
    )
    parser.add_argument("--openrouter-max-requests", type=int, default=0)
    parser.add_argument(
        "--category",
        choices=["all", *CATEGORY_FILES],
        default="all",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RagasTestsetConfig(
        chunks_path=args.chunks,
        output_dir=args.output_dir,
        llm_model=args.model,
        embedding_model=args.embedding_model,
        multi_hop_count=args.multi_hop_count,
        overview_count=args.overview_count,
        table_visual_count=args.table_visual_count,
        overview_window_pages=args.overview_window_pages,
        overview_stride_pages=args.overview_stride_pages,
        max_workers=max(1, args.max_workers),
        openrouter_min_interval=max(0.0, args.openrouter_min_interval),
        openrouter_max_requests=args.openrouter_max_requests,
    )
    if args.dry_run:
        print(json.dumps(dry_run_summary(config), ensure_ascii=False, indent=2))
        return
    categories = list(CATEGORY_FILES) if args.category == "all" else [args.category]
    generated = generate_testset(
        config,
        categories=categories,
        force=args.force,
    )
    print(
        json.dumps(
            {category: len(records) for category, records in generated.items()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
