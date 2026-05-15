from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict, Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.retrieval.index_raptor import (
    BM25Index,
    DEFAULT_CACHE_DIR,
    DEFAULT_EMBED_MODEL,
    DEFAULT_INDEX_DIR,
    normalize_text,
    read_jsonl,
    tokenize,
)


DEFAULT_SYNONYMS = {
    "khó thở": ["kho tho", "dyspnea", "suy hô hấp", "suy ho hap"],
    "sốt": ["sot", "tăng thân nhiệt", "tang than nhiet", "nhiệt độ cao"],
    "đau ngực": ["dau nguc", "chest pain"],
    "đau bụng": ["dau bung", "abdominal pain"],
    "co giật": ["co giat", "seizure", "động kinh"],
    "hôn mê": ["hon me", "rối loạn ý thức", "roi loan y thuc"],
    "nôn": ["non", "buồn nôn", "buon non", "ói"],
    "tiêu chảy": ["tieu chay", "ỉa chảy"],
    "ngộ độc": ["ngo doc", "nhiễm độc", "nhiem doc", "poisoning"],
    "viêm đa khớp dạng thấp": ["viem da khop dang thap", "rheumatoid arthritis"],
}

RERANK_PRESETS = {
    "off": None,
    "fast": "cross-encoder/ms-marco-MiniLM-L-6-v2",
    "balanced": "BAAI/bge-reranker-base",
    "quality": "BAAI/bge-reranker-v2-m3",
}


@dataclass
class RetrievalConfig:
    index_dir: Path = DEFAULT_INDEX_DIR
    embed_model: str = DEFAULT_EMBED_MODEL
    cache_folder: Optional[Path] = DEFAULT_CACHE_DIR
    bm25_top_n: int = 30
    dense_top_m: int = 60
    final_top_k: int = 10
    rrf_k: int = 60
    bm25_weight: float = 2.0
    dense_weight: float = 5.0
    rerank_model: Optional[str] = None
    rerank_top_k: int = 20
    rerank_batch_size: int = 16
    rerank_max_length: int = 384
    local_files_only: bool = False
    device: Optional[str] = None


MODE_HYBRID_COLLAPSED = "hybrid_collapsed"
MODE_DENSE_COLLAPSED = "dense_collapsed"

VISUAL_QUERY_TERMS = (
    "hinh",
    "anh",
    "so do",
    "diagram",
    "figure",
    "image",
    "visual",
    "chart",
    "bieu do",
    "minh hoa",
)


def auto_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def expand_query(query: str, synonyms: Optional[Dict[str, Sequence[str]]] = None) -> str:
    normalized = normalize_text(query)
    expansions: List[str] = [normalized]
    synonym_map = synonyms or DEFAULT_SYNONYMS
    for phrase, values in synonym_map.items():
        if normalize_text(phrase) in normalized:
            expansions.extend(values)
    return " ".join(expansions)


def compact_snippet(text: str, limit: int = 450) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def fold_text(text: str) -> str:
    normalized = normalize_text(text)
    decomposed = unicodedata.normalize("NFD", normalized)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks)


def query_mentions_visual(query: str) -> bool:
    folded = fold_text(query)
    return any(term in folded for term in VISUAL_QUERY_TERMS)


class RaptorRetriever:
    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config
        self.index_dir = config.index_dir
        self.manifest = json.loads((self.index_dir / "manifest.json").read_text(encoding="utf-8"))
        self.nodes = list(read_jsonl(self.index_dir / "nodes.jsonl"))
        self.node_by_id = {node["node_id"]: node for node in self.nodes}
        self.node_ids = [node["node_id"] for node in self.nodes]
        self.parent_map = defaultdict(list)
        for node in self.nodes:
            for child_id in node.get("child_node_ids") or []:
                self.parent_map[child_id].append(node["node_id"])

        self.depth_by_id = {
            node["node_id"]: int(node.get("depth", 0))
            for node in self.nodes
        }
        self.bm25 = BM25Index.load(self.index_dir / "bm25.pkl")
        self.faiss_index = self._load_faiss(self.index_dir / "faiss.index")
        self.embedder = self._load_embedder(config.embed_model)
        self.reranker: Optional[Any] = None

    def _load_faiss(self, path: Path) -> Any:
        import faiss

        return faiss.read_index(str(path))

    def _load_embedder(self, model_name: str) -> Any:
        from sentence_transformers import SentenceTransformer

        kwargs: Dict[str, Any] = {}
        if self.config.cache_folder is not None and self.config.cache_folder.exists():
            kwargs["cache_folder"] = str(self.config.cache_folder)
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.local_files_only:
            kwargs["local_files_only"] = True
        return SentenceTransformer(model_name, **kwargs)

    def _load_reranker(self, model_name: str) -> Any:
        from sentence_transformers import CrossEncoder

        kwargs: Dict[str, Any] = {}
        if self.config.device:
            kwargs["device"] = self.config.device
        if self.config.cache_folder is not None and self.config.cache_folder.exists():
            kwargs["cache_folder"] = str(self.config.cache_folder)
        if self.config.local_files_only:
            kwargs["local_files_only"] = True
        kwargs["max_length"] = self.config.rerank_max_length
        return CrossEncoder(model_name, **kwargs)

    def _get_reranker(self) -> Optional[Any]:
        if not self.config.rerank_model:
            return None
        if self.reranker is None:
            self.reranker = self._load_reranker(self.config.rerank_model)
        return self.reranker

    def embed_query(self, query: str) -> np.ndarray:
        vector = self.embedder.encode(
            [normalize_text(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vector, dtype="float32")

    def bm25_search(self, query: str, top_n: Optional[int] = None) -> List[Tuple[str, float]]:
        expanded = expand_query(query)
        query_tokens = tokenize(expanded)
        return self.bm25.search(query_tokens, top_k=top_n or self.config.bm25_top_n)

    def dense_search(self, query: str, top_m: Optional[int] = None) -> List[Tuple[str, float]]:
        query_vector = self.embed_query(query)
        scores, indices = self.faiss_index.search(query_vector, top_m or self.config.dense_top_m)
        results: List[Tuple[str, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self.node_ids[int(idx)], float(score)))
        return results

    def modality_weight(self, query: str, node_id: str) -> float:
        node = self.node_by_id[node_id]
        chunk_type = str(node.get("chunk_type") or "").lower()
        layer = str(node.get("layer") or "").lower()
        depth = int(node.get("depth", 0))
        metadata = node.get("metadata") or {}
        child_types = {str(value).lower() for value in metadata.get("child_chunk_types") or []}
        is_visual = chunk_type == "visual" or child_types == {"visual"}

        weight = 1.0

        if is_visual and not query_mentions_visual(query):
            weight *= 0.35

        # Boost nhẹ cho summary nodes để RAPTOR hiện rõ hơn
        # if layer == "summary":
        #     weight *= 1.18

        # Summary ở tầng cao hơn được boost nhẹ thêm
        if depth == 1 and layer == "summary":
            weight *= 1.8

        return weight

    def apply_modality_priority(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        base_field: str = "score",
    ) -> List[Dict[str, Any]]:
        adjusted: List[Dict[str, Any]] = []
        for item in candidates:
            updated = dict(item)
            weight = self.modality_weight(query, updated["node_id"])
            weight *= self.structural_weight(query, updated["node_id"])
            base_score = float(updated.get(base_field) or updated.get("score") or 0.0)
            if base_field == "rerank_score" and updated.get(base_field) is not None:
                adjusted_score = base_score + math.log(max(weight, 1e-6))
            else:
                adjusted_score = base_score * weight
            updated["modality_weight"] = weight
            updated["modality_adjusted_score"] = adjusted_score
            adjusted.append(updated)
        return sorted(adjusted, key=lambda item: item["modality_adjusted_score"], reverse=True)

    def structural_weight(self, query: str, node_id: str) -> float:
        node = self.node_by_id[node_id]
        layer = str(node.get("layer") or "").lower()
        q = normalize_text(query)

        weight = 1.0

        # Query thiên về symptom thì summary/symptom-like nodes đáng được ưu tiên hơn
        if "triệu chứng" in q or "trieu chung" in q:
            if layer == "summary":
                weight *= 1.12

        if "chẩn đoán" in q or "chan doan" in q:
            if layer == "summary":
                weight *= 1.10

        if "cận lâm sàng" in q or "cls" in q:
            if layer == "summary":
                weight *= 1.08

        return weight
    
    def hybrid_collapsed_search(
        self,
        query: str,
        *,
        bm25_top_n: Optional[int] = None,
        dense_top_m: Optional[int] = None,
        final_top_k: Optional[int] = None,
        rerank: bool = True,
    ) -> List[Dict[str, Any]]:
        bm25_results = self.bm25_search(query, bm25_top_n)
        dense_results = self.dense_search(query, dense_top_m)
        fused = self.apply_modality_priority(query, self.weighted_rrf(bm25_results, dense_results))
        candidates = fused[: max(final_top_k or self.config.final_top_k, self.config.rerank_top_k)]

        reranker = self._get_reranker() if rerank else None
        if rerank and reranker is not None and candidates:
            reranked = self.rerank(query, candidates[: self.config.rerank_top_k])
            reranked = self.apply_modality_priority(query, reranked, base_field="rerank_score")
            candidates = reranked + candidates[self.config.rerank_top_k :]

        return [self.format_result(item) for item in candidates[: final_top_k or self.config.final_top_k]]


    def weighted_rrf(
        self,
        bm25_results: Sequence[Tuple[str, float]],
        dense_results: Sequence[Tuple[str, float]],
    ) -> List[Dict[str, Any]]:
        by_id: Dict[str, Dict[str, Any]] = {}

        def add(results: Sequence[Tuple[str, float]], source: str, weight: float) -> None:
            for rank, (node_id, raw_score) in enumerate(results, start=1):
                item = by_id.setdefault(
                    node_id,
                    {
                        "node_id": node_id,
                        "score": 0.0,
                        "bm25_rank": None,
                        "dense_rank": None,
                        "bm25_score": None,
                        "dense_score": None,
                        "bm25_scope": "leaf_only",
                        "dense_scope": "all_raptor_nodes",
                    },
                )
                item["score"] += weight / (self.config.rrf_k + rank)
                item[f"{source}_rank"] = rank
                item[f"{source}_score"] = raw_score

        add(bm25_results, "bm25", self.config.bm25_weight)
        add(dense_results, "dense", self.config.dense_weight)
        return sorted(by_id.values(), key=lambda item: item["score"], reverse=True)

    def rerank(self, query: str, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        reranker = self._get_reranker()
        if reranker is None:
            return list(candidates)
        pairs = [(query, self.node_by_id[item["node_id"]]["text"]) for item in candidates]
        scores = reranker.predict(
            pairs,
            batch_size=self.config.rerank_batch_size,
            show_progress_bar=False,
        )
        reranked: List[Dict[str, Any]] = []
        for item, score in zip(candidates, scores):
            updated = dict(item)
            updated["rerank_score"] = float(score)
            reranked.append(updated)
        return sorted(reranked, key=lambda item: item["rerank_score"], reverse=True)

    def expand_with_ancestors(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        *,
        max_hops: int = 2,
        max_extra: int = 12,
        ancestor_decay: float = 0.92,
    ) -> List[Dict[str, Any]]:
        expanded: List[Dict[str, Any]] = []
        seen: set[str] = set()

        for item in candidates:
            base_item = dict(item)
            expanded.append(base_item)
            seen.add(base_item["node_id"])

            frontier = [base_item["node_id"]]
            for hop in range(max_hops):
                next_frontier: List[str] = []
                for nid in frontier:
                    for pid in self.parent_map.get(nid, []):
                        if pid in seen:
                            continue
                        parent_item = {
                            "node_id": pid,
                            "score": float(base_item.get("score", 0.0)) * (ancestor_decay ** (hop + 1)),
                            "source_node_id": base_item["node_id"],
                            "ancestor_hop": hop + 1,
                        }
                        expanded.append(parent_item)
                        seen.add(pid)
                        next_frontier.append(pid)

                        if len(expanded) >= len(candidates) + max_extra:
                            return self.apply_modality_priority(query, expanded)
                frontier = next_frontier

        return self.apply_modality_priority(query, expanded)
    
    def dense_collapsed_search(
        self,
        query: str,
        *,
        token_limit: int = 2000,
        max_nodes: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        dense_results = self.dense_search(query, top_m=len(self.node_ids))
        dense_candidates = self.apply_modality_priority(
            query,
            [
                {
                    "node_id": node_id,
                    "score": score,
                    "dense_rank": rank,
                }
                for rank, (node_id, score) in enumerate(dense_results, start=1)
            ],
        )
        selected: List[Dict[str, Any]] = []
        total_tokens = 0
        for selection_rank, item in enumerate(dense_candidates, start=1):
            node_id = item["node_id"]
            node = self.node_by_id[node_id]
            node_tokens = int(node.get("token_estimate") or len(node.get("text", "").split()))
            if selected and total_tokens + node_tokens > token_limit:
                continue
            selected.append(
                {
                    **item,
                    "node_id": node_id,
                    "selection_rank": selection_rank,
                    "token_estimate": node_tokens,
                    "total_tokens_after": total_tokens + node_tokens,
                }
            )
            total_tokens += node_tokens
            if max_nodes is not None and len(selected) >= max_nodes:
                break
            if total_tokens >= token_limit:
                break
        return [self.format_result(item) for item in selected]

    def format_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        node = self.node_by_id[item["node_id"]]
        return {
            **item,
            "layer": node.get("layer"),
            "depth": node.get("depth"),
            "source_guideline": node.get("source_guideline"),
            "source_docs": node.get("source_docs"),
            "source_pages": node.get("source_pages"),
            "chunk_type": node.get("chunk_type"),
            "chunk_index": node.get("chunk_index"),
            "token_estimate": node.get("token_estimate"),
            "citation": node.get("citation"),
            "text": node.get("text"),
            "snippet": compact_snippet(node.get("text") or ""),
        }


def print_results(results: Iterable[Dict[str, Any]]) -> None:
    for rank, item in enumerate(results, start=1):
        pages = ",".join(str(page) for page in item.get("source_pages") or [])
        doc = item.get("source_guideline") or ""
        bm25_rank = item.get("bm25_rank")
        dense_rank = item.get("dense_rank")
        rerank_score = item.get("rerank_score")
        score_bits = [f"score={item.get('score', 0):.4f}"]
        if bm25_rank:
            score_bits.append(f"bm25_rank={bm25_rank}")
        if dense_rank:
            score_bits.append(f"dense_rank={dense_rank}")
        if rerank_score is not None:
            score_bits.append(f"rerank={rerank_score:.4f}")
        print(f"\n#{rank} {item['node_id']} [{item.get('layer')} depth={item.get('depth')}] {' '.join(score_bits)}")
        print(f"source={doc} pages={pages}")
        if item.get("layer"):
            score_bits.append(f"layer={item.get('layer')}")
        if item.get("chunk_type"):
            score_bits.append(f"chunk_type={item.get('chunk_type')}")
        print(item["snippet"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hybrid collapsed retrieval over RAPTOR indexes.")
    parser.add_argument("query")
    parser.add_argument("--index", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument(
        "--mode",
        choices=[MODE_HYBRID_COLLAPSED, MODE_DENSE_COLLAPSED],
        default=MODE_HYBRID_COLLAPSED,
    )
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--bm25-top-n", type=int, default=30)
    parser.add_argument("--dense-top-m", type=int, default=60)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--bm25-weight", type=float, default=2.0)
    parser.add_argument("--dense-weight", type=float, default=5.0)
    parser.add_argument("--rerank-preset", choices=sorted(RERANK_PRESETS.keys()), default=None)
    parser.add_argument("--rerank-model", default=None)
    parser.add_argument("--rerank-top-k", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=16)
    parser.add_argument("--rerank-max-length", type=int, default=384)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--interactive", action="store_true", help="Keep models in memory and accept multiple queries.")
    parser.add_argument("--show-timing", action="store_true", help="Print startup and query latency.")
    parser.add_argument("--token-limit", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--debug-layers", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of readable text.")
    return parser.parse_args()


def build_retriever(args: argparse.Namespace) -> RaptorRetriever:
    rerank_model = args.rerank_model
    if args.rerank_preset is not None:
        rerank_model = RERANK_PRESETS[args.rerank_preset]
    device = args.device or auto_device()
    return RaptorRetriever(
        RetrievalConfig(
            index_dir=args.index,
            embed_model=args.embed_model,
            cache_folder=args.cache_folder,
            bm25_top_n=args.bm25_top_n,
            dense_top_m=args.dense_top_m,
            final_top_k=args.top_k,
            rrf_k=args.rrf_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            rerank_model=rerank_model,
            rerank_top_k=args.rerank_top_k,
            rerank_batch_size=args.rerank_batch_size,
            rerank_max_length=args.rerank_max_length,
            local_files_only=args.local_files_only,
            device=device,
        )
    )


def run_query(retriever: RaptorRetriever, args: argparse.Namespace, query: str) -> List[Dict[str, Any]]:
    if args.mode == MODE_DENSE_COLLAPSED:
        return retriever.dense_collapsed_search(query, token_limit=args.token_limit, max_nodes=args.top_k)
    return retriever.hybrid_collapsed_search(query, final_top_k=args.top_k, rerank=not args.no_rerank)


def print_query_results(args: argparse.Namespace, results: List[Dict[str, Any]]) -> None:
    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
    else:
        print_results(results)


def interactive_loop(retriever: RaptorRetriever, args: argparse.Namespace) -> None:
    print("Interactive mode. Type a query and press Enter. Type 'exit' to quit.")
    while True:
        try:
            query = input("\nquery> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break
        started = time.perf_counter()
        results = run_query(retriever, args, query)
        print_query_results(args, results)
        if args.show_timing:
            print(f"\n[query_time] {time.perf_counter() - started:.3f}s")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    args = parse_args()
    startup_started = time.perf_counter()
    retriever = build_retriever(args)
    startup_elapsed = time.perf_counter() - startup_started
    if args.show_timing:
        print(f"[startup_time] {startup_elapsed:.3f}s")

    if args.interactive:
        interactive_loop(retriever, args)
        return

    query_started = time.perf_counter()
    results = run_query(retriever, args, args.query)
    print_query_results(args, results)
    if args.show_timing:
        print(f"\n[query_time] {time.perf_counter() - query_started:.3f}s")


if __name__ == "__main__":
    main()
