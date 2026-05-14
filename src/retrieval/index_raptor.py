from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import time
import unicodedata
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

try:
    from ftfy import fix_text
except Exception:  # pragma: no cover - optional dependency at runtime
    fix_text = None


DEFAULT_TREE_PATH = Path("data/processed/raptor_tree/raptor_nodes.jsonl")
DEFAULT_INDEX_DIR = Path("data/processed/raptor_index")
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
DEFAULT_CACHE_DIR = Path("data/cache/huggingface/sentence_transformers")

MOJIBAKE_MARKERS = ("Ã", "Â", "Ä", "Æ", "á»", "áº", "ï¿½", "\ufffd")
VIETNAMESE_CHARS = set(
    "àáạảãâầấậẩẫăằắặẳẵ"
    "èéẹẻẽêềếệểễ"
    "ìíịỉĩ"
    "òóọỏõôồốộổỗơờớợởỡ"
    "ùúụủũưừứựửữ"
    "ỳýỵỷỹđ"
    "ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴ"
    "ÈÉẸẺẼÊỀẾỆỂỄ"
    "ÌÍỊỈĨ"
    "ÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠ"
    "ÙÚỤỦŨƯỪỨỰỬỮ"
    "ỲÝỴỶỸĐ"
)

SYMBOL_REPLACEMENTS = {
    "≥": " >= ",
    "≤": " <= ",
    "≧": " >= ",
    "≦": " <= ",
    "≠": " != ",
    "±": " +/- ",
    "→": " tang ",
    "↑": " tang ",
    "↓": " giam ",
    "µ": " micro ",
    "μ": " micro ",
    "℃": " do c ",
    "°c": " do c ",
}

DEFAULT_STOPWORDS = {
    "a",
    "ai",
    "anh",
    "bao",
    "bi",
    "cac",
    "cach",
    "can",
    "chi",
    "cho",
    "co",
    "con",
    "cua",
    "dang",
    "de",
    "den",
    "duoc",
    "gi",
    "hay",
    "hoac",
    "khi",
    "la",
    "lam",
    "mot",
    "nay",
    "neu",
    "nguoi",
    "nhieu",
    "nhung",
    "o",
    "phai",
    "qua",
    "ra",
    "sau",
    "se",
    "tai",
    "thi",
    "trong",
    "tu",
    "va",
    "ve",
    "voi",
    "benh_nhan",
    "nguoi_benh",
    "chan_doan",
    "dieu_tri",
    "trieu_chung",
}

TOKEN_RE = re.compile(r"[0-9]+(?:[.,][0-9]+)?|[^\W\d_]+(?:_[^\W\d_]+)*", re.UNICODE)
_VI_TOKENIZER: Any = None
_VI_TOKENIZER_LOADED = False


@dataclass
class IndexConfig:
    nodes_path: Path = DEFAULT_TREE_PATH
    output_dir: Path = DEFAULT_INDEX_DIR
    embed_model: str = DEFAULT_EMBED_MODEL
    cache_folder: Optional[Path] = DEFAULT_CACHE_DIR
    batch_size: int = 32
    k1: float = 1.7
    b: float = 0.83
    include_stopwords: bool = False
    device: Optional[str] = None


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha1_text(value: str) -> str:
    return hashlib.sha1(value.encode("utf-8")).hexdigest()


def strip_accents(text: str) -> str:
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if unicodedata.category(ch) != "Mn")
    return unicodedata.normalize("NFC", without_marks).replace("đ", "d").replace("Đ", "D")


def repair_mojibake(text: str) -> str:
    if not text:
        return ""

    candidates = [str(text)]
    if fix_text is not None:
        try:
            candidates.append(fix_text(text))
        except Exception:
            pass

    queue = list(candidates)
    for _ in range(2):
        next_queue: List[str] = []
        for value in queue:
            for encoding in ("cp1252", "latin1"):
                try:
                    repaired = value.encode(encoding).decode("utf-8")
                except UnicodeError:
                    continue
                if repaired not in candidates:
                    candidates.append(repaired)
                    next_queue.append(repaired)
        queue = next_queue
        if not queue:
            break

    def score(value: str) -> int:
        marker_penalty = sum(value.count(marker) for marker in MOJIBAKE_MARKERS) * 80
        replacement_penalty = value.count("\ufffd") * 200
        vietnamese_bonus = sum(1 for ch in value if ch in VIETNAMESE_CHARS) * 4
        ascii_bonus = sum(1 for ch in value if ch.isascii()) // 30
        return vietnamese_bonus + ascii_bonus - marker_penalty - replacement_penalty

    return unicodedata.normalize("NFC", max(candidates, key=score)).strip()


def normalize_text(text: str) -> str:
    text = repair_mojibake(text)
    text = unicodedata.normalize("NFC", text).lower()
    for source, target in SYMBOL_REPLACEMENTS.items():
        text = text.replace(source, target)
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def segment_vietnamese(text: str) -> str:
    global _VI_TOKENIZER, _VI_TOKENIZER_LOADED
    if not _VI_TOKENIZER_LOADED:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from pyvi import ViTokenizer as tokenizer

            _VI_TOKENIZER = tokenizer
        except Exception:
            _VI_TOKENIZER = None
        _VI_TOKENIZER_LOADED = True
    return _VI_TOKENIZER.tokenize(text) if _VI_TOKENIZER is not None else text


def tokenize(text: str, *, include_stopwords: bool = False) -> List[str]:
    normalized = normalize_text(text)
    segmented = segment_vietnamese(normalized)
    raw_tokens = TOKEN_RE.findall(segmented)
    tokens: List[str] = []

    for token in raw_tokens:
        token = token.strip("_.,;:!?()[]{}\"'")
        if not token:
            continue
        folded = strip_accents(token)
        if len(folded) <= 1 and not folded.isdigit():
            continue
        if not include_stopwords and folded in DEFAULT_STOPWORDS:
            continue
        tokens.append(folded)
        if token != folded and (include_stopwords or token not in DEFAULT_STOPWORDS):
            tokens.append(token)
    return tokens


class BM25Index:
    def __init__(self, *, k1: float = 1.7, b: float = 0.83) -> None:
        self.k1 = k1
        self.b = b
        self.node_ids: List[str] = []
        self.doc_lengths: np.ndarray = np.empty(0, dtype="float32")
        self.avgdl = 0.0
        self.idf: Dict[str, float] = {}
        self.postings: Dict[str, List[Tuple[int, int]]] = {}

    def fit(self, node_ids: Sequence[str], tokenized_docs: Sequence[Sequence[str]]) -> None:
        self.node_ids = list(node_ids)
        self.doc_lengths = np.asarray([len(doc) for doc in tokenized_docs], dtype="float32")
        self.avgdl = float(np.mean(self.doc_lengths)) if len(self.doc_lengths) else 0.0
        self.idf = {}
        self.postings = {}
        n_docs = len(tokenized_docs)

        for doc_idx, tokens in enumerate(tokenized_docs):
            term_counts: Dict[str, int] = {}
            for token in tokens:
                term_counts[token] = term_counts.get(token, 0) + 1
            for token, tf in term_counts.items():
                self.postings.setdefault(token, []).append((doc_idx, tf))

        for token, posting in self.postings.items():
            df = len(posting)
            self.idf[token] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    def search(self, query_tokens: Sequence[str], top_k: int = 20) -> List[Tuple[str, float]]:
        if not self.node_ids or not query_tokens:
            return []

        scores = np.zeros(len(self.node_ids), dtype="float32")
        unique_terms = set(query_tokens)
        for term in unique_terms:
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = self.idf.get(term, 0.0)
            for doc_idx, tf in posting:
                dl = float(self.doc_lengths[doc_idx])
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / max(self.avgdl, 1e-6))
                scores[doc_idx] += idf * (tf * (self.k1 + 1.0)) / denom

        if not np.any(scores):
            return []
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            (self.node_ids[int(idx)], float(scores[int(idx)]))
            for idx in top_indices
            if scores[int(idx)] > 0
        ]

    def save(self, path: Path) -> None:
        with path.open("wb") as handle:
            pickle.dump(self, handle)

    @staticmethod
    def load(path: Path) -> "BM25Index":
        with path.open("rb") as handle:
            return pickle.load(handle)


def prepare_node(raw: Dict[str, Any]) -> Dict[str, Any]:
    text = repair_mojibake(str(raw.get("text") or ""))
    source_docs = raw.get("source_docs") or []
    source_pages = raw.get("source_pages") or []
    metadata = raw.get("metadata") or {}
    reconstruction = metadata.get("reconstruction") or {}

    return {
        "node_id": raw["node_id"],
        "layer": "leaf" if raw.get("is_leaf") else "summary",
        "depth": int(raw.get("depth", 0)),
        "is_leaf": bool(raw.get("is_leaf")),
        "text": text,
        "token_estimate": int(raw.get("token_estimate") or max(1, len(text.split()))),
        "source_docs": source_docs,
        "source_pages": source_pages,
        "source_guideline": "; ".join(source_docs),
        "chunk_type": raw.get("chunk_type"),
        "chunk_index": raw.get("chunk_index"),
        "child_node_ids": raw.get("child_node_ids") or [],
        "source_chunk_id": raw.get("source_chunk_id"),
        "citation": {
            "source_docs": source_docs,
            "source_pages": source_pages,
            "page_markdown_path": reconstruction.get("page_markdown_path"),
            "bbox": reconstruction.get("bbox"),
        },
        "metadata": metadata,
    }


def load_nodes(path: Path) -> List[Dict[str, Any]]:
    nodes = [prepare_node(row) for row in read_jsonl(path)]
    if not nodes:
        raise ValueError(f"No RAPTOR nodes found at {path}")
    return nodes


def create_embeddings(
    texts: Sequence[str],
    *,
    model_name: str,
    batch_size: int,
    cache_folder: Optional[Path],
    device: Optional[str],
) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    kwargs: Dict[str, Any] = {}
    if cache_folder is not None:
        kwargs["cache_folder"] = str(cache_folder)
    if device:
        kwargs["device"] = device
    model = SentenceTransformer(model_name, **kwargs)
    vectors = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return np.asarray(vectors, dtype="float32")


def write_faiss_index(path: Path, embeddings: np.ndarray) -> None:
    import faiss

    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(embeddings)
    faiss.write_index(index, str(path))


def load_existing_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def build_index(config: IndexConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    nodes = load_nodes(config.nodes_path)
    leaf_nodes = [node for node in nodes if node.get("is_leaf")]
    if not leaf_nodes:
        raise ValueError("Cannot build BM25: no leaf chunks found in RAPTOR nodes.")
    node_ids = [node["node_id"] for node in nodes]
    texts = [node["text"] for node in nodes]
    leaf_node_ids = [node["node_id"] for node in leaf_nodes]
    leaf_texts = [node["text"] for node in leaf_nodes]
    text_hash = sha1_text("\n".join(f"{node_id}\t{text}" for node_id, text in zip(node_ids, texts)))
    manifest_path = config.output_dir / "manifest.json"
    existing_manifest = load_existing_manifest(manifest_path)

    print(f"Loaded {len(nodes)} RAPTOR nodes from {config.nodes_path}")
    print(f"Building BM25 index over {len(leaf_nodes)} leaf chunks with k1={config.k1}, b={config.b}")
    tokenized_docs = [
        tokenize(text, include_stopwords=config.include_stopwords)
        for text in tqdm(leaf_texts, desc="BM25 leaf tokenization")
    ]
    bm25 = BM25Index(k1=config.k1, b=config.b)
    bm25.fit(leaf_node_ids, tokenized_docs)
    bm25.save(config.output_dir / "bm25.pkl")

    embeddings_path = config.output_dir / "embeddings.npy"
    faiss_path = config.output_dir / "faiss.index"
    can_reuse_dense = (
        existing_manifest is not None
        and existing_manifest.get("text_hash") == text_hash
        and existing_manifest.get("embed_model") == config.embed_model
        and embeddings_path.exists()
        and faiss_path.exists()
    )
    if can_reuse_dense:
        print(f"Reusing existing dense index from {config.output_dir}")
        embeddings = np.load(embeddings_path)
    else:
        print(f"Embedding {len(texts)} RAPTOR nodes with {config.embed_model}")
        embeddings = create_embeddings(
            texts,
            model_name=config.embed_model,
            batch_size=config.batch_size,
            cache_folder=config.cache_folder if config.cache_folder and config.cache_folder.exists() else None,
            device=config.device,
        )
        np.save(embeddings_path, embeddings)
        write_faiss_index(faiss_path, embeddings)
    write_jsonl(config.output_dir / "nodes.jsonl", nodes)
    write_jsonl(config.output_dir / "leaf_nodes.jsonl", leaf_nodes)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nodes_path": str(config.nodes_path),
        "node_count": len(nodes),
        "leaf_node_count": len(leaf_nodes),
        "text_hash": text_hash,
        "embed_model": config.embed_model,
        "embedding_dim": int(embeddings.shape[1]),
        "bm25": {
            "scope": "leaf_only",
            "k1": config.k1,
            "b": config.b,
            "avgdl": bm25.avgdl,
            "include_stopwords": config.include_stopwords,
            "tokenizer": "pyvi" if _VI_TOKENIZER is not None else "regex",
            "normalization": "mojibake repair, lowercase, symbol folding, accent-folded variants",
        },
        "dense": {
            "scope": "all_raptor_nodes",
            "node_count": len(nodes),
        },
        "files": {
            "nodes": "nodes.jsonl",
            "leaf_nodes": "leaf_nodes.jsonl",
            "bm25": "bm25.pkl",
            "faiss": "faiss.index",
            "embeddings": "embeddings.npy",
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return manifest


def parse_args() -> IndexConfig:
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indexes for RAPTOR nodes.")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_TREE_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_INDEX_DIR)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--cache-folder", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--k1", type=float, default=1.7)
    parser.add_argument("--b", type=float, default=0.83)
    parser.add_argument("--include-stopwords", action="store_true")
    parser.add_argument("--device", default=None, help="Optional sentence-transformers device, e.g. cpu or cuda.")
    args = parser.parse_args()
    return IndexConfig(
        nodes_path=args.nodes,
        output_dir=args.output,
        embed_model=args.embed_model,
        cache_folder=args.cache_folder,
        batch_size=args.batch_size,
        k1=args.k1,
        b=args.b,
        include_stopwords=args.include_stopwords,
        device=args.device,
    )


def main() -> None:
    config = parse_args()
    manifest = build_index(config)
    print(f"Saved retrieval index to {manifest['files']} under {config.output_dir}")
    print(json.dumps({k: manifest[k] for k in ("node_count", "embed_model", "embedding_dim", "bm25")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
