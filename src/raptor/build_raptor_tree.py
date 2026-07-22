from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from ftfy import fix_text
from tqdm import tqdm

try:
    from ..openrouter_client import DEFAULT_OPENROUTER_MODEL, OpenRouterClient
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from openrouter_client import DEFAULT_OPENROUTER_MODEL, OpenRouterClient


DEFAULT_INPUT = Path("data/processed/preprocess_output")
DEFAULT_OUTPUT = Path("data/processed/raptor_tree")
DEFAULT_EMBED_MODEL = "BAAI/bge-m3"
MODAL_APP_NAME = "raptor-bge-m3-embedder"
MODAL_EMBED_FUNCTION_NAME = "embed_batch"
_LOCAL_EMBEDDERS: Dict[Tuple[str, str], Any] = {}


@dataclass
class BuildConfig:
    input_path: Path
    output_dir: Path
    embed_model: str = DEFAULT_EMBED_MODEL
    embedding_backend: str = "auto"
    batch_size: int = 16
    max_depth: int = 4
    umap_neighbors: int = 15
    umap_components: int = 8
    bic_max_k: int = 6
    min_cluster_size: int = 3
    max_cluster_tokens: int = 9000
    soft_threshold: float = 0.30
    max_memberships: int = 1
    summary_model: str = DEFAULT_OPENROUTER_MODEL
    summary_timeout: float = 120.0
    modal_embed_batch_size: int = 96
    max_summary_tokens: int = 384
    random_state: int = 42


def find_chunks_jsonl(root: Path) -> List[Path]:
    if root.is_file() and root.name.endswith(".jsonl"):
        return [root]
    return sorted(root.rglob("chunks.jsonl"))


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


def count_words(text: str) -> int:
    return len(re.findall(r"\S+", text or ""))


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(count_words(text) * 1.3))


def stable_id(prefix: str, *parts: Any) -> str:
    raw = "||".join(str(part) for part in parts)
    return f"{prefix}_{hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]}"


def clean_text(text: str) -> str:
    return re.sub(r"\n{3,}", "\n\n", fix_text(str(text or "")).strip())


def load_leaf_nodes(input_path: Path) -> List[Dict[str, Any]]:
    paths = find_chunks_jsonl(input_path)
    if not paths:
        raise FileNotFoundError(f"Cannot find chunks.jsonl under {input_path}")

    leaves: List[Dict[str, Any]] = []
    seq = 0
    for jsonl_path in paths:
        for item in read_jsonl(jsonl_path):
            text = clean_text(item.get("content") or item.get("search_text") or "")
            if not text:
                continue
            source_doc = item.get("doc_name") or jsonl_path.parent.name
            page_number = item.get("page_number")
            chunk_index = item.get("chunk_index")
            source_id = item.get("id") or stable_id("chunk", source_doc, page_number, chunk_index, text[:80])
            node_id = stable_id("leaf", seq, source_id)
            metadata = item.get("metadata") or {}
            leaves.append(
                {
                    "node_id": node_id,
                    "source_chunk_id": source_id,
                    "depth": 0,
                    "text": text,
                    "token_estimate": estimate_tokens(text),
                    "is_leaf": True,
                    "child_node_ids": [],
                    "source_docs": [source_doc] if source_doc else [],
                    "source_pages": [int(page_number)] if page_number is not None else [],
                    "chunk_type": item.get("chunk_type"),
                    "chunk_index": chunk_index,
                    "metadata": {
                        "source_jsonl": str(jsonl_path),
                        "original_metadata": metadata,
                        "reconstruction": item.get("reconstruction"),
                    },
                }
            )
            seq += 1
    return leaves


def load_or_create_embeddings(
    nodes: Sequence[Dict[str, Any]],
    cache_path: Path,
    model_name: str,
    batch_size: int,
    backend: str,
    modal_batch_size: int,
) -> np.ndarray:
    meta_path = cache_path.with_suffix(".meta.json")
    text_hash = hashlib.sha1(
        "\n".join(f"{node['node_id']}\t{node['text']}" for node in nodes).encode("utf-8")
    ).hexdigest()

    if cache_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_name") == model_name and meta.get("text_hash") == text_hash:
            print(f"Loading cached embeddings: {cache_path}")
            return np.load(cache_path)

    texts = [node["text"] for node in nodes]
    embeddings, backend_used = create_embeddings(
        texts,
        model_name=model_name,
        batch_size=batch_size,
        backend=backend,
        modal_batch_size=modal_batch_size,
    )
    np.save(cache_path, embeddings)
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "backend": backend_used,
                "text_hash": text_hash,
                "shape": list(embeddings.shape),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return embeddings


def _torch_cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def local_embed_texts(model_name: str, texts: Sequence[str], batch_size: int) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _torch_cuda_available() else "cpu"
    print(f"Embedding {len(texts)} texts locally with {model_name} on {device}")
    key = (model_name, device)
    embedder = _LOCAL_EMBEDDERS.get(key)
    if embedder is None:
        embedder = SentenceTransformer(model_name, device=device)
        _LOCAL_EMBEDDERS[key] = embedder
    return embedder.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")


def modal_embed_texts(
    model_name: str,
    texts: Sequence[str],
    local_batch_size: int,
    remote_batch_size: int,
) -> np.ndarray:
    import modal

    fn = modal.Function.from_name(MODAL_APP_NAME, MODAL_EMBED_FUNCTION_NAME)
    chunks: List[np.ndarray] = []
    for start in tqdm(range(0, len(texts), remote_batch_size), desc="Modal embeddings"):
        batch = list(texts[start : start + remote_batch_size])
        vectors = fn.remote(batch, model_name=model_name, batch_size=local_batch_size)
        chunks.append(np.asarray(vectors, dtype="float32"))
    return np.vstack(chunks) if chunks else np.empty((0, 0), dtype="float32")


def create_embeddings(
    texts: Sequence[str],
    model_name: str,
    batch_size: int,
    backend: str,
    modal_batch_size: int,
) -> Tuple[np.ndarray, str]:
    if backend in {"modal", "auto"}:
        try:
            print(f"Embedding {len(texts)} texts on Modal GPU with {model_name}")
            return (
                modal_embed_texts(
                    model_name=model_name,
                    texts=texts,
                    local_batch_size=batch_size,
                    remote_batch_size=modal_batch_size,
                ),
                "modal",
            )
        except Exception as exc:
            if backend == "modal":
                raise
            print(f"Modal embedding failed, falling back to local embedding: {exc}")

    return local_embed_texts(model_name, texts, batch_size), "local"


def load_or_create_level_embeddings(
    nodes: Sequence[Dict[str, Any]],
    cache_path: Path,
    config: BuildConfig,
) -> np.ndarray:
    return load_or_create_embeddings(
        nodes=nodes,
        cache_path=cache_path,
        model_name=config.embed_model,
        batch_size=config.batch_size,
        backend=config.embedding_backend,
        modal_batch_size=config.modal_embed_batch_size,
    )


def reduce_embeddings(embeddings: np.ndarray, config: BuildConfig) -> np.ndarray:
    from sklearn.decomposition import PCA

    n_samples = len(embeddings)
    if n_samples <= 2:
        return embeddings

    if n_samples <= config.umap_components + 2:
        n_components = min(config.umap_components, embeddings.shape[1], max(1, n_samples - 1))
        return PCA(n_components=n_components, random_state=config.random_state).fit_transform(embeddings)

    n_neighbors = min(config.umap_neighbors, max(2, n_samples - 1))
    n_components = min(config.umap_components, max(2, n_samples - 1))
    import umap

    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        n_components=n_components,
        min_dist=0.0,
        metric="cosine",
        random_state=config.random_state,
    )
    return reducer.fit_transform(embeddings)


def choose_gmm_k(X: np.ndarray, config: BuildConfig) -> Tuple[int, List[Dict[str, float]]]:
    from sklearn.mixture import GaussianMixture

    n_samples = len(X)
    if n_samples < config.min_cluster_size:
        return 1, []

    max_k = min(config.bic_max_k, n_samples - 1)
    if max_k < 2:
        return 1, []

    records: List[Dict[str, float]] = []
    for k in range(1, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=config.random_state,
                reg_covar=1e-5,
                n_init=3,
            )
            gmm.fit(X)
            records.append({"k": float(k), "bic": float(gmm.bic(X)), "aic": float(gmm.aic(X))})
        except ValueError:
            continue

    if not records:
        return 1, []
    best = min(records, key=lambda item: item["bic"])
    return int(best["k"]), records


def gmm_soft_clusters(
    nodes: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    config: BuildConfig,
) -> Tuple[List[List[int]], int, List[Dict[str, float]]]:
    from sklearn.mixture import GaussianMixture

    if len(nodes) <= config.min_cluster_size:
        return [list(range(len(nodes)))], 1, []

    reduced = reduce_embeddings(embeddings, config)
    best_k, bic_records = choose_gmm_k(reduced, config)
    if best_k <= 1:
        return [list(range(len(nodes)))], best_k, bic_records

    gmm = GaussianMixture(
        n_components=best_k,
        covariance_type="full",
        random_state=config.random_state,
        reg_covar=1e-5,
        n_init=5,
    )
    probs = gmm.fit(reduced).predict_proba(reduced)
    clusters: Dict[int, List[int]] = {idx: [] for idx in range(best_k)}

    for row_idx, prob_row in enumerate(probs):
        ranked = np.argsort(prob_row)[::-1]
        chosen = [int(ranked[0])]
        for cluster_idx in ranked[1 : config.max_memberships]:
            if prob_row[cluster_idx] >= config.soft_threshold:
                chosen.append(int(cluster_idx))
        for cluster_idx in chosen:
            clusters[cluster_idx].append(row_idx)

    non_empty = [members for members in clusters.values() if members]
    return non_empty, best_k, bic_records


def cached_layer_nodes(
    path: Path,
    clusters: List[List[int]],
    current_nodes: Sequence[Dict[str, Any]],
) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None

    cached = list(read_jsonl(path))
    by_children = {tuple(node.get("child_node_ids", [])): node for node in cached}
    matched: List[Dict[str, Any]] = []
    for members in clusters:
        child_ids = tuple(current_nodes[idx]["node_id"] for idx in members)
        node = by_children.get(child_ids)
        if node is None or not node.get("text"):
            return None
        matched.append(node)
    return matched


def cached_backend_label(nodes: Sequence[Dict[str, Any]]) -> str:
    backends = sorted(
        {
            str(node.get("metadata", {}).get("summary_backend"))
            for node in nodes
            if node.get("metadata", {}).get("summary_backend")
        }
    )
    return "cached:" + ",".join(backends) if backends else "cached"


def split_large_clusters(
    clusters: List[List[int]],
    nodes: Sequence[Dict[str, Any]],
    embeddings: np.ndarray,
    config: BuildConfig,
) -> List[List[int]]:
    final: List[List[int]] = []
    queue = list(clusters)

    while queue:
        members = queue.pop(0)
        total_tokens = sum(int(nodes[idx]["token_estimate"]) for idx in members)
        if total_tokens <= config.max_cluster_tokens or len(members) < config.min_cluster_size * 2:
            final.append(members)
            continue

        sub_nodes = [nodes[idx] for idx in members]
        sub_embeddings = embeddings[members]
        sub_clusters, best_k, _ = gmm_soft_clusters(sub_nodes, sub_embeddings, config)
        if best_k <= 1 or len(sub_clusters) <= 1:
            final.append(members)
            continue
        for sub in sub_clusters:
            queue.append([members[idx] for idx in sub])

    return final


def build_summary_prompt(texts: Sequence[str], max_input_chars: int) -> str:
    parts: List[str] = []
    used_chars = 0
    for index, text in enumerate(texts, start=1):
        remaining = max_input_chars - used_chars
        if remaining <= 0:
            break
        clipped = str(text).strip()[:remaining]
        if clipped:
            parts.append(f"[{index}] {clipped}")
            used_chars += len(clipped)

    return (
        "Hãy tóm tắt cụm đoạn văn y khoa dưới đây bằng tiếng Việt có dấu. "
        "Giữ lại bệnh/can thiệp, triệu chứng, tiêu chí chẩn đoán, xét nghiệm, "
        "ngưỡng giá trị, xử trí và cảnh báo quan trọng. Không thêm thông tin "
        "ngoài nguồn. Trả về một đoạn tóm tắt ngắn gọn, không dùng lời dẫn.\n\n"
        + "\n\n".join(parts)
    )


def openrouter_summarize_clusters(
    clusters: Sequence[Sequence[str]],
    config: BuildConfig,
) -> List[str]:
    client = OpenRouterClient(
        model=config.summary_model,
        timeout=config.summary_timeout,
    )
    max_input_chars = max(6000, config.max_cluster_tokens * 5)
    summaries: List[str] = []
    for texts in tqdm(clusters, desc="OpenRouter summaries"):
        summaries.append(
            client.complete(
                system_prompt=(
                    "Bạn là trợ lý tóm tắt tài liệu y khoa. Chỉ sử dụng thông tin "
                    "trong các đoạn nguồn và ưu tiên độ chính xác."
                ),
                user_prompt=build_summary_prompt(texts, max_input_chars),
                max_tokens=config.max_summary_tokens,
            )
        )
    return summaries


def summary_cache_key(child_ids: Sequence[str], texts: Sequence[str], config: BuildConfig) -> str:
    payload = {
        "child_ids": list(child_ids),
        "text_hash": hashlib.sha1("\n".join(texts).encode("utf-8")).hexdigest(),
        "summary_model": config.summary_model,
        "max_summary_tokens": config.max_summary_tokens,
        "max_cluster_tokens": config.max_cluster_tokens,
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def read_summary_cache(cache_dir: Path, key: str) -> Optional[Dict[str, Any]]:
    path = cache_dir / f"{key}.json"
    if not path.exists():
        return None
    try:
        item = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not item.get("summary"):
        return None
    return item


def write_summary_cache(cache_dir: Path, key: str, summary: str, backend: str) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{key}.json").write_text(
        json.dumps(
            {
                "summary": summary,
                "backend": backend,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def summarize_clusters(
    cluster_texts: List[List[str]],
    config: BuildConfig,
) -> Tuple[List[str], str]:
    summaries = openrouter_summarize_clusters(cluster_texts, config)
    return summaries, f"openrouter:{config.summary_model}"


def summarize_clusters_cached(
    clusters: List[List[int]],
    current_nodes: Sequence[Dict[str, Any]],
    config: BuildConfig,
) -> Tuple[List[str], str]:
    cache_dir = config.output_dir / "summary_cache"
    summaries: List[Optional[str]] = []
    cache_keys: List[str] = []
    missing_texts: List[List[str]] = []
    missing_positions: List[int] = []
    cached_backends = set()

    for pos, members in enumerate(clusters):
        child_nodes = [current_nodes[idx] for idx in members]
        child_ids = [node["node_id"] for node in child_nodes]
        texts = [node["text"] for node in child_nodes]
        key = summary_cache_key(child_ids, texts, config)
        cache_keys.append(key)
        cached = read_summary_cache(cache_dir, key)
        if cached is not None:
            summaries.append(str(cached["summary"]))
            cached_backends.add(str(cached.get("backend", "cached")))
        else:
            summaries.append(None)
            missing_texts.append(texts)
            missing_positions.append(pos)

    backend_used = "cached"
    if missing_texts:
        fresh_summaries, backend_used = summarize_clusters(missing_texts, config)
        for pos, summary in zip(missing_positions, fresh_summaries):
            summaries[pos] = summary
            write_summary_cache(cache_dir, cache_keys[pos], summary, backend_used)

    if not missing_texts and cached_backends:
        backend_used = "cached:" + ",".join(sorted(cached_backends))
    elif cached_backends:
        backend_used = f"{backend_used}+cached"

    return [str(summary) for summary in summaries], backend_used


def make_parent_node(
    depth: int,
    cluster_idx: int,
    child_nodes: Sequence[Dict[str, Any]],
    summary: str,
    backend_used: str,
    best_k: int,
) -> Dict[str, Any]:
    child_ids = [node["node_id"] for node in child_nodes]
    source_docs = sorted({doc for node in child_nodes for doc in node.get("source_docs", [])})
    source_pages = sorted({page for node in child_nodes for page in node.get("source_pages", [])})
    child_chunk_types = sorted({str(node.get("chunk_type")) for node in child_nodes if node.get("chunk_type")})
    chunk_type = child_chunk_types[0] if len(child_chunk_types) == 1 else "mixed"
    node_id = stable_id("node", depth, cluster_idx, "|".join(child_ids))
    return {
        "node_id": node_id,
        "depth": depth,
        "text": clean_text(summary),
        "token_estimate": estimate_tokens(summary),
        "is_leaf": False,
        "child_node_ids": child_ids,
        "source_docs": source_docs,
        "source_pages": source_pages,
        "chunk_type": chunk_type,
        "cluster_id": cluster_idx,
        "metadata": {
            "n_children": len(child_ids),
            "child_chunk_types": child_chunk_types,
            "children_token_estimate": sum(int(node["token_estimate"]) for node in child_nodes),
            "summary_backend": backend_used,
            "gmm_best_k": best_k,
        },
    }


def build_tree(config: BuildConfig) -> Dict[str, Any]:
    config.output_dir.mkdir(parents=True, exist_ok=True)

    leaves = load_leaf_nodes(config.input_path)
    print(f"Loaded {len(leaves)} leaf chunks from {config.input_path}")
    write_jsonl(config.output_dir / "leaf_nodes.jsonl", leaves)

    leaf_embeddings = load_or_create_level_embeddings(
        leaves,
        config.output_dir / "leaf_embeddings.npy",
        config,
    )

    all_nodes: Dict[str, Dict[str, Any]] = {node["node_id"]: node for node in leaves}
    nodes_by_level: List[List[Dict[str, Any]]] = [leaves]
    layer_summaries: List[Dict[str, Any]] = []
    current_nodes = leaves
    current_embeddings = leaf_embeddings

    for depth in range(1, config.max_depth + 1):
        total_tokens = sum(int(node["token_estimate"]) for node in current_nodes)
        if len(current_nodes) < config.min_cluster_size:
            if len(current_nodes) > 1:
                clusters = [list(range(len(current_nodes)))]
                layer_path = config.output_dir / f"layer_{depth}_nodes.jsonl"
                cached_nodes = cached_layer_nodes(layer_path, clusters, current_nodes)
                if cached_nodes is not None:
                    next_nodes = cached_nodes
                    backend_used = cached_backend_label(cached_nodes)
                else:
                    summaries, backend_used = summarize_clusters_cached(
                        clusters,
                        current_nodes,
                        config,
                    )
                    next_nodes = [
                        make_parent_node(depth, 0, current_nodes, summaries[0], backend_used, best_k=1)
                    ]
                    write_jsonl(layer_path, next_nodes)

                for node in next_nodes:
                    all_nodes[node["node_id"]] = node
                nodes_by_level.append(next_nodes)
                layer_summaries.append(
                    {
                        "depth": depth,
                        "status": "ok",
                        "reason": "final_root_summary",
                        "input_nodes": len(current_nodes),
                        "output_nodes": len(next_nodes),
                        "input_token_estimate": total_tokens,
                        "best_k": 1,
                        "cluster_sizes": [len(current_nodes)],
                        "summary_backend": backend_used,
                        "bic": [],
                    }
                )
                print(
                    f"Depth {depth}: {len(current_nodes)} nodes -> {len(next_nodes)} final root "
                    f"using {backend_used}"
                )
                break

            layer_summaries.append(
                {"depth": depth, "status": "stop", "reason": "too_few_nodes", "input_nodes": len(current_nodes)}
            )
            break

        clusters, best_k, bic_records = gmm_soft_clusters(current_nodes, current_embeddings, config)
        clusters = split_large_clusters(clusters, current_nodes, current_embeddings, config)
        if len(clusters) == 1 and len(current_nodes) == 1:
            break

        layer_path = config.output_dir / f"layer_{depth}_nodes.jsonl"
        cached_nodes = cached_layer_nodes(layer_path, clusters, current_nodes)
        if cached_nodes is not None:
            next_nodes = cached_nodes
            backend_used = cached_backend_label(cached_nodes)
        else:
            summaries, backend_used = summarize_clusters_cached(clusters, current_nodes, config)
            next_nodes = []

        for cluster_idx, members in enumerate(clusters):
            if cached_nodes is not None:
                all_nodes[next_nodes[cluster_idx]["node_id"]] = next_nodes[cluster_idx]
                continue

            summary = summaries[cluster_idx]
            child_nodes = [current_nodes[idx] for idx in members]
            parent = make_parent_node(depth, cluster_idx, child_nodes, summary, backend_used, best_k)
            next_nodes.append(parent)
            all_nodes[parent["node_id"]] = parent

        if cached_nodes is None:
            write_jsonl(layer_path, next_nodes)
        layer_summaries.append(
            {
                "depth": depth,
                "status": "ok",
                "input_nodes": len(current_nodes),
                "output_nodes": len(next_nodes),
                "input_token_estimate": total_tokens,
                "best_k": best_k,
                "cluster_sizes": [len(members) for members in clusters],
                "summary_backend": backend_used,
                "bic": bic_records,
            }
        )
        nodes_by_level.append(next_nodes)
        print(
            f"Depth {depth}: {len(current_nodes)} nodes -> {len(next_nodes)} summaries "
            f"using {backend_used}"
        )

        if len(next_nodes) <= 1:
            break

        current_nodes = next_nodes
        current_embeddings = load_or_create_level_embeddings(
            current_nodes,
            config.output_dir / f"layer_{depth}_embeddings.npy",
            config,
        )

    if len(nodes_by_level[-1]) > 1:
        final_children = nodes_by_level[-1]
        final_depth = max(int(node.get("depth", 0)) for node in final_children) + 1
        clusters = [list(range(len(final_children)))]
        layer_path = config.output_dir / f"layer_{final_depth}_nodes.jsonl"
        cached_nodes = cached_layer_nodes(layer_path, clusters, final_children)
        if cached_nodes is not None:
            final_nodes = cached_nodes
            backend_used = cached_backend_label(cached_nodes)
        else:
            summaries, backend_used = summarize_clusters_cached(clusters, final_children, config)
            final_nodes = [
                make_parent_node(final_depth, 0, final_children, summaries[0], backend_used, best_k=1)
            ]
            write_jsonl(layer_path, final_nodes)

        for node in final_nodes:
            all_nodes[node["node_id"]] = node
        nodes_by_level.append(final_nodes)
        layer_summaries.append(
            {
                "depth": final_depth,
                "status": "ok",
                "reason": "forced_single_root",
                "input_nodes": len(final_children),
                "output_nodes": len(final_nodes),
                "input_token_estimate": sum(int(node["token_estimate"]) for node in final_children),
                "best_k": 1,
                "cluster_sizes": [len(final_children)],
                "summary_backend": backend_used,
                "bic": [],
            }
        )
        print(
            f"Depth {final_depth}: {len(final_children)} nodes -> {len(final_nodes)} forced root "
            f"using {backend_used}"
        )

    tree = {
        "config": {
            "input_path": str(config.input_path),
            "output_dir": str(config.output_dir),
            "embed_model": config.embed_model,
            "embedding_backend": config.embedding_backend,
            "embedding_dim": int(leaf_embeddings.shape[1]),
            "max_depth": config.max_depth,
            "umap_neighbors": config.umap_neighbors,
            "umap_components": config.umap_components,
            "bic_max_k": config.bic_max_k,
            "min_cluster_size": config.min_cluster_size,
            "max_cluster_tokens": config.max_cluster_tokens,
            "soft_threshold": config.soft_threshold,
            "max_memberships": config.max_memberships,
            "summary_model": config.summary_model,
            "summary_timeout": config.summary_timeout,
            "modal_embed_batch_size": config.modal_embed_batch_size,
        },
        "stats": {
            "leaf_nodes": len(leaves),
            "total_nodes": len(all_nodes),
            "levels": len(nodes_by_level),
            "root_node_ids": [node["node_id"] for node in nodes_by_level[-1]],
        },
        "layer_summaries": layer_summaries,
        "nodes_by_level": [[node["node_id"] for node in level] for level in nodes_by_level],
        "nodes": all_nodes,
    }
    (config.output_dir / "raptor_tree.json").write_text(
        json.dumps(tree, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_jsonl(config.output_dir / "raptor_nodes.jsonl", all_nodes.values())
    return tree


def parse_args() -> BuildConfig:
    parser = argparse.ArgumentParser(description="Build a RAPTOR tree from preprocessed chunks.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    parser.add_argument("--embedding-backend", choices=["auto", "modal", "local"], default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--umap-neighbors", type=int, default=15)
    parser.add_argument("--umap-components", type=int, default=8)
    parser.add_argument("--bic-max-k", type=int, default=6)
    parser.add_argument("--min-cluster-size", type=int, default=3)
    parser.add_argument("--max-cluster-tokens", type=int, default=9000)
    parser.add_argument("--soft-threshold", type=float, default=0.30)
    parser.add_argument("--max-memberships", type=int, default=1)
    parser.add_argument("--summary-model", default=DEFAULT_OPENROUTER_MODEL)
    parser.add_argument("--summary-timeout", type=float, default=120.0)
    parser.add_argument("--modal-embed-batch-size", type=int, default=96)
    parser.add_argument("--max-summary-tokens", type=int, default=384)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    return BuildConfig(
        input_path=args.input,
        output_dir=args.output,
        embed_model=args.embed_model,
        embedding_backend=args.embedding_backend,
        batch_size=args.batch_size,
        max_depth=args.max_depth,
        umap_neighbors=args.umap_neighbors,
        umap_components=args.umap_components,
        bic_max_k=args.bic_max_k,
        min_cluster_size=args.min_cluster_size,
        max_cluster_tokens=args.max_cluster_tokens,
        soft_threshold=args.soft_threshold,
        max_memberships=args.max_memberships,
        summary_model=args.summary_model,
        summary_timeout=args.summary_timeout,
        modal_embed_batch_size=args.modal_embed_batch_size,
        max_summary_tokens=args.max_summary_tokens,
        random_state=args.random_state,
    )


def main() -> None:
    config = parse_args()
    tree = build_tree(config)
    print(f"Saved RAPTOR tree to {config.output_dir / 'raptor_tree.json'}")
    print(json.dumps(tree["stats"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
