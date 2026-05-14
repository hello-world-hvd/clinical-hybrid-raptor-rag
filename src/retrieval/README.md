# RAPTOR Retrieval

Architecture:

- BM25 over leaf chunks only
- Dense retrieval over all RAPTOR nodes
- Weighted RRF fusion across the two ranked lists
- Visual chunks are demoted unless the query asks for image/figure/diagram content
- This is the default `hybrid_collapsed` query mode

Build BM25 + FAISS indexes from the generated RAPTOR tree:

```powershell
.\venv\Scripts\python.exe -m src.retrieval.index_raptor `
  --nodes data\processed\raptor_tree\raptor_nodes.jsonl `
  --output data\processed\raptor_index `
  --k1 1.7 `
  --b 0.83 `
  --batch-size 16
```

Run hybrid collapsed retrieval with Weighted RRF:

```powershell
.\venv\Scripts\python.exe -m src.retrieval.hybrid_retriever `
  "triệu chứng ngộ độc thuốc chuột Fluoroacetat" `
  --index data\processed\raptor_index `
  --bm25-top-n 20 `
  --dense-top-m 20 `
  --bm25-weight 1 `
  --dense-weight 5 `
  --rrf-k 40 `
  --top-k 10
```

Run dense-only collapsed retrieval for comparison:

```powershell
.\venv\Scripts\python.exe -m src.retrieval.hybrid_retriever `
  "triệu chứng ngộ độc thuốc chuột Fluoroacetat" `
  --index data\processed\raptor_index `
  --mode dense_collapsed `
  --token-limit 2000 `
  --top-k 20
```

Optional cross-encoder reranking can be enabled by passing a reranker model:

```powershell
.\venv\Scripts\python.exe -m src.retrieval.hybrid_retriever `
  "triệu chứng ngộ độc thuốc chuột Fluoroacetat" `
  --index data\processed\raptor_index `
  --rerank-model BAAI/bge-reranker-v2-m3 `
  --rerank-top-k 15
```

For CPU-only machines, prefer the fast reranker preset instead of the 2.27GB quality model:

```powershell
.\venv\Scripts\python.exe -m src.retrieval.hybrid_retriever `
  "triệu chứng ngộ độc thuốc chuột Fluoroacetat" `
  --index data\processed\raptor_index `
  --rerank-preset fast `
  --rerank-top-k 8 `
  --rerank-batch-size 16 `
  --rerank-max-length 256
```

Useful presets:

- `--rerank-preset off`: skip reranking, fastest overall.
- `--rerank-preset fast`: small cross-encoder, best option for CPU latency.
- `--rerank-preset balanced`: stronger than `fast`, lighter than `quality`.
- `--rerank-preset quality`: `BAAI/bge-reranker-v2-m3`, highest cost.

If a reranker is already cached locally, add `--local-files-only` to avoid network checks.

For repeated testing, use interactive mode so the embedding and reranker stay in memory:

```powershell
.\venv\Scripts\python.exe -m src.retrieval.hybrid_retriever `
  dummy `
  --index data\processed\raptor_index `
  --rerank-preset fast `
  --rerank-top-k 5 `
  --rerank-batch-size 16 `
  --rerank-max-length 256 `
  --local-files-only `
  --interactive `
  --show-timing
```

This avoids paying model startup cost on every query.

The indexer repairs common PDF mojibake before tokenization/embedding, keeps all RAPTOR
nodes in both indexes, and writes citation metadata from each node into
`data/processed/raptor_index/nodes.jsonl`.
