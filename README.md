# 🏥 Clinical Hybrid RAPTOR RAG

> **Hệ thống Truy xuất và Sinh câu trả lời thông minh cho Tài liệu Y khoa Lâm sàng**  
> Học viện Công nghệ Bưu chính Viễn thông (PTIT) — Môn Truy xuất Thông tin, Năm 4 Kỳ 2

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Qdrant](https://img.shields.io/badge/VectorDB-Qdrant-red?logo=qdrant)](https://qdrant.tech)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![RAGAS](https://img.shields.io/badge/Eval-RAGAS-orange)](https://docs.ragas.io)

---

## 📖 Giới thiệu

**Clinical Hybrid RAPTOR RAG** là hệ thống RAG (Retrieval-Augmented Generation) lai ghép nhiều kỹ thuật tiên tiến, được thiết kế chuyên biệt cho bài toán trả lời câu hỏi y khoa lâm sàng tiếng Việt. Hệ thống tích hợp:

- **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval): Xây dựng cây phân cấp tóm tắt tài liệu
- **Graph RAG**: Trích xuất thực thể y khoa và xây dựng đồ thị tri thức
- **Hybrid Retrieval**: Kết hợp BM25 (sparse), Dense (BAAI/BGE-M3), và ColBERT (late interaction)
- **Query Optimization**: HyDE, Multi-query, Step-back query
- **LLM Reranking**: Tái xếp hạng kết quả bằng LLM

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────────────┐
│                     CLINICAL HYBRID RAPTOR RAG                  │
├─────────────┬───────────────────────┬───────────────────────────┤
│  DATA LAYER │   INDEXING LAYER      │   RETRIEVAL & GEN LAYER   │
│             │                       │                           │
│ PDF/DOCX    │  ┌─ RAPTOR Tree ─┐    │  ┌─ Query Optimizer ─┐   │
│ Guidelines  │  │ Hierarchical  │    │  │  HyDE / MultiQ   │   │
│             │  │ Summarization │    │  │  Step-back       │   │
│ Crawl Data  │  └───────────────┘    │  └──────────────────┘   │
│             │                       │                           │
│             │  ┌─ Graph RAG ───┐    │  ┌─ Ensemble Retriever ┐ │
│             │  │ Entity Extract│    │  │  BM25 (sparse)     │ │
│             │  │ Graph Builder │    │  │  Dense BGE-M3      │ │
│             │  │ Community Sum │    │  │  ColBERT (late)    │ │
│             │  └───────────────┘    │  └────────────────────┘ │
│             │                       │                           │
│             │  ┌─ Qdrant DB ───┐    │  ┌─ LLM Reranker ────┐  │
│             │  │ Dense vectors │    │  │  OpenRouter API   │  │
│             │  │ Sparse BM25   │    │  └───────────────────┘  │
│             │  │ ColBERT multi │    │                           │
│             │  └───────────────┘    │  ┌─ Answer Generator ─┐  │
│             │                       │  │  Qwen2.5 / LLM    │  │
│             │                       │  └───────────────────┘  │
└─────────────┴───────────────────────┴───────────────────────────┘
```

---

## 📁 Cấu trúc thư mục

```
project/
├── data/
│   ├── raw/                    # Dữ liệu gốc (PDF, DOCX hướng dẫn lâm sàng)
│   ├── processed/
│   │   ├── preprocess_output/  # Chunks sau tiền xử lý
│   │   ├── raptor_tree/        # Cây RAPTOR (JSONL)
│   │   ├── graph_rag/          # Đồ thị thực thể y khoa
│   │   └── qdrant_index/       # Manifest Qdrant
│   └── cache/
│       ├── embeddings/         # Cache embedding vectors
│       └── query_optimization/ # Cache tối ưu truy vấn
│
├── src/
│   ├── crawldata/              # Thu thập dữ liệu
│   ├── preprocess/             # Tiền xử lý tài liệu
│   │   ├── document.py         # Xử lý PDF/DOCX
│   │   ├── text.py             # Chunking văn bản
│   │   ├── tables.py           # Trích xuất bảng
│   │   ├── contextual_chunking.py  # Chunking có ngữ cảnh
│   │   └── normalization.py    # Chuẩn hóa văn bản
│   │
│   ├── embedding/              # Module embedding
│   │   ├── dense.py            # Dense embedder (BGE-M3)
│   │   ├── sparse.py           # Sparse BM25 embedder
│   │   └── late_interaction.py # ColBERT embedder
│   │
│   ├── raptor/                 # Xây dựng cây RAPTOR
│   │   └── build_raptor_tree.py
│   │
│   ├── graph_rag/              # Graph RAG
│   │   ├── entity_extractor.py # Trích xuất thực thể y khoa
│   │   ├── graph_builder.py    # Xây dựng đồ thị
│   │   ├── graph_indexer.py    # Index đồ thị
│   │   ├── graph_retriever.py  # Truy xuất từ đồ thị
│   │   └── community_summarizer.py
│   │
│   ├── retrieval/              # Module truy xuất
│   │   ├── ensemble_retriever.py  # Ensemble retriever chính
│   │   ├── hybrid_retriever.py    # Hybrid BM25+Dense
│   │   ├── query_optimizer.py     # HyDE, Multi-query, Step-back
│   │   ├── qdrant_indexer.py      # Index vào Qdrant
│   │   └── index_raptor.py        # Index cây RAPTOR
│   │
│   ├── vectorstore/            # Wrapper Qdrant
│   │   └── qdrant_store.py
│   │
│   ├── generation/             # Sinh câu trả lời
│   │   └── llm_answer.py       # Qwen2.5 generator
│   │
│   ├── evaluation/             # Đánh giá hệ thống
│   │   └── ragas_testset_generator.py
│   │
│   └── openrouter_client.py    # Client OpenRouter API
│
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter notebooks
├── frontend/                   # Giao diện người dùng
├── docker-compose.yml          # Docker cho Qdrant
├── requirements.txt
└── .env                        # API keys
```

---

## ⚙️ Cài đặt

### Yêu cầu hệ thống
- Python 3.10+
- CUDA (khuyến nghị, hoặc chạy trên CPU)
- Docker (cho Qdrant)
- RAM: tối thiểu 8GB (khuyến nghị 16GB+)

### 1. Clone repository

```bash
git clone <repo-url>
cd project
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình môi trường

Tạo file `.env` từ template:

```bash
cp .env.example .env
```

Cấu hình các biến:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=clinical_raptor
```

### 5. Khởi động Qdrant

```bash
docker-compose up -d
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Tiền xử lý tài liệu

```bash
python -m src.preprocess.pipeline --input data/raw/ --output data/processed/preprocess_output/
```

### Bước 2: Xây dựng cây RAPTOR

```bash
python -m src.raptor.build_raptor_tree \
    --input data/processed/preprocess_output/ \
    --output data/processed/raptor_tree/ \
    --max-depth 4 \
    --embed-model BAAI/bge-m3
```

### Bước 3: Xây dựng Graph RAG

```bash
python -m src.graph_rag.entity_extractor \
    --chunks data/processed/preprocess_output/chunks.jsonl \
    --output data/processed/graph_rag/

python -m src.graph_rag.graph_builder \
    --entities data/processed/graph_rag/entities.jsonl \
    --output data/processed/graph_rag/graph.json
```

### Bước 4: Index vào Qdrant

```bash
python -m src.retrieval.qdrant_indexer \
    --nodes data/processed/raptor_tree/nodes.jsonl \
    --qdrant-url http://localhost:6333 \
    --collection clinical_raptor
```

### Bước 5: Truy vấn

```python
from src.retrieval.ensemble_retriever import EnsembleRetriever, EnsembleConfig

config = EnsembleConfig(
    qdrant_url="http://localhost:6333",
    collection_name="clinical_raptor",
    final_top_k=10,
    rerank=True,
)
retriever = EnsembleRetriever(config=config)
results = retriever.retrieve("Xử trí ngộ độc Paracetamol như thế nào?")
```

---

## 🧩 Chi tiết các module

### Query Optimizer

Tối ưu câu truy vấn thông qua 4 kỹ thuật:
- **HyDE (Hypothetical Document Embeddings)**: Sinh tài liệu giả định để tăng recall
- **Multi-Query**: Tạo 3 cách diễn đạt khác nhau của cùng câu hỏi
- **Sub-questions**: Phân rã câu hỏi phức tạp thành câu hỏi con
- **Step-back Query**: Tạo câu hỏi tổng quát hơn

### Ensemble Retriever

Kết hợp 3 phương pháp retrieval:
| Phương pháp | Model | Top-k | Weight |
|-------------|-------|-------|--------|
| Dense | BAAI/bge-m3 | 40 | 5.0 |
| Sparse (BM25) | Qdrant FastEmbed | 30 | 2.0 |
| ColBERT | BAAI/bge-m3 (late) | 30 | 3.0 |

Tổng hợp kết quả bằng **RRF (Reciprocal Rank Fusion)** với k=60.

### RAPTOR Tree Builder

- **Clustering**: UMAP + Gaussian Mixture Model (BIC để chọn số cluster)
- **Max depth**: 4 cấp
- **Soft clustering**: Mỗi node có thể thuộc tối đa 1 cluster (configurable)
- **Summarization**: LLM (Nemotron Ultra) tóm tắt mỗi cluster

### Graph RAG

Nhận diện 13 loại thực thể y khoa:
`POISON`, `DISEASE`, `SYMPTOM`, `DRUG`, `TOXIN`, `LAB_TEST`, `PROCEDURE`, `DOSAGE`, `THRESHOLD`, `ORGAN_SYSTEM`, `CONTRAINDICATION`, `RISK_FACTOR`, `TIME`

---

## 🧪 Chạy tests

```bash
# Toàn bộ test suite
pytest tests/ -v

# Test cụ thể
pytest tests/test_ensemble_retriever.py -v
pytest tests/test_raptor_summarization.py -v
pytest tests/test_graph_rag.py -v
pytest tests/test_preprocess_chunking.py -v
```

---

## 📊 Đánh giá

Hệ thống được đánh giá bằng **RAGAS** framework:

```bash
python -m src.evaluation.ragas_testset_generator \
    --input data/processed/preprocess_output/ \
    --output data/evaluation/testset.json
```

### Metrics đánh giá
- **Answer Faithfulness**: Độ trung thực của câu trả lời với ngữ cảnh
- **Answer Relevancy**: Độ liên quan câu trả lời với câu hỏi
- **Context Precision**: Độ chính xác ngữ cảnh được truy xuất
- **Context Recall**: Độ bao phủ ngữ cảnh cần thiết

---

## 🛠️ Công nghệ sử dụng

| Thành phần | Công nghệ |
|-----------|-----------|
| Embedding | BAAI/bge-m3, ColBERT |
| Vector DB | Qdrant |
| BM25 | Qdrant FastEmbed |
| LLM | Qwen2.5-3B-Instruct, Nvidia Nemotron Ultra |
| LLM API | OpenRouter |
| Reranker | Nvidia LLaMA Nemotron Rerank |
| Graph | NetworkX |
| Evaluation | RAGAS |
| Framework | FastAPI, LangChain-core |

---

## 👥 Nhóm phát triển

| Họ tên | MSSV | Vai trò |
|--------|------|---------|
| *(Nhóm sinh viên PTIT)* | | |

---

## 📄 License

MIT License — xem file [LICENSE](LICENSE) để biết thêm chi tiết.
