from __future__ import annotations

from typing import List

import modal


APP_NAME = "raptor-bge-m3-embedder"
MODEL_NAME = "BAAI/bge-m3"
CACHE_DIR = "/cache"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "huggingface-hub>=0.24.0",
        "safetensors>=0.4.4",
        "sentence-transformers>=3.0.0",
        "torch>=2.4.0",
    )
)

cache_volume = modal.Volume.from_name("raptor-bge-m3-cache", create_if_missing=True)
app = modal.App(APP_NAME)
_embedder = None


def _load_embedder(model_name: str = MODEL_NAME):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(
            model_name,
            cache_folder=CACHE_DIR,
            device="cuda",
        )
    return _embedder


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume},
    timeout=1800,
    scaledown_window=300,
    max_containers=2,
)
def embed_batch(
    texts: List[str],
    model_name: str = MODEL_NAME,
    batch_size: int = 64,
) -> List[List[float]]:
    embeddings = _load_embedder(model_name).encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32").tolist()


@app.local_entrypoint()
def smoke():
    vectors = embed_batch.remote(["Thử nghiệm embedding tài liệu y khoa."])
    print(len(vectors), len(vectors[0]))
