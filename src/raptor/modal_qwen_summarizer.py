from __future__ import annotations

from typing import List

import modal


APP_NAME = "raptor-qwen25-summarizer"
MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
EMBED_MODEL_NAME = "BAAI/bge-m3"
CACHE_DIR = "/cache"


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "accelerate>=0.33.0",
        "huggingface-hub>=0.24.0",
        "safetensors>=0.4.4",
        "sentence-transformers>=3.0.0",
        "torch>=2.4.0",
        "transformers>=4.44.0",
    )
)

cache_volume = modal.Volume.from_name("raptor-qwen25-cache", create_if_missing=True)
app = modal.App(APP_NAME)

_model = None
_tokenizer = None
_embedder = None


def _load_model():
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        trust_remote_code=True,
    )
    _model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=CACHE_DIR,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    _model.eval()
    return _model, _tokenizer


def _load_embedder(model_name: str = EMBED_MODEL_NAME):
    global _embedder
    if _embedder is not None:
        return _embedder

    from sentence_transformers import SentenceTransformer

    _embedder = SentenceTransformer(
        model_name,
        cache_folder=CACHE_DIR,
        device="cuda",
    )
    return _embedder


def _build_prompt(texts: List[str], max_input_chars: int) -> str:
    clipped = []
    used = 0
    for idx, text in enumerate(texts, start=1):
        piece = str(text).strip()
        if not piece:
            continue
        remaining = max_input_chars - used
        if remaining <= 0:
            break
        piece = piece[:remaining]
        used += len(piece)
        clipped.append(f"[{idx}] {piece}")

    body = "\n\n".join(clipped)
    return (
        "Tom tat cac muc quan trong trong cac doan tai lieu y khoa sau. "
        "Giu lai benh/can thiep, trieu chung, tieu chi chan doan, xet nghiem, "
        "nguong gia tri, xu tri va canh bao quan trong neu co. "
        "Viet bang tieng Viet co dau, ngan gon, khong them thong tin ngoai van ban.\n\n"
        f"{body}"
    )


@app.function(
    image=image,
    gpu="A10G",
    volumes={CACHE_DIR: cache_volume},
    timeout=1800,
    scaledown_window=300,
    max_containers=6,
)
def summarize_batch(
    clusters: List[List[str]],
    max_new_tokens: int = 384,
    max_input_chars: int = 18000,
) -> List[str]:
    import torch

    model, tokenizer = _load_model()
    outputs: List[str] = []

    for texts in clusters:
        prompt = _build_prompt(texts, max_input_chars=max_input_chars)
        messages = [
            {"role": "system", "content": "Ban la tro ly tom tat tai lieu y khoa bang tieng Viet."},
            {"role": "user", "content": prompt},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer([chat_text], return_tensors="pt").to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = generated[:, inputs.input_ids.shape[-1] :]
        summary = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
        outputs.append(summary)

    return outputs


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
    model_name: str = EMBED_MODEL_NAME,
    batch_size: int = 64,
) -> List[List[float]]:
    embedder = _load_embedder(model_name)
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32").tolist()


@app.local_entrypoint()
def smoke():
    result = summarize_batch.remote(
        [["Benh nhan ngo doc paracetamol can danh gia thoi diem uong, lieu va xet nghiem men gan."]]
    )
    print(result[0])
