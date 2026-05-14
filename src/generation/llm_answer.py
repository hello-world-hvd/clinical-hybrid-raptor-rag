# src/generation/llm_answer.py
from __future__ import annotations

import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from dotenv import load_dotenv
load_dotenv()

@dataclass
class RetrievedContext:
    node_id: str
    text: str
    snippet: str
    source_guideline: str = ""
    source_pages: str = ""
    citation: Optional[Dict[str, Any]] = None
    score: Optional[float] = None
    rerank_score: Optional[float] = None


def _safe_join_pages(pages: Any) -> str:
    if not pages:
        return ""
    if isinstance(pages, (list, tuple)):
        return ",".join(str(p) for p in pages)
    return str(pages)


def format_context_block(results: Sequence[Dict[str, Any]], max_items: int = 5) -> str:
    blocks: List[str] = []
    for i, item in enumerate(results[:max_items], start=1):
        pages = _safe_join_pages(item.get("source_pages"))
        src = item.get("source_guideline") or ""
        snippet = item.get("snippet") or item.get("text") or ""
        blocks.append(
            f"[{i}] source={src} pages={pages}\n{snippet}".strip()
        )
    return "\n\n".join(blocks)


def build_prompt(query: str, results: Sequence[Dict[str, Any]], max_items: int = 5) -> str:
    context_block = format_context_block(results, max_items=max_items)
    return textwrap.dedent(
        f"""
        Bạn là trợ lý y khoa. Chỉ trả lời dựa trên ngữ cảnh được cung cấp.
        Nếu chưa đủ thông tin để kết luận, hãy nói rõ “chưa đủ cơ sở” và nêu các tiêu chí/xét nghiệm cần bổ sung.
        Luôn ưu tiên triệu chứng, chẩn đoán, cận lâm sàng và xử trí phù hợp với guideline.

        Câu hỏi:
        {query}

        Ngữ cảnh:
        {context_block}

        Yêu cầu đầu ra:
        1) Trả lời ngắn gọn, rõ ràng.
        2) Nếu có thể, liệt kê các bằng chứng hỗ trợ.
        3) Nếu chưa đủ dữ liệu, nêu những gì cần hỏi/khám/xét nghiệm thêm.
        """
    ).strip()


class QwenLocalGenerator:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-3B-Instruct",
        device_map: str = "auto",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        trust_remote_code: bool = True,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
        )

        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device_map,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
        self.model.eval()

    @torch.inference_mode()
    def generate(self, prompt: str) -> str:
        messages = [
            {
                "role": "system",
                "content": "Bạn là trợ lý y khoa, trả lời ngắn gọn, chính xác, có cấu trúc.",
            },
            {"role": "user", "content": prompt},
        ]

        if hasattr(self.tokenizer, "apply_chat_template"):
            input_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            input_text = f"System: {messages[0]['content']}\nUser: {messages[1]['content']}\nAssistant:"

        inputs = self.tokenizer(input_text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )

        generated = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated[len(input_text):].strip() if generated.startswith(input_text) else generated.strip()


class CohereGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "command-r-plus-08-2024",
        temperature: float = 0.2,
        max_tokens: int = 512,
    ) -> None:
        api_key = api_key or os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY is missing")

        import cohere

        self.client = cohere.ClientV2(api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str) -> str:
        resp = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        return resp.message.content[0].text.strip()

def answer_with_llm(
    query: str,
    results: Sequence[Dict[str, Any]],
    backend: Literal["qwen", "cohere"] = "qwen",
    qwen_model: str = "Qwen/Qwen2.5-3B-Instruct",
    cohere_model: str = "command-r-plus-08-2024",
    max_context_items: int = 10,
) -> str:
    prompt = build_prompt(query, results, max_items=max_context_items)

    if backend == "qwen":
        llm = QwenLocalGenerator(model_name=qwen_model)
    elif backend == "cohere":
        llm = CohereGenerator(model=cohere_model)
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    return llm.generate(prompt)