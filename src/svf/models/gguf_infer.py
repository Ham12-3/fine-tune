"""CPU inference on the merged, quantised GGUF with llama.cpp (llama-cpp-python). Used by Phases 7 and 8.

The prompt is the training prompt rendered with Qwen3.5's chat template exactly as the HF tokenizer renders it
(one user turn, generation prompt, thinking off). Phase 7 checks this string against apply_chat_template in Colab,
so the laptop needs no transformers install. Generation stops at end of turn (<|im_end|>) or end of text.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from svf.data import prompts

TEMPLATE = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
STOP = ["<|im_end|>", "<|endoftext|>"]


def chat_prompt(text: str) -> str:
    return TEMPLATE.format(user=prompts.finetune_prompt(text))


def physical_cores() -> int:
    try:
        import psutil
        return psutil.cpu_count(logical=False) or os.cpu_count() or 4
    except ImportError:
        return max(1, (os.cpu_count() or 8) // 2)


class GGUFClassifier:
    def __init__(self, path: str | Path, n_threads: int | None = None, n_ctx: int = 256, verbose: bool = False):
        from llama_cpp import Llama

        self.n_threads = n_threads or physical_cores()
        self.llm = Llama(model_path=str(path), n_ctx=n_ctx, n_threads=self.n_threads, n_batch=256, seed=0,
                         verbose=verbose)

    def prompt_tokens(self, text: str) -> int:
        return len(self.llm.tokenize(chat_prompt(text).encode("utf-8"), add_bos=False, special=True))

    def generate(self, text: str, max_tokens: int = 24) -> tuple[str, float, int]:
        """Greedy completion. Returns (text, wall seconds, completion tokens)."""
        t = time.perf_counter()
        out = self.llm(chat_prompt(text), max_tokens=max_tokens, temperature=0.0, top_k=1, stop=STOP)
        return out["choices"][0]["text"], time.perf_counter() - t, out["usage"]["completion_tokens"]
