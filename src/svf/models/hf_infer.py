"""GPU inference with Unsloth: the 4-bit base model (zero-shot, Phase 4) or base + LoRA adapter (Phase 6)."""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers/peft)

import time

import torch
from unsloth import FastModel


def load(model_name: str, max_seq_length: int = 4096):
    """model_name is a hub id or a saved adapter directory. Returns (model, tokenizer) ready for generation."""
    model, proc = FastModel.from_pretrained(model_name, max_seq_length=max_seq_length, load_in_4bit=True)
    FastModel.for_inference(model)
    tok = getattr(proc, "tokenizer", proc)
    tok.padding_side = "left"  # decoder-only batching: pad on the left so every row ends at the prompt
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def chat_prompt(tok, user: str, system: str | None = None) -> str:
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    try:  # thinking off for classification (Qwen3.5 template); other templates ignore or reject the kwarg
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def stop_ids(model, tok) -> list[int]:
    """End-of-turn and end-of-text ids. Qwen3.5's config eos is <|endoftext|> and the repo ships no
    generation_config, so without <|im_end|> here a fine-tuned model keeps generating after its label."""
    ids = {tok.eos_token_id, tok.convert_tokens_to_ids("<|im_end|>")}
    cfg_eos = getattr(model.generation_config, "eos_token_id", None)
    ids.update(cfg_eos if isinstance(cfg_eos, list) else [cfg_eos])
    return sorted(i for i in ids if isinstance(i, int) and i != tok.unk_token_id)


@torch.no_grad()
def generate(model, tok, prompts: list[str], max_new_tokens: int, batch_size: int, log=None) -> tuple[list[str], list[float]]:
    """Greedy decoding, stopping at end of turn. Returns (decoded continuations, wall seconds per batch)."""
    eos = stop_ids(model, tok)
    outs, secs = [], []
    for i in range(0, len(prompts), batch_size):
        enc = tok(text=prompts[i:i + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        g = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id,
                           eos_token_id=eos)
        torch.cuda.synchronize()
        secs.append(time.perf_counter() - t)
        outs.extend(tok.batch_decode(g[:, enc["input_ids"].shape[1]:], skip_special_tokens=True))
        if log and (i // batch_size) % 10 == 0:
            log(f"  {min(i + batch_size, len(prompts))}/{len(prompts)}")
    return outs, secs
