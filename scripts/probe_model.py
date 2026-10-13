"""Phase 3: probe a candidate base model on the GPU before choosing it.

Loads the model in 4-bit with Unsloth, adds LoRA (r=16, attention + MLP, language layers only), then measures:
VRAM after load, trainable params, LoRA target module names, time and peak memory per training step at the
sweep's batch shape, and greedy zero-shot generation latency with the full 77-label prompt.
Writes one JSON per model. Run in Colab: PYTHONPATH=src python scripts/probe_model.py unsloth/Qwen3.5-2B --out ...
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers/peft)

import argparse
import importlib.util
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from unsloth import FastModel

ROOT = Path(__file__).resolve().parents[1]


def zero_shot_prompt(labels: list[dict], text: str) -> str:
    """Rough stand-in for the Phase 4 prompt: same label list, so the length is realistic."""
    lines = "\n".join(f"- {lab['name']}: {lab['description']}" for lab in labels)
    return (f"Classify the banking customer message into exactly one of these intents:\n{lines}\n\n"
            f"Message: {text.strip()}\nAnswer with the intent name only.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6)
    args = ap.parse_args()
    labels = json.loads((ROOT / "data/processed/labels.json").read_text(encoding="utf-8"))
    res: dict = {"model": args.model, "gpu": torch.cuda.get_device_name(0),
                 "kernels": {m: importlib.util.find_spec(m) is not None
                             for m in ("fla", "causal_conv1d", "flash_attn", "xformers")}}

    t0 = time.perf_counter()
    model, proc = FastModel.from_pretrained(args.model, max_seq_length=2560, load_in_4bit=True,
                                            full_finetuning=False)
    tok = getattr(proc, "tokenizer", proc)
    res["load_s"] = round(time.perf_counter() - t0, 1)
    res["vram_after_load_gb"] = round(torch.cuda.memory_allocated() / 1e9, 2)
    res["dtype"] = str(model.dtype)

    # Linear layers in the language model only (skip vision/audio towers), by name suffix.
    lin = Counter(name.rsplit(".", 1)[-1] for name, mod in model.named_modules()
                  if isinstance(mod, torch.nn.Linear) and not re.search(r"vision|visual|audio|embed_|lm_head", name))
    res["linear_suffixes"] = dict(lin)

    model = FastModel.get_peft_model(model, r=16, lora_alpha=32, lora_dropout=0.0, bias="none",
                                     finetune_vision_layers=False, finetune_language_layers=True,
                                     finetune_attention_modules=True, finetune_mlp_modules=True,
                                     random_state=42, use_gradient_checkpointing="unsloth")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    res["trainable_params_m"] = round(trainable / 1e6, 2)
    res["lora_targets"] = sorted({n.split(".lora_A")[0].rsplit(".", 1)[-1] for n, _ in model.named_parameters()
                                  if ".lora_A" in n})

    # Training steps at the sweep's shape (random tokens: we only time it here).
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    ids = torch.randint(100, 20000, (args.batch, args.seq), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize(); t = time.perf_counter()
        out = model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids)
        out.loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); times.append(time.perf_counter() - t)
    res["train_step_s"] = round(sorted(times[2:])[len(times[2:]) // 2], 3)  # median after warm-up
    res["train_peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    res["train_loss_finite"] = bool(torch.isfinite(out.loss).item())
    del opt, out

    # Zero-shot generation, batch 1, greedy, thinking off where the template supports it.
    FastModel.for_inference(model)
    msgs = [{"role": "user", "content": zero_shot_prompt(labels, "I am still waiting on my card?")}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(text=prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
    res["prompt_tokens"] = int(enc["input_ids"].shape[1])
    res["prompt_tail"] = prompt[-160:]
    gen_times = []
    for _ in range(4):
        torch.cuda.synchronize(); t = time.perf_counter()
        with torch.no_grad():
            g = model.generate(**enc, max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize(); gen_times.append(time.perf_counter() - t)
    res["gen16_latency_s"] = round(sorted(gen_times[1:])[1], 3)
    res["zero_shot_output"] = tok.decode(g[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
