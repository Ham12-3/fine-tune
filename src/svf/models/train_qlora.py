"""QLoRA fine-tuning of the base model on one training subset (Phase 5), resumable from Drive checkpoints.

Each example is one chat turn: user = FINETUNE_INSTRUCTION + message (no label list), assistant = the label name.
Thinking is off in the chat template, and the loss covers only the label tokens + end-of-turn (prompt masked).
"""

from __future__ import annotations

import unsloth  # noqa: F401  (must be imported before transformers/peft)

import json
import math
import time
from pathlib import Path

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments
from transformers.trainer_utils import get_last_checkpoint
from unsloth import FastModel

from svf import runinfo
from svf.data import prompts
from svf.eval import metrics
from svf.eval.parse import LabelParser
from svf.models import hf_infer


def encode(tok, text: str, label: str, max_len: int) -> dict:
    prompt = hf_infer.chat_prompt(tok, prompts.finetune_prompt(text))
    p_ids = tok(text=prompt, add_special_tokens=False)["input_ids"]
    c_ids = tok(text=label + tok.eos_token, add_special_tokens=False)["input_ids"]
    if len(p_ids) + len(c_ids) > max_len:
        raise ValueError(f"example longer than max_seq_len={max_len}: {len(p_ids) + len(c_ids)} tokens")
    return {"input_ids": p_ids + c_ids, "attention_mask": [1] * (len(p_ids) + len(c_ids)),
            "labels": [-100] * len(p_ids) + c_ids}


class Collator:
    """Right-pads a batch; padded label positions are -100."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        pad = lambda seq, v: seq + [v] * (n - len(seq))  # noqa: E731
        return {"input_ids": torch.tensor([pad(b["input_ids"], self.pad_id) for b in batch]),
                "attention_mask": torch.tensor([pad(b["attention_mask"], 0) for b in batch]),
                "labels": torch.tensor([pad(b["labels"], -100) for b in batch])}


class Sessions(TrainerCallback):
    """Appends one record per (possibly resumed) training session, so total wall time survives disconnects."""

    def __init__(self, path: Path):
        self.path = path

    def _load(self) -> list:
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def on_train_begin(self, args, state, control, **kw):
        s = self._load()
        s.append({"start": runinfo.now(), "t0": time.time(), "start_step": state.global_step})
        self.path.write_text(json.dumps(s, indent=1))

    def _close(self, state):
        s = self._load()
        s[-1].update(end=runinfo.now(), seconds=time.time() - s[-1]["t0"], end_step=state.global_step)
        self.path.write_text(json.dumps(s, indent=1))

    def on_save(self, args, state, control, **kw):
        self._close(state)  # keeps the record current if the runtime dies before on_train_end

    def on_train_end(self, args, state, control, **kw):
        self._close(state)


def load_for_training(cfg: dict):
    model, proc = FastModel.from_pretrained(cfg["base_model"], max_seq_length=cfg["max_seq_len"],
                                            load_in_4bit=cfg["load_in_4bit"], full_finetuning=False)
    tok = getattr(proc, "tokenizer", proc)
    lora = cfg["lora"]
    model = FastModel.get_peft_model(
        model, r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"], bias="none",
        finetune_vision_layers=False, finetune_language_layers=True,
        finetune_attention_modules=True, finetune_mlp_modules=True,
        random_state=cfg["seed"], use_gradient_checkpointing=cfg["train"]["gradient_checkpointing"])
    targets = sorted({n.split(".lora_A")[0].rsplit(".", 1)[-1] for n, _ in model.named_parameters() if ".lora_A" in n})
    if targets != sorted(lora["target_modules"]):
        raise ValueError(f"LoRA attached to {targets}, config expects {sorted(lora['target_modules'])}")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def train_one(cfg: dict, rows: list[dict], ckpt_dir: Path, adapter_dir: Path, max_steps: int = -1, log=print) -> dict:
    """Trains (or resumes) one run and saves the adapter. Returns training stats."""
    t = cfg["train"]
    model, tok = load_for_training(cfg)
    data = [encode(tok, r["text"], r["label"], cfg["max_seq_len"]) for r in rows]
    log(f"{len(data)} examples, tokens/example mean {sum(len(d['input_ids']) for d in data) / len(data):.1f}, "
        f"max {max(len(d['input_ids']) for d in data)}; eos={tok.eos_token!r}")
    steps_per_epoch = math.ceil(len(data) / (t["per_device_batch_size"] * t["grad_accum_steps"]))
    total_steps = max_steps if max_steps > 0 else steps_per_epoch * t["epochs"]
    args = TrainingArguments(
        output_dir=str(ckpt_dir), num_train_epochs=t["epochs"], max_steps=max_steps,
        per_device_train_batch_size=t["per_device_batch_size"], gradient_accumulation_steps=t["grad_accum_steps"],
        learning_rate=t["learning_rate"], lr_scheduler_type=t["lr_scheduler"],
        warmup_steps=math.ceil(t["warmup_ratio"] * total_steps), weight_decay=t["weight_decay"], optim=t["optim"],
        logging_steps=t["logging_steps"], save_steps=t["save_steps"], save_total_limit=t["save_total_limit"],
        seed=cfg["seed"], data_seed=cfg["seed"], fp16=False, bf16=False, report_to="none",
        disable_tqdm=True, remove_unused_columns=False, dataloader_num_workers=0)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = Trainer(model=model, args=args, train_dataset=data, data_collator=Collator(tok.pad_token_id),
                      callbacks=[Sessions(ckpt_dir / "sessions.json")])
    last = get_last_checkpoint(str(ckpt_dir))
    log(f"resuming from {last}" if last else "starting from scratch")
    torch.cuda.reset_peak_memory_stats()
    out = trainer.train(resume_from_checkpoint=last)
    model.save_pretrained(str(adapter_dir))
    tok.save_pretrained(str(adapter_dir))
    sessions = json.loads((ckpt_dir / "sessions.json").read_text())
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    n_tokens = sum(len(d["input_ids"]) for d in data) * (t["epochs"] if max_steps <= 0 else 1)
    stats = {
        "n_train": len(data), "steps": trainer.state.global_step, "epochs": t["epochs"],
        "train_seconds_total": sum(s.get("seconds", 0) for s in sessions), "sessions": len(sessions),
        "resumed_from": last, "trainer_runtime_s_last_session": out.metrics.get("train_runtime"),
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
        "loss_first": losses[0] if losses else None, "loss_last": losses[-1] if losses else None,
        "loss_history": [(h["step"], round(h["loss"], 4)) for h in trainer.state.log_history if "loss" in h],
        "trainable_params_m": sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6,
    }
    stats["train_tokens_per_s"] = n_tokens / stats["train_seconds_total"] if stats["train_seconds_total"] else None
    return {"stats": stats, "model": model, "tok": tok}


def evaluate(model, tok, rows: list[dict], labels: list[dict], batch_size: int, max_new_tokens: int) -> dict:
    """Greedy generation with the in-memory adapter on rows (validation here; test in Phase 6)."""
    FastModel.for_inference(model)
    tok.padding_side = "left"
    parser = LabelParser(labels)
    texts = [hf_infer.chat_prompt(tok, prompts.finetune_prompt(r["text"])) for r in rows]
    outs, secs = hf_infer.generate(model, tok, texts, max_new_tokens, batch_size)
    parsed = [parser.parse(o) for o in outs]
    gold, pred = [r["label"] for r in rows], [p for p, _ in parsed]
    return {
        "metrics": metrics.classification_metrics(gold, pred, [lab["name"] for lab in labels]),
        "confused_pairs": metrics.confused_pairs(gold, pred),
        "parse_methods": {m: sum(x == m for _, x in parsed) for m in {x for _, x in parsed}},
        "seconds": sum(secs), "examples_per_s": len(rows) / sum(secs),
        "invalid_examples": [o[:60] for o, (p, _) in zip(outs, parsed) if p is None][:10],
    }
