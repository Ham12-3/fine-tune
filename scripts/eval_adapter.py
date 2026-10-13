"""Phase 6: score fine-tuned adapters on a test split (Colab GPU).

Prompt = the short fine-tune prompt (no label list, no system prompt), greedy, stops at end of turn, parsed with the
same LabelParser as every other model. Before timing, one warm-up batch is run at each batch size (the first call
compiles kernels). Then:
  latency     the first --latency-n examples one at a time (batch 1), per-example wall time
  throughput  every example in batches of --batch-size; predictions come from this pass
Writes results/raw/ft_<adapter>_<split>.jsonl and results/summary/ft_<adapter>_<split>.json under the code root.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

from svf import runinfo
from svf.data import prompts
from svf.data.load_banking77 import read_jsonl
from svf.eval import cost, metrics
from svf.eval.parse import LabelParser
from svf.models import hf_infer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def evaluate(adapter_dir: Path, split: str, batch_size: int, latency_n: int, max_new_tokens: int) -> dict:
    labels = json.loads((PROC / "labels.json").read_text(encoding="utf-8"))
    parser = LabelParser(labels)
    rows = read_jsonl(PROC / f"{split}.jsonl")
    model, tok = hf_infer.load(str(adapter_dir), max_seq_length=256)
    texts = [hf_infer.chat_prompt(tok, prompts.finetune_prompt(r["text"])) for r in rows]
    n_lat = min(latency_n, len(rows))

    hf_infer.generate(model, tok, texts[:3], max_new_tokens, 1)                    # warm-up, batch 1
    hf_infer.generate(model, tok, texts[:batch_size], max_new_tokens, batch_size)  # warm-up, batched
    lat_out, lat_s = hf_infer.generate(model, tok, texts[:n_lat], max_new_tokens, 1)
    outs, batch_s = hf_infer.generate(model, tok, texts, max_new_tokens, batch_size, log=print)

    recs = []
    for i, (r, raw) in enumerate(zip(rows, outs)):
        pred, method = parser.parse(raw)
        recs.append({"id": r["id"], "gold": r["label"], "raw": raw, "pred": pred, "parse_method": method,
                     "latency_s": lat_s[i] if i < n_lat else None})
    name = f"ft_{adapter_dir.name}_{split}"
    for d in ("raw", "summary"):
        (ROOT / "results" / d).mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "raw" / f"{name}.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in recs), encoding="utf-8")

    pricing = cost.load_pricing()
    total_s = sum(batch_s)
    gold, pred = [x["gold"] for x in recs], [x["pred"] for x in recs]
    summary = {
        "run": name, "adapter": adapter_dir.name, "split": split, "n": len(recs),
        "decoding": {"greedy": True, "max_new_tokens": max_new_tokens, "thinking": False, "load_in_4bit": True,
                     "stop_ids": hf_infer.stop_ids(model, tok)},
        "metrics": metrics.classification_metrics(gold, pred, [lab["name"] for lab in labels]),
        "confused_pairs": metrics.confused_pairs(gold, pred),
        "parse_methods": {m: sum(x["parse_method"] == m for x in recs) for m in {x["parse_method"] for x in recs}},
        "prompt_tokens_mean": sum(len(tok(text=t, add_special_tokens=False)["input_ids"]) for t in texts) / len(texts),
        "latency_batch1": metrics.latency_stats(lat_s),
        "batch1_vs_batched_agreement": sum(parser.parse(a)[0] == parser.parse(b)[0]
                                           for a, b in zip(lat_out, outs[:n_lat])) / n_lat,
        "throughput": {"batch_size": batch_size, "seconds": total_s, "examples_per_s": len(recs) / total_s},
        "gpu_cost_per_1k_usd": cost.gpu_cost(total_s, pricing) / len(recs) * 1000,
        "gpu_cost_per_1k_usd_batch1": cost.gpu_cost(sum(lat_s), pricing) / n_lat * 1000,
        "gpu_hourly_usd": pricing["gpu"]["hourly_usd"],
        "data_sha256": runinfo.data_hashes(ROOT, [split]),
        "hardware": runinfo.hardware(), "finished": runinfo.now(),
    }
    (ROOT / "results" / "summary" / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    m = summary["metrics"]
    print(f"{name}: acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  invalid {m['invalid_rate']:.3f}  "
          f"batch-1 p50 {summary['latency_batch1']['p50_s']:.3f}s  throughput "
          f"{summary['throughput']['examples_per_s']:.1f}/s  GPU $/1k {summary['gpu_cost_per_1k_usd']:.4f}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "train" / "base.yaml").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapters", nargs="+", default=["f0.10", "f0.25", "f0.50", "f1.00"])
    ap.add_argument("--splits", nargs="+", default=["test_sample"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=cfg["eval"]["max_new_tokens"])
    args = ap.parse_args()
    for name in args.adapters:
        for split in args.splits:
            out = ROOT / "results" / "summary" / f"ft_{name}_{split}.json"
            if out.exists():
                print(f"{out.name} exists, skipping")
                continue
            evaluate(Path(cfg["drive_root"]) / "adapters" / name, split, args.batch_size, args.latency_n,
                     args.max_new_tokens)
    return 0


if __name__ == "__main__":
    sys.exit(main())
