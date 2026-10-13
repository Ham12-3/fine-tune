"""Phase 4: zero-shot baseline for the small base model on the GPU (Colab).

Same system prompt (instructions + 77 labels with descriptions) and user turn as Claude zero-shot, greedy decoding,
thinking off, parsed with the same LabelParser. Two passes:
  latency     the first --latency-n examples one at a time (batch 1), per-example wall time
  throughput  every example in batches of --batch-size; predictions come from this pass
The batch-1 and batched predictions are compared on the overlap to check that left padding changes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from svf import runinfo
from svf.data import prompts
from svf.data.load_banking77 import read_jsonl
from svf.eval import cost, metrics
from svf.eval.parse import LabelParser
from svf.models import hf_infer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> int:
    base = yaml.safe_load((ROOT / "configs" / "train" / "base.yaml").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=base["base_model"])
    ap.add_argument("--split", default="test_sample")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--latency-n", type=int, default=100)
    ap.add_argument("--max-new-tokens", type=int, default=32)  # longest label is ~15 tokens
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    labels = json.loads((PROC / "labels.json").read_text(encoding="utf-8"))
    parser = LabelParser(labels)
    rows = read_jsonl(PROC / f"{args.split}.jsonl")[: args.limit]
    system = prompts.system_prompt(labels)

    model, tok = hf_infer.load(args.model)
    texts = [hf_infer.chat_prompt(tok, prompts.user_message(r["text"]), system) for r in rows]
    n_lat = min(args.latency_n, len(rows))

    print(f"latency pass: {n_lat} examples at batch 1")
    hf_infer.generate(model, tok, texts[:2], args.max_new_tokens, 1)  # warm-up
    lat_out, lat_s = hf_infer.generate(model, tok, texts[:n_lat], args.max_new_tokens, 1)
    print(f"throughput pass: {len(rows)} examples at batch {args.batch_size}")
    outs, batch_s = hf_infer.generate(model, tok, texts, args.max_new_tokens, args.batch_size, log=print)

    recs = []
    for i, (r, raw) in enumerate(zip(rows, outs)):
        pred, method = parser.parse(raw)
        recs.append({"id": r["id"], "gold": r["label"], "raw": raw, "pred": pred, "parse_method": method,
                     "latency_s": lat_s[i] if i < n_lat else None})
    agree = sum(parser.parse(a)[0] == parser.parse(b)[0] for a, b in zip(lat_out, outs[:n_lat])) / n_lat

    name = f"small_zeroshot_{args.model.split('/')[-1]}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    for d in ("raw", "summary"):
        (ROOT / "results" / d).mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "raw" / f"{name}.jsonl").write_text(
        "".join(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in recs), encoding="utf-8")

    pricing = cost.load_pricing()
    total_s = sum(batch_s)
    gold, pred = [x["gold"] for x in recs], [x["pred"] for x in recs]
    summary = {
        "run": name, "model": args.model, "mode": "zero_shot", "split": args.split, "n": len(recs),
        "decoding": {"greedy": True, "max_new_tokens": args.max_new_tokens, "thinking": False, "load_in_4bit": True},
        "metrics": metrics.classification_metrics(gold, pred, [lab["name"] for lab in labels]),
        "confused_pairs": metrics.confused_pairs(gold, pred),
        "parse_methods": {m: sum(x["parse_method"] == m for x in recs) for m in {x["parse_method"] for x in recs}},
        "prompt_tokens_mean": sum(len(tok(text=t, add_special_tokens=False)["input_ids"]) for t in texts[:20]) / min(20, len(texts)),
        "latency_batch1": metrics.latency_stats(lat_s),
        "throughput": {"batch_size": args.batch_size, "seconds": total_s, "examples_per_s": len(recs) / total_s},
        "batch1_vs_batched_agreement": agree,
        "gpu_cost_per_1k_usd": cost.gpu_cost(total_s, pricing) / len(recs) * 1000,
        "gpu_cost_per_1k_usd_batch1": cost.gpu_cost(sum(lat_s), pricing) / n_lat * 1000,
        "gpu_hourly_usd": pricing["gpu"]["hourly_usd"],
        "data_sha256": runinfo.data_hashes(ROOT, [args.split]),
        "hardware": runinfo.hardware(), "finished": runinfo.now(),
    }
    (ROOT / "results" / "summary" / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    m = summary["metrics"]
    print(f"{name}: acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  invalid {m['invalid_rate']:.3f}  "
          f"parse {summary['parse_methods']}\n  batch-1 p50 {summary['latency_batch1']['p50_s']:.3f}s  "
          f"throughput {summary['throughput']['examples_per_s']:.1f}/s  agreement {agree:.3f}  "
          f"GPU $/1k {summary['gpu_cost_per_1k_usd']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
