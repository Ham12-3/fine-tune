"""Phase 7: benchmark the quantised fine-tuned GGUF on CPU with llama.cpp.

Runs every example one at a time (batch 1, the realistic CPU setting), greedy, stopping at end of turn, and parses
with the same LabelParser as every other model. Also used in Colab on a small validation slice to check the file
before it is downloaded.
Writes results/raw/gguf_<tag>_<split>.jsonl and results/summary/gguf_<tag>_<split>.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from svf import runinfo
from svf.data.load_banking77 import read_jsonl
from svf.eval import metrics
from svf.eval.parse import LabelParser
from svf.models.gguf_infer import GGUFClassifier

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, default=ROOT / "models" / "qwen3.5-2b-banking77-f1.00-Q4_K_M.gguf")
    ap.add_argument("--split", default="test_sample")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--tag", default="laptop")
    ap.add_argument("--max-tokens", type=int, default=24)
    args = ap.parse_args()

    labels = json.loads((PROC / "labels.json").read_text(encoding="utf-8"))
    parser = LabelParser(labels)
    rows = read_jsonl(PROC / f"{args.split}.jsonl")[: args.limit]

    t0 = time.perf_counter()
    clf = GGUFClassifier(args.gguf, n_threads=args.threads)
    load_s = time.perf_counter() - t0
    for r in rows[:3]:  # warm-up
        clf.generate(r["text"], args.max_tokens)

    recs, t_all = [], time.perf_counter()
    for i, r in enumerate(rows):
        raw, sec, n_out = clf.generate(r["text"], args.max_tokens)
        pred, method = parser.parse(raw)
        recs.append({"id": r["id"], "gold": r["label"], "raw": raw, "pred": pred, "parse_method": method,
                     "latency_s": sec, "completion_tokens": n_out})
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)}  running acc {sum(x['pred'] == x['gold'] for x in recs) / len(recs):.3f}",
                  flush=True)
    wall = time.perf_counter() - t_all

    name = f"gguf_{args.tag}_{args.split}" + (f"_n{args.limit}" if args.limit else "")
    for d in ("raw", "summary"):
        (ROOT / "results" / d).mkdir(parents=True, exist_ok=True)
    with (ROOT / "results" / "raw" / f"{name}.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        f.writelines(json.dumps(x, ensure_ascii=False, sort_keys=True) + "\n" for x in recs)

    h = hashlib.sha256()
    with args.gguf.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    gold, pred = [x["gold"] for x in recs], [x["pred"] for x in recs]
    summary = {
        "run": name, "gguf": args.gguf.name, "gguf_sha256": h.hexdigest(), "gguf_bytes": args.gguf.stat().st_size,
        "split": args.split, "n": len(recs), "threads": clf.n_threads, "load_s": load_s,
        "decoding": {"greedy": True, "max_tokens": args.max_tokens, "stop": ["<|im_end|>", "<|endoftext|>"], "batch": 1},
        "metrics": metrics.classification_metrics(gold, pred, [lab["name"] for lab in labels]),
        "confused_pairs": metrics.confused_pairs(gold, pred),
        "parse_methods": {m: sum(x["parse_method"] == m for x in recs) for m in {x["parse_method"] for x in recs}},
        "prompt_tokens_mean": sum(clf.prompt_tokens(r["text"]) for r in rows[:50]) / min(50, len(rows)),
        "completion_tokens_mean": sum(x["completion_tokens"] for x in recs) / len(recs),
        "hit_max_tokens": sum(x["completion_tokens"] >= args.max_tokens for x in recs),
        "latency_batch1": metrics.latency_stats([x["latency_s"] for x in recs]),
        "throughput": {"seconds": wall, "examples_per_s": len(recs) / wall},
        "llama_cpp_python": __import__("llama_cpp").__version__,
        "data_sha256": runinfo.data_hashes(ROOT, [args.split]),
        "hardware": runinfo.hardware(), "finished": runinfo.now(),
    }
    with (ROOT / "results" / "summary" / f"{name}.json").open("w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(summary, indent=1) + "\n")
    m, lat = summary["metrics"], summary["latency_batch1"]
    print(f"{name}: acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  invalid {m['invalid_rate']:.3f}  "
          f"p50 {lat['p50_s']:.3f}s  p95 {lat['p95_s']:.3f}s  {summary['throughput']['examples_per_s']:.2f}/s  "
          f"out tokens {summary['completion_tokens_mean']:.1f} (hit max {summary['hit_max_tokens']})  "
          f"threads {clf.n_threads}  cpu {summary['hardware']['cpu']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
