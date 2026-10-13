"""Phase 4: Claude baselines on Banking77 (zero-shot and few-shot).

  --estimate          count tokens (free) and project the cost of the run; makes no Messages calls
  --limit N           evenly spaced N-example subset (the pilot), written under a separate run name
  --batch             Message Batches API (50% off, no per-request latency)
Results: results/raw/<run>.jsonl (one line per example, appended as calls finish, so an interrupted run resumes
without paying twice) and results/summary/<run>.json (metrics, token totals, cost, cache and latency stats).
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
import yaml
from dotenv import load_dotenv

from svf import runinfo
from svf.data import prompts
from svf.data.load_banking77 import read_jsonl
from svf.eval import cost, metrics
from svf.eval.parse import LabelParser
from svf.models import claude_client as cc

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def evenly_spaced(rows: list[dict], n: int) -> list[dict]:
    step = len(rows) / n
    return [rows[int(i * step)] for i in range(n)]


def run_name(model: str, mode: str, split: str, limit: int | None, batch: bool) -> str:
    return f"claude_{model}_{mode}_{split}" + (f"_n{limit}" if limit else "") + ("_batch" if batch else "")


def load_done(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    return {r["id"]: r for r in read_jsonl(path) if not r.get("error")}  # errored rows are retried


def finish(row: dict, res: dict, parser: LabelParser) -> dict:
    pred, method = parser.parse(res.get("raw")) if res.get("stop_reason") != "refusal" else (None, "refusal")
    return {"id": row["id"], "gold": row["label"], "pred": pred, "parse_method": method, **res}


def summarise(name: str, model: str, mode: str, rows: list[dict], labels: list[str], pricing: dict,
              batch: bool, meta: dict) -> dict:
    gold = [r["gold"] for r in rows]
    pred = [r["pred"] for r in rows]
    usage = cost.sum_usage(rows)
    n = len(rows)
    cfg_ttl = meta["config"]["cache_ttl"]
    return {
        "run": name, "model": model, "mode": mode, "batch": batch, **meta,
        "metrics": metrics.classification_metrics(gold, pred, labels),
        "confused_pairs": metrics.confused_pairs(gold, pred),
        "parse_methods": {m: sum(r["parse_method"] == m for r in rows) for m in {r["parse_method"] for r in rows}},
        "stop_reasons": {s: sum(r.get("stop_reason") == s for r in rows) for s in {r.get("stop_reason") for r in rows}},
        "errors": sum(bool(r.get("error")) for r in rows),
        "usage_total": usage,
        "usage_mean": {k: v / n for k, v in usage.items()},
        "calls_with_cache_read": sum((r.get("usage") or {}).get("cache_read_input_tokens", 0) > 0 for r in rows),
        "cost_usd": cost.claude_cost(usage, model, pricing, batch=batch, cache_ttl=cfg_ttl),
        "cost_usd_standard": cost.claude_cost(usage, model, pricing, batch=False, cache_ttl=cfg_ttl),
        "cost_usd_batch": cost.claude_cost(usage, model, pricing, batch=True, cache_ttl=cfg_ttl),
        "cost_per_1k_usd_standard": cost.claude_cost(usage, model, pricing, cache_ttl=cfg_ttl) / n * 1000,
        "cost_per_1k_usd_batch": cost.claude_cost(usage, model, pricing, batch=True, cache_ttl=cfg_ttl) / n * 1000,
        "latency": metrics.latency_stats([r["latency_s"] for r in rows if r.get("latency_s") is not None]),
    }


def run_standard(client, todo, params_for, parser, out: Path, workers: int, log=print) -> None:
    lock = threading.Lock()

    def write(rec: dict) -> None:
        with lock, out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")

    if not todo:
        return
    # First call alone, so it writes the cache before the concurrent calls read it.
    first = todo[0]
    write(finish(first, cc.classify(client, params_for(first)), parser))
    done = 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(cc.classify, client, params_for(r)): r for r in todo[1:]}
        for fut in as_completed(futs):
            write(finish(futs[fut], fut.result(), parser))
            done += 1
            if done % 50 == 0 or done == len(todo):
                log(f"  {done}/{len(todo)}")


def run_batch(client, todo, params_for, parser, out: Path, state: Path, log=print) -> None:
    if not todo:
        return
    st = json.loads(state.read_text()) if state.exists() else {}
    if not st.get("batch_id"):
        st["batch_id"] = cc.submit_batch(client, [(r["id"], params_for(r)) for r in todo])
        state.write_text(json.dumps(st))
        log(f"  submitted batch {st['batch_id']} ({len(todo)} requests)")
    cc.wait_batch(client, st["batch_id"], log=log)
    res = cc.batch_results(client, st["batch_id"])
    with out.open("a", encoding="utf-8") as f:
        for r in todo:
            rec = finish(r, res.get(r["id"], {"raw": None, "usage": {}, "error": "missing from batch results"}), parser)
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    state.unlink()


def estimate(client, models, modes, systems, rows, pricing, cfg) -> None:
    """Free: count_tokens for the cached system prompt and the mean user turn, then project the cost."""
    sample = evenly_spaced(rows, min(10, len(rows)))
    print(f"{'model':<20}{'mode':<11}{'system':>8}{'user':>7}  per-call output assumed 50 / 200 / 400 tokens -> USD")
    for model in models:
        for mode in modes:
            sys_tok = client.messages.count_tokens(
                model=model, system=systems[mode], messages=[{"role": "user", "content": "x"}]).input_tokens
            user = sum(client.messages.count_tokens(model=model, system=systems[mode], messages=[
                {"role": "user", "content": prompts.user_message(r["text"])}]).input_tokens - sys_tok + 1
                for r in sample) / len(sample)
            n = len(rows)
            costs = []
            for out_tok in (50, 200, 400):
                usage = {"cache_creation_input_tokens": sys_tok, "cache_read_input_tokens": sys_tok * (n - 1),
                         "input_tokens": user * n, "output_tokens": out_tok * n}
                costs.append(cost.claude_cost(usage, model, pricing, cache_ttl=cfg["cache_ttl"]))
            print(f"{model:<20}{mode:<11}{sys_tok:>8}{user:>7.0f}  " + " / ".join(f"${c:.3f}" for c in costs))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "claude.yaml")
    ap.add_argument("--models", nargs="+")
    ap.add_argument("--modes", nargs="+")
    ap.add_argument("--split", default="test_sample")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--estimate", action="store_true")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    models, modes = args.models or cfg["models"], args.modes or cfg["modes"]
    pricing = cost.load_pricing()
    labels = json.loads((PROC / "labels.json").read_text(encoding="utf-8"))
    label_names = [lab["name"] for lab in labels]
    parser = LabelParser(labels)
    schema = prompts.label_schema(labels)
    systems = {"zero_shot": prompts.system_prompt(labels),
               "few_shot": prompts.system_prompt(labels, read_jsonl(PROC / "few_shot.jsonl"))}
    rows = read_jsonl(PROC / f"{args.split}.jsonl")
    if args.limit:
        rows = evenly_spaced(rows, args.limit)
    client = anthropic.Anthropic(max_retries=cfg["max_retries"])

    if args.estimate:
        estimate(client, models, modes, systems, rows, pricing, cfg)
        return 0

    meta_base = {"split": args.split, "n": len(rows), "config": cfg, "git_sha": runinfo.git_sha(ROOT),
                 "data_sha256": runinfo.data_hashes(ROOT, [args.split, "few_shot"]),
                 "pricing_date_checked": str(pricing["date_checked"]), "hardware": runinfo.hardware()}
    (ROOT / "results" / "raw").mkdir(parents=True, exist_ok=True)
    (ROOT / "results" / "summary").mkdir(parents=True, exist_ok=True)
    for model in models:
        for mode in modes:
            name = run_name(model, mode, args.split, args.limit, args.batch)
            out = ROOT / "results" / "raw" / f"{name}.jsonl"
            done = load_done(out)
            if out.exists() and len(done) < sum(1 for _ in out.open(encoding="utf-8")):
                out.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in done.values()),
                               encoding="utf-8")  # drop errored rows before retrying them
            todo = [r for r in rows if r["id"] not in done]
            print(f"{name}: {len(done)} done, {len(todo)} to run")
            params_for = lambda r, m=model, md=mode: cc.request_params(  # noqa: E731
                m, systems[md], prompts.user_message(r["text"]), schema, cfg)
            started = runinfo.now()
            if args.batch:
                run_batch(client, todo, params_for, parser, out, out.with_suffix(".batch.json"))
            else:
                run_standard(client, todo, params_for, parser, out, cfg["concurrency"])
            by_id = {r["id"]: r for r in read_jsonl(out)}
            results = [by_id[r["id"]] for r in rows if r["id"] in by_id]
            summary = summarise(name, model, mode, results, label_names, pricing, args.batch,
                                {**meta_base, "started": started, "finished": runinfo.now()})
            (ROOT / "results" / "summary" / f"{name}.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
            m, u, lat = summary["metrics"], summary["usage_mean"], summary["latency"]
            print(f"  acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  invalid {m['invalid_rate']:.3f}  "
                  f"errors {summary['errors']}  stop {summary['stop_reasons']}\n"
                  f"  mean tokens: in {u['input_tokens']:.0f}  cache_w {u['cache_creation_input_tokens']:.0f}  "
                  f"cache_r {u['cache_read_input_tokens']:.0f}  out {u['output_tokens']:.0f}  "
                  f"| cache hits {summary['calls_with_cache_read']}/{len(results)}\n"
                  f"  cost ${summary['cost_usd']:.4f}  (per 1k: ${summary['cost_per_1k_usd_standard']:.3f} standard, "
                  f"${summary['cost_per_1k_usd_batch']:.3f} batch)  latency p50 {lat.get('p50_s', float('nan')):.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
