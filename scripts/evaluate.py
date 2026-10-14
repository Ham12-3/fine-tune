"""Phase 6: one comparison table and the break-even analysis, built only from results/summary/*.json.

Rows: every Claude model x mode, the small model zero-shot, and each fine-tuned adapter, all on the same 1k test
sample (the fine-tuned best adapter also on the full test set). Writes results/summary/comparison.{json,csv}.

Break-even, two ways of running the fine-tuned model:
  pay-per-use  GPU billed only while predicting (batched throughput) + the one-off training run;
               result = total predictions after which fine-tuning is cheaper
  always-on    one rented GPU running all month (pricing.yaml gpu.hourly_usd x hours_per_month);
               result = monthly volume above which it beats Claude, if one GPU can serve that volume
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from svf.data.load_banking77 import read_jsonl
from svf.eval import cost, metrics

ROOT = Path(__file__).resolve().parents[1]
SUM = ROOT / "results" / "summary"


def load(name: str) -> dict | None:
    p = SUM / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main() -> int:
    pricing = cost.load_pricing()
    gpu = pricing["gpu"]
    sweep = load("phase5_sweep")
    train_s = {r["run"]: r["train_seconds"] for r in sweep["runs"]} if sweep else {}
    rows = []

    for model in pricing["claude"]:
        for mode in ("zero_shot", "few_shot"):
            s = load(f"claude_{model}_{mode}_test_sample")
            if not s:
                continue
            m, lat = s["metrics"], s["latency"]
            rows.append({"system": model, "mode": mode, "split": "test_sample", "n": m["n"],
                         "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "invalid_rate": m["invalid_rate"],
                         "latency_p50_s": lat["p50_s"], "latency_p95_s": lat["p95_s"], "latency_note": "API round trip",
                         "cost_per_1k_usd": s["cost_per_1k_usd_standard"], "cost_per_1k_usd_batch_api": s["cost_per_1k_usd_batch"],
                         "train_usd": 0.0})

    small = [("small_zeroshot_Qwen3.5-2B_test_sample", "Qwen3.5-2B (4-bit, T4)", "zero_shot", None)]
    small += [(f"ft_{a}_{split}", "Qwen3.5-2B + QLoRA (4-bit, T4)", f"fine-tuned {a}", a)
              for a in ("f0.10", "f0.25", "f0.50", "f1.00") for split in ("test_sample", "test_full")]
    for name, system, mode, adapter in small:
        s = load(name)
        if not s:
            continue
        m, lat = s["metrics"], s["latency_batch1"]
        rows.append({"system": system, "mode": mode, "split": s["split"], "n": m["n"],
                     "accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "invalid_rate": m["invalid_rate"],
                     "latency_p50_s": lat["p50_s"], "latency_p95_s": lat["p95_s"], "latency_note": "GPU, batch 1",
                     "cost_per_1k_usd": s["gpu_cost_per_1k_usd"], "cost_per_1k_usd_batch_api": None,
                     "throughput_per_s": s["throughput"]["examples_per_s"],
                     "train_usd": cost.gpu_cost(train_s[adapter], pricing) if adapter in train_s else 0.0})

    s = load("gguf_laptop_test_sample")
    if s:
        m, lat = s["metrics"], s["latency_batch1"]
        rows.append({"system": f"Qwen3.5-2B + QLoRA Q4_K_M ({s['threads']}-thread laptop CPU)", "mode": "fine-tuned f1.00",
                     "split": s["split"], "n": m["n"], "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "invalid_rate": m["invalid_rate"], "latency_p50_s": lat["p50_s"], "latency_p95_s": lat["p95_s"],
                     "latency_note": f"CPU batch 1, {s['hardware']['cpu']}", "cost_per_1k_usd": 0.0,
                     "cost_note": "owned laptop: no marginal compute cost (electricity not counted)",
                     "cost_per_1k_usd_batch_api": None, "throughput_per_s": s["throughput"]["examples_per_s"],
                     "train_usd": cost.gpu_cost(train_s["f1.00"], pricing) if "f1.00" in train_s else 0.0})

    best = next((r for r in rows if r["mode"] == "fine-tuned f1.00" and r["split"] == "test_sample"), None)
    break_even = []
    if best:
        g = best["cost_per_1k_usd"] / 1000
        hours = gpu["hours_per_month"]
        monthly = gpu["hourly_usd"] * hours
        capacity = best["throughput_per_s"] * 3600 * hours
        for r in rows:
            if not r["system"].startswith("claude"):
                continue
            for api, c1k in (("standard", r["cost_per_1k_usd"]), ("batch_api", r["cost_per_1k_usd_batch_api"])):
                c = c1k / 1000
                break_even.append({
                    "vs": f"{r['system']} {r['mode']}", "claude_pricing": api, "claude_per_1k_usd": c1k,
                    "ft_gpu_per_1k_usd": best["cost_per_1k_usd"], "ft_train_usd": best["train_usd"],
                    "pay_per_use_predictions_to_break_even": cost.break_even_pay_per_use(best["train_usd"], g, c),
                    "always_on_monthly_volume_to_break_even": cost.break_even_always_on(monthly, c, capacity),
                    "accuracy_delta_ft_minus_claude": best["accuracy"] - r["accuracy"]})
        assumptions = {"gpu": gpu["name"], "gpu_hourly_usd": gpu["hourly_usd"], "hours_per_month": hours,
                       "always_on_monthly_usd": monthly, "ft_capacity_per_month": capacity,
                       "ft_throughput_per_s": best["throughput_per_s"],
                       "sweep_total_train_usd": cost.gpu_cost(sum(train_s.values()), pricing)}
    else:
        assumptions = {}

    # Paired significance on the shared 1k sample (needs the raw predictions in results/raw/)
    paired = []
    ft_raw = ROOT / "results" / "raw" / "ft_f1.00_test_sample.jsonl"
    if ft_raw.exists():
        ft = {r["id"]: r["pred"] == r["gold"] for r in read_jsonl(ft_raw)}
        for model in pricing["claude"]:
            for mode in ("zero_shot", "few_shot"):
                p = ROOT / "results" / "raw" / f"claude_{model}_{mode}_test_sample.jsonl"
                if not p.exists():
                    continue
                cl = {r["id"]: r["pred"] == r["gold"] for r in read_jsonl(p)}
                ids = sorted(ft.keys() & cl.keys())
                paired.append({"a": "fine-tuned f1.00", "b": f"{model} {mode}",
                               **metrics.mcnemar([ft[i] for i in ids], [cl[i] for i in ids])})

    out = {"pricing_date_checked": str(pricing["date_checked"]), "rows": rows, "break_even": break_even,
           "break_even_assumptions": assumptions, "paired_mcnemar": paired}
    (SUM / "comparison.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    with (SUM / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    print(f"{'system':<32}{'mode':<18}{'split':<12}{'acc':>6}{'F1':>7}{'inv':>6}{'p50 s':>7}{'$/1k':>9}{'batch $/1k':>11}")
    for r in rows:
        b = f"{r['cost_per_1k_usd_batch_api']:.4f}" if r["cost_per_1k_usd_batch_api"] is not None else "-"
        print(f"{r['system']:<32}{r['mode']:<18}{r['split']:<12}{r['accuracy']:>6.3f}{r['macro_f1']:>7.3f}"
              f"{r['invalid_rate']:>6.3f}{r['latency_p50_s']:>7.2f}{r['cost_per_1k_usd']:>9.4f}{b:>11}")
    if break_even:
        print(f"\nbreak-even vs fine-tuned f1.00 (assumptions: {json.dumps(assumptions)})")
        for b in break_even:
            ppu, ao = b["pay_per_use_predictions_to_break_even"], b["always_on_monthly_volume_to_break_even"]
            print(f"  {b['vs']:<32}{b['claude_pricing']:<10} pay-per-use: {f'{ppu:,.0f} preds' if ppu is not None else 'never':<16}"
                  f" always-on: {f'{ao:,.0f}/month' if ao is not None else 'beyond 1 GPU capacity':<22}"
                  f" acc delta {b['accuracy_delta_ft_minus_claude']:+.3f}")
    if paired:
        print("\npaired McNemar on the 1k sample (only_a = fine-tuned right, Claude wrong):")
        for p in paired:
            print(f"  vs {p['b']:<28} both {p['both']:>4}  only_ft {p['only_a']:>3}  only_claude {p['only_b']:>3}"
                  f"  p = {p['p_value']:.2g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
