"""Phase 9: charts and the results table, rebuilt from results/summary/ alone (run scripts/evaluate.py first).

  results/figures/accuracy_vs_training_size.png   fine-tuned test accuracy by training-set size vs Claude
  results/figures/cost_per_1k.png                 cost per 1,000 predictions (log scale)
  results/figures/latency.png                     median latency with p95, per system
  results/figures/break_even.png                  monthly cost vs monthly volume (log-log)
  results/summary/results_table.md                the headline table for the README
Colours follow one entity each, in the reference palette's fixed order; zero-/few-shot differ by line style.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SUM, FIG = ROOT / "results" / "summary", ROOT / "results" / "figures"

SURFACE, INK, INK2, GRID, NEUTRAL = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#8a8984"
COLOR = {"fine-tuned": "#2a78d6", "claude-haiku-5-5": "#eb6834", "claude-sonnet-5-5": "#1baf7a",
         "claude-opus-5-5": "#eda100"}
NAME = {"claude-haiku-5-5": "Haiku 5.5", "claude-sonnet-5-5": "Sonnet 5.5", "claude-opus-5-5": "Opus 5.5"}
MODE = {"zero_shot": "zero-shot", "few_shot": "few-shot"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 12,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "text.parse_math": False, "axes.axisbelow": True,  # "$" is currency here, not mathtext
})


def usd(v: float, _=None) -> str:
    return f"${v:,.0f}" if v >= 1 else f"${v:g}"


def count(v: float, _=None) -> str:
    for div, suffix in ((1e6, "M"), (1e3, "k")):
        if v >= div:
            return f"{v / div:g}{suffix}"
    return f"{v:g}"


def right_labels(ax, items: list[tuple[float, str]], x: float, gap: float, log: bool = True) -> None:
    """Direct labels at the right edge, nudged apart so none overlap (gap in log10 units if log)."""
    pos = sorted((np.log10(y) if log else y, t) for y, t in items)
    placed = []
    for y, t in pos:
        if placed and y - placed[-1][0] < gap:
            y = placed[-1][0] + gap
        placed.append((y, t))
    for y, t in placed:
        ax.text(x, 10 ** y if log else y, t, va="center", ha="left", fontsize=8.5, color=INK2)


def load(name: str) -> dict | None:
    p = SUM / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def claude_rows(comp: dict) -> list[dict]:
    return [r for r in comp["rows"] if r["system"].startswith("claude")]


def fig_accuracy(comp: dict) -> None:
    ft = [load(f"ft_{a}_test_sample") for a in ("f0.10", "f0.25", "f0.50", "f1.00")]
    sweep = {r["run"]: r for r in load("phase5_sweep")["runs"]}
    xs = [sweep[s["adapter"]]["n_train"] for s in ft]
    ys = [s["metrics"]["accuracy"] * 100 for s in ft]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x_end = 13000
    for r in claude_rows(comp):
        if r["mode"] != "few_shot":
            continue
        y = r["accuracy"] * 100
        ax.axhline(y, color=COLOR[r["system"]], lw=1.5, ls="--", zorder=1)
        ax.text(x_end, y, f"{NAME[r['system']]} few-shot {y:.1f}%", va="center", ha="left", fontsize=9, color=INK2)
    zs = load("small_zeroshot_Qwen3.5-2B_test_sample")["metrics"]["accuracy"] * 100
    ax.axhline(zs, color=NEUTRAL, lw=1.5, ls=":", zorder=1)
    ax.text(x_end, zs, f"Qwen3.5-2B untrained {zs:.1f}%", va="center", ha="left", fontsize=9, color=INK2)
    ax.plot(xs, ys, color=COLOR["fine-tuned"], marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
    for i, (x, y) in enumerate(zip(xs, ys)):
        # the 25% point sits between two reference lines, so its value goes to the right instead of above
        off, ha = ((12, -4), "left") if i == 1 else ((0, 9), "center")
        ax.annotate(f"{y:.1f}%", (x, y), xytext=off, textcoords="offset points", ha=ha,
                    fontsize=9, color=INK, fontweight="bold")
    ax.annotate("Fine-tuned Qwen3.5-2B (QLoRA)", (xs[0], ys[0]), xytext=(14, -16), textcoords="offset points",
                ha="left", fontsize=9.5, color=INK)
    ax.set_xscale("log")
    ax.set_xticks(xs, [f"{x:,}" for x in xs])
    ax.minorticks_off()
    ax.set_xlim(700, x_end)
    ax.set_ylim(60, 96)
    ax.set_xlabel("Training examples (log scale)")
    ax.set_ylabel("Accuracy on the 1,000-example test sample (%)")
    ax.set_title("Fine-tuned 2B model vs Claude few-shot, by training-set size")
    fig.savefig(FIG / "accuracy_vs_training_size.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def system_label(r: dict) -> tuple[str, str]:
    if r["system"].startswith("claude"):
        return f"{NAME[r['system']]} {MODE[r['mode']]}", r["system"]
    if "laptop" in r["system"]:
        return "Fine-tuned, laptop CPU", "fine-tuned"
    if r["mode"] == "zero_shot":
        return "Qwen3.5-2B untrained, T4", "neutral"
    return "Fine-tuned, T4 GPU", "fine-tuned"


def fig_cost(comp: dict) -> None:
    rows = [r for r in comp["rows"] if r["split"] == "test_sample" and r["mode"] in
            ("zero_shot", "few_shot", "fine-tuned f1.00") and "laptop" not in r["system"]]
    items = []
    for r in rows:
        label, ent = system_label(r)
        items.append((label + (" (batched)" if ent == "fine-tuned" else ""), r["cost_per_1k_usd"], ent))
    ft = load("ft_f1.00_test_sample")
    items.append(("Fine-tuned, T4 GPU (one at a time)", ft["gpu_cost_per_1k_usd_batch1"], "fine-tuned"))
    items.sort(key=lambda t: t[1])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ys = np.arange(len(items))
    colors = [NEUTRAL if e == "neutral" else COLOR[e] for _, _, e in items]
    ax.barh(ys, [v for _, v, _ in items], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for y, (_, v, _) in zip(ys, items):
        ax.text(v * 1.12, y, f"${v:,.4f}" if v < 0.01 else f"${v:,.2f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [n for n, _, _ in items])
    ax.set_xscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(usd))
    ax.set_xlim(min(v for _, v, _ in items) / 2, max(v for _, v, _ in items) * 6)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("USD per 1,000 predictions (log scale)")
    ax.set_title("Cost per 1,000 predictions")
    ax.text(0, -0.16, "GPU at $0.35/h (rented T4). Laptop CPU: no marginal compute cost. Claude at standard API "
            "prices with prompt caching; Batch API halves them.", transform=ax.transAxes, fontsize=8, color=INK2)
    fig.savefig(FIG / "cost_per_1k.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_latency(comp: dict) -> None:
    rows = [r for r in comp["rows"] if r["split"] == "test_sample" and r["mode"] in
            ("zero_shot", "few_shot", "fine-tuned f1.00")]
    items = sorted(((*system_label(r), r["latency_p50_s"], r["latency_p95_s"], r["latency_note"]) for r in rows),
                   key=lambda t: t[2])
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ys = np.arange(len(items))
    for y, (label, ent, p50, p95, _) in zip(ys, items):
        c = NEUTRAL if ent == "neutral" else COLOR[ent]
        ax.plot([p50, p95], [y, y], color=c, lw=2, alpha=0.5)
        ax.plot(p50, y, "o", color=c, ms=9, mec=SURFACE, mew=2, zorder=3)
        ax.plot(p95, y, "|", color=c, ms=12, mew=2)
        ax.text(p95 + 0.05, y, f"{p50:.2f} s", va="center", fontsize=9, color=INK)
    ax.set_yticks(ys, [t[0] for t in items])
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(t[3] for t in items) * 1.2)
    ax.set_xlabel("Seconds per prediction: dot = median, bar end = p95")
    ax.set_title("Latency, one request at a time")
    ax.text(0, -0.16, "Claude: API round trip from the laptop. T4: GPU time. Laptop CPU: llama.cpp, Q4_K_M, "
            "while other apps were open.", transform=ax.transAxes, fontsize=8, color=INK2)
    fig.savefig(FIG / "latency.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_break_even(comp: dict) -> None:
    a = comp["break_even_assumptions"]
    ft = next(r for r in comp["rows"] if r["mode"] == "fine-tuned f1.00" and r["split"] == "test_sample")
    v = np.logspace(3, 8, 200)
    x_end = 1e8
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = []
    shown = {("claude-haiku-5-5", "zero_shot"), ("claude-haiku-5-5", "few_shot"),
             ("claude-sonnet-5-5", "few_shot"), ("claude-opus-5-5", "few_shot")}
    for r in claude_rows(comp):
        if (r["system"], r["mode"]) not in shown:
            continue
        c = r["cost_per_1k_usd"] / 1000
        ax.plot(v, v * c, color=COLOR[r["system"]], ls="-" if r["mode"] == "few_shot" else "--", lw=1.8)
        labels.append((x_end * c, f"{NAME[r['system']]} {MODE[r['mode']]}"))
    gpus = np.ceil(v / a["ft_capacity_per_month"])
    ax.plot(v, gpus * a["always_on_monthly_usd"], color=COLOR["fine-tuned"], lw=2.4)
    labels.append((2 * a["always_on_monthly_usd"], "Fine-tuned, always-on T4"))
    ax.plot(v, ft["train_usd"] + v * ft["cost_per_1k_usd"] / 1000, color=COLOR["fine-tuned"], lw=2.4, ls=":")
    labels.append((ft["train_usd"] + x_end * ft["cost_per_1k_usd"] / 1000, "Fine-tuned, pay-per-use GPU"))
    right_labels(ax, labels, x_end * 1.12, gap=0.24)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(count))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(usd))
    ax.set_xlim(1e3, x_end)
    ax.set_ylim(0.05, 5e5)
    ax.set_xlabel("Predictions per month (log scale)")
    ax.set_ylabel("USD per month (log scale)")
    ax.set_title("Where the cost trade-off flips")
    ax.text(0, -0.17, f"Always-on: rented T4s at ${a['gpu_hourly_usd']}/h x {a['hours_per_month']} h, "
            f"~{a['ft_capacity_per_month'] / 1e6:.0f}M predictions/month each. Pay-per-use: batched GPU time + "
            f"${ft['train_usd']:.2f} training.\nSonnet and Opus zero-shot sit between the lines shown; "
            "Claude at standard prices (Batch API halves them). Laptop CPU: no marginal cost.",
            transform=ax.transAxes, fontsize=8, color=INK2)
    fig.savefig(FIG / "break_even.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def table(comp: dict) -> str:
    lines = ["| System | Accuracy | Macro-F1 | Median latency | Cost per 1k |", "|---|---|---|---|---|"]
    rows = [r for r in comp["rows"] if r["split"] == "test_sample" and r["mode"] in
            ("zero_shot", "few_shot", "fine-tuned f1.00")]
    for r in sorted(rows, key=lambda r: -r["accuracy"]):
        label, _ = system_label(r)
        c = r["cost_per_1k_usd"]
        cost_s = "$0 (owned)" if "laptop" in r["system"] else (f"${c:.4f}" if c < 0.01 else f"${c:.2f}")
        if r["system"].startswith("claude"):
            cost_s += f" (batch ${r['cost_per_1k_usd_batch_api']:.3f})"
        lines.append(f"| {label} | {r['accuracy'] * 100:.1f}% | {r['macro_f1']:.3f} | {r['latency_p50_s']:.2f} s "
                     f"| {cost_s} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    comp = load("comparison")
    if comp is None:
        print("run scripts/evaluate.py first")
        return 1
    FIG.mkdir(parents=True, exist_ok=True)
    fig_accuracy(comp)
    fig_cost(comp)
    fig_latency(comp)
    fig_break_even(comp)
    md = table(comp)
    (SUM / "results_table.md").write_text(md, encoding="utf-8", newline="\n")
    print(md)
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
