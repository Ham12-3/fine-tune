"""Token and GPU-time costs from configs/pricing.yaml."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[3]
USAGE_FIELDS = ("input_tokens", "cache_creation_input_tokens", "cache_read_input_tokens", "output_tokens")


def load_pricing(path: Path = ROOT / "configs" / "pricing.yaml") -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def claude_cost(usage: dict, model: str, pricing: dict, batch: bool = False, cache_ttl: str = "5m") -> float:
    """USD for one request's (or a summed) usage. Batch discount stacks with the cache multipliers."""
    p = pricing["claude"][model]
    usd = (usage.get("input_tokens", 0) * p["input"]
           + usage.get("cache_creation_input_tokens", 0) * p[f"cache_write_{cache_ttl}"]
           + usage.get("cache_read_input_tokens", 0) * p["cache_read"]
           + usage.get("output_tokens", 0) * p["output"]) / 1e6
    return usd * (pricing["batch_discount"] if batch else 1.0)


def sum_usage(rows: list[dict]) -> dict:
    return {f: sum(int(r.get("usage", {}).get(f) or 0) for r in rows) for f in USAGE_FIELDS}


def gpu_cost(seconds: float, pricing: dict) -> float:
    return seconds / 3600 * pricing["gpu"]["hourly_usd"]


def break_even_pay_per_use(train_usd: float, gpu_per_pred: float, claude_per_pred: float) -> float | None:
    """Total predictions after which fine-tuning (one-off training + GPU time per prediction, billed only while
    running) costs less than Claude. None if the GPU is not cheaper per prediction, so it never pays back."""
    saving = claude_per_pred - gpu_per_pred
    return train_usd / saving if saving > 0 else None


def break_even_always_on(monthly_gpu_usd: float, claude_per_pred: float, capacity_per_month: float | None = None,
                         train_usd_per_month: float = 0.0) -> float | None:
    """Monthly volume above which an always-on GPU server (fixed monthly cost) beats paying Claude per prediction.
    None if that volume exceeds what one GPU can serve in a month."""
    if claude_per_pred <= 0:
        return None
    v = (monthly_gpu_usd + train_usd_per_month) / claude_per_pred
    return None if capacity_per_month is not None and v > capacity_per_month else v
