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
