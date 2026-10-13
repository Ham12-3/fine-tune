"""Claude classification calls: standard Messages API and the Message Batches API.

Every request has the same shape: a cached system prompt (instructions + labels [+ few-shot block]), the
customer message as the user turn, structured output constrained to the 77 label names, and a fixed effort.
Thinking is left at the model default (adaptive). It can't be disabled on Opus 5.5, so low effort is used on all
three models for parity, and thinking tokens are billed as output. Server-side refusal fallbacks are deliberately
not enabled: a fallback would answer with a different model and blur the comparison. Refusals are recorded instead.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from svf.eval.cost import USAGE_FIELDS


def request_params(model: str, system: str, text: str, schema: dict, cfg: dict) -> dict:
    cache = {"type": "ephemeral"} if cfg.get("cache_ttl", "5m") == "5m" else {"type": "ephemeral", "ttl": cfg["cache_ttl"]}
    return {
        "model": model,
        "max_tokens": cfg["max_tokens"],
        "system": [{"type": "text", "text": system, "cache_control": cache}],
        "messages": [{"role": "user", "content": text}],
        "output_config": {"effort": cfg["effort"], "format": {"type": "json_schema", "schema": schema}},
    }


def _usage(msg: Any) -> dict:
    u = msg.usage
    return {f: int(getattr(u, f, 0) or 0) for f in USAGE_FIELDS}


def _record(msg: Any) -> dict:
    text = next((b.text for b in msg.content if b.type == "text"), None)
    return {
        "raw": text,
        "stop_reason": msg.stop_reason,
        "refusal_category": getattr(msg.stop_details, "category", None) if msg.stop_reason == "refusal" else None,
        "usage": _usage(msg),
        "request_id": getattr(msg, "_request_id", None),
        "model_served": msg.model,
    }


def classify(client: anthropic.Anthropic, params: dict) -> dict:
    """One standard call. Never raises for API errors: they are recorded so a run can continue and be resumed."""
    t0 = time.perf_counter()
    try:
        msg = client.messages.create(**params)
    except anthropic.APIStatusError as e:
        return {"raw": None, "error": f"{type(e).__name__} {e.status_code}: {e.message}",
                "request_id": getattr(e, "request_id", None), "latency_s": time.perf_counter() - t0, "usage": {}}
    except anthropic.APIConnectionError as e:
        return {"raw": None, "error": f"{type(e).__name__}: {e}", "latency_s": time.perf_counter() - t0, "usage": {}}
    rec = _record(msg)
    rec["latency_s"] = time.perf_counter() - t0
    return rec


def submit_batch(client: anthropic.Anthropic, items: list[tuple[str, dict]]) -> str:
    batch = client.messages.batches.create(
        requests=[Request(custom_id=cid, params=MessageCreateParamsNonStreaming(**p)) for cid, p in items])
    return batch.id


def wait_batch(client: anthropic.Anthropic, batch_id: str, poll_s: float = 30.0, log=print) -> None:
    while True:
        b = client.messages.batches.retrieve(batch_id)
        if b.processing_status == "ended":
            log(f"batch {batch_id} ended: {b.request_counts}")
            return
        log(f"batch {batch_id} {b.processing_status}: {b.request_counts}")
        time.sleep(poll_s)


def batch_results(client: anthropic.Anthropic, batch_id: str) -> dict[str, dict]:
    """Results keyed by custom_id (they arrive in any order)."""
    out = {}
    for r in client.messages.batches.results(batch_id):
        if r.result.type == "succeeded":
            out[r.custom_id] = _record(r.result.message)
        else:
            err = getattr(r.result, "error", None)
            out[r.custom_id] = {"raw": None, "usage": {}, "error": f"batch {r.result.type}: {getattr(err, 'type', '')}"}
    return out
