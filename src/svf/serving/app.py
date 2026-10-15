"""Phase 8: FastAPI app serving the fine-tuned model from the laptop CPU, with a side-by-side Claude comparison.

  GET  /health    model file, threads, whether /compare is available
  POST /classify  {"text": ...} -> local GGUF label + latency
  POST /compare   {"text": ...} -> local and Claude labels side by side (run concurrently), both latencies, the
                  Claude call's token usage and cost, and whether they agree. Each call is a paid API request.

Settings (environment): SVF_GGUF (model path), SVF_THREADS, SVF_COMPARE_MODEL (default claude-haiku-5-5),
SVF_COMPARE_MODE (zero_shot | few_shot, default few_shot), ANTHROPIC_API_KEY (from .env; /compare only).
The model is loaded once at startup; llama.cpp contexts aren't thread-safe, so local calls share a lock.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from svf.data import prompts
from svf.data.load_banking77 import read_jsonl
from svf.eval import cost
from svf.eval.parse import LabelParser

ROOT = Path(__file__).resolve().parents[3]
PROC = ROOT / "data" / "processed"
DEFAULT_GGUF = ROOT / "models" / "qwen3.5-2b-banking77-f1.00-Q4_K_M.gguf"


class Classifier(Protocol):
    n_threads: int

    def generate(self, text: str, max_tokens: int = 24) -> tuple[str, float, int]: ...


class ClaudeCaller(Protocol):
    model: str

    def classify(self, text: str) -> dict[str, Any]: ...


class Message(BaseModel):
    text: str = Field(min_length=1, max_length=1000, description="A customer message to classify")


class ClaudeBaseline:
    """The Phase 4 Claude setup (cached system prompt, label enum, effort from configs/claude.yaml)."""

    def __init__(self, model: str, mode: str, labels: list[dict]):
        import anthropic

        from svf.models import claude_client as cc

        self.cc, self.model, self.mode = cc, model, mode
        cfg = yaml.safe_load((ROOT / "configs" / "claude.yaml").read_text(encoding="utf-8"))
        self.cfg = {**cfg, "max_retries": 2}
        few = read_jsonl(PROC / "few_shot.jsonl") if mode == "few_shot" else None
        self.system, self.schema = prompts.system_prompt(labels, few), prompts.label_schema(labels)
        self.client = anthropic.Anthropic(max_retries=2)

    def classify(self, text: str) -> dict[str, Any]:
        params = self.cc.request_params(self.model, self.system, prompts.user_message(text), self.schema, self.cfg)
        return self.cc.classify(self.client, params)


def create_app(classifier: Classifier | None = None, claude: ClaudeCaller | None = None) -> FastAPI:
    """Pass classifier/claude to inject test doubles; otherwise they are built from the environment at startup."""
    labels = json.loads((PROC / "labels.json").read_text(encoding="utf-8"))
    parser = LabelParser(labels)
    pricing = cost.load_pricing()
    lock = threading.Lock()
    state: dict[str, Any] = {"classifier": classifier, "claude": claude}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        load_dotenv(ROOT / ".env")
        if state["classifier"] is None:
            from svf.models.gguf_infer import GGUFClassifier

            path = Path(os.environ.get("SVF_GGUF", DEFAULT_GGUF))
            threads = int(os.environ["SVF_THREADS"]) if os.environ.get("SVF_THREADS") else None
            state["classifier"] = GGUFClassifier(path, n_threads=threads)
            state["model_file"] = path.name
        if state["claude"] is None and os.environ.get("ANTHROPIC_API_KEY"):
            state["claude"] = ClaudeBaseline(os.environ.get("SVF_COMPARE_MODEL", "claude-haiku-5-5"),
                                             os.environ.get("SVF_COMPARE_MODE", "few_shot"), labels)
        yield

    app = FastAPI(title="small-vs-frontier", version="0.1.0", lifespan=lifespan,
                  description="Banking77 intent classification: fine-tuned Qwen3.5-2B on CPU vs Claude.")

    def local(text: str) -> dict[str, Any]:
        with lock:
            raw, seconds, n_tokens = state["classifier"].generate(text)
        label, method = parser.parse(raw)
        return {"label": label, "valid": label is not None, "raw": raw, "latency_ms": round(seconds * 1000, 1),
                "completion_tokens": n_tokens, "model": state.get("model_file", "local")}

    def remote(text: str) -> dict[str, Any]:
        c = state["claude"]
        t = time.perf_counter()
        res = c.classify(text)
        if res.get("error"):
            return {"label": None, "valid": False, "error": res["error"], "model": c.model}
        label = None if res.get("stop_reason") == "refusal" else parser.parse(res.get("raw"))[0]
        usage = res.get("usage") or {}
        return {"label": label, "valid": label is not None, "latency_ms": round((time.perf_counter() - t) * 1000, 1),
                "model": c.model, "stop_reason": res.get("stop_reason"), "usage": usage,
                "cost_usd": cost.claude_cost(usage, c.model, pricing)}  # 5-minute cache TTL, as in claude.yaml

    @app.get("/health")
    def health() -> dict[str, Any]:
        clf = state["classifier"]
        return {"status": "ok" if clf is not None else "loading", "model": state.get("model_file", "local"),
                "threads": getattr(clf, "n_threads", None), "compare_available": state["claude"] is not None,
                "compare_model": getattr(state["claude"], "model", None)}

    @app.post("/classify")
    async def classify(msg: Message) -> dict[str, Any]:
        return await asyncio.to_thread(local, msg.text)

    @app.post("/compare")
    async def compare(msg: Message) -> dict[str, Any]:
        if state["claude"] is None:
            raise HTTPException(503, "Claude comparison unavailable: set ANTHROPIC_API_KEY in .env")
        mine, theirs = await asyncio.gather(asyncio.to_thread(local, msg.text), asyncio.to_thread(remote, msg.text))
        return {"text": msg.text, "local": mine, "claude": theirs,
                "agree": mine["label"] is not None and mine["label"] == theirs["label"]}

    return app


app = create_app()
