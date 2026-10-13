# small-vs-frontier. GPU steps (zero-shot small model, QLoRA sweep, GGUF export) run in Colab:
# see notebooks/train_colab.ipynb. Everything here runs on the laptop CPU.

PY := uv run python
MODELS ?= claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5
MODES ?= zero_shot few_shot

.PHONY: help setup setup-cpu data baseline baseline-pilot fetch-results eval bench-cpu report serve test

help:
	@echo "setup          install core + dev deps (uv sync)"
	@echo "setup-cpu      also install llama-cpp-python (CPU wheel)"
	@echo "data           build splits, 1k test sample, few-shot pool (Phase 2)"
	@echo "baseline-pilot 20-example Claude pilot to measure tokens (cheap)"
	@echo "baseline       Claude zero-shot + few-shot on the test sample (PAID, asks first)"
	@echo "fetch-results  copy results/GGUF from Google Drive into results/ and models/"
	@echo "eval           compute metrics + costs for every run in results/raw"
	@echo "bench-cpu      benchmark the GGUF model on this laptop CPU"
	@echo "report         tables + charts into results/summary and results/figures"
	@echo "serve          FastAPI app on http://127.0.0.1:8000"
	@echo "test           run pytest"

setup:
	uv sync

setup-cpu:
	uv sync --extra cpu

data:
	$(PY) scripts/prepare_data.py

baseline-pilot:
	$(PY) scripts/run_claude.py --models $(MODELS) --modes $(MODES) --limit 20

baseline:
	$(PY) scripts/run_claude.py --models $(MODELS) --modes $(MODES)

fetch-results:
	$(PY) scripts/fetch_results.py

eval:
	$(PY) scripts/evaluate.py

bench-cpu:
	$(PY) scripts/bench_cpu.py

report:
	$(PY) scripts/make_report.py

serve:
	uv run uvicorn svf.serving.app:app --host 127.0.0.1 --port 8000

test:
	uv run pytest -q
