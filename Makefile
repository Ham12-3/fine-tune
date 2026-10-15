# small-vs-frontier. GPU steps (zero-shot small model, QLoRA sweep, GGUF export) run in Colab:
# see notebooks/train_colab.ipynb. Everything here runs on the laptop CPU.

PY := uv run python
PY_CPU := uv run --extra cpu python
MODELS ?= claude-haiku-5-5 claude-sonnet-5-5 claude-opus-5-5
MODES ?= zero_shot few_shot
GGUF ?= models/qwen3.5-2b-banking77-f1.00-Q4_K_M.gguf

.PHONY: help setup setup-cpu data baseline baseline-pilot fix-gguf eval bench-cpu report serve test

help:
	@echo "setup          install core + dev deps (uv sync)"
	@echo "setup-cpu      also install llama-cpp-python (CPU wheel)"
	@echo "data           build splits, 1k test sample, few-shot pool (Phase 2)"
	@echo "baseline-pilot 20-example Claude pilot to measure tokens (cheap)"
	@echo "baseline       Claude zero-shot + few-shot on the test sample (PAID, asks first)"
	@echo "fix-gguf       drop the absent nextn layer from a GGUF made before the merge fix (IN=... OUT=...)"
	@echo "eval           comparison table, break-even and paired tests from results/summary"
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

fix-gguf:
	$(PY) scripts/fix_gguf_nextn.py $(IN) $(OUT)

eval:
	$(PY) scripts/evaluate.py

bench-cpu:
	$(PY_CPU) scripts/bench_cpu.py --gguf $(GGUF)

report:
	$(PY) scripts/make_report.py

serve:
	SVF_GGUF=$(GGUF) uv run --extra cpu uvicorn svf.serving.app:app --host 127.0.0.1 --port 8000

test:
	uv run --extra cpu pytest -q
