# small-vs-frontier

A QLoRA fine-tuned small open model vs Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 on
[Banking77](https://huggingface.co/datasets/PolyAI/banking77), a 77-intent classification task. It measures where the
trade-off flips on accuracy, latency and cost, including running the fine-tuned model on a laptop CPU.
The full write-up is in [REPORT.md](REPORT.md).

## Results

Same 1,000-example stratified test sample for every system:

| System | Accuracy | Macro-F1 | Median latency | Cost per 1k predictions |
|---|---|---|---|---|
| **Fine-tuned Qwen3.5-2B, laptop CPU** (Q4_K_M, i5-1135G7) | **93.8%** | 0.939 | 1.65 s | $0 marginal |
| **Fine-tuned Qwen3.5-2B, T4 GPU** (4-bit) | **93.4%** | 0.935 | 0.57 s | $0.0046 (batched) |
| Claude Opus 5.5, few-shot | 91.1% | 0.910 | 1.40 s | $2.82 |
| Claude Opus 5.5, zero-shot | 88.9% | 0.883 | 1.36 s | $1.16 |
| Claude Sonnet 5.5, few-shot | 88.8% | 0.887 | 1.32 s | $1.41 |
| Claude Sonnet 5.5, zero-shot | 86.4% | 0.858 | 1.36 s | $0.61 |
| Claude Haiku 5.5, few-shot | 86.0% | 0.858 | 1.21 s | $0.13 |
| Claude Haiku 5.5, zero-shot | 81.8% | 0.811 | 1.08 s | $0.050 |
| Qwen3.5-2B untrained | 63.6% | 0.631 | 0.72 s | $0.052 |

- **Accuracy:** the fine-tuned 2B model beats every Claude setup; the closest, Opus few-shot, by 2.3 points
  (paired McNemar p = 0.0095). Full test set: 94.0%. 4-bit GGUF on a laptop CPU loses nothing measurable.
- **Data:** a quarter of the training set (2,282 examples) already passes Sonnet few-shot.
- **Cost:** batched on a GPU it is ~11x cheaper per prediction than Haiku zero-shot. As an always-on rented T4
  ($256/month) it beats Opus few-shot above ~91k predictions/month and Haiku zero-shot only above ~5.1M.
  One request at a time on a GPU, it costs the same as Haiku.
- **Effort:** Claude needs no training data or infrastructure and returned a valid label every time.

![Accuracy by training-set size](results/figures/accuracy_vs_training_size.png)
![Break-even](results/figures/break_even.png)

More charts in `results/figures/`; every number comes from `results/summary/`. Claude prices checked 2026-10-13
(`configs/pricing.yaml`); GPU priced at a rented-T4 rate of $0.35/h.

## Why Qwen3.5-2B

I probed four small open models before choosing one (`scripts/probe_model.py`, results in
`results/summary/phase3_model_probe.json`). Llama 3.2 3B was dropped first because its download is gated behind
manual approval. The other three all loaded, trained and generated on a Colab T4 in 4-bit. **Qwen3.5-2B**
(Apache-2.0) was the lightest to run: 2.4 GB of VRAM loaded and 3.7 GB peak while training, against 9 GB for
Gemma 4 E2B, whose per-layer embeddings make it a 5.1B-parameter checkpoint. It also had the fastest training
step, tied with Gemma (1.34 s vs 1.96 s for Phi-4-mini at 3.8B), and it gives the smallest GGUF for the laptop CPU.
It is a hybrid model: 18 Gated DeltaNet linear-attention layers and 6 full-attention layers. So LoRA targets the
DeltaNet projections as well as the usual attention and MLP ones, and only the language model is fine-tuned, not
its vision tower. Thinking mode is switched off in the chat template.

## Reproduce

| Step | Where | Command |
|---|---|---|
| Data splits (deterministic, hash-checked) | laptop | `make data` |
| Claude pilot, then full baselines (**paid**, ~$6.40) | laptop | `make baseline-pilot`, `make baseline` |
| Small-model zero-shot, QLoRA sweep, test eval, GGUF export | Colab T4 | `notebooks/train_colab.ipynb` (`run_small_zeroshot.py`, `train_sweep.py`, `eval_adapter.py`, `merge_and_gguf.py`) |
| Fix a GGUF made before the merge fix | laptop | `make fix-gguf IN=... OUT=...` |
| Laptop CPU benchmark | laptop | `make setup-cpu`, `make bench-cpu` |
| Comparison table, break-even, paired tests | laptop | `make eval` |
| Charts and README table | laptop | `make report` |
| API: `/health`, `/classify`, `/compare` (Claude side by side, **paid per call**) | laptop | `make serve` |
| Tests | laptop | `make test` |

```bash
curl -s -X POST localhost:8000/classify -H "content-type: application/json" -d '{"text": "I am still waiting on my card?"}'
# {"label": "card_arrival", "valid": true, "latency_ms": 958.8, ...}
```

## Layout

| Path | What |
|---|---|
| `src/svf/data` | Banking77 loading, splits, prompt builders, the 77 label descriptions |
| `src/svf/models` | Claude client, HF and GGUF inference, QLoRA training |
| `src/svf/eval` | label parsing, metrics, McNemar, cost and break-even |
| `src/svf/serving` | FastAPI app |
| `configs/` | data, Claude, pricing and training settings (no hardcoded hyperparameters) |
| `scripts/` | entry points called by the Makefile and the Colab notebook |
| `notebooks/train_colab.ipynb` | every GPU step, runnable by hand in Colab |
| `results/` | summaries (committed), raw predictions (not committed), figures |

## Setup (laptop)

```bash
uv sync --extra cpu     # core + dev deps + llama-cpp-python CPU wheel (Python 3.12)
cp .env.example .env    # then fill in ANTHROPIC_API_KEY and HF_TOKEN yourself
make test
```

The fine-tuned GGUF goes in `models/` (not committed; 1.27 GB, sha256 in `results/summary/gguf_model.json`).

## Setup (Colab GPU)

GPU work runs in Google Colab on a T4. Claude Code drives the notebook through
[colab-mcp](https://github.com/googlecolab/colab-mcp), and every cell is recorded in
`notebooks/train_colab.ipynb` so it can also be run by hand.

1. Runtime → Change runtime type → **T4 GPU**.
2. Run the Drive mount cell yourself (it needs a Google sign-in popup). It creates
   `MyDrive/small-vs-frontier/{checkpoints,adapters,results,gguf}`.
3. Add `HF_TOKEN` in Colab Secrets (🔑, notebook access on) and run the token cell from the Colab UI.
   Secrets can't be read from cells run remotely, so re-run it after every runtime restart.
4. Install the pinned GPU stack from `requirements-colab.txt` (verified on a T4: Python 3.13, torch 2.11 + CUDA 13.0,
   Unsloth 2026.10.5).
