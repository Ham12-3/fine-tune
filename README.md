# small-vs-frontier

A QLoRA fine-tuned small open model vs Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 on
[Banking77](https://huggingface.co/datasets/PolyAI/banking77), a 77-intent classification task. It measures where the trade-off flips on
accuracy, latency and cost, including running the fine-tuned model on a laptop CPU.

> Work in progress. The results table, write-up and reproduction steps are added as each phase lands.

## Why Qwen3.5-2B

I probed four small open models before choosing one (`scripts/probe_model.py`, results in
`results/summary/phase3_model_probe.json`). Llama 3.2 3B was dropped first because its download is gated behind
manual approval. The other three all loaded, trained and generated on a Colab T4 in 4-bit. **Qwen3.5-2B**
(Apache-2.0) was the lightest to run: 2.4 GB of VRAM loaded and 3.7 GB peak while training, against 9 GB for
Gemma 4 E2B, whose per-layer embeddings make it a 5.1B-parameter checkpoint. It also had the fastest training
step, tied with Gemma (1.34 s vs 1.96 s for Phi-4-mini at 3.8B), and it will be the smallest GGUF on the laptop
CPU. It is a hybrid model: 18 Gated DeltaNet linear-attention layers and 6 full-attention layers. So LoRA targets
the DeltaNet projections as well as the usual attention and MLP ones, and only the language model is fine-tuned,
not its vision tower. Thinking mode is switched off in the chat template. llama.cpp, and the llama-cpp-python
0.3.36 CPU wheel, support the architecture. Phi-4-mini (MIT, plain transformer) is the fallback if the hybrid
layers cause trouble in GGUF export.

## Layout

| Path | What |
|---|---|
| `src/svf/data` | Banking77 loading, splits, prompt builders |
| `src/svf/models` | Claude client, HF/GGUF inference, QLoRA training |
| `src/svf/eval` | label parsing, metrics, cost and break-even |
| `src/svf/serving` | FastAPI app (`/classify`, `/compare`) |
| `configs/` | data, Claude, pricing and training settings (no hardcoded hyperparameters) |
| `scripts/` | entry points called by the Makefile and the Colab notebook |
| `notebooks/train_colab.ipynb` | every GPU step, runnable by hand in Colab |
| `results/` | raw predictions, summaries and figures |

## Setup (laptop)

```bash
uv sync                 # core + dev deps (Python 3.12)
cp .env.example .env    # then fill in ANTHROPIC_API_KEY and HF_TOKEN yourself
make test
```

## Setup (Colab GPU)

GPU work runs in Google Colab on a T4. Claude Code drives the notebook through
[colab-mcp](https://github.com/googlecolab/colab-mcp), and every cell is recorded in
`notebooks/train_colab.ipynb` so it can also be run by hand.

1. Runtime → Change runtime type → **T4 GPU**.
2. Run the Drive mount cell yourself (it needs a Google sign-in popup). It creates
   `MyDrive/small-vs-frontier/{checkpoints,adapters,results,gguf}`.
3. Add `HF_TOKEN` in Colab Secrets (🔑, notebook access on) and run the token cell from the Colab UI.
   Secrets can't be read from cells run remotely, so re-run it after every runtime restart.
4. Install the pinned GPU stack from `requirements-colab.txt` (verified on a T4: Python 3.13, torch 2.11 + CUDA 13.0, Unsloth 2026.10.5).
