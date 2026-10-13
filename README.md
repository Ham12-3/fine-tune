# small-vs-frontier

A QLoRA fine-tuned small open model vs Claude Haiku 5.5, Sonnet 5.5 and Opus 5.5 on
[Banking77](https://huggingface.co/datasets/PolyAI/banking77), a 77-intent classification task. It measures where the trade-off flips on
accuracy, latency and cost, including running the fine-tuned model on a laptop CPU.

> Work in progress. The results table, write-up and reproduction steps are added as each phase lands.

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

GPU work runs in Google Colab (T4). Setup for that is documented with Phase 1.
