| System | Accuracy | Macro-F1 | Median latency | Cost per 1k |
|---|---|---|---|---|
| Fine-tuned, laptop CPU | 93.8% | 0.939 | 1.65 s | $0 (owned) |
| Fine-tuned, T4 GPU | 93.4% | 0.935 | 0.57 s | $0.0046 |
| Opus 5.5 few-shot | 91.1% | 0.910 | 1.40 s | $2.82 (batch $1.409) |
| Opus 5.5 zero-shot | 88.9% | 0.883 | 1.36 s | $1.16 (batch $0.578) |
| Sonnet 5.5 few-shot | 88.8% | 0.887 | 1.32 s | $1.41 (batch $0.703) |
| Sonnet 5.5 zero-shot | 86.4% | 0.858 | 1.36 s | $0.61 (batch $0.303) |
| Haiku 5.5 few-shot | 86.0% | 0.858 | 1.21 s | $0.13 (batch $0.064) |
| Haiku 5.5 zero-shot | 81.8% | 0.811 | 1.08 s | $0.05 (batch $0.025) |
| Qwen3.5-2B untrained, T4 | 63.6% | 0.631 | 0.72 s | $0.05 |
