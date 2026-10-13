"""Classification and latency metrics. Invalid predictions (None) count as wrong for every metric."""

from __future__ import annotations

from collections import Counter

import numpy as np
from sklearn.metrics import f1_score

INVALID = "__invalid__"


def classification_metrics(gold: list[str], pred: list[str | None], labels: list[str]) -> dict:
    if len(gold) != len(pred):
        raise ValueError("gold and pred differ in length")
    p = [x if x is not None else INVALID for x in pred]
    per_class = f1_score(gold, p, labels=labels, average=None, zero_division=0)
    return {
        "n": len(gold),
        "accuracy": float(np.mean([g == x for g, x in zip(gold, p)])),
        # macro over the 77 real labels only, so an invalid answer lowers recall without adding a class
        "macro_f1": float(np.mean(per_class)),
        "invalid_rate": float(np.mean([x == INVALID for x in p])),
        "per_class_f1": {lab: float(f) for lab, f in zip(labels, per_class)},
    }


def confused_pairs(gold: list[str], pred: list[str | None], top: int = 10) -> list[dict]:
    pairs = Counter((g, x if x is not None else INVALID) for g, x in zip(gold, pred) if g != x)
    return [{"gold": g, "pred": x, "count": c} for (g, x), c in pairs.most_common(top)]


def latency_stats(seconds: list[float]) -> dict:
    if not seconds:
        return {}
    a = np.asarray(seconds, dtype=float)
    return {"n": int(a.size), "mean_s": float(a.mean()), "p50_s": float(np.percentile(a, 50)),
            "p95_s": float(np.percentile(a, 95)), "max_s": float(a.max())}
