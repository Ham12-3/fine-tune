"""Banking77: download, splits, few-shot pool and nested training subsets.

Every function that draws randomly takes an explicit seed and works on records sorted by id,
so the same raw CSVs and config always give byte-identical JSONL files.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sklearn.model_selection import train_test_split

LABELS_FILE = Path(__file__).with_name("labels.yaml")
RAW_URL = "https://raw.githubusercontent.com/{repo}/{commit}/{path}"


def load_labels() -> list[dict]:
    """The 77 labels in canonical id order, each {"id", "name", "description"}."""
    labels = yaml.safe_load(LABELS_FILE.read_text(encoding="utf-8"))
    return [{"id": i, **lab} for i, lab in enumerate(labels)]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch_raw(source: dict, raw_dir: Path) -> dict[str, Path]:
    """Download the pinned CSVs once into raw_dir and verify their sha256."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, spec in source["files"].items():
        dest = raw_dir / f"{split}.csv"
        if not dest.exists():
            url = RAW_URL.format(repo=source["repo"], commit=source["commit"], path=spec["path"])
            with urllib.request.urlopen(url, timeout=60) as resp:
                dest.write_bytes(resp.read())
        got = sha256_bytes(dest.read_bytes())
        if got != spec["sha256"]:
            raise ValueError(f"{dest}: sha256 {got} != expected {spec['sha256']}")
        paths[split] = dest
    return paths


def read_csv(path: Path, split: str, labels: list[dict]) -> list[dict]:
    """Rows as {"id", "text", "label", "label_id"}; id is the split plus the row index in the CSV."""
    name_to_id = {lab["name"]: lab["id"] for lab in labels}
    # Same CSV dialect as the HF loading script.
    reader = csv.reader(io.StringIO(path.read_text(encoding="utf-8")), quotechar='"',
                        delimiter=",", quoting=csv.QUOTE_ALL, skipinitialspace=True)
    next(reader)  # header: text,category
    rows = []
    for i, (text, label) in enumerate(reader):
        if label not in name_to_id:
            raise ValueError(f"{path}:{i + 2}: unknown label {label!r}")
        rows.append({"id": f"{split}-{i:05d}", "text": text, "label": label, "label_id": name_to_id[label]})
    return rows


def sort_by_id(records: list[dict]) -> list[dict]:
    return sorted(records, key=lambda r: r["id"])


def stratified_split(records: list[dict], test_size: float | int, seed: int) -> tuple[list[dict], list[dict]]:
    """Split records into (rest, held_out), stratified by label."""
    records = sort_by_id(records)
    rest, held = train_test_split(records, test_size=test_size, random_state=seed,
                                  stratify=[r["label_id"] for r in records])
    return sort_by_id(rest), sort_by_id(held)


def few_shot_pool(train: list[dict], per_label: int, seed: int) -> list[dict]:
    """`per_label` examples per label from train, in label order."""
    by_label = defaultdict(list)
    for r in sort_by_id(train):
        by_label[r["label_id"]].append(r)
    rng = np.random.default_rng(seed)
    pool = []
    for label_id in sorted(by_label):
        rows = by_label[label_id]
        idx = sorted(rng.choice(len(rows), size=per_label, replace=False))
        pool.extend(rows[i] for i in idx)
    return pool


def nested_subsets(train: list[dict], fractions: list[float], seed: int) -> dict[float, list[dict]]:
    """Stratified training subsets where each smaller fraction is contained in every larger one.

    Each label's examples are shuffled once; fraction f keeps the first ceil(f * n) of them.
    """
    by_label = defaultdict(list)
    for r in sort_by_id(train):
        by_label[r["label_id"]].append(r)
    rng = np.random.default_rng(seed)
    order = {label_id: [by_label[label_id][i] for i in rng.permutation(len(by_label[label_id]))]
             for label_id in sorted(by_label)}
    # round() first: 0.1 * 30 is 3.0000000000000004 in floating point, which would ceil to 4.
    return {f: sort_by_id([r for rows in order.values() for r in rows[: math.ceil(round(f * len(rows), 6))]])
            for f in fractions}


def to_jsonl(records: list[dict]) -> bytes:
    lines = (json.dumps(r, ensure_ascii=False, sort_keys=True) for r in records)
    return ("\n".join(lines) + "\n").encode("utf-8")


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def label_counts(records: list[dict]) -> dict[str, int]:
    counts = defaultdict(int)
    for r in records:
        counts[r["label"]] += 1
    return dict(counts)
