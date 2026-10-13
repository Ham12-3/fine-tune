"""Phase 2: build every data split from the pinned Banking77 CSVs.

Writes to data/processed/:
  train.jsonl, val.jsonl          official train split minus a stratified 10% validation hold-out
  test_full.jsonl                 the official test split (3,080)
  test_sample.jsonl               fixed stratified 1k sample of test (used for every model)
  few_shot.jsonl                  3 examples per label, from train.jsonl only
  train_f{0.10,...}.jsonl         nested stratified training subsets for the sweep
  labels.json                     77 labels with descriptions, in id order
  manifest.json                   sha256, row count and per-label counts of every file
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from svf.data import load_banking77 as b77

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=ROOT / "configs" / "data.yaml", type=Path)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    seed = cfg["seed"]
    raw = b77.fetch_raw(cfg["source"], ROOT / cfg["raw_dir"])
    labels = b77.load_labels()

    official = {split: b77.read_csv(path, split, labels) for split, path in raw.items()}
    for split, rows in official.items():
        expected = cfg["source"]["files"][split]["rows"]
        if len(rows) != expected:
            raise ValueError(f"{split}: {len(rows)} rows, expected {expected}")

    train, val = b77.stratified_split(official["train"], cfg["val_frac"], seed)
    _, test_sample = b77.stratified_split(official["test"], cfg["test_sample_size"], seed)
    files = {
        "train": train,
        "val": val,
        "test_full": b77.sort_by_id(official["test"]),
        "test_sample": test_sample,
        "few_shot": b77.few_shot_pool(train, cfg["few_shot_per_label"], seed),
    }
    for frac, rows in b77.nested_subsets(train, cfg["sweep_fractions"], seed).items():
        files[f"train_f{frac:.2f}"] = rows

    out = ROOT / cfg["out_dir"]
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": cfg["source"],
        "seed": seed,
        "labels_sha256": b77.sha256_bytes(b77.LABELS_FILE.read_bytes()),
        "files": {},
    }
    for name, rows in files.items():
        data = b77.to_jsonl(rows)
        (out / f"{name}.jsonl").write_bytes(data)
        counts = b77.label_counts(rows)
        manifest["files"][name] = {
            "sha256": b77.sha256_bytes(data),
            "rows": len(rows),
            "labels": len(counts),
            "min_per_label": min(counts.values()),
            "max_per_label": max(counts.values()),
        }
    (out / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    # Exact-duplicate texts across splits (case/whitespace-insensitive): reported, not removed,
    # so the official splits stay comparable with published Banking77 numbers.
    norm = lambda rows: {" ".join(r["text"].lower().split()) for r in rows}  # noqa: E731
    manifest["overlap"] = {
        "train_test_full": len(norm(official["train"]) & norm(official["test"])),
        "train_test_sample": len(norm(train) & norm(test_sample)),
        "train_val": len(norm(train) & norm(val)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    print(f"{'file':<16}{'rows':>6}{'labels':>8}{'min':>5}{'max':>5}  sha256")
    for name, m in manifest["files"].items():
        print(f"{name:<16}{m['rows']:>6}{m['labels']:>8}{m['min_per_label']:>5}{m['max_per_label']:>5}  {m['sha256'][:12]}")
    print("duplicate texts across splits:", manifest["overlap"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
