"""Phase 5: QLoRA data-efficiency sweep (Colab GPU).

For each training fraction (configs/data.yaml sweep_fractions, nested subsets from Phase 2): train or resume
from <drive_root>/checkpoints/f<frac>, save the adapter to <drive_root>/adapters/f<frac>, score it on the
validation split, and write <drive_root>/adapters/f<frac>/done.json. A run with done.json is skipped.
  --fractions 0.10            run only these fractions
  --tag resume_test --max-steps 150   throwaway run under a separate name (used to test resuming)
  --dry-run                   print one encoded example and the token stats, no training
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import torch
import yaml

from svf import runinfo
from svf.data.load_banking77 import read_jsonl
from svf.models import train_qlora

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=ROOT / "configs" / "train" / "base.yaml")
    ap.add_argument("--fractions", type=float, nargs="+")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg = yaml.safe_load((ROOT / "configs" / "data.yaml").read_text(encoding="utf-8"))
    drive = Path(cfg["drive_root"])
    labels = json.loads((PROC / "labels.json").read_text(encoding="utf-8"))
    val = read_jsonl(PROC / "val.jsonl")

    if args.dry_run:
        model, tok = train_qlora.load_for_training(cfg)
        rows = read_jsonl(PROC / "train.jsonl")
        enc = [train_qlora.encode(tok, r["text"], r["label"], cfg["max_seq_len"]) for r in rows]
        e = enc[0]
        print("prompt+completion:", repr(tok.decode(e["input_ids"])))
        print("loss on:", repr(tok.decode([i for i, l in zip(e["input_ids"], e["labels"]) if l != -100])))
        lens = sorted(len(x["input_ids"]) for x in enc)
        print(f"tokens/example: mean {sum(lens) / len(lens):.1f}, p99 {lens[int(0.99 * len(lens))]}, max {lens[-1]}")
        return 0

    for frac in args.fractions or data_cfg["sweep_fractions"]:
        name = f"f{frac:.2f}" + (f"_{args.tag}" if args.tag else "")
        adapter_dir, ckpt_dir = drive / "adapters" / name, drive / "checkpoints" / name
        done = adapter_dir / "done.json"
        if done.exists():
            print(f"{name}: done, skipping")
            continue
        split = f"train_f{frac:.2f}"
        rows = read_jsonl(PROC / f"{split}.jsonl")
        print(f"== {name}: {len(rows)} training examples", flush=True)
        res = train_qlora.train_one(cfg, rows, ckpt_dir, adapter_dir, max_steps=args.max_steps)
        ev = train_qlora.evaluate(res["model"], res["tok"], val, labels, cfg["eval"]["batch_size"],
                                  cfg["eval"]["max_new_tokens"])
        m = ev["metrics"]
        ev["metrics"] = {k: v for k, v in m.items() if k != "per_class_f1"} | {"per_class_f1": m["per_class_f1"]}
        record = {"run": name, "fraction": frac, "base_model": cfg["base_model"], "config": cfg,
                  "train": res["stats"], "val": ev, "data_sha256": runinfo.data_hashes(ROOT, [split, "val"]),
                  "hardware": runinfo.hardware(), "finished": runinfo.now()}
        (drive / "results" / "sweep").mkdir(parents=True, exist_ok=True)
        (drive / "results" / "sweep" / f"{name}.json").write_text(json.dumps(record, indent=1))
        done.write_text(json.dumps({"finished": record["finished"], "val_accuracy": m["accuracy"]}))
        s = res["stats"]
        print(f"{name}: val acc {m['accuracy']:.3f}  macroF1 {m['macro_f1']:.3f}  invalid {m['invalid_rate']:.3f} | "
              f"{s['steps']} steps in {s['train_seconds_total'] / 60:.1f} min ({s['sessions']} session(s)), "
              f"peak {s['peak_vram_gb']:.1f} GB, loss {s['loss_first']:.3f} -> {s['loss_last']:.3f}", flush=True)
        del res
        gc.collect()
        torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
