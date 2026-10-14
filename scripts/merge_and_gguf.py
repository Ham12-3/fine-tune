"""Phase 7 (Colab): merge the best adapter into the 16-bit base, convert to GGUF with llama.cpp, quantise.

llama.cpp is pinned to the commit vendored by llama-cpp-python 0.3.36 (the laptop runtime), so the converter
writes exactly the metadata that runtime reads. Steps are skipped when their output already exists.
  1. merge     PEFT merge_and_unload on CPU in bfloat16 -> /content/merged_<adapter> (local disk)
  2. convert   llama.cpp convert_hf_to_gguf.py -> f16 GGUF (text model only; the vision tower is not exported)
  3. quantise  llama-quantize (built CPU-only, just that target) -> <quant> GGUF, copied to <drive_root>/gguf/
Writes <drive_root>/gguf/<name>.json with sizes, sha256, the llama.cpp commit and timings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

from svf import runinfo

ROOT = Path(__file__).resolve().parents[1]
LLAMA_CPP_COMMIT = "0c1e57098bba43ac29e6e3b677cdceebdd22334f"  # vendored by llama-cpp-python 0.3.36


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def run(cmd: list[str] | str, **kw) -> None:
    print("$", cmd if isinstance(cmd, str) else " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


def merge(adapter_dir: Path, base: str, out: Path) -> None:
    """Merge on CPU in bfloat16, the base checkpoint's own dtype (~5 GB of RAM). Unsloth's merged_16bit upcasts
    Qwen3.5 to float32 and ran Colab out of its 12 GB of system RAM (the runtime restarted)."""
    import torch
    import transformers
    from peft import PeftModel

    arch = transformers.AutoConfig.from_pretrained(base).architectures[0]  # Qwen3_5ForConditionalGeneration
    model = getattr(transformers, arch).from_pretrained(base, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(out), max_shard_size="2GB")
    transformers.AutoTokenizer.from_pretrained(str(adapter_dir)).save_pretrained(str(out))
    # transformers doesn't load the multi-token-prediction (mtp.*) weights, so the merged checkpoint has none;
    # without this the converter still declares the extra layer and llama.cpp fails to load the GGUF.
    cfg_path = out / "config.json"
    c = json.loads(cfg_path.read_text())
    c.get("text_config", c)["mtp_num_hidden_layers"] = 0
    cfg_path.write_text(json.dumps(c, indent=2))


def main() -> int:
    cfg = yaml.safe_load((ROOT / "configs" / "train" / "base.yaml").read_text(encoding="utf-8"))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", default="f1.00")
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--work", type=Path, default=Path("/content"))
    args = ap.parse_args()

    drive = Path(cfg["drive_root"])
    name = f"qwen3.5-2b-banking77-{args.adapter}"
    merged, llama, work = args.work / f"merged_{args.adapter}", args.work / "llama.cpp", args.work / "gguf"
    work.mkdir(exist_ok=True)
    f16, quant = work / f"{name}-f16.gguf", work / f"{name}-{args.quant}.gguf"
    timings = {}

    t = time.time()
    if not (merged / "config.json").exists():
        merge(drive / "adapters" / args.adapter, cfg["base_model"], merged)
    timings["merge_s"] = time.time() - t

    if not (llama / "convert_hf_to_gguf.py").exists():
        run(["git", "clone", "--quiet", "https://github.com/ggml-org/llama.cpp", str(llama)])
    run(["git", "-C", str(llama), "checkout", "--quiet", LLAMA_CPP_COMMIT])

    t = time.time()
    if not f16.exists():
        run([sys.executable, str(llama / "convert_hf_to_gguf.py"), str(merged), "--outtype", "f16", "--outfile", str(f16)])
    timings["convert_s"] = time.time() - t

    t = time.time()
    qbin = llama / "build" / "bin" / "llama-quantize"
    if not qbin.exists():
        run(["cmake", "-S", str(llama), "-B", str(llama / "build"), "-DGGML_NATIVE=OFF", "-DGGML_CUDA=OFF",
             "-DLLAMA_CURL=OFF", "-DCMAKE_BUILD_TYPE=Release"], stdout=subprocess.DEVNULL)
        run(["cmake", "--build", str(llama / "build"), "--target", "llama-quantize", "-j", "4"], stdout=subprocess.DEVNULL)
    timings["build_s"] = time.time() - t

    t = time.time()
    if not quant.exists():
        run([str(qbin), str(f16), str(quant), args.quant], stdout=subprocess.DEVNULL)
    timings["quantize_s"] = time.time() - t

    (drive / "gguf").mkdir(exist_ok=True)
    shutil.copy(quant, drive / "gguf" / quant.name)
    record = {
        "name": quant.name, "adapter": args.adapter, "base_model": cfg["base_model"], "quant": args.quant,
        "llama_cpp_commit": LLAMA_CPP_COMMIT, "bytes": quant.stat().st_size, "sha256": sha256(quant),
        "f16_bytes": f16.stat().st_size, "merged_dir_bytes": sum(p.stat().st_size for p in merged.rglob("*") if p.is_file()),
        "timings_s": {k: round(v, 1) for k, v in timings.items()}, "finished": runinfo.now(),
    }
    (drive / "gguf" / f"{quant.stem}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
