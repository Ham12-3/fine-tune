"""Phase 7: drop the declared-but-absent multi-token-prediction (nextn) layer from a Qwen3.5 GGUF.

Qwen3.5's config has mtp_num_hidden_layers=1. transformers does not load the base checkpoint's mtp.* weights, so
the merged checkpoint has none, but convert_hf_to_gguf.py still wrote block_count=25 and nextn_predict_layers=1.
llama.cpp then looks for blk.24.* and refuses to load. The fix rewrites three metadata fields; every tensor is
copied byte for byte. (merge_and_gguf.py now sets mtp_num_hidden_layers=0 before converting, so new conversions
don't need this.)
  python scripts/fix_gguf_nextn.py models/in.gguf models/out.gguf
"""

from __future__ import annotations

import sys
from pathlib import Path

import gguf
from gguf.scripts.gguf_new_metadata import MetadataDetails, copy_with_new_metadata


def main() -> int:
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    reader = gguf.GGUFReader(src)
    arch = reader.fields[gguf.Keys.General.ARCHITECTURE].contents()
    blocks = reader.fields[f"{arch}.block_count"].contents()
    nextn = reader.fields[f"{arch}.nextn_predict_layers"].contents()
    present = {int(t.name.split(".")[1]) for t in reader.tensors if t.name.startswith("blk.")}
    n = len(present)
    if not (nextn and blocks == n + nextn and present == set(range(n))):
        print(f"nothing to fix: block_count={blocks}, nextn={nextn}, blocks with tensors={n}")
        return 1
    recurrent = reader.fields[f"{arch}.attention.recurrent_layers"].contents()[:n]
    new = {
        f"{arch}.block_count": MetadataDetails(gguf.GGUFValueType.UINT32, n),
        f"{arch}.nextn_predict_layers": MetadataDetails(gguf.GGUFValueType.UINT32, 0),
        f"{arch}.attention.recurrent_layers": MetadataDetails(gguf.GGUFValueType.ARRAY, recurrent,
                                                              sub_type=gguf.GGUFValueType.BOOL),
    }
    writer = gguf.GGUFWriter(dst, arch=arch, endianess=reader.endianess)
    align = reader.fields.get(gguf.Keys.General.ALIGNMENT)
    if align is not None:
        writer.data_alignment = align.contents()
    copy_with_new_metadata(reader, writer, new, remove_metadata=[])
    print(f"wrote {dst}: block_count {blocks} -> {n}, nextn_predict_layers {nextn} -> 0, {len(reader.tensors)} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
