"""Metadata written next to every run, so each result file says exactly what produced it."""

from __future__ import annotations

import datetime as dt
import json
import os
import platform
import subprocess
from pathlib import Path


def git_sha(root: Path) -> str | None:
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True,
                             text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True,
                               text=True).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def data_hashes(root: Path, files: list[str]) -> dict:
    manifest = json.loads((root / "data" / "processed" / "manifest.json").read_text(encoding="utf-8"))
    return {f: manifest["files"][f]["sha256"] for f in files}


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
