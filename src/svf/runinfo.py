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


def cpu_name() -> str:
    """Human-readable CPU model (platform.processor() is only a family string on Windows)."""
    try:
        if platform.system() == "Windows":
            import winreg
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
        if Path("/proc/cpuinfo").exists():
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def hardware() -> dict:
    info = {"platform": platform.platform(), "python": platform.python_version(),
            "cpu": cpu_name(), "cpu_count": os.cpu_count()}
    try:
        import torch
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return info


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
