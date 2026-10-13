from pathlib import Path

import yaml

import svf

ROOT = Path(__file__).resolve().parents[1]


def test_package_imports():
    assert svf.__doc__


def test_pricing_covers_every_claude_model():
    claude = yaml.safe_load((ROOT / "configs" / "claude.yaml").read_text())
    pricing = yaml.safe_load((ROOT / "configs" / "pricing.yaml").read_text())
    fields = {"input", "cache_write_5m", "cache_read", "output"}
    for model in claude["models"]:
        assert fields <= pricing["claude"][model].keys(), model
    assert pricing["date_checked"]
