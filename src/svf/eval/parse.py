"""Map a model's raw output to one of the 77 labels, or None (counted as invalid).

Rules, in order (the method is recorded so the report can say how many answers needed which rule):
  1. json    {"label": "..."} (Claude's structured output)
  2. exact   first non-empty line, minus a leading "intent:"/"label:"/"answer:", quotes, backticks and
             a trailing full stop, matches a label after normalisation
  3. search  exactly one distinct label appears as a whole token anywhere in the output
Normalisation: lower case, spaces and hyphens to underscores, other punctuation dropped. So
"Refund not showing up" -> Refund_not_showing_up and "reverted_card_payment" -> reverted_card_payment?.
"""

from __future__ import annotations

import json
import re

_PREFIX = re.compile(r"^\s*(?:the\s+)?(?:intent|label|answer|category)\s*(?:is)?\s*[:=-]?\s*", re.I)


def normalise(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s\-]+", "_", text)
    text = re.sub(r"[^a-z0-9_]", "", text)
    return text.strip("_")


class LabelParser:
    def __init__(self, labels: list[dict]):
        self.names = [lab["name"] for lab in labels]
        self.by_key = {normalise(n): n for n in self.names}
        if len(self.by_key) != len(self.names):
            raise ValueError("labels collide after normalisation")
        # Longest first so the alternation prefers e.g. card_payment_wrong_exchange_rate over exchange_rate;
        # the lookarounds stop a label matching inside a longer one.
        alts = "|".join(re.escape(k) for k in sorted(self.by_key, key=len, reverse=True))
        self._search = re.compile(rf"(?<![a-z0-9_])({alts})(?![a-z0-9_])")

    def parse(self, raw: str | None) -> tuple[str | None, str]:
        """Returns (label or None, method) with method in json | exact | search | invalid."""
        if not raw or not raw.strip():
            return None, "invalid"
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and isinstance(obj.get("label"), str) and obj["label"] in self.names:
                return obj["label"], "json"
        except ValueError:
            pass
        first = next(line for line in raw.splitlines() if line.strip())
        first = _PREFIX.sub("", first).strip().strip("`'\"*").rstrip(".").strip()
        if (key := normalise(first)) in self.by_key:
            return self.by_key[key], "exact"
        found = set(self._search.findall(normalise_keep_spaces(raw)))
        if len(found) == 1:
            return self.by_key[found.pop()], "search"
        return None, "invalid"


def normalise_keep_spaces(text: str) -> str:
    """Like normalise, but per word, so whole-token search still sees word boundaries."""
    return " ".join(normalise(w) for w in re.split(r"[\s,;:()\[\]{}\"'`]+", text) if w)
