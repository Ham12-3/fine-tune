"""API tests with test doubles: no model file, no network, no API key needed."""

import pytest
from fastapi.testclient import TestClient

from svf.serving.app import create_app


class FakeLocal:
    n_threads = 2

    def __init__(self, raw="card_arrival"):
        self.raw, self.calls = raw, 0

    def generate(self, text, max_tokens=24):
        self.calls += 1
        return self.raw, 0.25, 3


class FakeClaude:
    model = "claude-haiku-5-5"

    def __init__(self, result):
        self.result = result

    def classify(self, text):
        return self.result


USAGE = {"input_tokens": 23, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 11578, "output_tokens": 19}


@pytest.fixture
def client_factory(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr("svf.serving.app.load_dotenv", lambda *a, **k: None)  # keep the real .env out of tests

    def make(local=None, claude=None):
        return TestClient(create_app(classifier=local or FakeLocal(), claude=claude))
    return make


def test_health_and_classify(client_factory):
    with client_factory() as c:
        h = c.get("/health").json()
        assert h["status"] == "ok" and h["threads"] == 2 and h["compare_available"] is False
        r = c.post("/classify", json={"text": "I am still waiting on my card?"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] == "card_arrival" and body["valid"] is True and body["latency_ms"] == 250.0


def test_classify_parses_and_flags_invalid(client_factory):
    with client_factory(local=FakeLocal("Refund not showing up")) as c:
        assert c.post("/classify", json={"text": "x"}).json()["label"] == "Refund_not_showing_up"
    with client_factory(local=FakeLocal("made_up_label")) as c:
        body = c.post("/classify", json={"text": "x"}).json()
        assert body["label"] is None and body["valid"] is False and body["raw"] == "made_up_label"


@pytest.mark.parametrize("payload", [{}, {"text": ""}, {"text": "x" * 1001}, {"txt": "hi"}])
def test_input_validation(client_factory, payload):
    with client_factory() as c:
        assert c.post("/classify", json=payload).status_code == 422


def test_compare_unavailable_without_key(client_factory):
    with client_factory() as c:
        assert c.post("/compare", json={"text": "hi"}).status_code == 503


def test_compare_agree_and_cost(client_factory):
    claude = FakeClaude({"raw": '{"label": "card_arrival"}', "stop_reason": "end_turn", "usage": USAGE})
    local = FakeLocal()
    with client_factory(local=local, claude=claude) as c:
        body = c.post("/compare", json={"text": "Where is my card?"}).json()
    assert body["agree"] is True and local.calls == 1
    assert body["local"]["label"] == body["claude"]["label"] == "card_arrival"
    # Haiku 5.5: 23 x $0.10 + 11,578 x $0.01 + 19 x $0.50 per MTok
    assert body["claude"]["cost_usd"] == pytest.approx((23 * 0.10 + 11578 * 0.01 + 19 * 0.50) / 1e6)


def test_compare_disagree_refusal_and_error(client_factory):
    refusal = FakeClaude({"raw": None, "stop_reason": "refusal", "usage": USAGE})
    with client_factory(claude=refusal) as c:
        body = c.post("/compare", json={"text": "x"}).json()
        assert body["claude"]["label"] is None and body["agree"] is False
    err = FakeClaude({"raw": None, "error": "APIConnectionError: down", "usage": {}})
    with client_factory(claude=err) as c:
        body = c.post("/compare", json={"text": "x"}).json()
        assert body["claude"]["error"].startswith("APIConnectionError") and body["local"]["label"] == "card_arrival"
