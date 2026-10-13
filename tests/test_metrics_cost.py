import pytest
from sklearn.metrics import accuracy_score, f1_score

from svf.data import prompts
from svf.data.load_banking77 import load_labels
from svf.eval import cost, metrics

LABELS = ["a", "b", "c"]


def test_metrics_match_sklearn_without_invalid():
    gold = ["a", "a", "b", "b", "c", "c", "c"]
    pred = ["a", "b", "b", "b", "c", "a", "c"]
    m = metrics.classification_metrics(gold, pred, LABELS)
    assert m["accuracy"] == pytest.approx(accuracy_score(gold, pred))
    assert m["macro_f1"] == pytest.approx(f1_score(gold, pred, average="macro"))
    assert m["invalid_rate"] == 0


def test_invalid_counts_as_wrong_and_is_not_a_class():
    gold = ["a", "b", "c", "c"]
    pred = ["a", None, "c", None]
    m = metrics.classification_metrics(gold, pred, LABELS)
    assert m["accuracy"] == 0.5
    assert m["invalid_rate"] == 0.5
    assert m["per_class_f1"]["b"] == 0  # b never predicted
    assert m["per_class_f1"]["c"] == pytest.approx(2 / 3)  # precision 1, recall 0.5
    assert set(m["per_class_f1"]) == set(LABELS)


def test_confused_pairs_and_latency():
    pairs = metrics.confused_pairs(["a", "a", "b", "c"], ["b", "b", "b", None])
    assert pairs[0] == {"gold": "a", "pred": "b", "count": 2}
    assert pairs[1]["pred"] == metrics.INVALID
    lat = metrics.latency_stats([1.0, 2.0, 3.0, 4.0])
    assert lat["p50_s"] == 2.5 and lat["n"] == 4
    assert metrics.latency_stats([]) == {}


def test_mcnemar():
    a = [True] * 10 + [True] * 8 + [False] * 2
    b = [True] * 10 + [False] * 8 + [True] * 2
    r = metrics.mcnemar(a, b)
    assert (r["both"], r["only_a"], r["only_b"]) == (10, 8, 2)
    assert r["p_value"] == pytest.approx(0.109375)  # exact binomial, 10 discordant pairs, 8 vs 2
    assert metrics.mcnemar([True, False], [True, False])["p_value"] == 1.0  # no discordant pairs


PRICING = {"batch_discount": 0.5, "gpu": {"hourly_usd": 0.36},
           "claude": {"m": {"input": 2.0, "cache_write_5m": 2.5, "cache_write_1h": 4.0, "cache_read": 0.1, "output": 10.0}}}


def test_claude_cost_each_component():
    mtok = 1_000_000
    assert cost.claude_cost({"input_tokens": mtok}, "m", PRICING) == 2.0
    assert cost.claude_cost({"cache_creation_input_tokens": mtok}, "m", PRICING) == 2.5
    assert cost.claude_cost({"cache_creation_input_tokens": mtok}, "m", PRICING, cache_ttl="1h") == 4.0
    assert cost.claude_cost({"cache_read_input_tokens": mtok}, "m", PRICING) == pytest.approx(0.1)
    assert cost.claude_cost({"output_tokens": mtok}, "m", PRICING) == 10.0
    assert cost.claude_cost({"input_tokens": mtok, "output_tokens": mtok}, "m", PRICING, batch=True) == 6.0
    assert cost.claude_cost({}, "m", PRICING) == 0


def test_sum_usage_and_gpu_cost():
    rows = [{"usage": {"input_tokens": 3, "output_tokens": 5}}, {"usage": {}}, {"usage": {"input_tokens": 1}}]
    assert cost.sum_usage(rows) == {"input_tokens": 4, "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0, "output_tokens": 5}
    assert cost.gpu_cost(3600, PRICING) == 0.36


def test_break_even_pay_per_use():
    # $1 training, GPU $0.001/pred, Claude $0.003/pred -> saves $0.002/pred -> 500 predictions to pay back
    assert cost.break_even_pay_per_use(1.0, 0.001, 0.003) == pytest.approx(500)
    assert cost.break_even_pay_per_use(1.0, 0.003, 0.003) is None  # equal cost: never pays back
    assert cost.break_even_pay_per_use(1.0, 0.004, 0.003) is None  # GPU dearer: never
    assert cost.break_even_pay_per_use(0.0, 0.001, 0.003) == 0


def test_break_even_always_on():
    assert cost.break_even_always_on(250.0, 0.001) == pytest.approx(250_000)
    assert cost.break_even_always_on(250.0, 0.001, train_usd_per_month=10.0) == pytest.approx(260_000)
    assert cost.break_even_always_on(250.0, 0.001, capacity_per_month=100_000) is None  # one GPU can't serve it
    assert cost.break_even_always_on(250.0, 0.0) is None


def test_real_pricing_file_loads():
    p = cost.load_pricing()
    assert p["claude"]["claude-sonnet-5-5"]["cache_read"] == 0.10


def test_prompts_are_stable_and_complete():
    labels = load_labels()
    s1, s2 = prompts.system_prompt(labels), prompts.system_prompt(labels)
    assert s1 == s2  # byte-stable, so the cache prefix never changes
    assert all(f"- {lab['name']}: " in s1 for lab in labels)
    fs = prompts.system_prompt(labels, [{"text": " hi \n", "label": "card_arrival"}])
    assert fs.startswith(s1) and fs.endswith("Message: hi\nIntent: card_arrival")
    schema = prompts.label_schema(labels)
    assert schema["properties"]["label"]["enum"][0] == "activate_my_card"
    assert len(schema["properties"]["label"]["enum"]) == 77 and schema["additionalProperties"] is False
