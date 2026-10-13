import pytest

from svf.data.load_banking77 import load_labels
from svf.eval.parse import LabelParser, normalise


@pytest.fixture(scope="module")
def parser():
    return LabelParser(load_labels())


@pytest.mark.parametrize("raw, label, method", [
    ('{"label": "card_arrival"}', "card_arrival", "json"),
    ("card_arrival", "card_arrival", "exact"),
    ("  Card_Arrival \n", "card_arrival", "exact"),
    ("card arrival", "card_arrival", "exact"),
    ("card-arrival.", "card_arrival", "exact"),
    ("`card_arrival`", "card_arrival", "exact"),
    ("Intent: card_arrival", "card_arrival", "exact"),
    ("The intent is card_arrival", "card_arrival", "exact"),
    ("refund not showing up", "Refund_not_showing_up", "exact"),
    ("reverted_card_payment", "reverted_card_payment?", "exact"),
    ("reverted_card_payment?", "reverted_card_payment?", "exact"),
    ("card_arrival\nBecause the customer is waiting.", "card_arrival", "exact"),
    ("I think this is card_arrival because they are waiting.", "card_arrival", "search"),
])
def test_valid(parser, raw, label, method):
    assert parser.parse(raw) == (label, method)


@pytest.mark.parametrize("raw", [
    None, "", "   ", "not_a_label", '{"label": "not_a_label"}',
    "either card_arrival or card_delivery_estimate",  # two labels: ambiguous
    "card",
])
def test_invalid(parser, raw):
    assert parser.parse(raw) == (None, "invalid")


def test_search_does_not_match_inside_longer_labels(parser):
    # exchange_rate is a substring of card_payment_wrong_exchange_rate; only the longer one is present
    assert parser.parse("Looks like card_payment_wrong_exchange_rate to me") == ("card_payment_wrong_exchange_rate", "search")


def test_every_label_round_trips(parser):
    for lab in load_labels():
        assert parser.parse(lab["name"])[0] == lab["name"]
        assert parser.parse(lab["name"].replace("_", " ").upper())[0] == lab["name"]


def test_normalise():
    assert normalise(" Top-up limits? ") == "top_up_limits"
