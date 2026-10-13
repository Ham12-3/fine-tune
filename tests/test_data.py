from collections import Counter

import pytest

from svf.data import load_banking77 as b77


def toy(n_labels=5, per_label=(30, 41, 52, 63, 74)):
    """Imbalanced toy records shaped like the real ones; ids in shuffled order on purpose."""
    rows = [
        {"id": f"train-{lab:02d}{i:03d}", "text": f"text {lab} {i}", "label": f"l{lab}", "label_id": lab}
        for lab in range(n_labels) for i in range(per_label[lab])
    ]
    return rows[::-1]


def test_labels_file_has_77_unique_described_labels():
    labels = b77.load_labels()
    assert len(labels) == 77
    assert len({lab["name"] for lab in labels}) == 77
    assert [lab["id"] for lab in labels] == list(range(77))
    assert all(lab["description"].strip() for lab in labels)
    # canonical ids from the HF ClassLabel: a few anchors
    names = [lab["name"] for lab in labels]
    assert names[0] == "activate_my_card" and names[11] == "card_arrival" and names[76].startswith("wrong_exchange")
    assert "Refund_not_showing_up" in names and "reverted_card_payment?" in names


def test_stratified_split_is_disjoint_stratified_and_deterministic():
    rows = toy()
    rest, held = b77.stratified_split(rows, 0.1, seed=42)
    assert len(rest) + len(held) == len(rows)
    assert not {r["id"] for r in rest} & {r["id"] for r in held}
    counts = Counter(r["label_id"] for r in held)
    assert set(counts) == set(range(5))
    for lab, n in counts.items():
        assert abs(n - 0.1 * (30 + 11 * lab)) <= 1
    # input order must not matter, only the seed
    assert b77.stratified_split(rows[::-1], 0.1, seed=42) == (rest, held)
    assert b77.stratified_split(rows, 0.1, seed=7) != (rest, held)


def test_fixed_size_sample():
    _, held = b77.stratified_split(toy(), 50, seed=42)
    assert len(held) == 50


def test_few_shot_pool_per_label_from_input_only():
    rows = toy()
    pool = b77.few_shot_pool(rows, per_label=3, seed=42)
    assert Counter(r["label_id"] for r in pool) == {lab: 3 for lab in range(5)}
    assert {r["id"] for r in pool} <= {r["id"] for r in rows}
    assert pool == b77.few_shot_pool(rows[::-1], per_label=3, seed=42)


def test_nested_subsets_are_nested_and_stratified():
    rows = toy()
    subsets = b77.nested_subsets(rows, [0.1, 0.25, 0.5, 1.0], seed=42)
    ids = {f: {r["id"] for r in s} for f, s in subsets.items()}
    assert ids[0.1] <= ids[0.25] <= ids[0.5] <= ids[1.0] == {r["id"] for r in rows}
    small = Counter(r["label_id"] for r in subsets[0.1])
    assert small == {0: 3, 1: 5, 2: 6, 3: 7, 4: 8}  # ceil(0.1 * n), and 0.1 * 30 must give 3, not 4


def test_jsonl_round_trip_and_stable_bytes(tmp_path):
    rows = b77.sort_by_id(toy())
    data = b77.to_jsonl(rows)
    assert data == b77.to_jsonl([dict(reversed(list(r.items()))) for r in rows])  # key order irrelevant
    path = tmp_path / "x.jsonl"
    path.write_bytes(data)
    assert b77.read_jsonl(path) == rows


def test_read_csv_rejects_unknown_labels(tmp_path):
    labels = [{"id": 0, "name": "card_arrival", "description": "d"}]
    path = tmp_path / "t.csv"
    path.write_text('text,category\n"where is my card?",card_arrival\n', encoding="utf-8")
    assert b77.read_csv(path, "test", labels) == [
        {"id": "test-00000", "text": "where is my card?", "label": "card_arrival", "label_id": 0}
    ]
    path.write_text('text,category\n"hi",not_a_label\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown label"):
        b77.read_csv(path, "test", labels)
