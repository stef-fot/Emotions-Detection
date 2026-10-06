"""Regression guard: the labelled sets must not get worse."""
import pytest

from evaluation.evaluate import evaluate, load


@pytest.mark.parametrize("split,floor", [
    ("dev", 0.95), ("test1", 0.90), ("holdout", 0.90), ("blind", 0.90),
])
def test_accuracy_floor(split, floor):
    r = evaluate(load(split))
    assert r["accuracy"] >= floor, [(e[0]["text"], e[1]) for e in r["errors"]]
    assert r["accuracy"] > r["baseline"] + 0.4
