"""Metrics on tiny hand-checkable examples."""

import math

import pytest
from PIL import Image

from inspector import metrics as M

LABELS = ["good", "good", "cut", "cut", "poke", "glue"]
DECISIONS = ["pass", "fail", "fail", "pass", "review", "fail"]


def test_accuracy_ignores_reviews():
    # kept: good/pass ok, good/fail wrong, cut/fail ok, cut/pass wrong, glue/fail ok -> 3/5
    assert M.accuracy(LABELS, DECISIONS) == pytest.approx(0.6)
    assert math.isnan(M.accuracy(["cut"], ["review"]))


def test_false_pass_and_false_fail():
    assert M.false_pass_rate(LABELS, DECISIONS) == pytest.approx(1 / 4)  # one passed cut of 4 defects
    assert M.false_fail_rate(LABELS, DECISIONS) == pytest.approx(1 / 2)
    assert M.review_rate(DECISIONS) == pytest.approx(1 / 6)


def test_false_pass_is_not_one_minus_accuracy():
    labels = ["good"] * 9 + ["cut"]
    decisions = ["pass"] * 10
    assert M.accuracy(labels, decisions) == pytest.approx(0.9)
    assert M.false_pass_rate(labels, decisions) == 1.0


def test_per_defect_recall_and_type_accuracy():
    assert M.per_defect_recall(LABELS, DECISIONS) == {"cut": 0.5, "glue": 1.0, "poke": 1.0}
    assert M.type_accuracy(["good", "cut", "poke"], ["none", "cut", "cut"]) == 0.5
    assert M.confusion(["cut", "cut", "good"], ["cut", "fold", "none"]) == {"cut": {"cut": 1, "fold": 1}, "good": {"none": 1}}


def test_auroc():
    assert M.auroc([0.1, 0.2, 0.8, 0.9], [False, False, True, True]) == 1.0
    assert M.auroc([0.9, 0.8, 0.2, 0.1], [False, False, True, True]) == 0.0
    assert M.auroc([0.5, 0.5], [False, True]) == 0.5
    assert math.isnan(M.auroc([0.1, 0.2], [False, False]))


def test_ece_and_reliability():
    # always says 0.9, right half the time -> ECE 0.4
    conf, ok = [0.9] * 4, [True, False, True, False]
    assert M.ece(conf, ok) == pytest.approx(0.4)
    rows = M.reliability_table(conf, ok)
    assert rows == [{"lo": 0.8, "hi": 1.0, "n": 4, "confidence": pytest.approx(0.9), "accuracy": 0.5}]
    assert M.ece([1.0, 0.0], [True, False]) == pytest.approx(0.0)


def test_risk_coverage_drops_when_uncertainty_is_informative():
    curve = M.risk_coverage([0.1, 0.2, 0.9, 0.8], [False, False, True, True])
    assert curve[0] == (1.0, 0.5)
    assert curve[2] == (0.5, 0.0)


def test_bootstrap_ci_brackets_point():
    labels = ["cut"] * 10 + ["good"] * 10
    decisions = ["fail"] * 7 + ["pass"] * 3 + ["pass"] * 10
    point, lo, hi = M.bootstrap_ci(M.false_pass_rate, labels, decisions, n=300)
    assert point == pytest.approx(0.3) and lo <= point <= hi and hi - lo > 0.05


def test_box_hits_mask(tmp_path):
    mask = Image.new("L", (100, 100), 0)
    mask.paste(255, (60, 60, 80, 80))
    path = tmp_path / "m.png"
    mask.save(path)
    assert M.box_hits_mask((0.6, 0.6, 0.8, 0.8), path) is True
    assert M.box_hits_mask((0.0, 0.0, 0.2, 0.2), path) is False
    assert M.box_hits_mask((0.57, 0.57, 0.59, 0.59), path) is True  # centre 0.02 outside: within the margin
    assert M.box_hits_mask((0.53, 0.53, 0.57, 0.57), path) is False  # centre 0.05 outside: a miss
    assert M.box_hits_mask(None, path) is False
    assert M.box_hits_mask((0, 0, 1, 1), None) is None
