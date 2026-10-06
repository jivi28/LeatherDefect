"""Escalation policy: signals, cost model, threshold choice. Hand-checkable numbers."""

import math

import pytest

from inspector import policy as P


def rec(decision="fail", conf="high", status="submitted", second=False, score=10.0, thr=3.0, dtype="cut", rid="a", k=0):
    return {"id": rid, "repeat": k, "decision": decision, "defect_type": dtype if decision == "fail" else "none",
            "confidence": conf, "status": status, "needs_second_look": second, "detector_score": score, "detector_threshold": thr}


def test_self_signal():
    assert P.self_signal(rec(conf="high")) == 0.0
    assert P.self_signal(rec(conf="medium")) == 0.5
    assert P.self_signal(rec(conf="high", second=True)) == 0.75
    assert P.self_signal(rec(status="text_only")) == 1.0


def test_detector_signal_peaks_at_threshold():
    assert P.detector_signal(3.0, 3.0) == 1.0
    assert P.detector_signal(3.0 * math.e, 3.0) == pytest.approx(0.0)
    assert P.detector_signal(3.0 / math.e, 3.0) == pytest.approx(0.0)
    assert 0 < P.detector_signal(4.0, 3.0) < 1
    assert P.detector_signal(float("nan"), 3.0) == 1.0


def test_agreement_signal():
    assert P.agreement_signal([rec()]) is None
    assert P.agreement_signal([rec(), rec(), rec()]) == 0.0
    # one of three flips pass/fail: decision agreement 2/3, pair agreement 2/3
    assert P.agreement_signal([rec(), rec(), rec(decision="pass")]) == pytest.approx(1 / 3, abs=1e-3)
    # same decision, one type differs: 1 - (1 + 2/3) / 2
    assert P.agreement_signal([rec(), rec(), rec(dtype="poke")]) == pytest.approx(1 / 6, abs=1e-3)


def test_disagreement_and_combined():
    assert P.disagreement_signal(rec(decision="pass", score=10, thr=3)) == 1.0
    assert P.disagreement_signal(rec(decision="fail", score=10, thr=3)) == 0.0
    s = P.signals_for([rec(conf="low", score=3.0, thr=3.0, decision="pass")])
    assert s["self"] == 1.0 and s["detector"] == 1.0 and s["disagreement"] == 0.0  # 3.0 is not > 3.0
    assert "agreement" not in s and s["combined"] == pytest.approx(2 / 3, abs=1e-3)
    assert "agreement" in P.signals_for([rec(), rec(k=1)])


def test_expected_cost():
    labels = ["good", "cut", "cut", "good"]
    assert P.expected_cost(labels, ["pass", "fail", "pass", "fail"]) == pytest.approx((0 + 0 + 10 + 1) / 4)
    assert P.expected_cost(labels, ["review"] * 4) == pytest.approx(0.3)


def test_choose_threshold_escalates_the_risky_ones():
    labels = ["good", "cut", "cut", "good"]
    decisions = ["pass", "fail", "pass", "pass"]  # the passed cut is the costly mistake
    u = [0.1, 0.2, 0.9, 0.3]
    t, c = P.choose_threshold(u, labels, decisions)
    assert P.apply(u, decisions, t) == ["pass", "fail", "review", "pass"]
    assert c == pytest.approx(0.3 / 4)


def test_choose_threshold_never_escalates_when_perfect():
    t, c = P.choose_threshold([0.5, 0.5], ["good", "cut"], ["pass", "fail"])
    assert t == float("inf") and c == 0.0


def test_useless_signal_cannot_beat_reviewing_errors_it_cannot_find():
    # the wrong answer has the LOWEST uncertainty: escalating it means escalating everything
    labels, decisions, u = ["cut", "good", "good"], ["pass", "pass", "pass"], [0.0, 0.5, 0.9]
    t, c = P.choose_threshold(u, labels, decisions)
    assert c == pytest.approx(0.3) and P.apply(u, decisions, t) == ["review"] * 3


def test_group_runs():
    g = P.group_runs([rec(rid="b", k=1), rec(rid="a"), rec(rid="b", k=0)])
    assert list(g) == ["a", "b"] and [r["repeat"] for r in g["b"]] == [0, 1]
