"""Confidence signals -> one uncertainty number -> escalate to human review or not.

Signals (each 0 = sure, 1 = unsure), computed from run records only, never from labels:
  self          the model's own confidence / needs_second_look / failed run
  agreement     disagreement across k repeated runs (needs --repeat > 1)
  detector      how close the detector score is to its threshold (log scale)
  disagreement  agent and detector disagree on pass/fail
  combined      mean of the available signals

The escalation threshold is chosen on dev with an explicit cost model, then applied unchanged to test.
choose_threshold() takes labels as an argument; only evaluation code passes them in.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

SELF_UNCERTAINTY = {"high": 0.0, "medium": 0.5, "low": 1.0}
SIGNALS = ("self", "agreement", "detector", "disagreement", "combined")


@dataclass(frozen=True)
class Costs:
    false_pass: float = 10.0  # a defective part ships
    false_fail: float = 1.0  # a good part is scrapped
    review: float = 0.3  # a human looks at it
    correct: float = 0.0


def self_signal(rec: dict) -> float:
    if rec.get("status") != "submitted":
        return 1.0
    u = SELF_UNCERTAINTY.get(rec.get("confidence") or "low", 1.0)
    if rec.get("needs_second_look"):
        u = max(u, 0.75)
    return u


def detector_signal(score: float, threshold: float, width: float = 1.0) -> float:
    """1 at the threshold, falling to 0 when the score is a factor e^width away from it."""
    if not (score > 0 and threshold > 0) or math.isnan(score):
        return 1.0
    return max(0.0, 1.0 - abs(math.log(score / threshold)) / width)


def agreement_signal(runs: Sequence[dict]) -> float | None:
    """Share of runs that disagree with the majority on (decision, defect_type). None with a single run."""
    if len(runs) < 2:
        return None
    keys = [(r.get("decision"), r.get("defect_type") if r.get("decision") == "fail" else "none") for r in runs]
    decisions = [k[0] for k in keys]
    top_decision = Counter(decisions).most_common(1)[0][1] / len(runs)
    top_pair = Counter(keys).most_common(1)[0][1] / len(runs)
    # pass/fail flips count fully, type-only flips count half
    return round(1.0 - (top_decision + top_pair) / 2, 4)


def disagreement_signal(rec: dict) -> float:
    detector_fail = rec["detector_score"] > rec["detector_threshold"]
    return float((rec.get("decision") == "fail") != detector_fail)


def signals_for(runs: Sequence[dict]) -> dict[str, float]:
    """All signals for one image, from its k runs (runs[0] is the primary answer)."""
    first = runs[0]
    out = {
        "self": self_signal(first),
        "detector": detector_signal(first["detector_score"], first["detector_threshold"]),
        "disagreement": disagreement_signal(first),
    }
    agree = agreement_signal(runs)
    if agree is not None:
        out["agreement"] = agree
    out["combined"] = round(sum(out.values()) / len(out), 4)
    return out


def group_runs(records: Sequence[dict]) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    for r in sorted(records, key=lambda r: (r["id"], r["repeat"])):
        by.setdefault(r["id"], []).append(r)
    return by


def apply(uncertainty: Sequence[float], decisions: Sequence[str], threshold: float) -> list[str]:
    """Escalate to 'review' when uncertainty is strictly above the threshold."""
    return ["review" if u > threshold else d for u, d in zip(uncertainty, decisions)]


def expected_cost(labels: Sequence[str], decisions: Sequence[str], costs: Costs = Costs()) -> float:
    total = 0.0
    for label, d in zip(labels, decisions):
        defective = label != "good"
        if d == "review":
            total += costs.review
        elif d == "pass":
            total += costs.false_pass if defective else costs.correct
        else:
            total += costs.correct if defective else costs.false_fail
    return total / len(labels) if labels else float("nan")


def choose_threshold(
    uncertainty: Sequence[float], labels: Sequence[str], decisions: Sequence[str], costs: Costs = Costs()
) -> tuple[float, float]:
    """Threshold minimising expected cost on THIS data (use dev). Ties go to the higher threshold (fewer reviews).

    Candidates: never escalate (inf), escalate everything (-inf), and every observed uncertainty value."""
    candidates = sorted({float("inf"), float("-inf"), *map(float, uncertainty)}, reverse=True)
    best_t, best_c = float("inf"), float("inf")
    for t in candidates:
        c = expected_cost(labels, apply(uncertainty, decisions, t), costs)
        if c < best_c - 1e-12:
            best_t, best_c = t, c
    return best_t, best_c
