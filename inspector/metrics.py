"""Evaluation metrics. Pure functions over plain lists, plus mask-based localisation.

Conventions:
  label     "good" or a defect name (ground truth)
  decision  "pass" | "fail" | "review" (system output); "review" = escalated to a human
  A false pass is a defective item that got "pass". It is the expensive error.
"""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
from PIL import Image, ImageFilter

GOOD = "good"


def _truth_fail(label: str) -> bool:
    return label != GOOD


def accuracy(labels: Sequence[str], decisions: Sequence[str]) -> float:
    """Share of automatic (non-review) decisions that are right. NaN if nothing was decided."""
    kept = [(l, d) for l, d in zip(labels, decisions) if d != "review"]
    if not kept:
        return float("nan")
    return sum((d == "fail") == _truth_fail(l) for l, d in kept) / len(kept)


def false_pass_rate(labels: Sequence[str], decisions: Sequence[str]) -> float:
    """Defective items that were passed, over ALL defective items (a reviewed defect is not a false pass)."""
    defective = [d for l, d in zip(labels, decisions) if _truth_fail(l)]
    return sum(d == "pass" for d in defective) / len(defective) if defective else float("nan")


def false_fail_rate(labels: Sequence[str], decisions: Sequence[str]) -> float:
    """Good items that were failed, over ALL good items."""
    good = [d for l, d in zip(labels, decisions) if not _truth_fail(l)]
    return sum(d == "fail" for d in good) / len(good) if good else float("nan")


def review_rate(decisions: Sequence[str]) -> float:
    return sum(d == "review" for d in decisions) / len(decisions) if decisions else float("nan")


def per_defect_recall(labels: Sequence[str], decisions: Sequence[str]) -> dict[str, float]:
    """For each defect type: share of its items that were NOT passed (failed or sent to review)."""
    out: dict[str, float] = {}
    for label in sorted(set(labels) - {GOOD}):
        ds = [d for l, d in zip(labels, decisions) if l == label]
        out[label] = sum(d != "pass" for d in ds) / len(ds)
    return out


def type_accuracy(labels: Sequence[str], predicted_types: Sequence[str]) -> float:
    """On defective items only: share where the predicted defect type equals the true one."""
    pairs = [(l, p) for l, p in zip(labels, predicted_types) if _truth_fail(l)]
    return sum(l == p for l, p in pairs) / len(pairs) if pairs else float("nan")


def confusion(labels: Sequence[str], predicted_types: Sequence[str]) -> dict[str, dict[str, int]]:
    """truth -> predicted -> count. Predicted 'none' means the model called it good."""
    table: dict[str, Counter] = {}
    for l, p in zip(labels, predicted_types):
        table.setdefault(l, Counter())[p] += 1
    return {l: dict(c) for l, c in sorted(table.items())}


def auroc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """Probability that a random positive scores above a random negative (ties count half). NaN if one class is empty."""
    s = np.asarray(scores, float)
    y = np.asarray(positives, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def auroc_for_errors(uncertainty: Sequence[float], was_wrong: Sequence[bool]) -> float:
    """How well an uncertainty signal ranks wrong answers above right ones. 0.5 = useless, 1.0 = perfect."""
    return auroc(uncertainty, was_wrong)


def reliability_table(confidence: Sequence[float], correct: Sequence[bool], bins: int = 5) -> list[dict]:
    """Bins of stated confidence vs observed accuracy. Confidence in 0..1."""
    c = np.asarray(confidence, float)
    ok = np.asarray(correct, bool)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (c >= lo) & ((c < hi) if i < bins - 1 else (c <= hi))
        if sel.any():
            rows.append({"lo": round(lo, 2), "hi": round(hi, 2), "n": int(sel.sum()),
                         "confidence": float(c[sel].mean()), "accuracy": float(ok[sel].mean())})
    return rows


def ece(confidence: Sequence[float], correct: Sequence[bool], bins: int = 5) -> float:
    """Expected calibration error: weighted mean |confidence - accuracy| over bins."""
    rows = reliability_table(confidence, correct, bins)
    n = sum(r["n"] for r in rows)
    return float(sum(r["n"] * abs(r["confidence"] - r["accuracy"]) for r in rows) / n) if n else float("nan")


def risk_coverage(uncertainty: Sequence[float], was_wrong: Sequence[bool]) -> list[tuple[float, float]]:
    """Escalate the most uncertain first. Returns (coverage, error rate among kept) from full coverage down."""
    u = np.asarray(uncertainty, float)
    w = np.asarray(was_wrong, bool)
    order = np.argsort(u, kind="stable")  # most certain first
    out = []
    for keep in range(len(u), 0, -1):
        kept = w[order[:keep]]
        out.append((keep / len(u), float(kept.mean())))
    return out


def bootstrap_ci(
    stat: Callable[..., float], *columns: Sequence, n: int = 2000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float, float]:
    """(point, lo, hi) by resampling rows with replacement. NaN resamples are dropped."""
    rows = list(zip(*columns))
    point = stat(*columns)
    rng = random.Random(seed)
    vals = []
    for _ in range(n):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        v = stat(*map(list, zip(*sample)))
        if v == v:  # not NaN
            vals.append(v)
    if not vals:
        return point, float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def box_hits_mask(box: Sequence[float] | None, mask_path: Path | None, margin: float = 0.025) -> bool | None:
    """Localisation hit: the box centre lies inside the ground-truth mask dilated by `margin` (share of width).

    None when there is nothing to score (no mask, i.e. a good image)."""
    if mask_path is None:
        return None
    if box is None:
        return False
    size = 256
    with Image.open(mask_path) as m:
        m = m.convert("L").resize((size, size), Image.NEAREST)
        k = max(3, int(margin * size) * 2 + 1)
        m = m.filter(ImageFilter.MaxFilter(k if k % 2 else k + 1))
        mask = np.asarray(m) > 127
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return bool(mask[min(int(cy * size), size - 1), min(int(cx * size), size - 1)])
