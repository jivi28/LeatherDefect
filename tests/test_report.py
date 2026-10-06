"""Report builder on synthetic records: tables render, threshold comes from dev, plots are written."""

from inspector.report import build_report


def _rec(i, label, decision, conf="high", dtype=None, k=0, score=None, method="agent"):
    return {"id": f"{label}/{i:03d}", "repeat": k, "label": label, "path": f"/x/test/{label}/{i:03d}.png", "method": method,
            "decision": decision, "defect_type": dtype or ("cut" if decision == "fail" else "none"), "confidence": conf,
            "needs_second_look": conf == "low", "status": "submitted", "region": None, "loc_hit": None,
            "detector_score": score if score is not None else (8.0 if label != "good" else 2.0), "detector_threshold": 3.0,
            "elapsed_s": 1.0, "steps": 3, "tool_calls": 2, "images_sent": 3, "usage": {}, "models": ["fake"]}


def _dev():
    recs = [_rec(i, "good", "pass") for i in range(6)] + [_rec(i, "cut", "fail") for i in range(6)]
    recs.append(_rec(7, "cut", "pass", conf="low", score=3.2))  # the dangerous miss, flagged as unsure
    return recs


def test_report_renders_and_policy_catches_the_miss(tmp_path):
    det = [dict(r, method="detector", decision="fail" if r["label"] != "good" else "pass", models=["detector"]) for r in _dev()]
    md, summary = build_report(_dev(), {"detector": det}, out_dir=tmp_path)
    assert "| detector |" in md and "agent (no policy)" in md and "AUROC for errors" in md
    assert summary["choice"]["cost"] < 10 / 13  # escalating the unsure miss beats shipping it
    policy_row = [k for k in summary["dev"] if k.startswith("agent + policy")][0]
    assert summary["dev"][policy_row]["false_pass"][0] == 0.0
    assert (tmp_path / "risk_coverage_dev.png").exists() and (tmp_path / "reliability_dev.png").exists()


def test_report_with_test_split(tmp_path):
    test = [_rec(i, "good", "pass") for i in range(3)] + [_rec(i, "cut", "pass", conf="low", score=3.1) for i in range(2)]
    md, summary = build_report(_dev(), {}, test, out_dir=tmp_path)
    assert "Test split" in md
    assert summary["test"]["agent + policy"]["review"] > 0
