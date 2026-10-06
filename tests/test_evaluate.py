"""Evaluation harness: resumable records, summary fields, test-split guard."""

from pathlib import Path

from inspector import evaluate as E
from inspector.data import Sample


def _samples(tmp_path: Path) -> list[Sample]:
    return [Sample(tmp_path / f"{l}{i}.png", l) for l in ("good", "cut") for i in range(3)]


def _fake_record(decision="fail", conf="high"):
    def fn(sample, k):
        return {"decision": decision, "defect_type": "cut" if decision == "fail" else "none", "confidence": conf,
                "needs_second_look": False, "region": None, "status": "submitted", "detector_score": 5.0 if sample.is_defect else 1.0,
                "detector_threshold": 3.0, "elapsed_s": 0.1, "steps": 2, "tool_calls": 1, "images_sent": 2,
                "usage": {"prompt_tokens": 100}, "models": ["fake"]}
    return fn


def test_evaluate_appends_and_resumes(tmp_path):
    out = tmp_path / "run.jsonl"
    calls = []

    def fn(s, k):
        calls.append((s.id, k))
        return _fake_record()(s, k)

    E.evaluate(_samples(tmp_path), "agent", out, fn, repeat=2, log=lambda *_: None)
    assert len(calls) == 12
    out.write_text(out.read_text() + '{"half-written')  # simulate a crash mid-line
    calls.clear()
    records = E.evaluate(_samples(tmp_path), "agent", out, fn, repeat=2, log=lambda *_: None)
    assert calls == [] and len(records) == 12
    assert {r["repeat"] for r in records} == {0, 1}


def test_summary_has_labelled_metrics(tmp_path):
    out = tmp_path / "run.jsonl"
    records = E.evaluate(_samples(tmp_path), "agent", out, _fake_record(), log=lambda *_: None)
    s = E.summarise(records, {"model": "fake", "split": "dev", "date": "2026-10-07"})
    assert s["n"] == 6 and s["n_defective"] == 3 and s["model"] == "fake"
    assert s["accuracy"]["value"] == 0.5 and s["false_pass_rate"]["value"] == 0.0
    assert s["false_fail_rate"]["value"] == 1.0
    assert s["type_accuracy"]["value"] == 1.0
    assert s["detector_auroc"] == 1.0
    assert s["ece_self_reported"] == 0.45  # says 0.95, right half the time
    assert s["cost_per_image"]["tool_calls"] == 1


def test_test_split_needs_final_flag(capsys):
    assert E.main(["--method", "detector", "--split", "test"]) == 1
    assert "--final" in capsys.readouterr().err


def test_run_name_is_filesystem_safe():
    assert E.run_name("agent", "dev", "ollama:qwen3-vl:8b", 3) == "agent_dev_ollama-qwen3-vl-8b_k3"
    assert "/" not in E.run_name("agent", "dev", "groq:qwen/qwen3.8-27b", 1)
