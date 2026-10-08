"""Demo server, offline: the inspection streams live events (scripted LLM, synthetic resources)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agentkit.testing import ScriptedLLM, text_reply, tool_reply
from inspector import agent as A
from inspector.app import server
from inspector.detector import Detector
from inspector.tools import ReferenceBank

SUBMIT = {"decision": "fail", "defect_type": "cut", "confidence": "high", "region": [0.25, 0.25, 0.5, 0.5],
          "reasons": ["slit"], "needs_second_look": False}


@pytest.fixture
def client(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    goods = []
    for i in range(6):
        p = tmp_path / f"g{i}.png"
        Image.fromarray((np.clip(0.4 + rng.normal(0, 0.03, (128, 128, 3)), 0, 1) * 255).astype(np.uint8)).save(p)
        goods.append(p)
    tax = ["cut"]
    monkeypatch.setitem(server._state, "res", A.Resources(Detector.fit(goods), ReferenceBank(goods, n=2), tax, A.make_verdict_model(tax)))
    monkeypatch.setattr(server, "REPORT", tmp_path / "missing.json")
    return TestClient(server.app), goods


def _script(monkeypatch, items):
    llm = ScriptedLLM(items)
    monkeypatch.setitem(server._state, "llm", llm)
    return llm


def _events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_index_and_samples(client):
    c, _ = client
    assert "Leather Inspector" in c.get("/").text
    assert c.get("/api/samples").json()["policy"] is None


def test_photo_rejects_traversal(client):
    c, _ = client
    assert c.get("/api/photo/../../.env").status_code == 404


def test_stream_shows_each_step_live(client, monkeypatch):
    c, goods = client
    llm = _script(monkeypatch, [
        tool_reply("scan_anomalies", {"top_k": 2}, content="Scanning the whole photo first."),
        tool_reply("zoom", {"cell": "C3"}, content="Checking C3 up close."),
        tool_reply("submit_answer", SUBMIT, content="It is a cut."),
    ])
    r = c.post("/api/stream/upload", files={"file": ("x.png", goods[0].read_bytes(), "image/png")})
    assert r.status_code == 200 and r.headers["content-type"].startswith("application/x-ndjson")
    ev = _events(r)
    kinds = [e["type"] for e in ev]
    assert kinds[0] == "start" and ev[0]["overview"].startswith("data:image/png")
    assert kinds[-1] == "done" and ev[-1]["final"] == "fail" and ev[-1]["verdict"]["defect_type"] == "cut"
    assert [e["text"] for e in ev if e["type"] == "thought"] == ["Scanning the whole photo first.", "Checking C3 up close.", "It is a cut."]
    zoom_call = next(e for e in ev if e["type"] == "tool_call" and e["name"] == "zoom")
    assert zoom_call["box"] == [0.25, 0.25, 0.375, 0.375]  # C3, resolved before the result arrives
    scan = next(e for e in ev if e["type"] == "tool_result" and e["name"] == "scan_anomalies")
    assert len(scan["boxes"]) == 2 and scan["images"][0].startswith("data:image/png")
    zoom = next(e for e in ev if e["type"] == "tool_result" and e["name"] == "zoom")
    assert zoom["images"] and kinds.index("tool_call") < kinds.index("tool_result")
    assert "submitting" in kinds
    assert "Narrate your work" in llm.calls[0]["messages"][0]["content"]  # UI runs narrate


def test_stream_sample_and_bad_tool_args(client, monkeypatch, tmp_path):
    c, goods = client
    root = tmp_path / "data"
    (root / "test" / "cut").mkdir(parents=True)
    (root / "test" / "cut" / "000.png").write_bytes(goods[1].read_bytes())
    monkeypatch.setattr(server.data, "DEFAULT_ROOT", root)
    _script(monkeypatch, [tool_reply("zoom", {"cell": "Z99"}), tool_reply("submit_answer", SUBMIT)])
    ev = _events(c.post("/api/stream/sample", json={"path": "test/cut/000.png"}))
    bad = next(e for e in ev if e["type"] == "tool_result")
    assert bad["ok"] is False and bad["images"] == [] and "A1" in bad["text"]
    assert next(e for e in ev if e["type"] == "tool_call")["box"] is None
    assert ev[-1]["type"] == "done"
    assert c.post("/api/stream/sample", json={"path": "../../.env"}).status_code == 404


def test_stream_reports_text_only_runs(client, monkeypatch):
    c, goods = client
    _script(monkeypatch, [text_reply("looks fine"), text_reply("still fine")])
    ev = _events(c.post("/api/stream/upload", files={"file": ("x.png", goods[0].read_bytes(), "image/png")}))
    assert any(e["type"] == "nudge" for e in ev)
    assert ev[-1]["type"] == "done" and ev[-1]["status"] == "text_only"


def test_unreadable_upload_ends_with_error(client, monkeypatch):
    c, _ = client
    _script(monkeypatch, [])
    ev = _events(c.post("/api/stream/upload", files={"file": ("x.png", b"not a png", "image/png")}))
    assert ev == [ev[0]] and ev[0]["type"] == "error"


def test_upload_rejects_other_files(client):
    c, _ = client
    assert c.post("/api/stream/upload", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
