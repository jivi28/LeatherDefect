"""Demo server endpoints, offline (scripted LLM, synthetic resources)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from agentkit.testing import ScriptedLLM, tool_reply
from inspector import agent as A
from inspector.app import server
from inspector.detector import Detector
from inspector.tools import ReferenceBank


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
    monkeypatch.setitem(server._state, "llm", ScriptedLLM([
        tool_reply("scan_anomalies", {}),
        tool_reply("submit_answer", {"decision": "pass", "defect_type": "none", "confidence": "high", "reasons": ["clean"], "needs_second_look": False}),
    ]))
    monkeypatch.setattr(server, "REPORT", tmp_path / "missing.json")
    return TestClient(server.app), goods


def test_index_and_samples(client):
    c, _ = client
    assert "Leather Inspector" in c.get("/").text
    assert c.get("/api/samples").json()["policy"] is None


def test_photo_rejects_traversal(client):
    c, _ = client
    assert c.get("/api/photo/../../.env").status_code == 404


def test_upload_runs_agent_and_returns_tool_images(client):
    c, goods = client
    r = c.post("/api/inspect/upload", files={"file": ("x.png", goods[0].read_bytes(), "image/png")})
    assert r.status_code == 200
    d = r.json()
    assert d["final"] == "pass" and d["status"] == "submitted"
    assert d["steps"][0]["tool"] == "scan_anomalies" and d["steps"][0]["images"][0].startswith("data:image/png")


def test_upload_rejects_other_files(client):
    c, _ = client
    assert c.post("/api/inspect/upload", files={"file": ("x.txt", b"hi", "text/plain")}).status_code == 400
