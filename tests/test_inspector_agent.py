"""The inspection agent end to end with a scripted LLM (plumbing only, no network)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from agentkit.llm import LLMError
from agentkit.testing import ScriptedLLM, text_reply, tool_reply
from inspector import agent as A
from inspector.detector import Detector
from inspector.tools import ReferenceBank

ROOT = Path(__file__).resolve().parent.parent


def _texture(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(np.array([0.45, 0.32, 0.22], np.float32) + rng.normal(0, 0.03, (size, size, 1)).astype(np.float32), 0, 1)


def _save(rgb, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return path


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    root = tmp_path_factory.mktemp("cat")
    goods = [_save(_texture(i), root / "train" / "good" / f"{i:03d}.png") for i in range(8)]
    for label in ("scratch", "stain"):
        _save(_texture(90), root / "test" / label / "000.png")
    bad = _texture(42)
    bad[150:180, 40:70] *= 0.3
    photo = _save(bad, root / "test" / "stain" / "001.png")
    tax = ["scratch", "stain"]
    res = A.Resources(Detector.fit(goods), ReferenceBank(goods, n=3), tax, A.make_verdict_model(tax))
    return res, photo


def submit(**overrides):
    v = {"decision": "fail", "defect_type": "stain", "confidence": "high", "region": [0.15, 0.58, 0.28, 0.7],
         "reasons": ["dark spot in B6"], "needs_second_look": False}
    v.update(overrides)
    return tool_reply("submit_answer", v)


def test_verdict_schema_uses_taxonomy_and_is_consistent(world):
    res, _ = world
    V = res.verdict_model
    ok = V(decision="fail", defect_type="stain", confidence="low", region=[0.5, 0.6, 0.1, 0.2], reasons=["x"], needs_second_look=True)
    assert ok.region.as_box() == (0.1, 0.2, 0.5, 0.6)
    for bad in (
        dict(decision="fail", defect_type="crack"),  # not in taxonomy
        dict(decision="pass", defect_type="stain"),  # pass with a defect
        dict(decision="fail", defect_type="none"),  # fail without a defect
        dict(decision="pass", defect_type="none", reasons=[]),  # needs a reason
        dict(decision="pass", defect_type="none", region=[0, 0, 2, 1]),  # out of range
    ):
        fields = dict(confidence="low", reasons=["x"], needs_second_look=False) | bad
        with pytest.raises(ValidationError):
            V(**fields)
    schema = A.make_verdict_model(["a", "b"]).model_json_schema()
    assert schema["properties"]["defect_type"]["enum"] == ["a", "b", "none", "unknown"]


def test_prompt_lists_taxonomy_from_folders():
    p = A.system_prompt(["alpha", "beta"], product="tile")
    assert "alpha, beta" in p and "tile" in p
    assert "ground_truth" not in p


def test_full_agent_run(world):
    res, photo = world
    llm = ScriptedLLM([
        tool_reply("scan_anomalies", {"top_k": 2}),
        tool_reply("zoom", {"box": [0.15, 0.58, 0.28, 0.7]}),
        submit(),
    ])
    run = A.run_one(photo, llm, res)
    assert run.status == "submitted" and run.ok
    assert run.verdict["defect_type"] == "stain" and run.verdict["decision"] == "fail"
    assert run.tool_calls == 2 and run.images_sent == 3  # overview + heatmap + zoom
    assert run.detector_score > run.detector_threshold
    assert run.models == ["fake"]
    first_user = llm.calls[0]["messages"][1]["content"]
    assert any(part["type"] == "image_url" for part in first_user)  # overview attached up front
    assert set(llm.calls[0]["tools"]) == set(A.AGENT_TOOLS) | {"submit_answer"}
    assert "b64" not in str(run.trace) and "base64" not in str(run.trace)  # no image bytes in traces


def test_inconsistent_answer_is_sent_back_and_fixed(world):
    res, photo = world
    llm = ScriptedLLM([submit(decision="pass"), submit()])
    run = A.run_one(photo, llm, res)
    assert run.ok
    feedback = [m for m in llm.calls[1]["messages"] if m["role"] == "tool"][0]["content"]
    assert "requires defect_type 'none'" in feedback


def test_text_only_falls_back_to_fail_safe_review(world):
    res, photo = world
    run = A.run_one(photo, ScriptedLLM([text_reply("looks fine"), text_reply("really fine")]), res)
    assert run.status == "text_only" and not run.ok
    assert run.verdict["decision"] == "fail" and run.verdict["defect_type"] == "unknown"
    assert run.verdict["needs_second_look"] is True and run.verdict["confidence"] == "low"


def test_provider_error_is_reported(world):
    res, photo = world
    run = A.run_one(photo, ScriptedLLM([LLMError("rate limited")]), res)
    assert run.status == "provider_error" and run.verdict["needs_second_look"] is True


def test_oneshot_offers_only_submit(world):
    res, photo = world
    llm = ScriptedLLM([submit(decision="pass", defect_type="none", region=None)])
    run = A.run_one(photo, llm, res, oneshot=True)
    assert run.ok and run.method == "oneshot" and run.tool_calls == 0
    assert llm.calls[0]["tools"] == ["submit_answer"]


def test_step_budget_ends_the_run(world):
    res, photo = world
    llm = ScriptedLLM([tool_reply("measure", {"cell": "B6"})] * 2 + [submit()])
    run = A.run_one(photo, llm, res, max_steps=3)
    assert run.ok and run.steps == 3
    assert llm.calls[-1]["tools"] == ["submit_answer"]


def test_bad_image_does_not_crash(world, tmp_path):
    res, _ = world
    broken = tmp_path / "x.png"
    broken.write_bytes(b"nope")
    run = A.run_one(broken, ScriptedLLM([]), res)
    assert run.status == "bad_image" and not run.ok


AGENT_FACING = ["agent.py", "tools.py", "detector.py", "imaging.py", "policy.py"]


@pytest.mark.parametrize("name", AGENT_FACING)
def test_agent_facing_code_never_touches_labels(name):
    path = ROOT / "inspector" / name
    if not path.exists():
        pytest.skip(f"{name} not written yet")
    src = path.read_text()
    for forbidden in ("ground_truth", "_fake_manifest", "get_split", "load_samples", "allow_test", ".mask"):
        assert forbidden not in src, f"{name} mentions {forbidden}"
