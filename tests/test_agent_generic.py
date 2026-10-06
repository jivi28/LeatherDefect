"""Tests for the kit's additions to the agent core: custom answer models, fallbacks,
statuses and image delivery. All offline, using ScriptedLLM."""

import json
from typing import Literal

import pytest
from pydantic import BaseModel

from agentkit import Agent, ImageData, ToolRegistry, ToolResult, image_from_bytes
from agentkit.testing import ScriptedLLM, text_reply, tool_reply

PNG = b"\x89PNG\r\n\x1a\nfake-bytes"


class Verdict(BaseModel):
    decision: Literal["pass", "fail", "review"]
    reason: str


def make_registry(n_images=1):
    registry = ToolRegistry()

    @registry.register
    def look(region: str = "full") -> ToolResult:
        """Returns a crop of the part."""
        return ToolResult(
            f"crop {region}",
            [image_from_bytes(PNG + str(i).encode(), label=f"{region} #{i}") for i in range(n_images)],
        )

    @registry.register
    def plain() -> str:
        """No image."""
        return "just text"

    return registry


def user_messages(call):
    return [m for m in call["messages"] if m["role"] == "user"]


def image_parts(message):
    content = message["content"]
    if isinstance(content, str):
        return []
    return [p for p in content if p["type"] == "image_url"]


def test_custom_answer_model_is_used_and_validated():
    llm = ScriptedLLM(
        [
            tool_reply("submit_answer", {"decision": "maybe", "reason": "x"}),  # invalid literal
            tool_reply("submit_answer", {"decision": "fail", "reason": "cut"}),
        ]
    )
    agent = Agent(llm, make_registry(), "sys", answer_model=Verdict)
    result = agent.run("q")
    assert result.ok and result.status == "submitted"
    assert isinstance(result.answer, Verdict) and result.answer.decision == "fail"
    # the invalid first attempt was bounced back as a tool message mentioning the field
    bounced = [m["content"] for m in llm.calls[1]["messages"] if m["role"] == "tool"]
    assert bounced and "decision" in bounced[0]
    # the submit spec carries the custom schema
    assert "decision" in json.dumps(llm.calls[0]["tools"]) or llm.calls[0]["tools"][-1] == "submit_answer"


def test_custom_model_without_fallback_returns_none_answer_on_failure():
    llm = ScriptedLLM([text_reply("hmm"), text_reply("still hmm")])
    result = Agent(llm, make_registry(), "sys", answer_model=Verdict).run("q")
    assert not result.ok and result.answer is None and result.status == "text_only"


def test_custom_fallback_factory_builds_review_verdict():
    def fb(message, confidence, caveats):
        return Verdict(decision="review", reason=f"agent failed: {message}")

    llm = ScriptedLLM([text_reply("hmm"), text_reply("still hmm")])
    result = Agent(llm, make_registry(), "sys", answer_model=Verdict, fallback=fb).run("q")
    assert not result.ok and result.answer.decision == "review"


def test_provider_error_status():
    from agentkit import LLMError

    llm = ScriptedLLM([LLMError("boom")])
    agent = Agent(llm, make_registry(), "sys", answer_model=Verdict, fallback=lambda m, c, k: Verdict(decision="review", reason=m))
    result = agent.run("q")
    assert result.status == "provider_error" and result.answer.decision == "review"


def test_step_budget_status_when_submit_is_never_valid():
    bad = lambda: tool_reply("submit_answer", {"decision": "nope", "reason": "x"})  # noqa: E731
    llm = ScriptedLLM([bad(), bad(), bad()])
    result = Agent(llm, make_registry(), "sys", answer_model=Verdict, max_steps=3).run("q")
    assert result.status == "step_budget" and not result.ok
    # last step only offers submit_answer
    assert llm.calls[-1]["tools"] == ["submit_answer"]


def test_tool_images_arrive_in_user_message_after_tool_messages():
    llm = ScriptedLLM([tool_reply("look", {"region": "top"}), tool_reply("submit_answer", {"decision": "pass", "reason": "clean"})])
    result = Agent(llm, make_registry(), "sys", answer_model=Verdict).run("inspect")
    assert result.ok
    second = llm.calls[1]["messages"]
    roles = [m["role"] for m in second]
    # system, user, assistant(tool call), tool, user(images)
    assert roles[-2:] == ["tool", "user"]
    assert second[-2]["content"] == "crop top"  # tool message stays text-only
    parts = second[-1]["content"]
    assert parts[0]["type"] == "text" and "top #0" in parts[0]["text"]
    assert len(image_parts(second[-1])) == 1
    assert image_parts(second[-1])[0]["image_url"]["url"].startswith("data:image/png;base64,")


def test_parallel_tool_calls_all_get_tool_messages_before_one_image_message():
    from agentkit.llm import Reply, ToolCall

    reply = Reply(
        content=None,
        tool_calls=[
            ToolCall(id="a", name="look", arguments='{"region": "left"}'),
            ToolCall(id="b", name="plain", arguments="{}"),
            ToolCall(id="c", name="look", arguments='{"region": "right"}'),
        ],
        model="fake",
    )
    llm = ScriptedLLM([reply, tool_reply("submit_answer", {"decision": "pass", "reason": "ok"})])
    Agent(llm, make_registry(), "sys", answer_model=Verdict).run("q")
    msgs = llm.calls[1]["messages"]
    roles = [m["role"] for m in msgs]
    assert roles[-4:] == ["tool", "tool", "tool", "user"]
    assert [m["tool_call_id"] for m in msgs if m["role"] == "tool"] == ["a", "b", "c"]
    assert len(image_parts(msgs[-1])) == 2
    caption = msgs[-1]["content"][0]["text"]
    assert "left #0" in caption and "right #0" in caption


def test_image_budget_drops_extras_and_tells_the_model():
    llm = ScriptedLLM([tool_reply("look", {}), tool_reply("look", {}), tool_reply("submit_answer", {"decision": "review", "reason": "r"})])
    events = []
    agent = Agent(llm, make_registry(n_images=3), "sys", answer_model=Verdict, max_images=4)
    agent.run("q", on_event=events.append)
    image_events = [e for e in events if e["type"] == "images"]
    assert [(e["shown"], e["dropped"]) for e in image_events] == [(3, 0), (1, 2)]
    last_images_msg = llm.calls[2]["messages"][-1]
    assert len(image_parts(last_images_msg)) == 1
    assert "NOT shown" in last_images_msg["content"][0]["text"]


def test_budget_exhausted_sends_text_only_notice():
    llm = ScriptedLLM([tool_reply("look", {}), tool_reply("submit_answer", {"decision": "review", "reason": "r"})])
    agent = Agent(llm, make_registry(n_images=2), "sys", answer_model=Verdict, max_images=0)
    agent.run("q")
    msg = llm.calls[1]["messages"][-1]
    assert image_parts(msg) == [] and "budget" in msg["content"][0]["text"].lower()


def test_initial_images_go_in_first_user_message_and_count_toward_budget():
    llm = ScriptedLLM([tool_reply("look", {}), tool_reply("submit_answer", {"decision": "pass", "reason": "ok"})])
    first = image_from_bytes(PNG, label="photo")
    agent = Agent(llm, make_registry(n_images=2), "sys", answer_model=Verdict, max_images=2)
    agent.run("inspect this", images=[first])
    first_user = user_messages(llm.calls[0])[0]
    assert first_user["content"][0] == {"type": "text", "text": "inspect this"}
    assert len(image_parts(first_user)) == 1
    # budget 2, one used by the initial photo -> only 1 of the tool's 2 images shown
    assert len(image_parts(llm.calls[1]["messages"][-1])) == 1


def test_trace_never_contains_image_bytes():
    llm = ScriptedLLM([tool_reply("look", {}), tool_reply("submit_answer", {"decision": "pass", "reason": "ok"})])
    result = Agent(llm, make_registry(), "sys", answer_model=Verdict).run("q")
    blob = json.dumps(result.trace)
    assert "base64" not in blob and "iVBOR" not in blob
    results = [e for e in result.trace if e["type"] == "tool_result" and e["name"] == "look"]
    assert results[0]["images"] == 1


def test_tool_without_images_adds_no_user_message():
    llm = ScriptedLLM([tool_reply("plain", {}), tool_reply("submit_answer", {"decision": "pass", "reason": "ok"})])
    Agent(llm, make_registry(), "sys", answer_model=Verdict).run("q")
    assert [m["role"] for m in llm.calls[1]["messages"]][-1] == "tool"


def test_registry_unwraps_toolresult_and_truncates_text():
    registry = ToolRegistry()

    @registry.register
    def big() -> ToolResult:
        """Big."""
        return ToolResult("x" * 10000, [ImageData("image/png", "AAAA", "l")])

    outcome = registry.execute("big", "{}")
    assert outcome.ok and len(outcome.images) == 1
    assert "truncated" in outcome.content and len(outcome.content) < 7000


def test_load_image_rejects_unknown_suffix(tmp_path):
    from agentkit import load_image
    from agentkit.tools import ToolError

    p = tmp_path / "a.bmp"
    p.write_bytes(b"x")
    with pytest.raises(ToolError):
        load_image(p)
    q = tmp_path / "a.PNG"
    q.write_bytes(PNG)
    assert load_image(q).mime == "image/png"


def test_default_answer_still_has_default_fallback():
    llm = ScriptedLLM([text_reply("a"), text_reply("b")])
    result = Agent(llm, make_registry(), "sys").run("q")
    assert result.answer is not None and result.answer.confidence == "low" and result.status == "text_only"
