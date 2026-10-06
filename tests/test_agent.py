import json

from agentkit import Agent, LLMError, ToolRegistry
from agentkit.llm import FallbackLLM, Reply, ToolCall
from agentkit.testing import ScriptedLLM, text_reply, tool_reply

GOOD_ANSWER = {"answer": "It is 42.", "value": "42", "confidence": "high"}


def make_registry():
    registry = ToolRegistry()

    @registry.register
    def ping() -> str:
        """Returns pong."""
        return "pong"

    @registry.register
    def explode() -> str:
        """Always raises."""
        raise RuntimeError("kaput")

    return registry


def agent_with(script, max_steps=8):
    llm = ScriptedLLM(script)
    return Agent(llm, make_registry(), "system prompt", max_steps=max_steps), llm


def tool_texts(llm_call):
    return [m["content"] for m in llm_call["messages"] if m["role"] == "tool"]


def test_happy_path():
    agent, llm = agent_with([tool_reply("ping", {}), tool_reply("submit_answer", GOOD_ANSWER)])
    result = agent.run("question?")
    assert result.ok and result.answer.value == "42" and result.answer.confidence == "high"
    assert result.steps == 2 and result.tool_calls == 1
    assert tool_texts(llm.calls[1]) == ["pong"]
    assert result.trace[-1]["type"] == "answer"


def test_malformed_tool_json_is_fed_back_and_recovered():
    agent, llm = agent_with([tool_reply("ping", "{broken"), tool_reply("ping", {}), tool_reply("submit_answer", GOOD_ANSWER)])
    result = agent.run("q")
    assert result.ok
    assert "not valid JSON" in tool_texts(llm.calls[1])[0]
    assert tool_texts(llm.calls[2])[-1] == "pong"


def test_tool_exception_is_fed_back_and_recovered():
    agent, llm = agent_with([tool_reply("explode", {}), tool_reply("submit_answer", GOOD_ANSWER)])
    result = agent.run("q")
    assert result.ok
    assert "RuntimeError: kaput" in tool_texts(llm.calls[1])[0]
    failed = [e for e in result.trace if e["type"] == "tool_result" and not e["ok"]]
    assert len(failed) == 1


def test_unknown_tool_is_reported():
    agent, llm = agent_with([tool_reply("nope", {}), tool_reply("submit_answer", GOOD_ANSWER)])
    assert agent.run("q").ok
    assert "Unknown tool 'nope'" in tool_texts(llm.calls[1])[0]


def test_invalid_submit_is_rejected_then_corrected():
    bad = {"answer": "x", "confidence": "very sure"}
    agent, llm = agent_with([tool_reply("submit_answer", bad), tool_reply("submit_answer", GOOD_ANSWER)])
    result = agent.run("q")
    assert result.ok and result.answer.value == "42"
    assert "Invalid submit_answer" in tool_texts(llm.calls[1])[0]


def test_submit_with_broken_json_is_rejected_then_corrected():
    agent, llm = agent_with([tool_reply("submit_answer", "{not json"), tool_reply("submit_answer", GOOD_ANSWER)])
    assert agent.run("q").ok
    assert "Invalid submit_answer" in tool_texts(llm.calls[1])[0]


def test_plain_text_reply_is_nudged_once_then_can_recover():
    agent, llm = agent_with([text_reply("I think it's 42"), tool_reply("submit_answer", GOOD_ANSWER)])
    result = agent.run("q")
    assert result.ok
    assert any(e["type"] == "nudge" for e in result.trace)
    assert "submit_answer" in llm.calls[1]["messages"][-1]["content"]


def test_model_that_never_calls_tools_ends_with_low_confidence():
    agent, _ = agent_with([text_reply("42 probably"), text_reply("42 probably")])
    result = agent.run("q")
    assert not result.ok and result.answer.confidence == "low"
    assert result.answer.answer == "42 probably" and result.answer.caveats


def test_last_step_only_offers_submit_answer():
    agent, llm = agent_with([tool_reply("ping", {}), tool_reply("ping", {}), tool_reply("submit_answer", GOOD_ANSWER)], max_steps=3)
    result = agent.run("q")
    assert result.ok
    assert "ping" in llm.calls[0]["tools"] and "submit_answer" in llm.calls[0]["tools"]
    assert llm.calls[2]["tools"] == ["submit_answer"]
    assert "out of steps" in llm.calls[2]["messages"][-1]["content"]


def test_step_budget_exhausted_gives_honest_none_answer():
    agent, _ = agent_with([tool_reply("ping", {}), tool_reply("ping", {})], max_steps=2)
    result = agent.run("q")
    assert not result.ok and result.answer.confidence == "none" and result.steps == 2


def test_provider_failure_does_not_crash():
    agent, _ = agent_with([LLMError("provider: 429 rate limited")])
    result = agent.run("q")
    assert not result.ok and result.answer.confidence == "none"
    assert "429" in result.answer.caveats[0]
    assert any(e["type"] == "error" for e in result.trace)


def test_parallel_tool_calls_all_get_a_response():
    parallel = Reply(
        tool_calls=[ToolCall("a", "ping", "{}"), ToolCall("b", "ping", "{}")],
        model="fake",
    )
    agent, llm = agent_with([parallel, tool_reply("submit_answer", GOOD_ANSWER)])
    assert agent.run("q").ok
    messages = llm.calls[1]["messages"]
    answered = {m["tool_call_id"] for m in messages if m["role"] == "tool"}
    assert answered == {"a", "b"}


def test_assistant_tool_calls_are_replayed_in_openai_format():
    agent, llm = agent_with([tool_reply("ping", {}), tool_reply("submit_answer", GOOD_ANSWER)])
    agent.run("q")
    assistant = llm.calls[1]["messages"][2]
    assert assistant["role"] == "assistant"
    call = assistant["tool_calls"][0]
    assert call["type"] == "function" and call["function"]["name"] == "ping"
    json.loads(call["function"]["arguments"])


def test_events_are_streamed_to_the_callback():
    seen = []
    agent, _ = agent_with([tool_reply("ping", {}), tool_reply("submit_answer", GOOD_ANSWER)])
    agent.run("q", on_event=seen.append)
    assert [e["type"] for e in seen][:3] == ["step", "tool_call", "tool_result"]
    assert seen[-1]["type"] == "answer"


def test_usage_is_summed():
    first = tool_reply("ping", {})
    first.usage = {"prompt_tokens": 10, "completion_tokens": 5}
    second = tool_reply("submit_answer", GOOD_ANSWER)
    second.usage = {"prompt_tokens": 20, "completion_tokens": 7}
    result = agent_with([first, second])[0].run("q")
    assert result.usage == {"llm_calls": 2, "prompt_tokens": 30, "completion_tokens": 12}


def test_fallback_switches_provider_and_remembers_the_failure():
    primary = ScriptedLLM([LLMError("429"), LLMError("429")])
    backup = ScriptedLLM([text_reply("a"), text_reply("b")])
    llm = FallbackLLM([primary, backup], cooldown=60)
    assert llm.chat([], None).content == "a"
    assert llm.chat([], None).content == "b"
    assert len(primary.calls) == 1  # skipped while cooling down


def test_fallback_raises_when_everyone_fails():
    llm = FallbackLLM([ScriptedLLM([LLMError("one")]), ScriptedLLM([LLMError("two")])])
    try:
        llm.chat([], None)
    except LLMError as exc:
        assert "one" in str(exc) and "two" in str(exc)
    else:
        raise AssertionError("expected LLMError")
