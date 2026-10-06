"""A scripted fake LLM, so the loop, tools and evals can be tested offline and for free."""

from __future__ import annotations

import json
from typing import Callable

from .llm import LLMError, Reply, ToolCall

ScriptItem = Reply | Exception | Callable[[list[dict], list[dict]], Reply]

_counter = 0


def tool_reply(name: str, args: dict | str, content: str | None = None) -> Reply:
    global _counter
    _counter += 1
    arguments = args if isinstance(args, str) else json.dumps(args)
    return Reply(content=content, tool_calls=[ToolCall(id=f"call_{_counter}", name=name, arguments=arguments)], model="fake")


def text_reply(text: str) -> Reply:
    return Reply(content=text, model="fake")


class ScriptedLLM:
    """Plays back `script` one item per chat() call. Items can be a Reply, an exception to raise,
    or a function (messages, tool_specs) -> Reply for replies that depend on earlier tool results."""

    def __init__(self, script: list[ScriptItem], label: str = "fake") -> None:
        self.script = list(script)
        self.label = label
        self.calls: list[dict] = []

    def chat(self, messages: list[dict], tools: list[dict] | None = None, tool_choice: str = "auto") -> Reply:
        names = [t["function"]["name"] for t in (tools or [])]
        self.calls.append({"messages": [dict(m) for m in messages], "tools": names})
        if not self.script:
            raise LLMError(f"{self.label}: script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        if callable(item):
            return item(messages, tools or [])
        return item
