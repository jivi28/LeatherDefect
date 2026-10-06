"""The agent loop: call model -> run tools -> feed results back -> until submit_answer.

Reliability rules baked in:
  * tool errors, bad JSON and bad arguments go back to the model as text so it can retry
  * the final answer is a validated pydantic object (any model you pass as `answer_model`)
  * a model that replies with plain text is nudged once, then the run ends as "text_only"
  * on the last step only submit_answer is offered, so the run always ends
  * provider failures end the run with status "provider_error" instead of crashing
  * images returned by tools are shown to the model in a user message after the tool
    results, within a per-run image budget (images cost tokens and money)

Every run ends with a `status`: submitted | text_only | provider_error | step_budget.
Only "submitted" means the model finished through a valid submit_answer call.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, ValidationError

from .llm import LLM, LLMError, Reply
from .schemas import Answer
from .tools import ImageData, ToolRegistry, model_to_spec

SUBMIT = "submit_answer"
SUBMIT_DESCRIPTION = "Submit your final answer. Call this exactly once, when you are done."

EventHandler = Callable[[dict], None]
# (message, confidence_hint: "low" | "none", caveats) -> an instance of answer_model
FallbackFactory = Callable[[str, str, list[str]], BaseModel]


def _default_answer_fallback(message: str, confidence: str, caveats: list[str]) -> BaseModel:
    return Answer(answer=message, confidence=confidence, caveats=caveats)  # type: ignore[arg-type]


@dataclass
class RunResult:
    answer: BaseModel | None  # None when the run did not submit and no fallback factory exists
    ok: bool  # True only if the model finished through a valid submit_answer call
    steps: int
    elapsed_s: float
    usage: dict
    trace: list[dict] = field(default_factory=list)
    status: str = "step_budget"  # submitted | text_only | provider_error | step_budget

    @property
    def tool_calls(self) -> int:
        return sum(1 for e in self.trace if e["type"] == "tool_call" and e["name"] != SUBMIT)


def _assistant_message(reply: Reply) -> dict:
    message: dict = {"role": "assistant", "content": reply.content or ""}
    if reply.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
            for c in reply.tool_calls
        ]
    return message


def _clip(text: str, limit: int = 1500) -> str:
    return text if len(text) <= limit else text[:limit] + f"... [{len(text) - limit} more chars]"


def _image_part(image: ImageData) -> dict:
    return {"type": "image_url", "image_url": {"url": image.data_url()}}


class Agent:
    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        system_prompt: str,
        *,
        max_steps: int = 10,
        answer_model: type[BaseModel] = Answer,
        fallback: FallbackFactory | None = None,
        max_images: int = 12,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self.answer_model = answer_model
        self.max_images = max_images
        if fallback is None and answer_model is Answer:
            fallback = _default_answer_fallback
        self.fallback = fallback
        self._submit_spec = model_to_spec(SUBMIT, SUBMIT_DESCRIPTION, answer_model)

    def run(
        self,
        question: str,
        on_event: EventHandler | None = None,
        images: list[ImageData] | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        trace: list[dict] = []
        usage = {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0}

        def emit(event: dict) -> None:
            trace.append(event)
            if on_event is not None:
                on_event(event)

        user_content: str | list[dict] = question
        images_sent = 0
        if images:
            user_content = [{"type": "text", "text": question}] + [_image_part(i) for i in images]
            images_sent = len(images)
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
        ]
        tool_specs = self.registry.specs()
        answer: BaseModel | None = None
        status = "step_budget"
        fail_text = "I could not produce a reliable answer within the step budget."
        fail_confidence = "none"
        fail_caveats = ["The model never produced a valid submit_answer call."]
        nudged = False
        steps = 0

        for step in range(1, self.max_steps + 1):
            steps = step
            if step == self.max_steps:
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "You are out of steps. Call submit_answer now with your best answer from what you "
                            "have. If it is incomplete, lower your confidence and say what is missing."
                        ),
                    }
                )
                specs = [self._submit_spec]
            else:
                specs = tool_specs + [self._submit_spec]

            emit({"type": "step", "step": step})
            try:
                reply = self.llm.chat(messages, specs)
            except LLMError as exc:
                emit({"type": "error", "message": str(exc)})
                status = "provider_error"
                fail_text = "The language model provider failed, so no answer could be produced."
                fail_confidence, fail_caveats = "none", [str(exc)]
                break

            usage["llm_calls"] += 1
            usage["prompt_tokens"] += reply.usage.get("prompt_tokens", 0)
            usage["completion_tokens"] += reply.usage.get("completion_tokens", 0)
            if reply.content:
                emit({"type": "assistant", "content": _clip(reply.content), "model": reply.model})
            messages.append(_assistant_message(reply))

            if not reply.tool_calls:
                if not nudged:
                    nudged = True
                    emit({"type": "nudge"})
                    messages.append(
                        {
                            "role": "user",
                            "content": "Use the tools to get the facts, then finish by calling submit_answer.",
                        }
                    )
                    continue
                status = "text_only"
                fail_text = reply.content or "No answer produced."
                fail_confidence = "low"
                fail_caveats = ["The model replied without calling submit_answer, so this answer is unstructured."]
                break

            pending_images: list[ImageData] = []
            for call in reply.tool_calls:
                emit({"type": "tool_call", "name": call.name, "args": _clip(call.arguments, 600)})
                if call.name == SUBMIT:
                    try:
                        answer = self.answer_model.model_validate_json(call.arguments or "{}")
                    except ValidationError as exc:
                        problems = "; ".join(
                            f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors()
                        )
                        text = f"Invalid submit_answer arguments: {problems}. Fix them and call submit_answer again."
                        emit({"type": "tool_result", "name": call.name, "ok": False, "content": text, "ms": 0, "images": 0})
                        messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
                        continue
                    status = "submitted"
                    emit({"type": "tool_result", "name": call.name, "ok": True, "content": "Answer recorded.", "ms": 0, "images": 0})
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": "Answer recorded."})
                    break
                outcome = self.registry.execute(call.name, call.arguments)
                emit(
                    {
                        "type": "tool_result",
                        "name": call.name,
                        "ok": outcome.ok,
                        "content": _clip(outcome.content),
                        "ms": outcome.ms,
                        "images": len(outcome.images),  # counts only: never put image bytes in the trace
                    }
                )
                messages.append({"role": "tool", "tool_call_id": call.id, "content": outcome.content})
                pending_images.extend(outcome.images)
            if answer is not None:
                break

            if pending_images:
                room = max(0, self.max_images - images_sent)
                shown = pending_images[:room]
                dropped = len(pending_images) - len(shown)
                images_sent += len(shown)
                captions = "; ".join(f"{n}) {img.label or 'image'}" for n, img in enumerate(shown, 1))
                note = f" {dropped} image(s) were NOT shown because the image budget ({self.max_images}) is used up." if dropped else ""
                if shown:
                    text = f"Images returned by your tool calls, in order: {captions}.{note}"
                    parts = [{"type": "text", "text": text}] + [_image_part(i) for i in shown]
                else:
                    parts = [{"type": "text", "text": f"Image budget ({self.max_images}) is used up; no more images can be shown.{note}"}]
                messages.append({"role": "user", "content": parts})
                emit({"type": "images", "shown": len(shown), "dropped": dropped, "total_sent": images_sent})

        if answer is None and self.fallback is not None:
            answer = self.fallback(fail_text, fail_confidence, fail_caveats)
        result = RunResult(
            answer=answer,
            ok=status == "submitted",
            steps=steps,
            elapsed_s=round(time.perf_counter() - started, 2),
            usage=usage,
            trace=trace,
            status=status,
        )
        emit(
            {
                "type": "answer",
                "answer": answer.model_dump() if answer is not None else None,
                "ok": result.ok,
                "status": status,
            }
        )
        return result
