#!/usr/bin/env python
"""Two cheap live checks that your model can do what the inspector needs. Costs a few cents or nothing.

    python scripts/smoke_llm.py                         # uses LLM_PROVIDERS from .env
    python scripts/smoke_llm.py --model gemini          # test one provider (or provider:model) on its own

Stage 1  vision input   : can it see an image you attach to the question?
Stage 2  tool + image   : can it call a tool and then see the image that tool returned?

Not every model, especially free ones, passes both. Run this before you pick a model for the
inspector, and again on every model you plan to use as a fallback.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402
from pydantic import BaseModel  # noqa: E402

from agentkit import Agent, ToolRegistry, ToolResult, image_from_bytes  # noqa: E402
from agentkit.llm import LLM  # noqa: E402

COLOURS = {"red": (220, 30, 30), "green": (30, 170, 60), "blue": (30, 70, 220), "yellow": (240, 220, 30)}


class Colour(BaseModel):
    colour: Literal["red", "green", "blue", "yellow"]


@dataclass
class StageResult:
    name: str
    ok: bool
    detail: str
    seconds: float


def solid_png(rgb: tuple[int, int, int], size: int = 96) -> bytes:
    """A grey frame with a big square of one colour in the middle."""
    img = Image.new("RGB", (size, size), (128, 128, 128))
    pad = size // 4
    img.paste(Image.new("RGB", (size - 2 * pad, size - 2 * pad), rgb), (pad, pad))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def stage_vision_input(llm: LLM, colour: str = "blue") -> StageResult:
    started = time.perf_counter()
    agent = Agent(
        llm,
        ToolRegistry(),
        "You are a careful visual assistant. Look at the image, then call submit_answer.",
        max_steps=3,
        answer_model=Colour,
    )
    result = agent.run(
        "The image shows a grey frame with a coloured square in the middle. What colour is the square?",
        images=[image_from_bytes(solid_png(COLOURS[colour]), label="test image")],
    )
    took = round(time.perf_counter() - started, 1)
    return _judge("vision input", result, colour, took, tool_expected=False)


def stage_tool_image(llm: LLM, colour: str = "green") -> StageResult:
    started = time.perf_counter()
    registry = ToolRegistry()

    @registry.register
    def get_swatch() -> ToolResult:
        """Returns a picture of a coloured square. Look at the attached image to see the colour."""
        return ToolResult("Swatch image attached.", [image_from_bytes(solid_png(COLOURS[colour]), label="swatch")])

    agent = Agent(
        llm,
        registry,
        "You are a careful visual assistant. Use the tools to get the picture, look at it, then call submit_answer.",
        max_steps=4,
        answer_model=Colour,
    )
    result = agent.run("Call get_swatch, look at the image it returns, and tell me the colour of the square.")
    took = round(time.perf_counter() - started, 1)
    return _judge("tool-returned image", result, colour, took, tool_expected=True)


def _judge(name: str, result, expected: str, took: float, tool_expected: bool) -> StageResult:
    if result.status == "provider_error":
        reason = next((e["message"] for e in result.trace if e["type"] == "error"), "unknown")
        return StageResult(name, False, f"provider error: {reason[:300]}", took)
    if tool_expected and result.tool_calls == 0:
        return StageResult(name, False, "the model never called the tool (weak tool calling)", took)
    if result.status != "submitted" or result.answer is None:
        return StageResult(name, False, f"no valid submit_answer (status={result.status})", took)
    if result.answer.colour != expected:
        return StageResult(name, False, f"answered '{result.answer.colour}', expected '{expected}': it likely cannot see images", took)
    return StageResult(name, True, f"saw '{expected}' correctly in {result.steps} step(s)", took)


def run_smoke(llm: LLM) -> list[StageResult]:
    return [stage_vision_input(llm), stage_tool_image(llm)]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", help="provider or provider:model, overriding LLM_PROVIDERS")
    a = p.parse_args(argv)
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    if a.model:
        os.environ["LLM_PROVIDERS"] = a.model
    from agentkit.llm import ConfigError, from_env

    try:
        llm = from_env()
    except ConfigError as exc:
        print(f"[FAIL] {exc}")
        return 1
    results = run_smoke(llm)
    for r in results:
        print(f"[{'ok' if r.ok else 'FAIL'}] {r.name}: {r.detail} ({r.seconds}s)")
    if all(r.ok for r in results):
        print("\nThis model can drive the inspector.")
        return 0
    if results[0].ok:
        print("\nIt can see images but not through tool results. Try another model, or ask Claude Code to attach images in the user turn.")
    else:
        print("\nThis model cannot be used for the inspector. Pick another one (a vision-capable model with tool calling).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
