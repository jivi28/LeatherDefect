"""The inspection agent: Verdict schema, system prompt, build_agent(), run_one(), and the one-shot baseline.

    python -m inspector.agent data/leather/test/cut/000.png            # agent, live trace
    python -m inspector.agent data/leather/test/cut/000.png --oneshot  # baseline: overview only, no tools

Agent-facing: the taxonomy comes from folder names (data.taxonomy); labels and masks are never read here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, create_model, model_validator

from agentkit import Agent
from agentkit.llm import LLM, Reply

from . import data
from . import imaging as im
from .detector import Detector, get_detector
from .tools import OVERVIEW_SIDE, Inspection, ReferenceBank, build_tools

MAX_STEPS = 8
MAX_IMAGES = int(os.environ.get("INSPECTOR_MAX_IMAGES", "10"))
AGENT_TOOLS = ("scan_anomalies", "zoom", "compare_reference", "measure")  # the overview is attached up front


class Region(BaseModel):
    x0: float = Field(ge=0, le=1)
    y0: float = Field(ge=0, le=1)
    x1: float = Field(ge=0, le=1)
    y1: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def _accept_list_and_sort(cls, v):
        if isinstance(v, (list, tuple)) and len(v) == 4:
            v = dict(zip(("x0", "y0", "x1", "y1"), v))
        if isinstance(v, dict) and all(k in v for k in ("x0", "y0", "x1", "y1")):
            try:
                x0, x1 = sorted((float(v["x0"]), float(v["x1"])))
                y0, y1 = sorted((float(v["y0"]), float(v["y1"])))
                v = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
            except (TypeError, ValueError):
                pass
        return v

    def as_box(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


class Verdict(BaseModel):
    """Base schema. make_verdict_model() narrows defect_type to the dataset's taxonomy."""

    decision: Literal["pass", "fail"] = Field(description="pass = acceptable leather, fail = has a defect.")
    defect_type: str = Field(description="The defect name, 'none' when the decision is pass, or 'unknown'.")
    confidence: Literal["low", "medium", "high"] = Field(
        description="high: the close-up clearly shows it (or clearly shows nothing). medium: visible but ambiguous. low: guessing."
    )
    region: Region | None = Field(
        default=None, description="Normalised box x0,y0,x1,y1 (0..1) of the main suspicious area. Omit for a clean pass."
    )
    reasons: list[str] = Field(description="1-4 short observations that justify the decision.", min_length=1, max_length=4)
    needs_second_look: bool = Field(description="true if you would want a human to check this photo.")

    @model_validator(mode="after")
    def _consistent(self):
        if self.decision == "pass" and self.defect_type not in ("none",):
            raise ValueError("decision 'pass' requires defect_type 'none'. If there is a defect, use decision 'fail'.")
        if self.decision == "fail" and self.defect_type == "none":
            raise ValueError("decision 'fail' needs a defect_type (a defect name or 'unknown'), not 'none'.")
        return self


def make_verdict_model(taxonomy: list[str]) -> type[Verdict]:
    names = tuple(taxonomy) + ("none", "unknown")
    return create_model(
        "Verdict",
        __base__=Verdict,
        defect_type=(Literal[names], Field(description="One of the defect names, 'none' for a pass, or 'unknown'.")),  # type: ignore[valid-type]
    )


def system_prompt(taxonomy: list[str], product: str = "leather") -> str:
    names = ", ".join(taxonomy) if taxonomy else "(none listed)"
    return f"""You are a quality inspector for {product}. You get one photo of a {product} surface and decide whether it
passes or fails inspection, which defect it has, and where.

Defect types for this product: {names}.
Use exactly one of these names, "none" for a pass, or "unknown" for a clear defect that matches none of them.

Procedure:
1. Look at the attached overview photo. It has an 8x8 grid: columns A-H left to right, rows 1-8 top to bottom.
2. Call scan_anomalies once. It is a statistical detector fitted only on good {product}. It is good at finding WHERE
   the texture is unusual, but it cannot name defects and can flag harmless variation.
3. Zoom into the top region if it is above normal, and into anything you noticed yourself. If you are unsure whether it
   is a defect or normal texture, call compare_reference. Use measure when numbers would settle it (colour shift,
   darkness, line-like shape).
4. Decide, then call submit_answer.

What counts as evidence:
- A defect is a local, visible change against the surrounding texture: an opening or slit, a hole or puncture mark,
  a crease line, foreign material or a glossy/stained spot, or a patch of different colour.
- Uniform texture, grain that looks like the reference, and gentle lighting changes across the whole photo are normal.
- A high detector score with a clean-looking close-up is not enough: compare with the reference first.
- Something clearly visible in the close-up counts even if the detector score is modest.

Answer rules:
- region: the box of the main defect in 0..1 coordinates (reuse the box from scan_anomalies, or read the x=/y= tick
  labels on the zoom). Omit it for a clean pass.
- confidence: high only if the close-up makes it obvious; medium if a defect is visible but its type is ambiguous;
  low if you are guessing. Say so honestly: low confidence is useful, wrong high confidence is costly.
- needs_second_look: true whenever a human should check.
- Be efficient: usually 3 to 5 tool calls. Never ask for pixel coordinates."""


ONESHOT_PROMPT_SUFFIX = """

For this run you have NO tools except submit_answer. Look at the overview photo carefully and answer directly."""

# UI only: makes the agent's reasoning visible. Off for evaluation runs, so measured results keep the original prompt.
NARRATE_SUFFIX = """

Narrate your work for the person watching: before every tool call, write one short sentence (max 20 words) saying
what you are checking and why. Before submit_answer, write one sentence with your conclusion."""


class _Recording:
    """Wraps an LLM to remember which model label actually answered (fallback chains can switch)."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm
        self.models: list[str] = []

    def chat(self, messages, tools=None, tool_choice="auto") -> Reply:
        reply = self.llm.chat(messages, tools, tool_choice)
        if reply.model and reply.model not in self.models:
            self.models.append(reply.model)
        return reply


@dataclass
class VerdictRun:
    path: str
    method: str  # agent | oneshot
    verdict: dict | None
    status: str  # submitted | text_only | provider_error | step_budget
    ok: bool
    steps: int
    tool_calls: int
    images_sent: int
    elapsed_s: float
    usage: dict
    models: list[str]
    detector_score: float
    detector_threshold: float
    trace: list[dict] = field(default_factory=list)
    started_at: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Resources:
    """Things shared across many runs: the fitted detector, reference images and taxonomy."""

    detector: Detector
    references: ReferenceBank
    taxonomy: list[str]
    verdict_model: type[Verdict]

    @classmethod
    def load(cls, root: Path = data.DEFAULT_ROOT) -> "Resources":
        good = data.train_good(root)
        tax = data.taxonomy(root)
        return cls(get_detector(good), ReferenceBank(good), tax, make_verdict_model(tax))


def _fallback_factory(verdict_model: type[Verdict]):
    def make(message: str, confidence: str, caveats: list[str]) -> Verdict:
        # No valid answer: fail-safe (never auto-pass) and ask for a human. The policy also escalates non-submitted runs.
        return verdict_model(
            decision="fail", defect_type="unknown", confidence="low", region=None,
            reasons=[("No valid answer: " + (caveats[0] if caveats else message))[:300]], needs_second_look=True,
        )
    return make


def overview_image(ins: Inspection):
    return im.to_image_data(im.draw_grid(im.downscale(ins.image, OVERVIEW_SIDE)), "overview with grid")


def build_agent(llm: LLM, image_path: Path, res: Resources, *, oneshot: bool = False, narrate: bool = False,
                max_steps: int = MAX_STEPS, max_images: int = MAX_IMAGES) -> tuple[Agent, Inspection]:
    ins = Inspection(Path(image_path), res.detector, res.references)
    registry = build_tools(ins, include=() if oneshot else AGENT_TOOLS)
    prompt = system_prompt(res.taxonomy) + (ONESHOT_PROMPT_SUFFIX if oneshot else "") + (NARRATE_SUFFIX if narrate else "")
    agent = Agent(
        llm, registry, prompt,
        max_steps=2 if oneshot else max_steps,
        answer_model=res.verdict_model,
        fallback=_fallback_factory(res.verdict_model),
        max_images=max_images,
    )
    return agent, ins


QUESTION = "Inspect this photo. The overview with the grid is attached."


def run_one(image_path: Path | str, llm: LLM, res: Resources, *, oneshot: bool = False, on_event=None,
            narrate: bool = False, max_steps: int = MAX_STEPS, max_images: int = MAX_IMAGES) -> VerdictRun:
    rec = _Recording(llm)
    agent, ins = build_agent(rec, Path(image_path), res, oneshot=oneshot, narrate=narrate,
                             max_steps=max_steps, max_images=max_images)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        first = [overview_image(ins)]
    except Exception as exc:  # noqa: BLE001 - unreadable photo: report, do not crash an eval loop
        return VerdictRun(str(image_path), "oneshot" if oneshot else "agent", None, "bad_image", False, 0, 0, 0, 0.0,
                          {}, [], float("nan"), res.detector.threshold, [{"type": "error", "message": str(exc)}], started)
    result = agent.run(QUESTION, on_event=on_event, images=first)
    images_sent = max([e.get("total_sent", 0) for e in result.trace if e["type"] == "images"], default=len(first))
    return VerdictRun(
        path=str(image_path),
        method="oneshot" if oneshot else "agent",
        verdict=result.answer.model_dump() if result.answer is not None else None,
        status=result.status,
        ok=result.ok,
        steps=result.steps,
        tool_calls=result.tool_calls,
        images_sent=images_sent,
        elapsed_s=result.elapsed_s,
        usage=result.usage,
        models=rec.models,
        detector_score=ins.detection.score,
        detector_threshold=res.detector.threshold,
        trace=result.trace,
        started_at=started,
    )


def _print_event(e: dict) -> None:
    t = e["type"]
    if t == "step":
        print(f"\n-- step {e['step']}")
    elif t == "assistant":
        print(f"   model: {e['content'][:300]}")
    elif t == "tool_call":
        print(f"   -> {e['name']}({e['args'][:200]})")
    elif t == "tool_result":
        print(f"   <- {'ok' if e['ok'] else 'ERR'} {e['content'][:300]}" + (f" [+{e['images']} image]" if e.get("images") else ""))
    elif t in ("nudge", "error", "images"):
        print(f"   [{t}] {json.dumps({k: v for k, v in e.items() if k != 'type'})[:200]}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--oneshot", action="store_true", help="baseline: overview only, no tools")
    p.add_argument("--model", help="provider or provider:model, overriding LLM_PROVIDERS")
    a = p.parse_args(argv)
    from dotenv import load_dotenv

    load_dotenv(data.ROOT / ".env")
    if a.model:
        os.environ["LLM_PROVIDERS"] = a.model
    from agentkit.llm import ConfigError, from_env

    try:
        llm = from_env()
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    run = run_one(a.image, llm, Resources.load(), oneshot=a.oneshot, on_event=_print_event)
    print("\n== verdict", json.dumps(run.verdict, indent=2))
    print(f"status={run.status} steps={run.steps} tools={run.tool_calls} images={run.images_sent} "
          f"time={run.elapsed_s}s models={run.models} detector={run.detector_score:.2f}/{run.detector_threshold:.2f}")
    return 0 if run.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
