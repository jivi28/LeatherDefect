"""Turn plain Python functions into LLM tools, and run them without ever raising.

The key property: every failure (bad JSON from the model, missing argument, exception
inside the tool) comes back as an error *string* the model can read and recover from.
"""

from __future__ import annotations

import base64
import inspect
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, get_type_hints

from pydantic import BaseModel, ValidationError, create_model

MAX_RESULT_CHARS = 6000
_MIME_BY_SUFFIX = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


class ToolError(Exception):
    """Raise inside a tool for a failure whose message the model should see verbatim."""


class ToolArgError(Exception):
    """The model sent arguments that are not valid JSON or do not match the schema."""


@dataclass(frozen=True)
class ImageData:
    """An image the model can look at. `label` is a short caption shown next to it."""

    mime: str
    b64: str
    label: str = ""

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64}"


def image_from_bytes(data: bytes, mime: str = "image/png", label: str = "") -> ImageData:
    return ImageData(mime=mime, b64=base64.b64encode(data).decode("ascii"), label=label)


def load_image(path: str | Path, label: str = "") -> ImageData:
    path = Path(path)
    mime = _MIME_BY_SUFFIX.get(path.suffix.lower())
    if mime is None:
        raise ToolError(f"Unsupported image type '{path.suffix}'. Use png, jpg or webp.")
    return image_from_bytes(path.read_bytes(), mime, label or path.name)


@dataclass
class ToolResult:
    """Return this from a tool to show the model text AND images (e.g. a zoomed crop).

    Chat APIs generally only accept text in tool messages, so the agent loop delivers the
    images in a user message right after the tool results.
    """

    text: str
    images: list[ImageData] = field(default_factory=list)


@dataclass
class ToolOutcome:
    ok: bool
    content: str
    ms: int
    images: list[ImageData] = field(default_factory=list)


def simplify_schema(node: Any) -> Any:
    """Make a pydantic JSON schema friendlier to strict/free providers.

    Drops 'title' noise, collapses Optional[X] (anyOf [X, null]) to X, drops null defaults.
    """
    if isinstance(node, list):
        return [simplify_schema(x) for x in node]
    if not isinstance(node, dict):
        return node
    out: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key == "properties" and isinstance(value, dict):
            out[key] = {name: simplify_schema(sub) for name, sub in value.items()}
        else:
            out[key] = simplify_schema(value)
    any_of = out.get("anyOf")
    if isinstance(any_of, list) and len(any_of) == 2:
        non_null = [x for x in any_of if x.get("type") != "null"]
        if len(non_null) == 1:
            merged = {k: v for k, v in out.items() if k != "anyOf"}
            merged.update(non_null[0])
            out = merged
    if "default" in out and out["default"] is None:
        del out["default"]
    return out


def model_to_spec(name: str, description: str, model: type[BaseModel]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": simplify_schema(model.model_json_schema()),
        },
    }


class Tool:
    def __init__(self, fn: Callable[..., Any], name: str | None = None, description: str | None = None):
        self.fn = fn
        self.name = name or fn.__name__
        self.description = (description or inspect.getdoc(fn) or self.name).strip()
        hints = get_type_hints(fn)
        fields: dict[str, Any] = {}
        for param in inspect.signature(fn).parameters.values():
            annotation = hints.get(param.name, str)
            default = ... if param.default is inspect.Parameter.empty else param.default
            fields[param.name] = (annotation, default)
        self.args_model = create_model(f"{self.name}_args", **fields)

    def spec(self) -> dict:
        return model_to_spec(self.name, self.description, self.args_model)

    def run(self, raw_args: str | dict | None) -> Any:
        if isinstance(raw_args, str):
            text = raw_args.strip() or "{}"
            try:
                data = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ToolArgError(
                    f"arguments are not valid JSON ({exc.msg} at char {exc.pos}); send a single JSON object"
                ) from exc
        else:
            data = raw_args or {}
        if not isinstance(data, dict):
            raise ToolArgError("arguments must be a JSON object")
        try:
            parsed = self.args_model(**data)
        except ValidationError as exc:
            problems = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'}: {e['msg']}" for e in exc.errors())
            raise ToolArgError(problems) from exc
        return self.fn(**parsed.model_dump())


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, fn: Callable[..., Any] | None = None, *, name: str | None = None):
        """Use as @registry.register or @registry.register(name="x")."""

        def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
            tool = Tool(f, name=name)
            self._tools[tool.name] = tool
            return f

        return decorator(fn) if fn is not None else decorator

    def add(self, fn: Callable[..., Any], *, name: str | None = None) -> None:
        tool = Tool(fn, name=name)
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return sorted(self._tools)

    def specs(self) -> list[dict]:
        return [t.spec() for t in self._tools.values()]

    def execute(self, name: str, raw_args: str | dict | None) -> ToolOutcome:
        start = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - start) * 1000)

        tool = self._tools.get(name)
        if tool is None:
            return ToolOutcome(False, f"Unknown tool '{name}'. Available tools: {', '.join(self.names())}.", elapsed())
        try:
            result = tool.run(raw_args)
        except ToolArgError as exc:
            return ToolOutcome(False, f"Invalid arguments for {name}: {exc}", elapsed())
        except ToolError as exc:
            return ToolOutcome(False, str(exc), elapsed())
        except Exception as exc:  # noqa: BLE001 - tools must never crash the agent loop
            return ToolOutcome(False, f"{type(exc).__name__}: {exc}", elapsed())

        images: list[ImageData] = []
        if isinstance(result, ToolResult):
            images = list(result.images)
            result = result.text
        content = result if isinstance(result, str) else json.dumps(result, default=str, ensure_ascii=False)
        if len(content) > MAX_RESULT_CHARS:
            extra = len(content) - MAX_RESULT_CHARS
            content = content[:MAX_RESULT_CHARS] + f"\n...[truncated {extra} chars; narrow your query]"
        return ToolOutcome(True, content, elapsed(), images)
