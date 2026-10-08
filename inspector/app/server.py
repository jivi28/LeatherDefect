"""Demo UI: pick or upload a photo and watch the agent work live: what it thinks, where it zooms, what it sees.

    .venv/bin/uvicorn inspector.app.server:app --port 8000      then open http://localhost:8000

Inspections stream as NDJSON (one JSON event per line) while the agent runs. The policy threshold comes from
results/report.json (chosen on dev); without it the raw agent decision is shown. The model never sees file names.
"""

from __future__ import annotations

import json
import os
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from agentkit import ToolError
from agentkit.llm import from_env

from .. import data
from .. import policy as P
from ..agent import Resources, VerdictRun, overview_image, run_one
from ..tools import Inspection, build_tools, resolve_region

load_dotenv(data.ROOT / ".env")
HERE = Path(__file__).resolve().parent
UPLOADS = Path(tempfile.gettempdir()) / "leather-inspector-uploads"
UPLOADS.mkdir(exist_ok=True)
REPORT = data.ROOT / "results" / "report.json"
MAX_UPLOAD = 15 * 1024 * 1024
REGION_TOOLS = ("zoom", "compare_reference", "measure")
SUBMIT = "submit_answer"

app = FastAPI(title="Leather Inspector")
_state: dict = {}


def resources() -> Resources:
    if "res" not in _state:
        _state["res"] = Resources.load()
    return _state["res"]


def llm():
    if "llm" not in _state:
        _state["llm"] = from_env()
    return _state["llm"]


def policy_choice() -> dict | None:
    try:
        choice = json.loads(REPORT.read_text())["choice"]
        return choice if choice.get("signal") in P.SIGNALS else None
    except (OSError, ValueError, KeyError):
        return None


def _samples() -> list[str]:
    """Demo photos: a few per folder under data/leather/test (paths relative to data/leather)."""
    root = data.DEFAULT_ROOT / "test"
    out = []
    if root.is_dir():
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            out += [str(p.relative_to(data.DEFAULT_ROOT)) for p in sorted(folder.glob("*.png"))[:3]]
    return out


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (HERE / "index.html").read_text()


@app.get("/api/samples")
def samples() -> dict:
    return {"samples": _samples(), "policy": policy_choice(), "model": os.environ.get("LLM_PROVIDERS", "?")}


def _sample_path(rel: str) -> Path:
    path = (data.DEFAULT_ROOT / rel).resolve()
    if not path.is_relative_to(data.DEFAULT_ROOT.resolve()) or not path.is_file():
        raise HTTPException(404, "unknown sample")
    return path


@app.get("/api/photo/{rel:path}")
def photo(rel: str) -> FileResponse:
    return FileResponse(_sample_path(rel))


def summary(run: VerdictRun) -> dict:
    """Verdict + policy decision for one finished run."""
    v = run.verdict or {}
    rec = {"decision": v.get("decision", "fail"), "defect_type": v.get("defect_type"), "confidence": v.get("confidence"),
           "needs_second_look": v.get("needs_second_look", True), "status": run.status,
           "detector_score": run.detector_score, "detector_threshold": run.detector_threshold}
    signals = P.signals_for([rec])
    choice = policy_choice()
    final = rec["decision"]
    if choice:
        final = P.apply([signals.get(choice["signal"], 1.0)], [rec["decision"]], choice["threshold"])[0]
    return {"verdict": run.verdict, "status": run.status, "final": final, "signals": signals, "policy": choice,
            "seconds": run.elapsed_s, "tool_calls": run.tool_calls, "models": run.models,
            "detector": {"score": round(run.detector_score, 2), "threshold": round(run.detector_threshold, 2)}}


def _region_box(name: str, args: str) -> list[float] | None:
    """The normalised box a region tool will look at, so the UI can highlight it before the result arrives."""
    if name not in REGION_TOOLS:
        return None
    try:
        a = json.loads(args or "{}")
        return list(resolve_region(a.get("cell") or "", a.get("box")))
    except (ToolError, ValueError, TypeError, AttributeError):
        return None


def _scan_boxes(content: str) -> list[list[float]]:
    try:
        return [r["box"] for r in json.loads(content).get("regions", [])]
    except (ValueError, KeyError, TypeError, AttributeError):
        return []


def inspect_events(path: Path) -> Iterator[dict]:
    """Run the agent in a thread and yield UI events as they happen. Always ends with 'done' or 'error'."""
    res = resources()
    started = time.perf_counter()
    # A second Inspection on the same photo replays each tool call to get the images the model saw
    # (tools are deterministic and fast; the trace itself never carries image bytes).
    replay_ins = Inspection(path, res.detector, res.references)
    replay = build_tools(replay_ins)
    try:
        yield {"type": "start", "overview": overview_image(replay_ins).data_url()}
    except Exception as exc:  # noqa: BLE001 - unreadable upload
        yield {"type": "error", "message": f"Could not read the photo: {exc}"}
        return

    events: queue.Queue = queue.Queue()
    done = object()
    holder: dict = {}

    def work() -> None:
        try:
            holder["run"] = run_one(path, llm(), res, narrate=True, on_event=events.put)
        except Exception as exc:  # noqa: BLE001 - report to the page instead of hanging it
            holder["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            events.put(done)

    threading.Thread(target=work, daemon=True).start()
    last_args = "{}"
    while True:
        e = events.get()
        if e is done:
            break
        t = round(time.perf_counter() - started, 1)
        kind = e["type"]
        if kind == "step":
            yield {"type": "step", "step": e["step"], "t": t}
        elif kind == "assistant":
            yield {"type": "thought", "text": e["content"], "t": t}
        elif kind == "tool_call":
            last_args = e["args"]
            if e["name"] == SUBMIT:
                yield {"type": "submitting", "t": t}
            else:
                yield {"type": "tool_call", "name": e["name"], "args": e["args"], "box": _region_box(e["name"], e["args"]), "t": t}
        elif kind == "tool_result":
            if e["name"] == SUBMIT:
                if not e["ok"]:
                    yield {"type": "retry", "text": e["content"], "t": t}
                continue
            out = replay.execute(e["name"], last_args)
            yield {"type": "tool_result", "name": e["name"], "ok": e["ok"], "text": e["content"][:1500],
                   "images": [i.data_url() for i in out.images] if e["ok"] else [],
                   "boxes": _scan_boxes(e["content"]) if e["name"] == "scan_anomalies" else [], "t": t}
        elif kind in ("nudge", "error"):
            yield {"type": kind, "text": e.get("message", "Model replied without using tools; nudged it."), "t": t}
    if "error" in holder:
        yield {"type": "error", "message": holder["error"]}
    else:
        yield {"type": "done", **summary(holder["run"])}


def _ndjson(path: Path) -> StreamingResponse:
    lines = (json.dumps(e) + "\n" for e in inspect_events(path))
    return StreamingResponse(lines, media_type="application/x-ndjson", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.post("/api/stream/sample")
def stream_sample(body: dict) -> StreamingResponse:
    return _ndjson(_sample_path(str(body.get("path", ""))))


@app.post("/api/stream/upload")
async def stream_upload(file: UploadFile = File(...)) -> StreamingResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(400, "upload a .png or .jpg photo")
    blob = await file.read()
    if len(blob) > MAX_UPLOAD:
        raise HTTPException(413, "photo too large (max 15 MB)")
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(blob)
    return _ndjson(path)
