"""Demo UI: upload or pick a photo, watch the agent's steps, see the verdict and the policy's pass/fail/review.

    .venv/bin/uvicorn inspector.app.server:app --port 8000      then open http://localhost:8000

The policy threshold comes from results/report.json (chosen on dev). Without it, everything is shown as the raw
agent decision and the page says so. The model never sees file names, only pixels.
"""

from __future__ import annotations

import json
import os
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from agentkit.llm import from_env

from .. import data
from .. import policy as P
from ..agent import Resources, run_one
from ..tools import Inspection, build_tools

load_dotenv(data.ROOT / ".env")
HERE = Path(__file__).resolve().parent
UPLOADS = Path(tempfile.gettempdir()) / "leather-inspector-uploads"
UPLOADS.mkdir(exist_ok=True)
REPORT = data.ROOT / "results" / "report.json"
MAX_UPLOAD = 15 * 1024 * 1024

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


@app.get("/api/photo/{rel:path}")
def photo(rel: str) -> FileResponse:
    path = (data.DEFAULT_ROOT / rel).resolve()
    if not path.is_relative_to(data.DEFAULT_ROOT.resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


def _inspect(path: Path) -> dict:
    res = resources()
    run = run_one(path, llm(), res)
    # Re-run each tool call (tools are deterministic) to recover the images the model saw.
    tools = build_tools(Inspection(path, res.detector, res.references))
    steps = []
    for e in run.trace:
        if e["type"] == "tool_call" and e["name"] != "submit_answer":
            out = tools.execute(e["name"], e["args"])
            steps.append({"tool": e["name"], "args": e["args"], "result": out.content[:1200],
                          "images": [i.data_url() for i in out.images]})
        elif e["type"] == "assistant":
            steps.append({"tool": None, "text": e["content"]})
    rec = {"decision": (run.verdict or {}).get("decision", "fail"), "defect_type": (run.verdict or {}).get("defect_type"),
           "confidence": (run.verdict or {}).get("confidence"), "needs_second_look": (run.verdict or {}).get("needs_second_look", True),
           "status": run.status, "detector_score": run.detector_score, "detector_threshold": run.detector_threshold}
    signals = P.signals_for([rec])
    choice = policy_choice()
    final = rec["decision"]
    if choice:
        final = P.apply([signals.get(choice["signal"], 1.0)], [rec["decision"]], choice["threshold"])[0]
    return {"verdict": run.verdict, "status": run.status, "final": final, "signals": signals, "policy": choice,
            "steps": steps, "seconds": run.elapsed_s, "tool_calls": run.tool_calls, "models": run.models,
            "detector": {"score": round(run.detector_score, 2), "threshold": round(run.detector_threshold, 2)}}


@app.post("/api/inspect/sample")
def inspect_sample(body: dict) -> dict:
    rel = str(body.get("path", ""))
    path = (data.DEFAULT_ROOT / rel).resolve()
    if not path.is_relative_to(data.DEFAULT_ROOT.resolve()) or not path.is_file():
        raise HTTPException(404, "unknown sample")
    return _inspect(path)


@app.post("/api/inspect/upload")
async def inspect_upload(file: UploadFile = File(...)) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in (".png", ".jpg", ".jpeg"):
        raise HTTPException(400, "upload a .png or .jpg photo")
    blob = await file.read()
    if len(blob) > MAX_UPLOAD:
        raise HTTPException(413, "photo too large (max 15 MB)")
    path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    path.write_bytes(blob)
    return _inspect(path)
