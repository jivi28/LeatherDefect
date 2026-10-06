"""Run a method over a split and write labelled, resumable results.

    python -m inspector.evaluate --method detector --split dev
    python -m inspector.evaluate --method oneshot  --split dev
    python -m inspector.evaluate --method agent    --split dev [--repeat 3] [--limit 10]
    python -m inspector.evaluate --method agent    --split test --final     # once per reported configuration

Per-image records go to results/runs/<run>.jsonl as they finish (re-running the same command resumes).
The summary goes to results/<run>.json with model label, date, split, n and bootstrap 95% intervals.
This module (with data.py and metrics.py) is allowed to read labels and masks.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path
from statistics import mean
from typing import Callable

from . import data
from . import metrics as M
from .agent import Resources, VerdictRun, run_one

RESULTS = data.ROOT / "results"
CONF_PROB = {"low": 0.55, "medium": 0.75, "high": 0.95}  # stated confidence -> probability, for ECE


def run_name(method: str, split: str, model: str, repeat: int) -> str:
    safe = model.replace("/", "_").replace(":", "-") if model else "nomodel"
    return f"{method}_{split}_{safe}" + (f"_k{repeat}" if repeat > 1 else "")


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a half-written last line from an interrupted run
    return out


def detector_record(sample: data.Sample, res: Resources) -> dict:
    t = time.perf_counter()
    d = res.detector.detect(sample.path)
    box = list(d.regions[0].box) if d.regions else None
    fail = d.score > res.detector.threshold
    return {
        "decision": "fail" if fail else "pass",
        "defect_type": "unknown" if fail else "none",
        "confidence": None,
        "needs_second_look": False,
        "region": box,
        "status": "submitted",
        "detector_score": d.score,
        "detector_threshold": res.detector.threshold,
        "elapsed_s": round(time.perf_counter() - t, 3),
        "steps": 0, "tool_calls": 0, "images_sent": 0, "usage": {}, "models": ["detector"],
    }


def llm_record(run: VerdictRun) -> dict:
    v = run.verdict or {}
    region = v.get("region")
    box = [region["x0"], region["y0"], region["x1"], region["y1"]] if isinstance(region, dict) else None
    return {
        "decision": v.get("decision", "fail"),
        "defect_type": v.get("defect_type", "unknown"),
        "confidence": v.get("confidence"),
        "needs_second_look": v.get("needs_second_look", True),
        "reasons": v.get("reasons", []),
        "region": box,
        "status": run.status,
        "detector_score": run.detector_score,
        "detector_threshold": run.detector_threshold,
        "elapsed_s": run.elapsed_s,
        "steps": run.steps, "tool_calls": run.tool_calls, "images_sent": run.images_sent,
        "usage": run.usage, "models": run.models,
        "trace": [e for e in run.trace if e["type"] in ("tool_call", "tool_result", "assistant", "nudge", "error")],
    }


def evaluate(
    samples: list[data.Sample],
    method: str,
    out_path: Path,
    record_fn: Callable[[data.Sample, int], dict],
    repeat: int = 1,
    rpm: float = 0.0,
    log: Callable[[str], None] = print,
) -> list[dict]:
    """Run record_fn on every (sample, repeat) not already in out_path, appending as we go."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["repeat"]) for r in load_records(out_path)}
    todo = [(s, k) for s in samples for k in range(repeat) if (s.id, k) not in done]
    log(f"{method}: {len(done)} done, {len(todo)} to run -> {out_path.relative_to(data.ROOT) if out_path.is_relative_to(data.ROOT) else out_path}")
    gap = 60.0 / rpm if rpm > 0 else 0.0
    for i, (s, k) in enumerate(todo, 1):
        started = time.perf_counter()
        rec = record_fn(s, k)
        rec.update({"id": s.id, "label": s.label, "path": str(s.path), "method": method, "repeat": k,
                    "loc_hit": M.box_hits_mask(rec.get("region"), s.mask)})
        with out_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        mark = "ok " if (rec["decision"] == "fail") == s.is_defect else "XX "
        log(f"  [{i}/{len(todo)}] {mark}{s.id:12s} -> {rec['decision']:4s} {str(rec['defect_type']):8s} "
            f"conf={rec.get('confidence')} status={rec['status']} {rec['elapsed_s']}s")
        wait = gap - (time.perf_counter() - started)
        if wait > 0 and i < len(todo):
            time.sleep(wait)
    return load_records(out_path)


def _ci(stat, *cols) -> dict:
    point, lo, hi = M.bootstrap_ci(stat, *cols, n=2000, seed=0)
    r = lambda v: None if v != v else round(v, 4)  # noqa: E731
    return {"value": r(point), "ci95": [r(lo), r(hi)]}


def summarise(records: list[dict], meta: dict) -> dict:
    """Metrics over the first repeat (k=0) of each image; agreement over repeats is computed in policy.py."""
    first = sorted((r for r in records if r["repeat"] == 0), key=lambda r: r["id"])
    labels = [r["label"] for r in first]
    decisions = [r["decision"] for r in first]
    types = [r["defect_type"] if r["decision"] == "fail" else "none" for r in first]
    out = dict(meta)
    out["n"] = len(first)
    out["n_defective"] = sum(l != "good" for l in labels)
    if not first:
        return out
    out["accuracy"] = _ci(M.accuracy, labels, decisions)
    out["false_pass_rate"] = _ci(M.false_pass_rate, labels, decisions)
    out["false_fail_rate"] = _ci(M.false_fail_rate, labels, decisions)
    out["per_defect_recall"] = M.per_defect_recall(labels, decisions)
    if any(t not in ("unknown", "none") for t in types):
        out["type_accuracy"] = _ci(M.type_accuracy, labels, types)
        out["confusion"] = M.confusion(labels, types)
    hits = [r["loc_hit"] for r in first if r["loc_hit"] is not None and r["decision"] == "fail"]
    out["localisation_hit_rate"] = round(sum(hits) / len(hits), 4) if hits else None
    out["detector_auroc"] = round(M.auroc([r["detector_score"] for r in first], [l != "good" for l in labels]), 4)
    out["valid_answer_rate"] = round(mean(r["status"] == "submitted" for r in first), 4)
    conf = [r["confidence"] for r in first]
    if all(c in CONF_PROB for c in conf):
        correct = [(d == "fail") == (l != "good") for l, d in zip(labels, decisions)]
        out["ece_self_reported"] = round(M.ece([CONF_PROB[c] for c in conf], correct), 4)
        out["reliability"] = M.reliability_table([CONF_PROB[c] for c in conf], correct)
        out["confidence_counts"] = {c: conf.count(c) for c in CONF_PROB}
    out["cost_per_image"] = {
        "seconds": round(mean(r["elapsed_s"] for r in first), 2),
        "tool_calls": round(mean(r["tool_calls"] for r in first), 2),
        "images": round(mean(r["images_sent"] for r in first), 2),
        "prompt_tokens": round(mean(r["usage"].get("prompt_tokens", 0) for r in first)),
        "completion_tokens": round(mean(r["usage"].get("completion_tokens", 0) for r in first)),
    }
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", choices=("detector", "oneshot", "agent"), required=True)
    p.add_argument("--split", choices=data.SPLITS, default="dev")
    p.add_argument("--final", action="store_true", help="required for --split test: the one reported run")
    p.add_argument("--limit", type=int, default=0, help="only the first N images (stratified order), for quick checks")
    p.add_argument("--repeat", type=int, default=1, help="runs per image (k>1 feeds the agreement signal)")
    p.add_argument("--rpm", type=float, default=0.0, help="max images per minute (free cloud tiers)")
    p.add_argument("--model", help="provider or provider:model, overriding LLM_PROVIDERS")
    p.add_argument("--name", help="run name (default from method/split/model)")
    a = p.parse_args(argv)

    from dotenv import load_dotenv

    load_dotenv(data.ROOT / ".env")
    if a.model:
        os.environ["LLM_PROVIDERS"] = a.model
    if a.split == "test" and not a.final:
        print("error: the test split is for the final reported run only. Add --final if that is what this is.", file=sys.stderr)
        return 1
    samples = data.get_split(a.split, allow_test=a.final)
    if a.limit:
        # interleave labels so a small limit still covers every class
        by: dict[str, list[data.Sample]] = {}
        for s in samples:
            by.setdefault(s.label, []).append(s)
        mixed = [grp[i] for i in range(max(map(len, by.values()))) for grp in by.values() if i < len(grp)]
        samples = mixed[: a.limit]
    res = Resources.load()

    model_label = "detector"
    if a.method == "detector":
        record_fn = lambda s, k: detector_record(s, res)  # noqa: E731
    else:
        from agentkit.llm import ConfigError, from_env

        try:
            llm = from_env()
        except ConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        model_label = os.environ.get("LLM_PROVIDERS", "?").split(",")[0]
        oneshot = a.method == "oneshot"
        record_fn = lambda s, k: llm_record(run_one(s.path, llm, res, oneshot=oneshot))  # noqa: E731

    name = a.name or run_name(a.method, a.split, model_label, a.repeat) + (f"_n{a.limit}" if a.limit else "")
    records = evaluate(samples, a.method, RESULTS / "runs" / f"{name}.jsonl", record_fn, a.repeat, a.rpm)
    keep = {s.id for s in samples}
    records = [r for r in records if r["id"] in keep]
    meta = {
        "run": name, "method": a.method, "split": a.split, "model": model_label, "date": date.today().isoformat(),
        "repeat": a.repeat, "data": "FAKE (plumbing check only)" if data.is_fake() else "real MVTec AD leather",
        "detector_threshold": round(res.detector.threshold, 4),
    }
    summary = summarise(records, meta)
    (RESULTS / f"{name}.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("reliability", "confusion")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
