# Leather Inspector: a vision agent that knows when it is unsure

Practice project for the EHL Finale (agentic build under time pressure). You build a vision agent that
inspects photos of leather, decides **pass / fail / review**, names the defect, shows where, and escalates
to a human when it is unsure. The part that matters most: **does "low confidence" actually predict being wrong?**

Data: the `leather` category of MVTec AD (real), or the fake generator in `scripts/` until the real data is in.
MVTec AD is CC BY-NC-SA 4.0: practice only, non-commercial.

## 1. What "agentic" means here

A one-shot model call looks at the whole photo once and answers. The agent instead decides what to look at next:

1. glance at a downscaled overview,
2. ask a deterministic detector where the photo differs from known-good leather,
3. zoom into suspicious regions, compare them with a clean reference patch, measure them,
4. decide, and say how sure it is, or route to a human.

Every one of those is a tool call chosen by the model, with images coming back in the conversation.
The loop, tool registry, image delivery, validated final answer and fallbacks already exist in `agentkit/`
and are tested. You build the inspector on top.

## 2. Output contract

The final answer is a validated pydantic model passed to `submit_answer`:

```python
class Verdict(BaseModel):
    decision: Literal["pass", "fail"]            # the agent's call; escalation is decided by the POLICY, not the model
    defect_type: str                              # one of the taxonomy names, "none", or "unknown"
    confidence: Literal["low", "medium", "high"]  # self-reported; calibrated later (see section 5)
    region: Region | None                         # normalised box x0,y0,x1,y1 in 0..1, the main suspicious area
    reasons: list[str]                            # 1-4 short observations that justify the call
    needs_second_look: bool                       # the model's own "I would ask a human"
```

The **taxonomy is read from the dataset folder names** (`test/<defect>/`), never hard-coded in agent code.
My expected leather names (color, cut, fold, glue, poke) came from memory; `scripts/check_setup.py` prints
the real ones. Build everything so a different category works by pointing at a different folder.

Final system output (after the policy) is one of `pass`, `fail`, `review`.

## 3. Architecture

```
inspector/
  data.py        Sample records, loader, deterministic dev/test split. Only place that knows about ground_truth/.
  detector.py    No-LLM anomaly detector fitted on train/good: patch statistics -> anomaly map -> top regions.
  imaging.py     Pure image helpers: downscale, crop by normalised box, grid overlay, side-by-side, heatmap.
  tools.py       Agent tools built on imaging/detector (they return ToolResult with images).
  agent.py       Verdict schema, system prompt, build_agent(), run_one(path) -> VerdictRun.
  policy.py      Confidence signals -> escalate or not. Cost model. Threshold selection on dev.
  metrics.py     accuracy, false-pass rate, per-defect recall, ECE, AUROC, risk-coverage, bootstrap CIs.
  evaluate.py    CLI: run a method over dev/test, write results/<run>.json (+ model label, date, cost).
  report.py      Turn results into a Markdown/PNG report (reliability diagram, risk-coverage, confusion).
  app/           (M7) tiny demo UI.
tests/           offline tests using ScriptedLLM; no network, no keys.
results/         real run outputs only (git-ignored).
```

Rules of the architecture:
- Only `data.py` and `evaluate.py`/`metrics.py` may touch `ground_truth/` or the fake manifest. Agent-facing code never does.
- Tools never raise; failures return short actionable strings (already enforced by `ToolRegistry`).
- Tool coordinates are **normalised 0..1**. Vision models are poor at pixel coordinates, so tools take
  a grid cell or normalised centre, and the zoom image carries a labelled grid so the model can refer to cells like `C4`.

## 4. Tools the agent gets

| Tool | What it does | Returns |
|---|---|---|
| `view_overview()` | The photo downscaled to at most ~768 px, with a labelled grid | image |
| `scan_anomalies(top_k=3)` | Runs the deterministic detector, lists the most unusual regions with score and box | text + heatmap image |
| `zoom(cell or box, scale)` | Full-resolution crop of a region with a finer grid | image |
| `compare_reference(box)` | The same region from a known-good image, side by side with this photo | image |
| `measure(box)` | Deterministic numbers for a region: anomalous-pixel area, elongation, mean colour shift vs reference, darkness, smoothness | text (JSON) |
| `submit_answer` | Final `Verdict` | (built in) |

Budget per photo: at most 8 steps and 10 images (the kit enforces both). A normal photo should cost about 3 to 5 calls.

## 5. Confidence and escalation (the core of the project)

The model's own "confidence" is just a claim. Build several signals and test which ones predict errors:

1. **Self-reported** `confidence` and `needs_second_look`.
2. **Agreement**: run the agent k times (k=3, some temperature) and measure the share agreeing on pass/fail and on defect type.
3. **Detector margin**: distance between the max anomaly score and the dev-set threshold (small distance = uncertain).
4. **Measurement margin**: how close `measure` outputs are to the class boundaries seen on dev.
5. **Disagreement**: detector says anomalous but the agent says pass (or the reverse).

`policy.py` combines signals into one `uncertainty` number, then escalates to `review` above a threshold chosen
**on the dev split only** with an explicit cost model (defaults, change them and say why):

| Outcome | Cost |
|---|---|
| false pass (defective part ships) | 10 |
| false fail (good part scrapped) | 1 |
| sent to human review | 0.3 |
| correct automatic decision | 0 |

Report on the **test split, once**: automatic-decision accuracy, false-pass rate, share escalated, expected cost per part,
compared with always-trust-the-model and with the detector alone.

### Metrics
- Image-level: accuracy, false-pass rate (most important), false-fail rate, per-defect recall, confusion matrix on defect type.
- Calibration: reliability table (bins of confidence vs observed accuracy), ECE, **AUROC of the uncertainty signal for predicting errors**,
  risk-coverage curve (error rate among kept decisions as you escalate more).
- Localisation (bonus): does `region` overlap the ground-truth mask? (hit if the box centre lies in the mask dilated by a margin.)
- Cost: tokens, images, wall time per part.
- Everything with a bootstrap 95% interval. The test set is small, so intervals will be wide. Say so in the report.

### Baselines (the agent must beat these to justify itself)
1. Detector only (max anomaly score vs threshold).
2. One-shot LLM: overview image, no tools, same `Verdict`.
3. Agent without escalation policy.
4. Agent with policy.

If the agent does not beat the detector, that is a finding. Write it down and explain why; do not tune until it looks good.

## 6. Milestones

Each is one session. See `COURSE.md` for the teaching plan.

| # | Milestone | Done when |
|---|---|---|
| M0 | Orientation and setup | `check_setup` clean, tests pass, smoke test run on the chosen model, you can explain the loop in `agentkit/agent.py` |
| M1 | Data and splits | `inspector/data.py` + tests; stratified dev/test split with a fixed seed; test split guarded |
| M2 | Deterministic detector | image-level AUROC on dev reported; top regions overlap masks; no LLM involved |
| M3 | Image tools | all tools implemented and tested for coordinate maths and never raising |
| M4 | The agent | scripted-LLM tests pass; 10 real dev images traced by hand; one-shot baseline exists |
| M5 | Evaluation harness | `evaluate.py` runs detector / one-shot / agent on dev, saves labelled results |
| M6 | Confidence and escalation | signals built, calibration metrics computed, threshold picked on dev, test run once |
| M7 | Demo and pitch | upload a photo, see the trace and verdict, escalate button; 6-minute pitch outline with a failure slide |
| M8 | Stretch | second category or real/fake swap; Entire dry run; challenge-day playbook |

## 7. Honesty rules

- Results from the fake dataset prove plumbing only. Never put them in a results table or a pitch.
- Numbers in `PROGRESS.md` and reports come from real runs, with model label, date and split.
- The test split is run once per configuration you intend to report. No peeking, no tuning on it.
- Do not claim a model "understands" a defect unless the evaluation shows it; report what was measured.
- Free-tier models log prompts and rate-limit. Use only the open MVTec data with them, never anything private.
