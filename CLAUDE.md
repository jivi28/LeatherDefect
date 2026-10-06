# Working agreements: Leather Inspector (build mode)

Read `SPEC.md` (what we build) and `PROGRESS.md` (where we are) at the start of every session.
Then say in two sentences which milestone we are on and what the first step is.

## Who you are working with
The owner of this repo: comfortable with Python. Building this for the EHL Finale hackathon (Oct 10-11, 2026).
They want the system built, not a course: no teaching steps, no quizzes, no "you write the CORE part".

## Mode
- **autopilot** (default): you write the whole milestone, run the tests, update PROGRESS.md, commit, then give a short
  summary: what was built, key decisions, what is untested. `COURSE.md` is kept for reference only.
- **explain <file or function>**: walkthrough on request, no changes.

## Process
- Build milestones back to back; no need to wait for a go between them unless something is ambiguous.
- Tests first for geometry/coordinates, splitting, metrics and anything touching `ground_truth/`.
- Run `pytest -q` before saying anything is done. Report exactly what passed and failed.
- At the end of a milestone: update `PROGRESS.md` (status, what was learned, open questions, decisions), commit with a clear message.
- If something in SPEC is ambiguous, ask once. If unanswered, pick the simplest option and log it in `PROGRESS.md` under Decisions.
- Do not add dependencies beyond `requirements.txt` without asking.

## Code
- Python 3.11+, type hints, pydantic for schemas, plain functions over class hierarchies, small modules.
- `agentkit/` is tested infrastructure. Extend it only with tests; do not rewrite it.
- Tools never raise; every failure becomes a short, actionable string for the model.
- Tool coordinates are normalised 0..1. Never ask the model for pixel coordinates.
- Never put image bytes in logs or traces (the kit already keeps them out).

## Integrity (non-negotiable)
- Never weaken, skip or delete a test to make it pass. If a test looks wrong, stop and say so.
- Agent-facing code (prompts, tools, agent, policy at inference time) never reads `ground_truth/`, `_fake_manifest.csv`, or the
  test split. Only `data.py`, `metrics.py` and `evaluate.py` may.
- Never hard-code defect names, file names or expected answers into prompts or tools. The taxonomy comes from folder names.
- The test split runs once per configuration we intend to report. Tune only on dev.
- Never report numbers from `FAKE_DATA.txt` datasets as results. Say "plumbing check on fake data".
- Never claim an LLM result you did not run. Offline tests with `ScriptedLLM` only prove the plumbing.
- Results in `PROGRESS.md` need: model label, date, split, n, and a bootstrap interval.

## Safety
- Never commit `.env`, `data/`, or `results/`. Never print API keys.
- Free-tier models may log prompts. Only MVTec images and nothing private go through them.
- MVTec AD is non-commercial (CC BY-NC-SA 4.0). Keep it practice-only and say so if the user starts talking about shipping a product.

## Communication
- Be direct. If something is flaky or half-working, say so in the first sentence.
- When a milestone ends: what is done, what is untested, what you are unsure about, the next step.
- Prefer short answers and one question at a time. The user prefers honest, unhedged assessments over diplomacy.
