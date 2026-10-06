# LeatherDefect: a vision agent that inspects leather and knows when it is unsure

A practice project for the EHL Finale hackathon (Oct 10-11, 2026). The agent looks at a photo of leather and decides
**pass / fail**, names the defect (`color`, `cut`, `fold`, `glue`, `poke`) and shows where it is. An escalation policy
then sends uncertain cases to a human (**review**). The core question: does "low confidence" actually predict being wrong?

Data: the `leather` category of [MVTec AD](https://www.mvtec.com/company/research/datasets/mvtec-ad), CC BY-NC-SA 4.0.
**Non-commercial, practice only.** Everything runs locally: a free Ollama vision model, no API keys, no rate limits.

**Hand-off status (2026-10-07):** milestones M0-M5 are built and tested (182 offline tests pass). M6 (escalation) and
M7 (demo) are coded. Still missing: the **full agent run on dev**, the **report**, the **one-time test run**,
and the **pitch**. Section 4 is the to-do list.

---

## 1. Setup from zero (about 30 min, mostly downloads)

You need: macOS (Apple Silicon recommended) or Linux, Python 3.11-3.13, ~10 GB free disk, ~10 GB free RAM.

```bash
git clone https://github.com/jivi28/LeatherDefect.git
cd LeatherDefect
python3 --version                      # 3.11, 3.12 or 3.13 (3.13 was used)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pytest -q                              # expect: 182 passed
cp .env.example .env                   # default is the local Ollama model, no key needed
```

### 1a. Real data (525 MB, 463 files)

```bash
python scripts/download_mvtec_leather.py
```

It downloads only `leather/` from the Hugging Face mirror `foersben/mvtec-ad` into `.cache/` (re-run it if interrupted,
it resumes), copies the files into `data/leather/`, then runs the checks. It should end with
`0 blocking problem(s)`, showing `color=19, cut=19, fold=17, glue=19, poke=18`, 245 train/good, 32 test/good.
**Do not run `scripts/make_fake_leather.py`.** That is fake data, and its numbers must never be reported.

### 1b. Local vision model (Ollama)

1. Install Ollama: `brew install ollama` on macOS, or see <https://ollama.com/download>.
2. Start the server **with a 16K context**. The default 4K context silently truncates conversations with several images:
   ```bash
   OLLAMA_CONTEXT_LENGTH=16384 ollama serve
   ```
   Leave that terminal open. If the Ollama desktop app is already running, quit it first (menu bar icon -> Quit),
   or the app's server keeps the 4K default.
3. In another terminal, pull the model (6.1 GB):
   ```bash
   ollama pull qwen3-vl:8b-instruct
   ```
   Use the **`-instruct`** tag. Plain `qwen3-vl:8b` is the *thinking* variant: it writes ~2,000 hidden reasoning tokens per
   step and takes ~3 min per photo instead of ~20 s. `think: false` does not switch that off.
4. Check that the model can see images and call tools:
   ```bash
   python scripts/check_setup.py         # no [FAIL]
   python scripts/smoke_llm.py           # both stages [ok]
   ```

### 1c. Try it on one photo

```bash
python -m inspector.agent data/leather/test/glue/000.png              # agent, prints every step
python -m inspector.agent data/leather/test/glue/000.png --oneshot    # baseline: photo only, no tools
```

---

## 2. How it works (the 2-minute version)

```
photo -> overview (8x8 grid, attached to the first message)
      -> model calls tools:   scan_anomalies   statistical detector fitted on good leather: where is it unusual?
                              zoom             full-res close-up of a cell ("C4") or box (0..1 coordinates)
                              compare_reference same region from a known-good photo, side by side
                              measure          numbers: anomaly area, elongation, colour/brightness/contrast z-scores
      -> submit_answer(Verdict)   decision, defect_type, confidence, region, reasons, needs_second_look
      -> policy.py               uncertainty signals -> "review" if above a threshold chosen on dev
```

| File | What it does |
|---|---|
| `agentkit/` | Provided, tested core: agent loop, tool registry, image delivery, LLM provider fallback. Extend only with tests. |
| `inspector/data.py` | Loads samples, reads the taxonomy from folder names, makes the stratified dev/test split (seed 7, 62/62). **The test split is locked.** |
| `inspector/detector.py` | No-LLM anomaly detector: patch colour/contrast/edge statistics, Mahalanobis distance to good patches. Cached in `results/cache/`. |
| `inspector/imaging.py`, `tools.py` | Grid, crops, heatmaps, and the 4 agent tools. Tools never raise; bad arguments come back as helpful text. |
| `inspector/agent.py` | Verdict schema (taxonomy from folders), system prompt, `run_one()`, one-shot baseline, CLI. |
| `inspector/evaluate.py` | Runs a method over a split and writes `results/runs/<run>.jsonl` (resumable) plus a `results/<run>.json` summary with bootstrap 95% CIs. |
| `inspector/metrics.py` | Accuracy, false-pass/false-fail rates, per-defect recall, type accuracy, AUROC, ECE, risk-coverage, bootstrap CIs, mask hits. |
| `inspector/policy.py` | Uncertainty signals (self-reported, k=3 agreement, detector margin, agent-vs-detector disagreement, combined), cost model, threshold search. |
| `inspector/report.py` | Methods table, which signal predicts errors (AUROC for errors), the dev-chosen threshold, plots. Writes `results/report.md` + `report.json`. |
| `inspector/app/` | Demo web page (FastAPI): pick or upload a photo, see each step with its images, the verdict and the policy decision. |

More detail: `SPEC.md` (the full spec), `PROGRESS.md` (status, decisions, results so far), `CLAUDE.md` (rules for Claude Code).

---

## 3. Results so far (real data, dev split, n=62, 2026-10-07)

| Method | Model | Accuracy | False pass | False fail | Defect-type acc | Time/photo |
|---|---|---|---|---|---|---|
| Detector only | none | 96.8% [91.9, 100] | **0.0%** | 12.5% [0, 30.8] | n/a (cannot name types) | 0.03 s |
| One-shot LLM | qwen3-vl:8b-instruct | 80.6% [71.0, 90.3] | **23.9%** [12.5, 36.8] | 6.3% [0, 20.0] | 34.8% [21.6, 48.9] | 6.5 s |
| Agent | qwen3-vl:8b-instruct | *not finished* | | | | ~20-25 s |

What this means:
- On leather the detector alone already separates good from defective perfectly on dev (AUROC 1.0, top box hits the
  defect mask 45/46). The agent **cannot beat it on pass/fail**. Its value has to come from naming the defect type,
  explaining, cutting false fails, and escalating the right cases.
- The one-shot model ships 1 in 4 defects and says "high" confidence 40 of 62 times. That is the failure to show in the pitch.
- Known confusion: the model calls hole-like **cuts** `poke`, even with tools. A good honest failure slide.

---

## 4. To-do for tomorrow, in order

Before each step: Ollama is running with the 16K context (1b), and the venv is active (`source .venv/bin/activate`).
`results/` is git-ignored, so on a fresh clone you re-run everything. Times are from an M5 Pro; slower machines take longer.

### Step 1: Baselines on dev (~8 min)
```bash
python -m inspector.evaluate --method detector --split dev        # ~10 s
python -m inspector.evaluate --method oneshot  --split dev        # ~7 min
```
Numbers should match section 3 closely (the LLM is not fully deterministic).

### Step 2: The agent on dev, 3 runs per photo (~75-90 min)
```bash
python -m inspector.evaluate --method agent --split dev --repeat 3
```
- `--repeat 3` feeds the **agreement** signal. Short on time? Use `--repeat 1` (~25 min) and drop that signal.
- It is resumable: Ctrl-C and run the same command again, finished photos are skipped.
- First try it on a few photos: `--limit 6` (that writes to a separate `_n6` run file).
- Each wrong pass/fail call prints a line starting with `XX`.

### Step 3: Report and policy (1 min)
```bash
python -m inspector.report \
  --agent    results/runs/agent_dev_ollama-qwen3-vl-8b-instruct_k3.jsonl \
  --baseline results/runs/detector_dev_detector.jsonl \
  --baseline results/runs/oneshot_dev_ollama-qwen3-vl-8b-instruct.jsonl
```
This writes `results/report.md` (open it), the plots, and `results/report.json`, which holds the chosen signal and threshold
that the demo uses. Read the "which signal predicts errors" table: AUROC near 0.5 means that signal is useless.
`report.py` is only tested on synthetic records so far, so check its numbers against `results/*.json` once.

### Step 4 (optional): improve on dev only
If something is clearly fixable (e.g. the system prompt in `inspector/agent.py` for the cut/poke confusion), change it,
then re-run Step 2 under a new name (`--name agent_dev_v2`) and compare. **Never tune on test.**
Never hard-code defect names, file names or answers into the prompt or tools: the taxonomy must come from folder names.

### Step 5: The test split, ONCE (~75-90 min)
Only when the configuration is final:
```bash
python -m inspector.evaluate --method agent --split test --final --repeat 3
python -m inspector.report \
  --agent    results/runs/agent_dev_ollama-qwen3-vl-8b-instruct_k3.jsonl \
  --baseline results/runs/detector_dev_detector.jsonl \
  --baseline results/runs/oneshot_dev_ollama-qwen3-vl-8b-instruct.jsonl \
  --test     results/runs/agent_test_ollama-qwen3-vl-8b-instruct_k3.jsonl
```
The threshold stays the one chosen on dev. Those test numbers are the ones you report.
Optionally also run the baselines on test (`--method detector --split test --final`, same for `oneshot`).

### Step 6: Record results
Add the numbers to `PROGRESS.md` → Results: date, method, model, split, n, and the 95% intervals.
Mark M4-M6 done there. Commit (never `.env`, `data/` or `results/`; they are git-ignored).

### Step 7: Demo (M7)
```bash
uvicorn inspector.app.server:app --port 8000          # then open http://localhost:8000
```
Pick a sample photo or upload one, then press Inspect (~20 s). It shows each tool call with its image, the verdict, and the
final pass/fail/review. Without `results/report.json` (Step 3) it shows the raw agent decision and says so.
Tested end to end on 2026-10-07: a glue photo came back fail/glue/high in 18 s.

### Step 8: Pitch (6 min including Q&A)
Problem 30 s → live demo 2 min → how it decides (tools + escalation) 1 min → results vs baselines with intervals 1 min →
one honest failure (cut→poke, or the one-shot's 24% false passes) and limits 1 min → Q&A prep 30 s.
Limits to say out loud: n=62 per split (wide intervals), one texture category, an 8B local model, non-commercial data.

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| ~3 min per photo | You pulled `qwen3-vl:8b` (thinking). Use `qwen3-vl:8b-instruct` and check `.env`. |
| Answers ignore earlier images / odd truncation | Ollama is running with the 4K default. Restart it with `OLLAMA_CONTEXT_LENGTH=16384 ollama serve`. |
| `No usable LLM provider` | `.env` is missing: `cp .env.example .env`. |
| `Connection refused` on port 11434 | Ollama isn't running (step 1b.2). |
| `The test split is locked` | Intended. Use `--split test --final` only for the final run. |
| `smoke_llm.py` stage 2 fails | The model can't see tool images. Use the instruct model above. |
| Using Groq instead | Free tier: ~100 images/day, max 3 images per request. Set `INSPECTOR_MAX_IMAGES=3` and add `GROQ_API_KEY` in `.env`. |

## 6. Rules (from CLAUDE.md, non-negotiable)

- Agent-facing code (`agent.py`, `tools.py`, `detector.py`, `imaging.py`, `policy.py`) never reads labels, masks or the
  test split. A test enforces this.
- Tune on dev. Run test once per configuration you report.
- Never weaken or delete a test to make it pass.
- Report only real runs, with model, date, split, n and a bootstrap interval.
- MVTec AD is non-commercial. This is practice, not a product.

Working with Claude Code: open the folder and say *"Read CLAUDE.md, SPEC.md and PROGRESS.md and continue from section 4 of
README.md"*. It builds in autopilot mode.
