# START HERE

A ready-to-build practice project for the EHL Finale: a **vision agent that inspects leather**, decides pass/fail/review, and
learns to know when it is unsure. You hand the build to Claude Code, and it teaches you as it goes.

What is in the box:
- `agentkit/` the agent loop, tool registry, image delivery, validated answers, provider fallback. Tested (88 offline tests).
- `scripts/` fake leather generator, MVTec leather importer, setup checker, vision smoke test.
- `SPEC.md` what to build · `COURSE.md` how you learn it · `CLAUDE.md` Claude Code's tutor rules · `PROGRESS.md` the log.
- `inspector/` is empty on purpose. That is the project.

## 1. Setup (about 10 minutes, macOS)

```bash
cd leather-kit
python3 --version            # need 3.11 or newer. If older: `conda create -n leather python=3.12 && conda activate leather`
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
cp .env.example .env         # then open .env and paste at least one API key
python scripts/make_fake_leather.py     # 120 fake images so you can start right now
python scripts/check_setup.py           # should show no [FAIL]
pytest -q                               # 88 passed
python scripts/smoke_llm.py             # does your model see images and call tools?
```

The smoke test matters. A model has to pass **both** stages (sees an attached image, and sees an image a tool returned).
Many free models do not. If your first choice fails, try another with `python scripts/smoke_llm.py --model gemini` (or `openai`, `openrouter:<slug>`).
Model names and free tiers change; check the provider's current model list when a name is rejected.

## 2. Get the real data (the leather folder only)

Do this in parallel; you do not need to wait for it to start.

1. Open <https://www.mvtec.com/company/research/datasets/mvtec-ad> and find the Downloads section.
2. MVTec asks for a short form and licence acceptance (CC BY-NC-SA 4.0, non-commercial: fine for practice).
   Take the leather category download if offered, otherwise the full archive (a few GB; it is fine).
3. Import only leather into the project:
   ```bash
   python scripts/import_mvtec_leather.py ~/Downloads/<the file or unpacked folder> --force
   ```
   `--force` replaces whatever is in `data/leather` (the fake data, or an earlier import). Without it the script refuses to touch a non-empty folder.
4. It prints the real defect folder names and image sizes. My expected names (color, cut, fold, glue, poke) came from memory.
   If they differ, just tell Claude Code; the project reads names from folders anyway.

I could not download this for you: my sandbox cannot reach mvtec.com, and the form needs you.

## 3. Hand it to Claude Code

```bash
git init && git add -A && git commit -m "leather kit"
claude
```

Paste this as your first message:

```
Read CLAUDE.md, SPEC.md, COURSE.md and PROGRESS.md. You are my tutor and pair-programmer for this project, in guided mode.
Start with M0. Before writing code, tell me in two sentences where we are and propose the plan for this session, then wait for my go.
I want to be able to rebuild a system like this alone at a hackathon, so make me write the CORE parts and quiz me at the end of each step.
```

Words you can say any time: **autopilot** (it builds the whole milestone, then walks you through it), **guided** (back to teaching),
**quiz** (5 questions on what you just did), **explain agentkit/agent.py** (walkthrough, no changes), **I'm short on time** (it cuts to the smallest useful slice).

## 4. Pacing

COURSE.md has a suggested plan: Wed night M0-M1, Thu M2-M4, Fri M5-M6, M7 if energy allows. M6 (confidence and escalation) is the
point of the whole exercise, so protect time for it and drop M7/M8 first.

## 5. Honest caveats

- Results on the fake data prove plumbing only. Real numbers need the real data and a real model run.
- The real leather defect names and counts in these docs came from memory; `check_setup` shows what is actually there.
- The test split will be small, so intervals are wide. Report them.
- Free-tier models may log prompts and throttle you. Only MVTec images go through them.
- Check early whether Entire (mandatory for hackathon submissions) works with how you run Claude Code (M8).
- MVTec AD is non-commercial. This is practice, not a product.

## 6. Layout

```
agentkit/      tested core (agent loop, tools, images, LLM clients)
scripts/       make_fake_leather.py  import_mvtec_leather.py  check_setup.py  smoke_llm.py
inspector/     YOU + Claude Code build this (see SPEC.md section 3)
tests/         offline tests, no keys needed
data/leather/  fake or real images (git-ignored)
results/       real run outputs (git-ignored)
```
