# Course: learn to build a trustworthy vision agent

Used by Claude Code in guided mode (see `CLAUDE.md`). Each milestone lists the idea, the **CORE** piece you write yourself,
prediction and check questions, and what transfers to the hackathon. Times are rough guesses for one focused session.

Suggested pacing to be ready by Sat 10:00 (adjust honestly, a slipped milestone beats a half-understood one):
Wed night M0-M1 · Thu M2-M4 · Fri M5-M6 (+M7 if energy) · M8 only if everything else is solid.

---

## M0 Orientation and setup (30-45 min)
**Idea.** An agent is a loop: model call, tool calls, results back, repeat until a validated final answer. Everything else is plumbing around that.
**Do.** `python scripts/check_setup.py`, `pytest -q`, `python scripts/make_fake_leather.py`, `python scripts/smoke_llm.py`. Then read `agentkit/agent.py` with Claude Code explaining it.
**Predict.** What happens if the model replies with plain text and never calls `submit_answer`? What if a tool raises?
**Check.** (1) Why is the final answer a tool call rather than "reply in JSON"? (2) Why are tool images sent in a user message after the tool messages?
**Transfer.** On the day you will wire an unknown API or model into this loop within minutes. Knowing which part is which is the speed.

## M1 Data and splits (45 min)
**Idea.** Decide how you will be judged before you build anything. A dev/test split with a fixed seed, and a test set you do not touch.
**CORE (you write).** `split(samples, seed)`: stratified by defect type, deterministic, roughly half dev / half test, good images spread across both.
**Predict.** If you tune thresholds on all the data, what happens to the reported accuracy, and why does it not show up until later?
**Check.** (1) Why stratify? (2) Why does only `data.py` know about `ground_truth/`?
**Transfer.** Judges ask "how do you know it works?" The honest answer starts here.

## M2 Deterministic detector (60-90 min)
**Idea.** Before using an LLM, see how far plain statistics get. Leather is a texture: split into patches, describe each (mean colour, local contrast, edge energy), learn what "normal" looks like from `train/good`, flag patches that are far from normal.
**CORE (you write).** `patch_scores(image)`: per-patch distance from the normal distribution (z-score or Mahalanobis on a few features). Claude Code writes feature extraction and the region grouping.
**Predict.** Which defects will this miss? (Guess before running: colour shifts? folds? tiny pokes?)
**Check.** (1) Image-level AUROC on dev: what is it, and what does 0.5 mean? (2) Why fit only on good images?
**Transfer.** A cheap deterministic baseline is both a tool for the agent and the yardstick that keeps you honest.

## M3 Image tools (60-90 min)
**Idea.** Tools are the agent's senses. Make them small, forgiving and well-described, because the description is the only manual the model gets.
**CORE (you write).** `crop_box(image, box, pad)`: normalised box to pixel crop, clamped to the image, never raising. Claude Code writes grid overlay, side-by-side, `measure`.
**Predict.** What does the model do when two tools return near-identical images? What happens if it asks for a box outside the image?
**Check.** (1) Why normalised coordinates? (2) What does a tool return when its arguments are bad, and why text instead of an exception?
**Transfer.** Any hackathon agent lives or dies on its tool design. Clear names, one job each, helpful error text.

## M4 The agent (60-90 min)
**Idea.** Prompt = job description + procedure + output rules. Verdict schema = contract. Offline tests with a scripted LLM prove the plumbing; only real runs prove behaviour.
**CORE (you write).** The system prompt: the inspection procedure, when to zoom, what counts as evidence, how to use confidence. Claude Code reviews it and asks what is missing.
**Predict.** After 10 real dev images, which failure do you expect most: not calling tools, wrong defect name, or over-confident passes?
**Check.** (1) Why cap steps and images? (2) What does the one-shot baseline tell you that the agent run cannot?
**Transfer.** The first 30 minutes of challenge day: write the job description, schema and a one-shot baseline before any fancy loop.

## M5 Evaluation harness (60-90 min)
**Idea.** One command, comparable numbers, labelled with model and date. Always compare against baselines.
**CORE (you write).** `false_pass_rate` and `per_defect_recall` in `metrics.py`, with tests on tiny hand-made examples you can verify by hand.
**Predict.** Which of detector / one-shot / agent has the lowest false-pass rate on dev? Write your guess down, then see.
**Check.** (1) Why is false-pass rate not the same as 1 - accuracy? (2) Why record cost per part?
**Transfer.** A results table with baselines is the strongest 20 seconds of a pitch.

## M6 Confidence and escalation (90-120 min) <- the heart of the project
**Idea.** Calibration: when the system says 80% sure, is it right about 80% of the time? A useful uncertainty signal ranks wrong answers above right ones. Escalation turns that into a decision with a cost.
**CORE (you write).** `auroc_for_errors(uncertainty, was_wrong)` and the threshold search over dev with the cost model (SPEC section 5).
**Predict.** Rank by how well they will predict errors: self-reported confidence, agreement over 3 runs, detector margin. Commit to an order.
**Check.** (1) What does ECE miss that AUROC catches, and vice versa? (2) Why choose the threshold on dev and report on test? (3) What does the risk-coverage curve show?
**Transfer.** "Knows when it doesn't know" is what separates a demo from something a company would trust. Judges, including investors, care.

## M7 Demo and pitch (60-90 min)
**Idea.** Show the agent working, show the trace, show one honest failure.
**CORE (you write).** The 6-minute pitch outline: problem (30s), demo (2m), how it decides (1m), results vs baselines (1m), failure and limits (1m), Q&A prep (30s).
**Check.** (1) What is the one sentence a judge repeats about your project? (2) Which failure do you show and why?
**Transfer.** Pitch slot is 6 minutes including Q&A, so rehearse with a timer.

## M8 Stretch (only if everything above is solid)
- Swap real/fake data or run a second MVTec texture category (tile, wood, carpet, grid) with zero agent-code changes. The taxonomy-from-folders rule makes this the test.
- Entire dry run: the hackathon requires Entire for submissions. Check how it works with your setup early.
- Write a one-page challenge-day playbook: first 30 minutes, who does what, baseline before cleverness, cut list.

---

## Quiz bank (Claude Code draws from this for "quiz" mode; add your own)
1. Name three ways the loop can end and what status each produces.
2. Why must the agent never see `ground_truth/`?
3. Your detector has AUROC 0.97 on dev. Is the agent worth building? What would convince you either way?
4. The agent is right 90% of the time and always says "high". What are its ECE and AUROC-for-errors roughly, and why is it useless for escalation?
5. A false pass costs 10, a false fail 1, a review 0.3. With uncertainty 0.4, what do you do if P(defect) is 0.2? What if 0.5?
6. Test results look much worse than dev. List three plausible causes, with a check for each.
7. Your model passes `smoke_llm` stage 1 but fails stage 2. What do you change?
8. Why do vision models struggle with pixel coordinates, and what do our tools do about it?
