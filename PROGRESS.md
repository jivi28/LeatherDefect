# Progress

Claude Code updates this at the end of every milestone. Real results only (model label, date, split, n, interval).

## Status
| # | Milestone | Status | Notes |
|---|---|---|---|
| M0 | Orientation and setup | done 2026-10-07 | venv (Python 3.13), 88 kit tests pass, real data imported, local model smoke-tested |
| M1 | Data and splits | done 2026-10-07 | `inspector/data.py`, seed 7, dev 62 / test 62, test locked |
| M2 | Deterministic detector | done 2026-10-07 | patch stats + Mahalanobis, 2 scales; dev AUROC 1.0 |
| M3 | Image tools | done 2026-10-07 | scan_anomalies, zoom, compare_reference, measure (+ view_overview for UI/one-shot) |
| M4 | The agent | done 2026-10-08 | real agent run on a 24-photo dev subset (k=1) | scripted tests pass; first real trace: correct fail, wrong type (cut -> poke) |
| M5 | Evaluation harness | done 2026-10-08 | detector, one-shot (62) and agent (24) on dev | resumable JSONL + labelled summaries |
| M6 | Confidence and escalation | dev done, test not run | disagreement signal chosen on dev (AUROC 1.0, but only 1 error in 24); test split untouched |
| M7 | Demo and pitch | demo done 2026-10-08 (live view), pitch not written | UI streams steps live (NDJSON): thoughts, zoom box on the photo, tool images |
| M8 | Stretch | dropped for now | |

## Data in use
- [ ] fake generated data (plumbing only) — never generated in this checkout
- [x] real MVTec AD leather, from the Hugging Face mirror `foersben/mvtec-ad` (CC BY-NC-SA 4.0), imported with
  `scripts/import_mvtec_leather.py`. Defect folders: color 19, cut 19, fold 17, glue 19, poke 18; test/good 32; train/good 245; 1024x1024 RGB.

## Model in use
- `ollama:qwen3-vl:8b-instruct` (local, free, no rate limits). Smoke test on `qwen3-vl:8b` (thinking variant): both stages ok,
  2026-10-07, but ~3 min/image because it reasons for ~2k tokens per step even with think=false, so switched to the instruct tag.
- Ollama server needs `OLLAMA_CONTEXT_LENGTH=16384` (default 4096 truncates multi-image conversations).
- Optional cloud cross-check: `groq:qwen/qwen3.8-27b` (free tier ~8K tokens/min, 200K/day, max 3 images/request: set INSPECTOR_MAX_IMAGES=3).

## Decisions
- 2026-10-07: Build mode, not tutor mode (owner's call). CLAUDE.md updated; COURSE.md kept for reference only.
- 2026-10-07: Real data from the HF mirror instead of the mvtec.com form (same files, same licence).
- 2026-10-07: Local Ollama model as primary: free cloud vision tiers (Gemini ~100 req/day, Groq ~100 images/day) are too small for evals.
- 2026-10-07: Split seed 7; odd-sized classes alternate their extra image between dev and test.
- 2026-10-07: Detector threshold is label-free: 1.05 x the max score over held-out train/good images (every 5th). Not tuned on dev.
- 2026-10-07: The overview (8x8 grid) is attached to the first message instead of being a tool call; saves a step and guarantees the model saw it.
- 2026-10-07: Verdict enforces pass <=> defect_type "none"; inconsistent answers go back to the model to fix.
- 2026-10-07: Runs that fail to submit fall back to fail/unknown/low/needs_second_look (never auto-pass); the self signal escalates them.
- 2026-10-07: Measurement-margin signal (SPEC 5.4) not built separately; the detector-margin signal covers the same idea with less code.
- 2026-10-07: Detector region picking suppresses a defect's whole halo, so ranked regions are distinct spots (was: 3 boxes around one cut).
- 2026-10-07: `scripts/download_mvtec_leather.py` fetches leather from the HF mirror and imports it (resumable, cache in `.cache/`).
- 2026-10-08: UI runs use `narrate=True` (one sentence before each tool call) so viewers see the reasoning; eval runs keep the original prompt, so measured results are unaffected.
- 2026-10-07: agentkit gained `LLM_EXTRA_BODY_<PROVIDER>` (tested) to pass Ollama-only fields.

## Results (real runs only)
| date | method | model | split | n | accuracy (auto) | false pass | false fail | notes |
|---|---|---|---|---|---|---|---|---|
| 2026-10-07 | detector only | — | dev | 62 | 96.8% [91.9, 100] | 0.0% [0, 0] | 12.5% [0, 30.8] | AUROC 1.0; top-1 box hits mask 45/46 (97.8%); 0.03 s/image |
| 2026-10-07 | one-shot LLM (overview only) | ollama:qwen3-vl:8b-instruct | dev | 62 | 80.6% [71.0, 90.3] | 23.9% [12.5, 36.8] | 6.3% [0, 20.0] | defect-type acc 34.8% [21.6, 48.9]; 40/62 answers 'high' confidence; calls most cuts 'poke'; 6.5 s/image |
| 2026-10-08 | detector only | — | dev subset (same 24) | 24 | 100% [100, 100] | 0.0% [0, 0] | 0.0% [0, 0] | |
| 2026-10-08 | one-shot LLM | ollama:qwen3-vl:8b-instruct | dev subset (same 24) | 24 | 75.0% [54.2, 91.7] | 30.0% [10.5, 52.4] | 0.0% [0, 0] | defect-type acc 30.0% [10.5, 50.0] |
| 2026-10-08 | agent, k=1 | ollama:qwen3-vl:8b-instruct | dev subset (`--limit 24`) | 24 | 95.8% [87.5, 100] | 0.0% [0, 0] | 25.0% [0, 100] (1 of 4 good) | defect-type acc 55.0% [31.6, 73.9]; 19.4 s/image; 21/24 'high' |
| 2026-10-08 | agent + policy (disagreement, chosen on same dev) | ollama:qwen3-vl:8b-instruct | dev subset | 24 | 100% (optimistic: chosen and scored on the same data) | 0.0% | 0.0% | 1 of 24 escalated (4%) |

Finding: on leather the deterministic detector already separates good from defective perfectly on dev (AUROC 1.0).
The agent cannot beat it on pass/fail; its job is to name the defect type, explain, and reduce false fails / escalate well.

## Open questions
- Not done: full dev agent run (62, k=3), the one-time test run, the pitch outline. Tonight's numbers are a 24-photo dev subset only.
- The agent's single error was a false fail (good -> color, medium confidence); the policy escalates it because the detector disagreed.
- Agent dev run (k=3) paused at 35/186 records; resume with `python -m inspector.evaluate --method agent --split dev --repeat 3`.
- Does the agent's defect-type accuracy justify ~20-30 s and several model calls per part?
- Which signal best predicts the agent's errors: self-reported, k=3 agreement, detector margin, or agent-vs-detector disagreement?
