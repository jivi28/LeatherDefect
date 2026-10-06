"""Turn run records into a Markdown report with plots: baselines, signal quality, the dev-chosen policy.

    python -m inspector.report --agent results/runs/agent_dev_X.jsonl \
        --baseline results/runs/detector_dev_detector.jsonl --baseline results/runs/oneshot_dev_X.jsonl
    # final: add --test results/runs/agent_test_X.jsonl (threshold still chosen on the dev agent run)

Writes results/report.md (+ PNGs next to it). Labels are read from the records (written by evaluate.py).
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

from . import metrics as M
from . import policy as P
from .evaluate import CONF_PROB, RESULTS, load_records

ERROR_NOTE = "error = wrong pass/fail decision (a wrong defect name on a correctly failed part is reported separately)"


def _fmt(point_lo_hi: tuple[float, float, float], pct: bool = True) -> str:
    p, lo, hi = point_lo_hi
    if p != p:
        return "n/a"
    f = (lambda v: f"{100 * v:.1f}%") if pct else (lambda v: f"{v:.3f}")
    return f"{f(p)} [{f(lo)}, {f(hi)}]"


def _primary(records: list[dict]) -> list[dict]:
    return [runs[0] for runs in P.group_runs(records).values()]


def _model(records: list[dict]) -> str:
    models = sorted({m for r in records for m in r.get("models", [])})
    return ", ".join(models) or "?"


def method_row(name: str, records: list[dict], decisions: list[str] | None = None, costs: P.Costs = P.Costs()) -> dict:
    first = _primary(records)
    labels = [r["label"] for r in first]
    decisions = decisions or [r["decision"] for r in first]
    types = [r["defect_type"] if d == "fail" else "none" for r, d in zip(first, decisions)]
    has_types = any(t not in ("none", "unknown") for t in types)
    return {
        "method": name,
        "model": _model(records),
        "n": len(first),
        "accuracy": M.bootstrap_ci(M.accuracy, labels, decisions),
        "false_pass": M.bootstrap_ci(M.false_pass_rate, labels, decisions),
        "false_fail": M.bootstrap_ci(M.false_fail_rate, labels, decisions),
        "type_accuracy": M.bootstrap_ci(M.type_accuracy, labels, types) if has_types else (float("nan"),) * 3,
        "review": M.review_rate(decisions),
        "cost": M.bootstrap_ci(lambda l, d: P.expected_cost(l, d, costs), labels, decisions),
        "seconds": sum(r["elapsed_s"] for r in first) / max(1, len(first)),
    }


def signal_table(records: list[dict]) -> tuple[list[dict], dict[str, list[float]], list[bool]]:
    groups = P.group_runs(records)
    sigs = [P.signals_for(runs) for runs in groups.values()]
    first = [runs[0] for runs in groups.values()]
    wrong = [(r["decision"] == "fail") != (r["label"] != "good") for r in first]
    names = [s for s in P.SIGNALS if all(s in x for x in sigs)]
    values = {s: [x[s] for x in sigs] for s in names}
    rows = [{"signal": s, "auroc_errors": M.auroc_for_errors(values[s], wrong)} for s in names]
    return rows, values, wrong


def _plots(records: list[dict], values: dict[str, list[float]], wrong: list[bool], out_dir: Path, tag: str) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    first = _primary(records)
    conf = [r.get("confidence") for r in first]
    if all(c in CONF_PROB for c in conf):
        correct = [not w for w in wrong]
        rows = M.reliability_table([CONF_PROB[c] for c in conf], correct, bins=5)
        fig, ax = plt.subplots(figsize=(4.2, 4))
        ax.plot([0, 1], [0, 1], ls="--", c="#999", lw=1)
        ax.scatter([r["confidence"] for r in rows], [r["accuracy"] for r in rows], s=[20 + 6 * r["n"] for r in rows], c="#2a6fdb")
        for r in rows:
            ax.annotate(f"n={r['n']}", (r["confidence"], r["accuracy"]), textcoords="offset points", xytext=(6, -10), fontsize=8)
        ax.set(xlim=(0.4, 1.0), ylim=(0, 1.05), xlabel="stated confidence", ylabel="observed accuracy", title="Reliability (self-reported)")
        fig.tight_layout()
        name = f"reliability_{tag}.png"
        fig.savefig(out_dir / name, dpi=130)
        plt.close(fig)
        files.append(name)
    fig, ax = plt.subplots(figsize=(5, 4))
    for s, u in values.items():
        curve = M.risk_coverage(u, wrong)
        ax.plot([c for c, _ in curve], [r for _, r in curve], label=s, lw=1.6)
    ax.set(xlabel="coverage (share decided automatically)", ylabel="error rate among kept", title="Risk-coverage", xlim=(1.0, 0.0))
    ax.legend(fontsize=8)
    fig.tight_layout()
    name = f"risk_coverage_{tag}.png"
    fig.savefig(out_dir / name, dpi=130)
    plt.close(fig)
    files.append(name)
    return files


def build_report(agent_dev: list[dict], baselines: dict[str, list[dict]], agent_test: list[dict] | None = None,
                 costs: P.Costs = P.Costs(), out_dir: Path = RESULTS) -> tuple[str, dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_sig, values, wrong = signal_table(agent_dev)
    dev_first = _primary(agent_dev)
    labels = [r["label"] for r in dev_first]
    decisions = [r["decision"] for r in dev_first]

    # pick the signal + threshold with the lowest expected cost on dev
    choice = None
    for s, u in values.items():
        t, c = P.choose_threshold(u, labels, decisions, costs)
        if choice is None or c < choice["cost"] - 1e-12:
            choice = {"signal": s, "threshold": t, "cost": c}
    assert choice is not None
    dev_policy = P.apply(values[choice["signal"]], decisions, choice["threshold"])

    split = dev_first[0]["path"].split("/test/")[0] if dev_first else ""
    lines = [
        "# Leather Inspector report",
        "",
        f"Generated {date.today().isoformat()}. Data: real MVTec AD leather ({split or 'data/leather'}), "
        f"licence CC BY-NC-SA 4.0 (practice only). Costs: false pass {costs.false_pass}, false fail {costs.false_fail}, "
        f"review {costs.review}. Intervals are bootstrap 95%. The split is small (n~62), so intervals are wide.",
        "",
        "## Dev split: methods compared",
        "",
        "| method | model | n | accuracy (auto) | false pass | false fail | defect-type acc | review | cost/part | s/image |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    table = [method_row(name, recs, costs=costs) for name, recs in baselines.items()]
    table.append(method_row("agent (no policy)", agent_dev, costs=costs))
    table.append(method_row(f"agent + policy ({choice['signal']})", agent_dev, dev_policy, costs))
    for r in table:
        lines.append(
            f"| {r['method']} | {r['model']} | {r['n']} | {_fmt(r['accuracy'])} | {_fmt(r['false_pass'])} | {_fmt(r['false_fail'])} | "
            f"{_fmt(r['type_accuracy'])} | {100 * r['review']:.0f}% | {_fmt(r['cost'], pct=False)} | {r['seconds']:.1f} |"
        )
    lines += [
        "",
        "Note: the policy row is chosen and scored on the same dev data, so it is optimistic. Only the test row below is an honest estimate.",
        "",
        "## Which uncertainty signal predicts errors? (dev)",
        "",
        f"AUROC for errors: 0.5 = no better than chance, 1.0 = every error ranked above every correct answer. {ERROR_NOTE}. "
        f"Errors on dev: {sum(wrong)} of {len(wrong)}.",
        "",
        "| signal | AUROC for errors | best dev threshold | dev cost/part |",
        "|---|---|---|---|",
    ]
    for row in rows_sig:
        t, c = P.choose_threshold(values[row["signal"]], labels, decisions, costs)
        auc = row["auroc_errors"]
        lines.append(f"| {row['signal']} | {'n/a (no errors)' if auc != auc else f'{auc:.3f}'} | {t:.3g} | {c:.3f} |")
    lines += ["", f"Chosen on dev: **{choice['signal']}** with threshold **{choice['threshold']:.3g}** (escalate when above)."]

    first_dev = _primary(agent_dev)
    conf = [r.get("confidence") for r in first_dev]
    if all(c in CONF_PROB for c in conf):
        lines.append(
            f"Self-reported confidence on dev: {dict((c, conf.count(c)) for c in CONF_PROB)}; "
            f"ECE {M.ece([CONF_PROB[c] for c in conf], [not w for w in wrong]):.3f}."
        )
    for f in _plots(agent_dev, values, wrong, out_dir, "dev"):
        lines += ["", f"![{f}]({f})"]

    summary = {"choice": choice, "dev": {r["method"]: _jsonable(r) for r in table}}
    if agent_test:
        test_first = _primary(agent_test)
        tg = P.group_runs(agent_test)
        u = [P.signals_for(runs).get(choice["signal"], 1.0) for runs in tg.values()]
        test_policy = P.apply(u, [r["decision"] for r in test_first], choice["threshold"])
        rows = [method_row("agent (no policy)", agent_test, costs=costs), method_row("agent + policy", agent_test, test_policy, costs)]
        lines += ["", "## Test split (run once, threshold fixed from dev)", "",
                  "| method | n | accuracy (auto) | false pass | false fail | defect-type acc | review | cost/part |", "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['method']} | {r['n']} | {_fmt(r['accuracy'])} | {_fmt(r['false_pass'])} | {_fmt(r['false_fail'])} | "
                         f"{_fmt(r['type_accuracy'])} | {100 * r['review']:.0f}% | {_fmt(r['cost'], pct=False)} |")
        summary["test"] = {r["method"]: _jsonable(r) for r in rows}
    md = "\n".join(lines) + "\n"
    return md, summary


def _jsonable(row: dict) -> dict:
    return {k: (list(v) if isinstance(v, tuple) else v) for k, v in row.items()}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True, help="agent run JSONL on dev (threshold is chosen here)")
    p.add_argument("--baseline", action="append", default=[], help="other run JSONLs on the same split")
    p.add_argument("--test", help="agent run JSONL on test (final)")
    p.add_argument("--out", default=str(RESULTS / "report.md"))
    a = p.parse_args(argv)
    agent_dev = load_records(Path(a.agent))
    ids = {r["id"] for r in agent_dev}
    baselines = {}
    for path in a.baseline:
        recs = [r for r in load_records(Path(path)) if r["id"] in ids]  # compare on the same images
        baselines[f"{recs[0]['method']}" if recs else Path(path).stem] = recs
    out = Path(a.out)
    md, summary = build_report(agent_dev, baselines, load_records(Path(a.test)) if a.test else None, out_dir=out.parent)
    out.write_text(md)
    out.with_suffix(".json").write_text(json.dumps(summary, indent=2, default=str))
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
