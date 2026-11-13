#!/usr/bin/env python3
"""
trace_report.py - error analysis of a blackboard run from its traces (for the paper's analysis section).

  python trace_report.py --run runs/popqa/blackboard --examples 5

Joins per_item.csv (labels) with traces.jsonl (which knowledge source ran, and why) and reports:
  * process statistics: knowledge-source usage, cycles, stop reasons, share of model outputs the code could
    not use ("failed:" notes, mostly broken JSON)
  * outcome categories:
      fixed_by_competing_answer   raw wrong -> correct, via a verified evidence-based hypothesis
      fixed_by_abstaining         raw wrong -> honest abstention
      wrong_answer_kept           raw wrong, verifier accepted it (verifier error or misleading evidence)
      wrong_competitor_accepted   a new hypothesis won but is wrong
      correct_answer_replaced     raw correct -> replaced by a wrong answer (harm)
      over_abstention             raw correct -> abstained (harm)
      no_evidence                 nothing retrieved (KB / retrieval coverage problem)
      unusable_model_output       a knowledge source failed on unparseable model output
      budget_exhausted            stopped by the LLM-call budget
  * a few examples per category, for reading by hand.
Writes <run>/trace_report.md.
"""
import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

CORRECT, ABSTAINED, HALLUCINATED = "CORRECT", "ABSTAINED", "HALLUCINATED"


def categorize(row, trace):
    cycles = trace.get("cycles") or []
    notes = [str(c.get("note") or "") for c in cycles]
    mode = next((c.get("note") for c in reversed(cycles) if c.get("ks") == "Composer"), "")
    passages = 0
    for c in cycles:
        if c.get("ks") in ("Retrieval", "WebSearch"):
            m = re.match(r"(\d+)", str(c.get("note") or ""))
            passages += int(m.group(1)) if m else 0
    raw, fin = row["raw_label"], row["final_label"]
    cats = []
    if any(n.startswith("failed:") for n in notes):
        cats.append("unusable_model_output")
    if row.get("stop_reason") == "budget":
        cats.append("budget_exhausted")
    if passages == 0:
        cats.append("no_evidence")
    if raw == HALLUCINATED and fin == CORRECT and mode == "replace":
        cats.append("fixed_by_competing_answer")
    elif raw == HALLUCINATED and fin == ABSTAINED:
        cats.append("fixed_by_abstaining")
    elif raw == HALLUCINATED and fin == HALLUCINATED and mode in ("keep", "keep_no_budget"):
        cats.append("wrong_answer_kept")
    elif fin == HALLUCINATED and mode == "replace" and raw != CORRECT:
        cats.append("wrong_competitor_accepted")
    if raw == CORRECT and fin == HALLUCINATED:
        cats.append("correct_answer_replaced")
    if raw == CORRECT and fin == ABSTAINED:
        cats.append("over_abstention")
    if any(c.get("ks") == "Confirmer" and not str(c.get("note") or "").startswith("SUPPORTED") for c in cycles):
        cats.append("second_model_disagreed")
    if any("not written in the evidence" in n for n in notes):
        cats.append("ungrounded_answer_discarded")
    return cats, mode, passages


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="eval_pipeline.py output dir of a blackboard run")
    ap.add_argument("--examples", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run)
    rows = {r["row_index"]: r for r in csv.DictReader(open(run / "per_item.csv", encoding="utf-8"))}
    traces = {str(t["id"]): t for t in (json.loads(l) for l in open(run / "traces.jsonl", encoding="utf-8") if l.strip())}
    if not traces:
        raise SystemExit("no traces in this run (traces exist only for --system blackboard)")

    ks_use, stops, n_cycles, n_failed = Counter(), Counter(), [], 0
    by_cat = defaultdict(list)
    for rid, t in traces.items():
        row = rows.get(rid)
        if row is None:
            continue
        cyc = [c for c in t.get("cycles") or [] if c.get("ks") != "Composer"]
        n_cycles.append(len(cyc))
        ks_use.update(c["ks"] for c in cyc)
        n_failed += sum(1 for c in cyc if str(c.get("note") or "").startswith("failed:"))
        stops[row.get("stop_reason") or "?"] += 1
        cats, mode, passages = categorize(row, t)
        for c in cats:
            by_cat[c].append((row, t, mode, passages))

    total_cycles = sum(n_cycles) or 1
    lines = [f"# Trace report: {run}", "", f"Mitigated items with a trace: {len(n_cycles)}; "
             f"mean cycles {sum(n_cycles) / max(1, len(n_cycles)):.2f}; "
             f"unusable model outputs: {n_failed}/{total_cycles} cycles ({100 * n_failed / total_cycles:.1f}%).", "",
             "Stop reasons: " + ", ".join(f"{k}={v}" for k, v in stops.most_common()), "",
             "Knowledge-source activations: " + ", ".join(f"{k}={v}" for k, v in ks_use.most_common()), "",
             "| category | items |", "|---|---|"]
    order = ["fixed_by_competing_answer", "fixed_by_abstaining", "wrong_answer_kept", "wrong_competitor_accepted",
             "correct_answer_replaced", "over_abstention", "no_evidence", "unusable_model_output", "budget_exhausted"]
    for c in order:
        lines.append(f"| {c} | {len(by_cat.get(c, []))} |")
    rng = random.Random(args.seed)
    for c in order:
        items = by_cat.get(c, [])
        if not items:
            continue
        lines += ["", f"## {c} ({len(items)})"]
        for row, t, mode, passages in rng.sample(items, min(args.examples, len(items))):
            seq = " > ".join(f"{x['ks']}[{x.get('note', '')}]" for x in t.get("cycles") or [])
            hyps = "; ".join(f"{h['text']!r}:{h['status']}" for h in t.get("hypotheses") or [])
            lines += ["", f"- **Q:** {row['question']}  **gold:** {row['gold']}",
                      f"  **raw:** {row['raw_answer']!r} ({row['raw_label']}) -> **final:** {row['final_response']!r} "
                      f"({row['final_label']}); passages {passages}; composer {mode}",
                      f"  hypotheses: {hyps}", f"  trace: {seq[:900]}"]
    (run / "trace_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:14 + len(order)]))
    print(f"\nfull report with examples -> {run / 'trace_report.md'}")


if __name__ == "__main__":
    main()
