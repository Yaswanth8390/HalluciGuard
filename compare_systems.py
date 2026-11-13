#!/usr/bin/env python3
"""
compare_systems.py - paired comparison of several eval_pipeline.py runs on the SAME generations.

  python compare_systems.py --runs blackboard=eval_runs/popqa/blackboard pipeline=eval_runs/popqa/pipeline \
      cove=eval_runs/popqa/cove crag=eval_runs/popqa/crag --reference pipeline --out eval_runs/popqa/comparison.md

Each run directory must hold per_item.csv (written by eval_pipeline.py). Rows are paired by row_index.
For every system: accuracy / hallucination / abstention (raw answers shown once as "no mitigation"),
LLM calls per mitigated item, and versus --reference: paired-bootstrap 95% CIs of the hallucination-rate
and accuracy differences and an exact McNemar test on hallucinations (fixed vs broken items).
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pipeline import ABSTAINED, CORRECT, HALLUCINATED, PROCESSING_ERROR, mcnemar_exact  # noqa: E402


def load(run_dir):
    rows = list(csv.DictReader(open(Path(run_dir) / "per_item.csv", encoding="utf-8")))
    return {r["row_index"]: r for r in rows}


def rates(labels):
    n = len(labels) or 1
    return {"n": len(labels), "accuracy": labels.count(CORRECT) / n, "hallucination": labels.count(HALLUCINATED) / n,
            "abstention": labels.count(ABSTAINED) / n, "errors": labels.count(PROCESSING_ERROR) / n}


def boot(a, b, fn, iters=2000, seed=0):
    rng = random.Random(seed)
    idx = list(range(len(a)))
    diffs = []
    for _ in range(iters):
        s = [rng.choice(idx) for _ in idx]
        diffs.append(fn([b[i] for i in s]) - fn([a[i] for i in s]))
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir ...")
    ap.add_argument("--reference", default=None, help="system to compare against (default: first)")
    ap.add_argument("--out", default=None, help="write the markdown table here (+ .json next to it)")
    args = ap.parse_args()

    runs = {}
    for spec in args.runs:
        name, _, d = spec.partition("=")
        runs[name] = load(d)
    ref = args.reference or next(iter(runs))
    common = sorted(set.intersection(*(set(r) for r in runs.values())), key=lambda x: int(x) if x.isdigit() else x)
    if not common:
        raise SystemExit("no common row_index across runs")
    first = runs[ref]
    raws = {k: first[k]["raw_answer"] for k in common}
    for name, r in runs.items():
        bad = [k for k in common if r[k]["raw_answer"] != raws[k]]
        if bad:
            raise SystemExit(f"{name}: raw answers differ from {ref} on {len(bad)} items; runs must share generations")

    raw_labels = [first[k]["raw_label"] for k in common]
    lab = {n: [r[k]["final_label"] for k in common] for n, r in runs.items()}
    hal = lambda L: L.count(HALLUCINATED) / len(L)
    acc = lambda L: L.count(CORRECT) / len(L)
    out = {"n": len(common), "reference": ref, "no_mitigation": rates(raw_labels), "systems": {}}
    pct = lambda v: f"{100 * v:.1f}"
    md = [f"# System comparison (n={len(common)} paired items, reference = {ref})", "",
          "| system | accuracy | hallucination | abstention | LLM calls / question | Δ halluc. vs ref [95% CI] | Δ acc. vs ref [95% CI] | fixed / broken vs ref (McNemar p) |",
          "|---|---|---|---|---|---|---|---|"]
    m0 = out["no_mitigation"]
    md.append(f"| no mitigation (raw) | {pct(m0['accuracy'])} | {pct(m0['hallucination'])} | {pct(m0['abstention'])} | 0 | | | |")
    for name, L in lab.items():
        m = rates(L)
        calls = [float(runs[name][k]["llm_calls"]) for k in common if runs[name][k].get("llm_calls") not in (None, "")]
        m["llm_calls_per_question"] = sum(calls) / len(calls) if calls else None
        if name != ref:
            R = lab[ref]
            fixed = sum(1 for a, b in zip(R, L) if a == HALLUCINATED and b != HALLUCINATED)
            broken = sum(1 for a, b in zip(R, L) if a != HALLUCINATED and b == HALLUCINATED)
            m["vs_reference"] = {"halluc_diff": hal(L) - hal(R), "halluc_ci95": boot(R, L, hal),
                                 "acc_diff": acc(L) - acc(R), "acc_ci95": boot(R, L, acc),
                                 "fixed": fixed, "broken": broken, "mcnemar_p": mcnemar_exact(fixed, broken)}
            v = m["vs_reference"]
            cmp_ = (f"{100 * v['halluc_diff']:+.1f} [{100 * v['halluc_ci95'][0]:+.1f}, {100 * v['halluc_ci95'][1]:+.1f}] | "
                    f"{100 * v['acc_diff']:+.1f} [{100 * v['acc_ci95'][0]:+.1f}, {100 * v['acc_ci95'][1]:+.1f}] | "
                    f"{fixed} / {broken} (p={v['mcnemar_p']:.4f})")
        else:
            cmp_ = "reference | | "
        lc = "-" if m["llm_calls_per_question"] is None else f"{m['llm_calls_per_question']:.2f}"
        md.append(f"| {name} | {pct(m['accuracy'])} | {pct(m['hallucination'])} | {pct(m['abstention'])} | {lc} | {cmp_} |")
        out["systems"][name] = m
    md += ["", "Δ = system minus reference; negative Δ hallucination = fewer hallucinations than the reference. "
               "Fixed/broken count items the reference got wrong and the system did not, and vice versa."]
    text = "\n".join(md)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        Path(args.out).with_suffix(".json").write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
