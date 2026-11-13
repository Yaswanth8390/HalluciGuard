#!/usr/bin/env python3
"""
choose_threshold.py - pick the probe's risk threshold on a DEV split (never on the eval rows).

  python eval_pipeline.py --dataset eval_data/popqa_dev.jsonl --generate_only --probe probe.joblib --out_dir runs/popqa_dev
  python choose_threshold.py --dev runs/popqa_dev/generations.jsonl --out runs/popqa_dev/threshold.json

Each dev answer is graded against its gold aliases (same lenient rule as eval_pipeline.py). Positive class =
HALLUCINATED; abstentions are excluded, because they are always routed to mitigation anyway. Prints the
probe's AUROC on the dev split and the threshold that maximizes the chosen criterion:
  f1      (default) F1 of "flag if score >= t" for catching wrong answers
  youden  TPR - FPR
  recall  the highest threshold that still catches >= --min_recall of the wrong answers
            (mitigation is cheap compared to a missed hallucination, so a recall target is often preferred)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_pipeline import ABSTAINED, HALLUCINATED, auroc, grade, load_dataset  # noqa: E402


def choose(scores, wrong, criterion="f1", min_recall=0.8):
    P = sum(wrong)
    N = len(wrong) - P
    best = None
    for t in sorted(set(scores)):
        tp = sum(1 for s, y in zip(scores, wrong) if s >= t and y)
        fp = sum(1 for s, y in zip(scores, wrong) if s >= t and not y)
        rec = tp / P if P else 0.0
        prec = tp / (tp + fp) if tp + fp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        fpr = fp / N if N else 0.0
        flagged = (tp + fp) / len(wrong)
        row = {"threshold": t, "precision": prec, "recall": rec, "f1": f1, "youden": rec - fpr, "flagged_rate": flagged}
        if criterion == "recall":
            key = (rec >= min_recall, t)          # highest threshold meeting the recall target
        else:
            key = (row[criterion], -flagged)      # ties: flag fewer answers
        if best is None or key > best[0]:
            best = (key, row)
    return best[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dev", required=True, help="dev generations.jsonl (answers + prob_hallucinated + gold)")
    ap.add_argument("--criterion", default="f1", choices=["f1", "youden", "recall"])
    ap.add_argument("--min_recall", type=float, default=0.8)
    ap.add_argument("--match", default="lenient", choices=["lenient", "strict"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = load_dataset(args.dev, "generic")
    pairs = []
    for r in recs:
        if r.get("prob_hallucinated") is None or "answer" not in r:
            continue
        lab = grade(r["answer"], r["gold"], args.match)
        if lab != ABSTAINED:
            pairs.append((r["prob_hallucinated"], lab == HALLUCINATED))
    if not pairs:
        raise SystemExit("no dev rows with prob_hallucinated: generate the dev split with --probe")
    scores, wrong = [p[0] for p in pairs], [p[1] for p in pairs]
    res = {"n": len(pairs), "n_wrong": sum(wrong), "n_abstained_excluded": len(recs) - len(pairs),
           "auroc": auroc(scores, [int(w) for w in wrong]), "criterion": args.criterion}
    if 0 < sum(wrong) < len(wrong):
        res.update(choose(scores, wrong, args.criterion, args.min_recall))
    else:
        res.update({"threshold": 0.70, "note": "dev split has only one class; falling back to 0.70"})
    print(json.dumps(res, indent=2))
    print(f"THRESHOLD={res['threshold']:.6f}")
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
