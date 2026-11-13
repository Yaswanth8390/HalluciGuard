#!/usr/bin/env python3
"""
plot_cost_accuracy.py - accuracy / hallucination vs. LLM cost for several eval_pipeline.py runs.

  python plot_cost_accuracy.py --runs blackboard=runs/popqa/blackboard pipeline=runs/popqa/pipeline \
      cove=runs/popqa/cove crag=runs/popqa/crag bb_b4=runs/popqa/bb_budget4 bb_b16=runs/popqa/bb_budget16 \
      --out runs/popqa/cost_accuracy

Reads each run's metrics.json. Cost = LLM calls per question over ALL questions (a probe gate that skips
answers makes a system cheaper; that is part of the result). The raw answers are drawn at 0 calls.
Writes <out>.md (table) and <out>.png (if matplotlib is available).
"""
import argparse
import json
from pathlib import Path


def load(run_dir):
    m = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    lc = m.get("llm_calls") or {}
    return {"accuracy": m["with_blackboard"]["accuracy"], "hallucination": m["with_blackboard"]["hallucination_rate"],
            "abstention": m["with_blackboard"]["abstention_rate"], "calls_per_question": lc.get("per_question"),
            "calls_per_mitigated": lc.get("mean"), "budget": m["config"].get("max_llm_calls"),
            "threshold": m["config"].get("threshold"),
            "raw": {"accuracy": m["without_blackboard"]["accuracy"],
                    "hallucination": m["without_blackboard"]["hallucination_rate"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="name=run_dir ...")
    ap.add_argument("--out", required=True, help="output path without extension")
    args = ap.parse_args()
    pts = {}
    for spec in args.runs:
        name, _, d = spec.partition("=")
        pts[name] = load(d)
    raw = next(iter(pts.values()))["raw"]
    pct = lambda v: "-" if v is None else f"{100 * v:.1f}"
    num = lambda v: "-" if v is None else f"{v:.2f}"
    lines = ["| system | LLM calls / question | calls / mitigated item | budget | threshold | accuracy | hallucination | abstention |",
             "|---|---|---|---|---|---|---|---|",
             f"| raw (no mitigation) | 0 | - | - | - | {pct(raw['accuracy'])} | {pct(raw['hallucination'])} | - |"]
    for n, p in sorted(pts.items(), key=lambda kv: kv[1]["calls_per_question"] or 0):
        lines.append(f"| {n} | {num(p['calls_per_question'])} | {num(p['calls_per_mitigated'])} | {p['budget'] or '-'} | "
                     f"{p['threshold']} | {pct(p['accuracy'])} | {pct(p['hallucination'])} | {pct(p['abstention'])} |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".md").write_text("# Cost vs. accuracy\n\n" + "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(matplotlib not installed: table only)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, key, label in ((axes[0], "accuracy", "accuracy (%)"), (axes[1], "hallucination", "hallucination rate (%)")):
        ax.scatter([0], [100 * raw[key]], marker="x", color="#555555")
        ax.annotate("raw", (0, 100 * raw[key]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        for n, p in pts.items():
            if p["calls_per_question"] is None:
                continue
            ax.scatter([p["calls_per_question"]], [100 * p[key]])
            ax.annotate(n, (p["calls_per_question"], 100 * p[key]), textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("LLM calls per question")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"saved {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
