#!/usr/bin/env python3
"""
eval_bio.py - long-form evaluation on FActScore biographies: raw Qwen vs. each mitigation system.

  # 1) data (once):            python prepare_factscore.py --n 50 --out eval_data/bio_50.jsonl
  # 2) generate (GPU, Qwen):   python eval_bio.py --dataset eval_data/bio_50.jsonl --generate_only --out_dir eval_runs/bio
  # 3) mitigate + score:       python eval_bio.py --answers_file eval_runs/bio/generations.jsonl \
  #                                --kb eval_data/bio_50_kb.jsonl --systems blackboard,pipeline,cove,crag \
  #                                --out_dir eval_runs/bio

Stage 2 reuses eval_pipeline.run_blackboard_stage (same KB loading, memory isolation, threads, retries and
per-item checkpoints as the PopQA eval). The KB is every passage of every entity's article in one shared
collection, so retrieval also has to find the right entity. Stage 3 scores the raw answers and each
system's answers with factscore_eval.py (judge = LLM_JUDGE_* role) and reports FActScore, FActScore with the
gamma=10 length penalty, % responding, facts per response, LLM calls per item, and a paired bootstrap CI of
each system's FActScore change vs. raw (items where both responded).

Everything is checkpointed under --out_dir; rerunning the same command resumes.
"""
import argparse
import json
import random
import sys
from argparse import Namespace
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

BIO_SYSTEM_PROMPT = ("You are a helpful assistant. Answer with a short factual biography of about 100-150 words "
                     "in plain prose. If you do not know who the person is, say so instead of guessing.")


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def generate(records, args):
    import joblib
    import torch
    sys.path.insert(0, str(_ROOT / "probe"))
    from llama_features import QwenResidualFeatureExtractor
    clf = scaler = None
    if args.probe:
        b = joblib.load(args.probe)
        scaler, clf = b["scaler"], b["classifier"]
    ex = QwenResidualFeatureExtractor(model_id=args.model_id, layer=args.layer)
    for s in range(0, len(records), args.batch_size):
        batch = records[s:s + args.batch_size]
        gens = ex.batch_generate_and_extract_features([r["question"] for r in batch], max_new_tokens=args.max_new_tokens,
                                                      stop_strings=(), system_prompt=BIO_SYSTEM_PROMPT)
        for r, g in zip(batch, gens):
            r["answer"] = g["answer_text"]
            if clf is not None:
                X = g["feature_vector"].cpu().to(torch.float32).numpy()[None, :]
                r["prob_hallucinated"] = float(clf.predict_proba(scaler.transform(X))[:, 1][0])
        print(f"[generate] {min(s + args.batch_size, len(records))}/{len(records)}", flush=True)
    return records


def paired_ci(a, b, iters=2000, seed=0):
    """Mean of b - a with a 95% bootstrap CI (paired)."""
    d = [y - x for x, y in zip(a, b)]
    if not d:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(iters))
    return {"mean_diff": sum(d) / len(d), "ci95_low": means[int(0.025 * iters)], "ci95_high": means[int(0.975 * iters) - 1],
            "n_pairs": len(d)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", help="bio items from prepare_factscore.py")
    ap.add_argument("--answers_file", help="generations.jsonl from a --generate_only run")
    ap.add_argument("--kb", help="<bio>_kb.jsonl from prepare_factscore.py")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    # stage 1
    ap.add_argument("--generate_only", action="store_true")
    ap.add_argument("--probe", default=None)
    ap.add_argument("--model_id", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--max_new_tokens", type=int, default=320)
    ap.add_argument("--batch_size", type=int, default=4)
    # stage 2
    ap.add_argument("--systems", default="blackboard,pipeline,cove,crag")
    ap.add_argument("--max_llm_calls", type=int, default=24, help="budget per biography for blackboard/cove/crag")
    ap.add_argument("--disable_ks", default="", help="ablation for the blackboard system (comma list)")
    ap.add_argument("--threshold", type=float, default=0.0, help="0 = mitigate every answer (no probe needed)")
    ap.add_argument("--default_score", type=float, default=1.0)
    ap.add_argument("--enable_web", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--chroma_dir", default="./eval_chroma_bio")
    ap.add_argument("--no_llm_abstention", action="store_true", default=True)
    # stage 3
    ap.add_argument("--judge_batch", type=int, default=8)
    ap.add_argument("--skip_scoring", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.answers_file:
        records = read_jsonl(args.answers_file)
    else:
        if not args.dataset:
            ap.error("--dataset or --answers_file is required")
        records = read_jsonl(args.dataset)
    if args.limit:
        records = records[: args.limit]
    if not args.answers_file:
        records = generate(records, args)
        write_jsonl(out / "generations.jsonl", records)
        print(f"saved {out / 'generations.jsonl'}")
        if args.generate_only:
            return
    if not args.kb:
        ap.error("--kb is required for mitigation/scoring")

    import eval_pipeline as ev
    for r in records:
        r.setdefault("gold", [""])
        r.setdefault("knowledge", "")
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    disabled = ",".join(sorted(x.strip() for x in args.disable_ks.split(",") if x.strip())) or None

    results = {}
    for system in systems:
        tag = system + (f"-minus-{disabled.replace(',', '-')}" if disabled and system == "blackboard" else "")
        sdir = out / tag
        sdir.mkdir(exist_ok=True)
        rows_path = sdir / "rows.jsonl"
        if rows_path.exists() and len(read_jsonl(rows_path)) == len(records):
            rows = [x["row"] for x in read_jsonl(rows_path)]
            print(f"[{tag}] reusing {rows_path}")
        else:
            a = Namespace(threshold=args.threshold, default_score=args.default_score, chroma_dir=args.chroma_dir,
                          no_knowledge=False, oracle_knowledge=False, kb_coverage=None, kb_seed=None,
                          kb_file=args.kb, kb_file_sha256=ev._sha256(args.kb), enable_web=args.enable_web,
                          isolate_memory=True, fallback_on_error=False, no_llm_abstention=args.no_llm_abstention,
                          workers=args.workers, retries=args.retries, seed=0, system=system,
                          max_llm_calls=None if system == "pipeline" else args.max_llm_calls,
                          disable_ks=disabled if system == "blackboard" else None, max_items=0)
            ev.KB_POOL[:] = records
            rows, bc = ev.run_blackboard_stage(records, a, partial_path=sdir / "bb_partial.jsonl")
            write_jsonl(rows_path, [{"id": r.get("id"), "entity": r.get("entity"), "row": x} for r, x in zip(records, rows)])
        results[tag] = rows

    if args.skip_scoring:
        return
    import factscore_eval as fe
    caller, judge_cfg = fe.get_judge_caller()
    kb = fe.load_kb_by_entity(args.kb)

    def scored(path, items, field):
        if path.exists() and len(read_jsonl(path)) == len(items):
            return read_jsonl(path)
        s = fe.score_many(items, field, kb, caller, args.judge_batch, args.workers)
        write_jsonl(path, s)
        return s

    print("[score] raw answers...")
    raw_scores = scored(out / "factscore_raw.jsonl", records, "answer")
    table = {"raw": {**fe.aggregate(raw_scores), "llm_calls_mean": 0.0}}
    paired = {}
    for tag, rows in results.items():
        print(f"[score] {tag}...")
        items = [{"entity": r["entity"], "final_response": x["final_response"]} for r, x in zip(records, rows)]
        s = scored(out / tag / "factscore.jsonl", items, "final_response")
        calls = [x.get("llm_calls") for x in rows if x.get("llm_calls") is not None]
        errors = sum(1 for x in rows if x["status"] in ("ERROR", "ERROR_FALLBACK"))
        table[tag] = {**fe.aggregate(s), "llm_calls_mean": (sum(calls) / len(calls)) if calls else None,
                      "processing_errors": errors}
        both = [(a_["score"], b_["score"]) for a_, b_ in zip(raw_scores, s)
                if a_.get("score") is not None and b_.get("score") is not None]
        paired[tag] = paired_ci([x for x, _ in both], [y for _, y in both])

    # does the probe score (trained on short answers) say anything about long-form factuality?
    probe_rep = None
    pp = [(r["prob_hallucinated"], s["score"]) for r, s in zip(records, raw_scores)
          if r.get("prob_hallucinated") is not None and s.get("score") is not None]
    if len(pp) >= 10:
        med = sorted(x for _, x in pp)[len(pp) // 2]
        low = [int(x < med) for _, x in pp]
        probe_rep = {"n": len(pp), "auroc_low_factscore": ev.auroc([p for p, _ in pp], low),
                     "definition": "positive = raw FActScore below the median", "median_factscore": med}
    metrics = {"probe_vs_factscore": probe_rep,
               "n": len(records), "systems": table, "paired_vs_raw": paired, "judge": judge_cfg,
               "max_llm_calls": args.max_llm_calls, "disable_ks": disabled, "threshold": args.threshold,
               "knowledge_source": "dskar/FActScore Wikipedia text (scraped 2025-07-29)"}
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    pct = lambda v: "n/a" if v is None else f"{100 * v:.1f}"
    lines = [f"# FActScore biographies (n={len(records)})", "",
             "| system | FActScore | FActScore (gamma=10) | % responding | facts / response | LLM calls / item | change vs raw [95% CI] |",
             "|---|---|---|---|---|---|---|"]
    for tag, m in table.items():
        p = paired.get(tag)
        ch = "-" if not p else f"{100 * p['mean_diff']:+.1f} [{100 * p['ci95_low']:+.1f}, {100 * p['ci95_high']:+.1f}] (n={p['n_pairs']})"
        lc = m.get("llm_calls_mean")
        lines.append(f"| {tag} | {pct(m['factscore'])} | {pct(m.get('factscore_gamma10'))} | {pct(m['response_ratio'])} | "
                     f"{m['facts_per_response']:.1f} | {'-' if lc is None else f'{lc:.1f}'} | {ch} |")
    if probe_rep and probe_rep["auroc_low_factscore"] is not None:
        lines += ["", f"Probe on long-form answers: AUROC {probe_rep['auroc_low_factscore']:.3f} for flagging "
                      f"below-median raw FActScore (n={probe_rep['n']}); the probe was trained on short answers."]
    lines += ["", f"Judge: {judge_cfg}. Knowledge source: {metrics['knowledge_source']}.",
              "FActScore is computed over responses that did not abstain; read it together with % responding."]
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
