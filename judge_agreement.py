#!/usr/bin/env python3
"""
judge_agreement.py - validate the FActScore judge against human labels (reviewers will ask for this).

1) export a stratified sample of judged atomic facts for annotation:
     python judge_agreement.py export --bio_dir runs/bio --answers_file runs/bio/generations.jsonl \
         --kb eval_data/bio_kb.jsonl --n 200 --out runs/bio/annotation
   -> annotation/annotate.csv   fact + Wikipedia context, EMPTY label columns (the judge's label is hidden,
                                so annotators are not biased by it)
      annotation/key.csv        sample_id -> judge label (do not show this to the annotators)
   Fill `human_label_1` (and, ideally, `human_label_2` by a second person) with 1 = supported by the
   context / 0 = not supported. Judge as FActScore does: "supported" only if the context supports it.

2) score the agreement:
     python judge_agreement.py score --dir runs/bio/annotation
   -> accuracy and Cohen's kappa of the judge vs. each annotator and vs. their consensus, plus
      inter-annotator kappa. Writes agreement.md / agreement.json in the same folder.
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

TRUE = {"1", "true", "t", "yes", "y", "s", "supported"}
FALSE = {"0", "false", "f", "no", "n", "ns", "not supported", "unsupported"}


def parse_label(v):
    v = str(v or "").strip().lower()
    if v in TRUE:
        return 1
    if v in FALSE:
        return 0
    return None


def cohen_kappa(a, b):
    n = len(a)
    if n == 0:
        return None
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa, pb = sum(a) / n, sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def _read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").split("\n") if l.strip()]


def export(args):
    import factscore_eval as fe
    items = _read_jsonl(args.answers_file)
    kb = fe.load_kb_by_entity(args.kb)
    bio = Path(args.bio_dir)
    sources = {"raw": bio / "factscore_raw.jsonl"}
    for d in sorted(p for p in bio.iterdir() if p.is_dir() and (p / "factscore.jsonl").exists()):
        sources[d.name] = d / "factscore.jsonl"
    pool = defaultdict(list)   # (system, judge_label) -> facts
    for system, path in sources.items():
        if not path.exists():
            continue
        for it, s in zip(items, _read_jsonl(path)):
            for fact, lab in zip(s.get("facts") or [], s.get("supported") or []):
                pool[(system, int(bool(lab)))].append({"system": system, "entity": it["entity"], "fact": fact,
                                                       "judge_label": int(bool(lab))})
    if not pool:
        raise SystemExit(f"no scored facts found under {bio} (run eval_bio.py with scoring first)")
    rng = random.Random(args.seed)
    keys = sorted(pool)
    per = max(1, args.n // len(keys))
    picked = []
    for k in keys:   # equal share per (system, judge label): both labels and every system are represented
        rng.shuffle(pool[k])
        picked += pool[k][:per]
    rest = [x for k in keys for x in pool[k][per:]]
    rng.shuffle(rest)
    picked += rest[: max(0, args.n - len(picked))]
    rng.shuffle(picked)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "annotate.csv", "w", newline="", encoding="utf-8") as fa, \
            open(out / "key.csv", "w", newline="", encoding="utf-8") as fk:
        wa, wk = csv.writer(fa), csv.writer(fk)
        wa.writerow(["sample_id", "entity", "fact", "wikipedia_context", "human_label_1", "human_label_2", "notes"])
        wk.writerow(["sample_id", "system", "judge_label"])
        for i, x in enumerate(picked):
            ctx = " || ".join(fe.rank_passages(x["fact"], kb.get(x["entity"], []), 3))
            wa.writerow([i, x["entity"], x["fact"], ctx, "", "", ""])
            wk.writerow([i, x["system"], x["judge_label"]])
    cnt = Counter((x["system"], x["judge_label"]) for x in picked)
    print(f"exported {len(picked)} facts -> {out / 'annotate.csv'} (key: {out / 'key.csv'})")
    print("per (system, judge label):", dict(sorted(cnt.items())))


def score(args):
    d = Path(args.dir)
    key = {r["sample_id"]: r for r in csv.DictReader(open(d / "key.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(d / "annotate.csv", encoding="utf-8")))
    res = {"n_exported": len(rows)}
    lines = ["# FActScore judge vs. human labels", ""]
    for col in ("human_label_1", "human_label_2"):
        pairs = [(int(key[r["sample_id"]]["judge_label"]), parse_label(r.get(col))) for r in rows]
        pairs = [(j, h) for j, h in pairs if h is not None]
        if not pairs:
            continue
        j, h = [p[0] for p in pairs], [p[1] for p in pairs]
        res[col] = {"n": len(pairs), "accuracy": sum(1 for a, b in pairs if a == b) / len(pairs),
                    "kappa": cohen_kappa(j, h), "human_supported_rate": sum(h) / len(h),
                    "judge_supported_rate": sum(j) / len(j)}
        lines.append(f"- judge vs {col}: accuracy {res[col]['accuracy']:.3f}, Cohen's kappa {res[col]['kappa']:.3f} "
                     f"(n={len(pairs)}; supported: human {res[col]['human_supported_rate']:.2f}, judge {res[col]['judge_supported_rate']:.2f})")
    both = [(parse_label(r.get("human_label_1")), parse_label(r.get("human_label_2")), int(key[r["sample_id"]]["judge_label"]))
            for r in rows]
    both = [b for b in both if b[0] is not None and b[1] is not None]
    if both:
        res["inter_annotator_kappa"] = cohen_kappa([b[0] for b in both], [b[1] for b in both])
        agree = [b for b in both if b[0] == b[1]]
        res["consensus"] = {"n": len(agree), "accuracy": sum(1 for a, _, j in agree if a == j) / max(1, len(agree)),
                            "kappa": cohen_kappa([b[2] for b in agree], [b[0] for b in agree]) if agree else None}
        lines.append(f"- inter-annotator kappa: {res['inter_annotator_kappa']:.3f} (n={len(both)})")
        if agree:
            lines.append(f"- judge vs consensus (both annotators agree, n={len(agree)}): accuracy "
                         f"{res['consensus']['accuracy']:.3f}, kappa {res['consensus']['kappa']:.3f}")
    by_sys = defaultdict(list)
    for r in rows:
        h = parse_label(r.get("human_label_1"))
        if h is not None:
            by_sys[key[r["sample_id"]]["system"]].append(int(key[r["sample_id"]]["judge_label"]) == h)
    if by_sys:
        res["accuracy_by_system"] = {k: sum(v) / len(v) for k, v in by_sys.items()}
        lines.append("- judge accuracy by system (vs annotator 1): " +
                     ", ".join(f"{k} {sum(v) / len(v):.2f} (n={len(v)})" for k, v in sorted(by_sys.items())))
    if len(lines) == 2:
        raise SystemExit("no human labels found: fill human_label_1 in annotate.csv first")
    lines += ["", "Kappa >= 0.6 is usually read as substantial agreement. Report it next to the FActScore numbers; "
                  "similar accuracy across systems shows the judge is not biased toward one system's style."]
    (d / "agreement.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (d / "agreement.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--bio_dir", required=True)
    e.add_argument("--answers_file", required=True)
    e.add_argument("--kb", required=True)
    e.add_argument("--n", type=int, default=200)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("--dir", required=True)
    args = ap.parse_args()
    export(args) if args.cmd == "export" else score(args)


if __name__ == "__main__":
    main()
