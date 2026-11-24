#!/usr/bin/env python3
"""Build the FActScore biography eval set + its knowledge base.

  python prepare_factscore.py --n 50 --out eval_data/bio_50.jsonl

Source: HuggingFace `dskar/FActScore` - the 500 entities of FActScore's unlabeled set (Min et al., 2023),
each with its Wikipedia article (scraped 2025-07-29). Note: the original FActScore used a 2023-04-01
Wikipedia dump; report that the knowledge source here is the 2025 scrape.

Writes
  <out>            one row per entity: {"id", "entity", "question": "Tell me a bio of <entity>.", "group": "bio"}
  <out>_kb.jsonl   the entities' articles split into passages: {"text", "entity"}
                   (= the retrieval KB for eval_bio.py AND the knowledge source of factscore_eval.py)
"""
import argparse
import json
import random
from pathlib import Path

from kb_utils import chunk_article

HF_URL = "https://huggingface.co/datasets/dskar/FActScore/resolve/main/factscore_eval_20250729_222645.jsonl"


def load_rows(path=None):
    if path:
        text = Path(path).read_text(encoding="utf-8")
    else:
        import requests
        r = requests.get(HF_URL, timeout=120)
        r.raise_for_status()
        text = r.text
    return [json.loads(l) for l in text.split("\n") if l.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=50, help="number of entities (max 500)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kb_out", default=None)
    ap.add_argument("--source_file", default=None, help="local copy of the dskar/FActScore jsonl (offline)")
    ap.add_argument("--passage_words", type=int, default=120)
    ap.add_argument("--max_passages", type=int, default=40, help="passages kept per article")
    args = ap.parse_args()

    rows = [r for r in load_rows(args.source_file) if r.get("entity") and r.get("wikipedia_text")]
    order = list(range(len(rows)))
    random.Random(args.seed).shuffle(order)   # seeded sample; ids are positions in the source file
    picked = sorted(order[: args.n])
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kb_path = Path(args.kb_out or str(out_path.with_suffix("")) + "_kb.jsonl")
    n_pass = 0
    with open(out_path, "w", encoding="utf-8") as fo, open(kb_path, "w", encoding="utf-8") as fk:
        for i in picked:
            r = rows[i]
            ent = r["entity"].strip()
            q = (r.get("factscore_prompt") or f"Tell me a bio of {ent}.").strip()
            fo.write(json.dumps({"id": i, "entity": ent, "question": q, "group": "bio"}, ensure_ascii=False) + "\n")
            for p in chunk_article(ent, r["wikipedia_text"], args.passage_words, args.max_passages):
                fk.write(json.dumps({"text": p, "entity": ent}, ensure_ascii=False) + "\n")
                n_pass += 1
    print(f"wrote {len(picked)} entities -> {out_path}; {n_pass} passages -> {kb_path}")


if __name__ == "__main__":
    main()
