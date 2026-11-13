#!/usr/bin/env python3
"""Build closed-book eval files for eval_pipeline.py from PopQA and TriviaQA.

  python prepare_eval_data.py --dataset popqa    --n 1000 --out eval_data/popqa_1000.jsonl
  python prepare_eval_data.py --dataset triviaqa --n 1000 --out eval_data/triviaqa_1000.jsonl

Then:
  python eval_pipeline.py --dataset eval_data/popqa_1000.jsonl --probe probe/probe.joblib ...

Output rows: {"question", "answers": [aliases...], "group", "s_pop"?, "prop"?}
  PopQA   : group = popularity tercile of the subject (head / torso / tail), computed over the
            FULL PopQA test set, then sampled equally from each tercile (seeded).
  TriviaQA: group = "triviaqa" (validation split of unfiltered.nocontext; train is not used).

With --wiki (PopQA only) the subject's Wikipedia article is fetched and split into passages:
  * `knowledge` = the passage that contains a gold answer (used by eval_pipeline --kb_coverage F)
  * all passages of all items -> <out>_kb.jsonl (used by eval_pipeline --kb_file: realistic retrieval)

Sources: HuggingFace `akariasai/PopQA` (split test) and `mandarjoshi/trivia_qa`
(config unfiltered.nocontext, split validation). Use --popqa_file / --triviaqa_file to read a
local json/jsonl/csv export instead (offline machines).
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path


def read_local(path):
    p = Path(path)
    if p.suffix.lower() == ".csv":
        with open(p, newline="", encoding="utf-8") as f:
            return list(csv.DictReader(f))
    text = p.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.split("\n") if l.strip()]


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def clean_aliases(aliases, min_len=2):
    """Dedupe, drop empties and 1-char aliases (they cause false 'correct' matches)."""
    seen, out = set(), []
    for a in aliases:
        a = str(a).strip()
        k = norm(a)
        if len(k) < min_len or k in seen:
            continue
        seen.add(k)
        out.append(a)
    return out


def parse_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            x = json.loads(v)
            return x if isinstance(x, list) else [x]
        except json.JSONDecodeError:
            return [v]
    return []


def load_popqa(path=None):
    if path:
        return read_local(path)
    from datasets import load_dataset
    return list(load_dataset("akariasai/PopQA", split="test"))


def load_triviaqa(path=None):
    if path:
        return read_local(path)
    from datasets import load_dataset
    return list(load_dataset("mandarjoshi/trivia_qa", "unfiltered.nocontext", split="validation"))


def build_popqa(rows, n, seed, n_dev=0):
    """n eval items + n_dev disjoint dev items (for choosing the probe threshold), equal per tercile.
    The eval sample is the same whatever n_dev is: dev items come after it in each shuffled pool."""
    items = []
    for r in rows:
        ans = clean_aliases(parse_list(r.get("possible_answers")))
        q = str(r.get("question", "")).strip()
        if not q or not ans:
            continue
        items.append({"question": q, "answers": ans, "s_pop": float(r["s_pop"]),
                      "prop": r.get("prop", ""), "popqa_id": r.get("id", ""),
                      "subj": r.get("subj", ""), "s_wiki_title": r.get("s_wiki_title", "") or r.get("subj", "")})
    # tercile cut points over the full set (not the sample)
    pops = sorted(x["s_pop"] for x in items)
    lo, hi = pops[len(pops) // 3], pops[2 * len(pops) // 3]
    for x in items:
        x["group"] = "tail" if x["s_pop"] < lo else ("torso" if x["s_pop"] < hi else "head")
    rng = random.Random(seed)
    per, dper = n // 3, n_dev // 3
    out, dev = [], []
    for g in ("head", "torso", "tail"):
        pool = [x for x in items if x["group"] == g]
        rng.shuffle(pool)
        out += pool[:per]
        dev += pool[per:per + dper]
    rng.shuffle(out)
    rng.shuffle(dev)
    return out, dev


def build_triviaqa(rows, n, seed):
    items = []
    for r in rows:
        a = r.get("answer")
        if isinstance(a, str):
            try:
                a = json.loads(a)
            except json.JSONDecodeError:
                a = {"value": a, "aliases": []}
        a = a or {}
        ans = clean_aliases([a.get("value", "")] + list(a.get("aliases", [])) + list(a.get("normalized_aliases", [])))
        q = str(r.get("question", "")).strip()
        if q and ans:
            items.append({"question": q, "answers": ans, "group": "triviaqa"})
    random.Random(seed).shuffle(items)
    return items[:n]


def add_wikipedia(items, args):
    """Attach Wikipedia knowledge to PopQA items and write the shared KB passages file."""
    from kb_utils import chunk_article, fetch_wikipedia, gold_passage
    Path(args.wiki_cache).parent.mkdir(parents=True, exist_ok=True)
    titles = [x["s_wiki_title"] for x in items if x.get("s_wiki_title")]
    print(f"fetching {len(set(titles))} Wikipedia articles (cached in {args.wiki_cache})...")
    failed = []
    texts = fetch_wikipedia(titles, cache_path=args.wiki_cache, failed=failed)
    kb_path = Path(args.kb_out or str(Path(args.out).with_suffix("")) + "_kb.jsonl")
    seen, n_gold, n_article = set(), 0, 0
    with open(kb_path, "w", encoding="utf-8") as f:
        for i, x in enumerate(items):
            title = x.get("s_wiki_title", "")
            passages = chunk_article(title, texts.get(title, ""), args.passage_words, args.max_passages) if title else []
            n_article += bool(passages)
            gp = gold_passage(passages, x["answers"])
            x["knowledge"] = gp or ""          # empty = the answer is not in the article: never "covered"
            n_gold += bool(gp)
            for p in passages:
                if p not in seen:
                    seen.add(p)
                    f.write(json.dumps({"text": p, "title": title}, ensure_ascii=False) + "\n")
    print(f"wikipedia: {n_article}/{len(items)} items have an article, {n_gold} contain a gold answer; "
          f"{len(seen)} passages -> {kb_path}")
    if failed:
        print(f"WARNING: {len(failed)} articles could not be fetched (Wikipedia rate limit), so their questions have "
              f"no passage in the KB. Rerun this command: only the missing articles are fetched again.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=["popqa", "triviaqa"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--popqa_file")
    ap.add_argument("--triviaqa_file")
    ap.add_argument("--dev", type=int, default=0,
                    help="PopQA: also write N disjoint dev questions to <out>_dev.jsonl (choose the probe threshold "
                         "there with choose_threshold.py, never on the eval rows)")
    ap.add_argument("--wiki", action="store_true",
                    help="PopQA: fetch each subject's Wikipedia article. Sets `knowledge` to the passage that contains "
                         "a gold answer (for --kb_coverage) and writes every passage to --kb_out (for --kb_file)")
    ap.add_argument("--kb_out", default=None, help="passages file for --wiki (default: <out>_kb.jsonl)")
    ap.add_argument("--wiki_cache", default="eval_data/wiki_cache.jsonl")
    ap.add_argument("--passage_words", type=int, default=120)
    ap.add_argument("--max_passages", type=int, default=20, help="passages kept per article")
    args = ap.parse_args()

    if args.dataset == "popqa":
        out, dev = build_popqa(load_popqa(args.popqa_file), args.n, args.seed, args.dev)
        if args.wiki:
            add_wikipedia(out, args)
        if dev:
            dev_path = Path(str(Path(args.out).with_suffix("")) + "_dev.jsonl")
            with open(dev_path, "w", encoding="utf-8") as f:
                for x in dev:
                    f.write(json.dumps(x, ensure_ascii=False) + "\n")
            print(f"wrote {len(dev)} dev rows (disjoint from the eval rows) -> {dev_path}")
    else:
        out = build_triviaqa(load_triviaqa(args.triviaqa_file), args.n, args.seed)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for x in out:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    groups = {}
    for x in out:
        groups[x["group"]] = groups.get(x["group"], 0) + 1
    print(f"wrote {len(out)} rows -> {args.out}  groups={groups}")


if __name__ == "__main__":
    main()
