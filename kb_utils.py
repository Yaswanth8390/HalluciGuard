"""
kb_utils.py - helpers for building knowledge bases from Wikipedia text (PopQA, FActScore bios).

  chunk_article(title, text)  -> passages of ~N words, each prefixed with the article title (helps both the
                                 retriever and the verifier know which entity a passage is about)
  fetch_wikipedia(titles)     -> {title: plain text}, via the public MediaWiki API, with an on-disk cache
"""
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

USER_AGENT = "HalluciGuard-research/0.2 (academic hallucination-mitigation evaluation)"
_SECTION_JUNK = re.compile(r"^(See also|References|External links|Further reading|Notes|Bibliography|Sources)\b", re.I)


def clean_wiki_text(text: str) -> str:
    """Drop trailing reference sections and wiki heading markup from plain-text extracts."""
    out = []
    for line in str(text).splitlines():
        s = line.strip()
        h = re.match(r"^=+\s*(.*?)\s*=+$", s)
        if h:
            if _SECTION_JUNK.match(h.group(1)):
                break
            continue
        if _SECTION_JUNK.match(s) and len(s) < 40:
            break
        out.append(s)
    return "\n".join(out).strip()


def chunk_article(title: str, text: str, words: int = 120, max_passages: Optional[int] = 20) -> List[str]:
    """Split an article into passages of about `words` words along sentence boundaries."""
    text = clean_wiki_text(text)
    sents = re.split(r"(?<=[.!?])\s+|\n+", text)
    passages, cur, n = [], [], 0
    for s in sents:
        s = s.strip()
        if not s:
            continue
        cur.append(s)
        n += len(s.split())
        if n >= words:
            passages.append(f"{title}: " + " ".join(cur))
            cur, n = [], 0
            if max_passages and len(passages) >= max_passages:
                return passages
    if cur and (n >= 15 or not passages):
        passages.append(f"{title}: " + " ".join(cur))
    return passages


def gold_passage(passages: List[str], answers: Iterable[str]) -> Optional[str]:
    """First passage that contains one of the gold answers (case-insensitive, word-bounded)."""
    for p in passages:
        low = p.lower()
        for a in answers:
            a = str(a).strip().lower()
            if len(a) >= 2 and re.search(r"(?<!\w)" + re.escape(a) + r"(?!\w)", low):
                return p
    return None


def fetch_wikipedia(titles: Iterable[str], cache_path: Optional[str] = None, sleep: float = 0.2,
                    verbose: bool = True) -> Dict[str, str]:
    """Plain-text article for each title (empty string if missing). Results are cached as jsonl."""
    import requests

    cache: Dict[str, str] = {}
    cp = Path(cache_path) if cache_path else None
    if cp and cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                cache[row["title"]] = row["text"]
    todo = [t for t in dict.fromkeys(titles) if t and t not in cache]
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    fh = open(cp, "a", encoding="utf-8") if cp else None
    try:
        for i, title in enumerate(todo):
            text = ""
            for attempt in range(3):
                try:
                    r = sess.get("https://en.wikipedia.org/w/api.php", timeout=30, params={
                        "action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
                        "redirects": 1, "titles": title})
                    r.raise_for_status()
                    pages = r.json().get("query", {}).get("pages", {})
                    text = next(iter(pages.values()), {}).get("extract", "") or ""
                    break
                except Exception as exc:
                    if attempt == 2:
                        print(f"  wikipedia fetch failed for {title!r}: {exc}")
                    time.sleep(2 * (attempt + 1))
            cache[title] = text
            if fh:
                fh.write(json.dumps({"title": title, "text": text}, ensure_ascii=False) + "\n")
                fh.flush()
            if verbose and (i + 1) % 50 == 0:
                print(f"  fetched {i + 1}/{len(todo)} articles")
            time.sleep(sleep)
    finally:
        if fh:
            fh.close()
    return cache
