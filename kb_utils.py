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

USER_AGENT = "HalluciGuard-research/0.3 (https://github.com/Yaswanth8390/HalluciGuard)"
WIKI_API = "https://en.wikipedia.org/w/api.php"
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


class _RateLimited(Exception):
    def __init__(self, wait: float):
        super().__init__(f"rate limited, retry in {wait:.0f}s")
        self.wait = wait


def _retry_after(r, attempt: int) -> float:
    try:
        return max(1.0, min(120.0, float(r.headers.get("Retry-After", ""))))
    except (TypeError, ValueError):
        return float(min(60, 5 * 2 ** attempt))


def _fetch_one(sess, title: str, attempts: int, log) -> Optional[Dict[str, object]]:
    """{"text", "missing"} for one title, or None if every attempt failed (never cached)."""
    for attempt in range(attempts):
        try:
            r = sess.get(WIKI_API, timeout=30, params={
                "action": "query", "format": "json", "prop": "extracts", "explaintext": 1,
                "redirects": 1, "maxlag": 5, "titles": title})
            if r.status_code in (429, 503):
                raise _RateLimited(_retry_after(r, attempt))
            r.raise_for_status()
            data = r.json()
            if (data.get("error") or {}).get("code") == "maxlag":   # Wikipedia's servers are lagging: back off
                raise _RateLimited(_retry_after(r, attempt))
            page = next(iter((data.get("query") or {}).get("pages", {}).values()), {})
            if "missing" in page or "invalid" in page:
                return {"text": "", "missing": True}
            return {"text": page.get("extract", "") or "", "missing": False}
        except _RateLimited as exc:
            if attempt == attempts - 1:
                log(f"  wikipedia fetch failed for {title!r}: {exc}")
                return None
            _PACE["sleep"] = min(5.0, _PACE["sleep"] * 2)          # slow down for the rest of the run
            time.sleep(exc.wait)
        except Exception as exc:                                    # network hiccup, bad JSON, 5xx
            if attempt == attempts - 1:
                log(f"  wikipedia fetch failed for {title!r}: {exc}")
                return None
            time.sleep(2 * (attempt + 1))
    return None


_PACE = {"sleep": 0.5}


def fetch_wikipedia(titles: Iterable[str], cache_path: Optional[str] = None, sleep: float = 0.5,
                    verbose: bool = True, attempts: int = 8, second_pass_wait: float = 60.0,
                    failed: Optional[List[str]] = None) -> Dict[str, str]:
    """Plain-text article for each title ("" if Wikipedia has no such page). Results are cached as jsonl.

    Only real answers are cached: a request that failed (rate limit, network) is NOT written, so a rerun
    retries it. Old cache rows with empty text and no `missing` flag (failures cached by earlier versions)
    are retried too. Titles that still fail after a second pass are left out of the result and appended to
    `failed` if given. Pacing adapts: the pause between requests doubles after a 429 (max 5 s) and
    relaxes again after successful requests."""
    import requests

    cache: Dict[str, str] = {}
    cp = Path(cache_path) if cache_path else None
    if cp and cp.exists():
        for line in cp.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                if row.get("text") or row.get("missing"):
                    cache[row["title"]] = row.get("text") or ""
    todo = [t for t in dict.fromkeys(titles) if t and t not in cache]
    sess = requests.Session()
    sess.headers["User-Agent"] = USER_AGENT
    log = print if verbose else (lambda *a, **k: None)
    _PACE["sleep"] = sleep
    fh = open(cp, "a", encoding="utf-8") if cp else None
    still_failed: List[str] = []
    try:
        for round_ in range(2):
            if round_ == 1:
                if not still_failed:
                    break
                log(f"  {len(still_failed)} articles failed; waiting {second_pass_wait:.0f}s, then retrying them")
                time.sleep(second_pass_wait)
                todo, still_failed = still_failed, []
            for i, title in enumerate(todo):
                got = _fetch_one(sess, title, attempts, log)
                if got is None:
                    still_failed.append(title)
                else:
                    cache[title] = got["text"]
                    if fh:
                        fh.write(json.dumps({"title": title, "text": got["text"], "missing": got["missing"]},
                                            ensure_ascii=False) + "\n")
                        fh.flush()
                    _PACE["sleep"] = max(sleep, _PACE["sleep"] * 0.9)
                if verbose and (i + 1) % 50 == 0:
                    log(f"  fetched {i + 1}/{len(todo)} articles")
                time.sleep(_PACE["sleep"])
    finally:
        if fh:
            fh.close()
    if still_failed:
        log(f"  {len(still_failed)} articles could not be fetched (rate limit); rerun the same command to "
            f"retry only those (everything fetched so far is cached)")
    if failed is not None:
        failed.extend(still_failed)
    return cache
