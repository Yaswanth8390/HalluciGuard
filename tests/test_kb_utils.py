"""Offline tests for the Wikipedia fetcher's rate-limit handling and cache (no network)."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import kb_utils  # noqa: E402

requests = pytest.importorskip("requests")


class FakeResp:
    def __init__(self, status, payload=None, headers=None):
        self.status_code, self._payload, self.headers = status, payload or {}, headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")

    def json(self):
        return self._payload


def page(text):
    return FakeResp(200, {"query": {"pages": {"1": {"title": "t", "extract": text}}}})


MISSING = FakeResp(200, {"query": {"pages": {"-1": {"title": "t", "missing": ""}}}})
LIMIT = FakeResp(429, headers={"Retry-After": "3"})


class FakeSession:
    """Serves a scripted list of responses per title; records every request."""
    script = {}
    calls = []

    def __init__(self):
        self.headers = {}

    def get(self, url, timeout=None, params=None):
        title = params["titles"]
        FakeSession.calls.append(title)
        seq = FakeSession.script[title]
        return seq.pop(0) if len(seq) > 1 else seq[0]


@pytest.fixture
def fake(monkeypatch):
    sleeps = []
    monkeypatch.setattr(requests, "Session", FakeSession)
    monkeypatch.setattr(kb_utils.time, "sleep", lambda s: sleeps.append(s))
    FakeSession.calls = []
    return sleeps


def _cache(path):
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def test_429_is_retried_and_failures_are_not_cached(fake, tmp_path):
    FakeSession.script = {"A": [LIMIT, LIMIT, page("Article A text.")], "Gone": [MISSING], "Never": [LIMIT]}
    cp = tmp_path / "cache.jsonl"
    failed = []
    out = kb_utils.fetch_wikipedia(["A", "Gone", "Never"], cache_path=str(cp), attempts=3,
                                   second_pass_wait=0, failed=failed, verbose=False)
    assert out["A"] == "Article A text." and out["Gone"] == "" and "Never" not in out
    assert failed == ["Never"]
    assert 3 in fake                                   # honoured Retry-After
    rows = {r["title"]: r for r in _cache(cp)}
    assert set(rows) == {"A", "Gone"} and rows["Gone"]["missing"] is True   # the failure was not cached
    assert FakeSession.calls.count("Never") == 6       # 3 attempts x (first pass + second pass)


def test_rerun_retries_old_failures_but_not_missing_pages(fake, tmp_path):
    cp = tmp_path / "cache.jsonl"
    cp.write_text("\n".join(json.dumps(r) for r in [
        {"title": "A", "text": "cached A"},
        {"title": "Old", "text": ""},                       # failure cached by the previous version
        {"title": "Gone", "text": "", "missing": True}]) + "\n", encoding="utf-8")
    FakeSession.script = {"Old": [page("Old article, fetched now.")]}
    out = kb_utils.fetch_wikipedia(["A", "Old", "Gone"], cache_path=str(cp), verbose=False)
    assert FakeSession.calls == ["Old"]
    assert out == {"A": "cached A", "Old": "Old article, fetched now.", "Gone": ""}
