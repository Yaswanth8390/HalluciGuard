"""
Unit tests for the blackboard control loop (blackboard/control.py, knowledge_sources.py, workspace.py).

Everything is faked (no ChromaDB, no network, no API key): the fake agents implement the same interface
as the real ones in blackboard_core.py, and fake LLM callers answer the new KS prompts by rule.
Run:  python -m pytest tests -q
"""
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "blackboard"))

from control import build_controller  # noqa: E402
from workspace import ACCEPTED, REJECTED, REVISED  # noqa: E402


def _tok(s):
    return set(re.findall(r"\w+", s.lower()))


STOP = {"what", "is", "the", "of", "who", "a", "an", "in", "to", "tell", "me", "bio", "answer", "question", "was", "s"}


def _between(text, tag):
    m = re.search(rf"<{tag}>\s*(.*?)\s*</{tag}>", text, re.S)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------------
# fake agents (same interface as blackboard_core's)
# ---------------------------------------------------------------------------
class FakeRetrieval:
    def __init__(self, docs):
        self.docs = docs       # list of (id, text)
        self.top_k = 5
        self.queries = []

    def retrieve(self, board, *, top_k=None, max_distance=0.7):
        q = board.read("claim")
        self.queries.append(q)
        scored = []
        for did, text in self.docs:
            overlap = len(_tok(q) & _tok(text)) / max(1, len(_tok(q)))
            if overlap > 0.2:
                scored.append((1 - overlap, did, text))
        scored.sort()
        ev = [{"id": d, "text": t, "metadata": {}, "distance": round(s, 3)} for s, d, t in scored[: top_k or 5]]
        board.write("retrieved_evidence", ev)
        return ev


class FakeVerifier:
    """SUPPORTED if the claim's answer (text after 'is:') appears in the evidence; CONTRADICTED if evidence
    exists that mentions the claim subject but not the answer; INSUFFICIENT without evidence.
    `overrides` maps an evidence source to a forced verdict (to create conflicts)."""
    def __init__(self, overrides=None):
        self.calls = 0
        self.overrides = overrides or {}

    def verify(self, board):
        self.calls += 1
        claim = board.read("claim")
        ev = board.read("retrieved_evidence", [])
        if not ev:
            return {"verdict": "INSUFFICIENT", "explanation": "none", "supporting_evidence_ids": [], "confidence": 0.3}
        srcs = {e["metadata"].get("source") for e in ev}
        for s, v in self.overrides.items():
            if srcs == {s}:
                return {"verdict": v, "explanation": "forced", "supporting_evidence_ids": [ev[0]["id"]], "confidence": 0.9}
        sup = lambda e: {"verdict": "SUPPORTED", "explanation": "found", "supporting_evidence_ids": [e["id"]], "confidence": 0.9}
        con = {"verdict": "CONTRADICTED", "explanation": "not found", "supporting_evidence_ids": [ev[0]["id"]], "confidence": 0.85}
        if "is:" in claim:   # short-answer template claim: one passage must contain the answer AND the question's topic
            q, ans = claim.split("is:")
            ans = ans.strip().lower().rstrip(".")
            qw = _tok(q) - STOP
            for e in ev:
                if ans in e["text"].lower() and len(qw & _tok(e["text"])) >= min(2, len(qw)):
                    return sup(e)
            return con
        cw = _tok(claim) - STOP   # long-form claim: almost all of its content words in one passage
        for e in ev:
            if len(cw & _tok(e["text"])) >= 0.8 * len(cw):
                return sup(e)
        return con


class FakeAnswer:
    """Answers from evidence via a lookup table question -> answer, if the answer occurs in the evidence."""
    def __init__(self, table):
        self.table = table
        self.calls = 0

    def attempt_answer(self, board):
        self.calls += 1
        q = board.read("prompt")
        ev = board.read("retrieved_evidence", [])
        ans = self.table.get(q, "")
        text = " ".join(e["text"].lower() for e in ev)
        if ans and ans.lower() in text:
            return {"verdict": "ANSWER_FOUND", "answer": ans, "explanation": "", "supporting_evidence_ids": [ev[0]["id"]]}
        return {"verdict": "INSUFFICIENT", "answer": "", "explanation": "", "supporting_evidence_ids": []}


class FakeCorrection:
    def __init__(self):
        self.calls = 0

    def abstain(self, board):
        self.calls += 1
        return {"corrected_response": "I'm not sure.", "correction_summary": "abstained", "mode": "abstain"}

    def correct(self, board):
        raise AssertionError("the blackboard never calls correct(); corrections are hypotheses")


class FakeWeb:
    enabled = True

    def __init__(self, results):
        self.results = results
        self.calls = 0

    def search(self, query):
        self.calls += 1
        return [{"id": f"web:{i}", "text": t, "metadata": {"source": "web_search"}, "distance": None}
                for i, t in enumerate(self.results)]


def fake_llm(prompt, **kw):
    if "You are the QueryReformulator" in prompt:
        q = _between(prompt, "QUESTION")
        return json.dumps({"queries": [f"{q} encyclopedia"]})
    if "You are the Decomposer" in prompt:
        resp = _between(prompt, "ASSISTANT_RESPONSE")
        sents = [s.strip() for s in re.split(r"(?<=\.)\s+", resp) if s.strip()]
        return json.dumps({"claims": [{"claim": s, "span": s} for s in sents]})
    if "You are the ClaimReviser" in prompt:
        claim = _between(prompt, "CLAIM")
        ev = _between(prompt, "EVIDENCE")
        m = re.search(r"born in (\d{4})", ev)
        if m and "born in" in claim:
            return json.dumps({"revised_claim": re.sub(r"born in \d{4}", f"born in {m.group(1)}", claim), "evidence_ids": []})
        return json.dumps({"revised_claim": "", "evidence_ids": []})
    if "You are the ConflictResolver" in prompt:
        return json.dumps({"verdict": "SUPPORTED", "confidence": 0.8, "explanation": "kb passage is specific"})
    if "You are the Composer" in prompt:
        keep = re.findall(r"^- (.+)$", _between(prompt, "KEEP"), re.M)
        corr = re.findall(r"CORRECT: (.+)$", _between(prompt, "REPLACE"), re.M)
        return json.dumps({"response": " ".join(x for x in keep + corr if x != "(none)")})
    raise AssertionError("unexpected prompt: " + prompt[:80])


class Counter:
    def __init__(self, fn):
        self.fn, self.n = fn, 0

    def __call__(self, *a, **k):
        self.n += 1
        return self.fn(*a, **k)


DOCS = [("d1", "Canberra is the capital city of Australia."),
        ("d2", "Sydney is the largest city in Australia."),
        ("d3", "Alexander Fleming discovered penicillin in 1928."),
        ("d4", "Ada Smith is a chemist. Ada Smith was born in 1950 in Leeds.")]
TABLE = {"What is the capital of Australia?": "Canberra", "Who discovered penicillin?": "Alexander Fleming"}


def make(docs=DOCS, web=None, verifier=None, max_calls=8, disabled=(), memory=None, confirmer=None,
         answer=None, **kw):
    """confirmer: fn(prompt) -> JSON for the Confirmer (VERIFIER2); None = no Confirmer, as before."""
    llm = Counter(fake_llm)
    agents = dict(retrieval=FakeRetrieval(docs), verifier=verifier or FakeVerifier(),
                  answer=answer or FakeAnswer(TABLE), correction=FakeCorrection(), llm=llm,
                  confirm=Counter(confirmer) if confirmer else None)
    callers = {r: llm for r in ("DECOMPOSER", "REFORMULATOR", "HYPOTHESIS", "CONFLICT", "COMPOSER")}
    if confirmer:
        callers["VERIFIER2"] = agents["confirm"]
    ctl = build_controller(retrieval_agent=agents["retrieval"], verifier_agent=agents["verifier"],
                           answer_agent=agents["answer"], correction_agent=agents["correction"],
                           memory_agent=memory, web_search_agent=web, callers=callers,
                           max_llm_calls=max_calls, disabled=disabled, **kw)
    return ctl, agents


def total_calls(a):
    return (a["verifier"].calls + a["answer"].calls + a["correction"].calls + a["llm"].n
            + (a["confirm"].n if a["confirm"] else 0))


# ---------------------------------------------------------------------------
def test_supported_answer_is_kept_with_one_call():
    ctl, a = make()
    out = ctl.run("What is the capital of Australia?", "Canberra", 0.9)
    assert out["final_response"] == "Canberra"
    assert out["budget"]["llm_calls"] == 1 == total_calls(a)
    assert out["budget"]["stop_reason"] == "solved"
    assert out["verification_result"]["verdict"] == "SUPPORTED"


def test_contradicted_answer_replaced_by_verified_competing_hypothesis():
    ctl, a = make()
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    hyps = out["blackboard_trace"]["hypotheses"]
    assert [h["status"] for h in hyps] == [REJECTED, ACCEPTED]
    assert hyps[1]["text"] == "Canberra" and hyps[1]["origin"] == "evidence"
    assert out["final_response"] == "Canberra"
    # the new hypothesis was itself verified (2 verifier calls), not trusted blindly
    assert a["verifier"].calls == 2 and a["answer"].calls == 1
    assert out["budget"]["llm_calls"] == total_calls(a) == 3
    assert out["verification_result"]["verdict"] == "CONTRADICTED"


def test_unverifiable_competitor_is_not_accepted():
    class PickyVerifier(FakeVerifier):   # rejects everything
        def verify(self, board):
            self.calls += 1
            ev = board.read("retrieved_evidence", [])
            return {"verdict": "CONTRADICTED", "explanation": "", "supporting_evidence_ids": [ev[0]["id"]] if ev else [],
                    "confidence": 0.9}
    ctl, a = make(verifier=PickyVerifier())
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    assert out["final_response"] == "I'm not sure."
    assert all(h["status"] == REJECTED for h in out["blackboard_trace"]["hypotheses"])
    assert out["budget"]["llm_calls"] <= 8


def test_closed_book_reformulates_then_abstains():
    ctl, a = make(docs=[])
    out = ctl.run("Who painted the Mona Lisa?", "Picasso", 0.9)
    ks = [c["ks"] for c in out["blackboard_trace"]["cycles"]]
    assert "QueryReformulator" in ks and "Verifier" not in ks
    assert out["final_response"] == "I'm not sure."
    assert out["budget"]["llm_calls"] == total_calls(a) == 2   # reformulate + abstain


@pytest.mark.parametrize("budget", [1, 2, 3])
def test_budget_is_never_exceeded(budget):
    ctl, a = make(max_calls=budget)
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    assert total_calls(a) == out["budget"]["llm_calls"] <= budget


class FakeMemory:
    """Always 'remembers' a grounded SUPPORTED verdict with moderate confidence."""
    def __init__(self, conf=0.7):
        self.conf = conf

    def check_memory(self, board):
        return {"match_found": True, "matches": [], "best_match": {"id": "m1", "distance": 0.1, "metadata": {
            "verdict": "SUPPORTED", "grounded": True, "verifier_confidence": self.conf, "explanation": "old"}}}

    def log_verified_claim(self, board):
        return None


def test_memory_vs_kb_disagreement_is_resolved_by_conflict_resolver():
    """Long answer: memory says a claim is SUPPORTED (conf 0.7), the KB evidence, judged as an independent
    second opinion, says CONTRADICTED -> the controller posts a conflict -> ConflictResolver settles it."""
    ctl, a = make(memory=FakeMemory(0.7), max_calls=20, disabled=["HypothesisGenerator"])
    out = ctl.run("Tell me a bio of Ada Smith.", "Ada Smith was born in 1960. Ada Smith is a chemist.", 0.8,
                  form="long")
    ws = out["blackboard_trace"]["workspace"]["entries"]
    conflicts = [e for e in ws if e["kind"] == "conflict"]
    assert len(conflicts) == 1 and conflicts[0]["status"] == "DONE"
    ks = [c["ks"] for c in out["blackboard_trace"]["cycles"]]
    assert "ConflictResolver" in ks and "Memory" in ks
    born = next(e for e in ws if e["kind"] == "claim" and "1960" in e["text"])
    assert born["status"] == "SUPPORTED"     # the fake resolver sides with SUPPORTED
    assert out["budget"]["llm_calls"] == total_calls(a)


def test_confident_memory_skips_reverification():
    ctl, a = make(memory=FakeMemory(0.95), max_calls=20)
    out = ctl.run("Tell me a bio of Ada Smith.", "Ada Smith was born in 1960. Ada Smith is a chemist.", 0.8,
                  form="long")
    assert a["verifier"].calls == 0      # both claims settled from memory
    assert not [e for e in out["blackboard_trace"]["workspace"]["entries"] if e["kind"] == "conflict"]


def test_abstention_resolved_from_evidence():
    ctl, a = make()
    out = ctl.run("Who discovered penicillin?", "Unknown", 0.95, abstention=True)
    assert out["status"] == "ABSTENTION_RESOLVED"
    assert out["final_response"] == "Alexander Fleming"


def test_abstention_stands_without_evidence():
    ctl, a = make(docs=[])
    out = ctl.run("Who discovered penicillin?", "Unknown", 0.95, abstention=True)
    assert out["status"] == "ABSTENTION" and out["final_response"] == "Unknown"


def test_ablation_without_hypothesis_generator_abstains():
    ctl, a = make(disabled=["HypothesisGenerator"])
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    assert out["final_response"] == "I'm not sure."
    assert "HypothesisGenerator" not in out["blackboard_trace"]["config"]["knowledge_sources"]


def test_long_answer_claims_are_revised_and_reverified():
    ctl, a = make(max_calls=20)
    bio = "Ada Smith is a chemist. Ada Smith was born in 1960."
    out = ctl.run("Tell me a bio of Ada Smith.", bio, 0.8, form="long")
    ws = out["blackboard_trace"]["workspace"]["entries"]
    claims = [e for e in ws if e["kind"] == "claim"]
    revised = [c for c in claims if c["status"] == REVISED]
    assert len(revised) == 1 and "1960" in revised[0]["text"]
    new = [c for c in claims if c.get("revises") == revised[0]["id"]]
    assert new and "1950" in new[0]["text"] and new[0]["status"] == "SUPPORTED"
    assert "1950" in out["final_response"] and "1960" not in out["final_response"]
    assert out["budget"]["llm_calls"] == total_calls(a) <= 20


def test_unknown_ks_name_rejected():
    with pytest.raises(ValueError):
        make(disabled=["NoSuchKS"])


def test_trace_explains_every_cycle():
    ctl, a = make()
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    for c in out["blackboard_trace"]["cycles"]:
        assert c["ks"] and c["reason"] is not None
    json.dumps(out["blackboard_trace"])   # serializable


# ---------------------------------------------------------------------------
# cross-model confirmation + entity / answer grounding
# ---------------------------------------------------------------------------
from knowledge_sources import grounded_in, on_topic  # noqa: E402


def confirm_says(label, conf=0.9):
    def fn(prompt, **kw):
        assert "You are the Confirmer" in prompt
        return json.dumps({"verdict": label, "confidence": conf, "evidence_ids": [], "explanation": "fake"})
    return fn


def confirm_if_written(prompt, **kw):
    """Agrees when the claimed answer is written in the evidence, else INSUFFICIENT."""
    assert "You are the Confirmer" in prompt
    ans = _between(prompt, "CLAIM").split("is:")[-1].strip().lower().rstrip(".")
    ok = ans and ans in _between(prompt, "EVIDENCE").lower()
    return json.dumps({"verdict": "SUPPORTED" if ok else "INSUFFICIENT", "confidence": 0.9, "explanation": ""})


class YesVerifier(FakeVerifier):
    """A lenient first verifier: SUPPORTED whenever there is any evidence (the failure seen on PopQA)."""
    def verify(self, board):
        self.calls += 1
        ev = board.read("retrieved_evidence", [])
        if not ev:
            return {"verdict": "INSUFFICIENT", "explanation": "", "supporting_evidence_ids": [], "confidence": 0.3}
        return {"verdict": "SUPPORTED", "explanation": "", "supporting_evidence_ids": [ev[0]["id"]], "confidence": 0.9}


def _ks(out):
    return [c["ks"] for c in out["blackboard_trace"]["cycles"]]


def test_helpers_entity_and_answer_grounding():
    q = "Who was the composer of Black Patch?"
    assert on_topic(q, "Black Patch: Black Patch is a 1957 western film.") is True
    assert on_topic(q, "Black Sabbath: Black Sabbath were an English rock band.") is False
    assert on_topic("Who was the director of Murder?", "Murder (1930 film): Murder! is a 1930 thriller.") is True
    assert on_topic(q, "Canberra is the capital city of Australia.") is None     # untitled: unknown
    ev = [{"id": "e1", "text": "Murder (1930 film): directed by Alfred Hitchcock."}]
    assert grounded_in("Alfred Hitchcock", ev) and grounded_in("Hitchcock", ev)
    assert grounded_in("footballer", [{"id": "e", "text": "X: X played association football."}])
    assert grounded_in("Jerry Goldsmith", ev) is None
    assert grounded_in("1951", [{"id": "e", "text": "X: born in 1950."}]) is None


def test_high_risk_answer_needs_a_second_model_to_agree():
    ctl, a = make(confirmer=confirm_says("SUPPORTED"))
    out = ctl.run("What is the capital of Australia?", "Canberra", 0.9)
    assert out["final_response"] == "Canberra" and out["budget"]["stop_reason"] == "solved"
    assert _ks(out).count("Confirmer") == 1 and a["confirm"].n == 1
    assert out["verification_result"]["claims"][0]["confirmation"] == "confirmed"
    assert out["budget"]["llm_calls"] == total_calls(a) == 2


def test_low_risk_answer_is_accepted_on_one_check():
    ctl, a = make(confirmer=confirm_says("SUPPORTED"), confirm_risk=0.5)
    out = ctl.run("What is the capital of Australia?", "Canberra", 0.3)
    assert out["final_response"] == "Canberra" and "Confirmer" not in _ks(out)
    assert out["budget"]["llm_calls"] == total_calls(a) == 1


def test_second_model_disagreeing_blocks_a_wrong_answer():
    """The lenient first verifier supports Sydney; the Confirmer does not -> no hallucinated answer."""
    ctl, a = make(verifier=YesVerifier(), confirmer=confirm_says("INSUFFICIENT"))
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    assert out["final_response"] == "I'm not sure."
    assert not [h for h in out["blackboard_trace"]["hypotheses"] if h["status"] == ACCEPTED]
    assert out["budget"]["llm_calls"] == total_calls(a) <= 8


def test_without_confirmer_the_lenient_verifier_lets_it_through():
    ctl, a = make(verifier=YesVerifier())
    assert ctl.run("What is the capital of Australia?", "Sydney", 0.9)["final_response"] == "Sydney"


def test_confirmer_contradiction_goes_to_the_resolver():
    ctl, a = make(confirmer=confirm_says("CONTRADICTED"))
    out = ctl.run("What is the capital of Australia?", "Canberra", 0.9)
    ws = out["blackboard_trace"]["workspace"]["entries"]
    assert [e for e in ws if e["kind"] == "conflict" and e["about"] == "claim"]
    assert "ConflictResolver" in _ks(out)
    assert out["final_response"] == "Canberra"     # the fake resolver sides with SUPPORTED


def test_evidence_answer_is_always_confirmed():
    """Even with a low-risk base answer, a competitor read off the evidence needs the second model."""
    ctl, a = make(confirmer=confirm_if_written, confirm_risk=0.95)
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.6)
    assert out["final_response"] == "Canberra"
    assert _ks(out).count("Confirmer") == 1        # only for the evidence answer, not for Sydney


BLACK_PATCH = [("b1", "Black Sabbath: Black Sabbath music was written by Tony Iommi, the composer of the band."),
               ("b2", "Black Patch: Black Patch is a 1957 American western film about a black patch.")]


class HallucinatingAnswer(FakeAnswer):
    """Reads an answer off ANY passage, or makes one up (what the 8B models did on PopQA)."""
    def attempt_answer(self, board):
        self.calls += 1
        ev = board.read("retrieved_evidence", [])
        text = " ".join(e["text"] for e in ev)
        ans = "Tony Iommi" if "Tony Iommi" in text else "Jerry Goldsmith"
        return {"verdict": "ANSWER_FOUND", "answer": ans, "explanation": "", "supporting_evidence_ids": [ev[0]["id"]]}


def test_off_topic_passage_cannot_propose_an_answer():
    q = "Who was the composer of Black Patch?"
    ctl, a = make(docs=BLACK_PATCH, answer=HallucinatingAnswer({}), grounding=False)
    assert ctl.run(q, "Unknown", 0.9, abstention=True)["final_response"] == "Tony Iommi"   # the failure
    ctl, a = make(docs=BLACK_PATCH, answer=HallucinatingAnswer({}))
    out = ctl.run(q, "Unknown", 0.9, abstention=True)
    assert out["status"] == "ABSTENTION" and out["final_response"] == "Unknown"
    notes = [c["note"] for c in out["blackboard_trace"]["cycles"] if c["ks"] == "HypothesisGenerator"]
    assert notes and "not written in the evidence" in notes[0]    # Jerry Goldsmith is not in b2


def test_confirmer_needs_a_passage_about_the_entity():
    """The first verifier supports Tony Iommi from the Black Sabbath passage; the Confirmer sees no
    passage about Black Patch that names him and rejects without an LLM call."""
    ctl, a = make(docs=BLACK_PATCH, verifier=YesVerifier(), confirmer=confirm_says("SUPPORTED"))
    out = ctl.run("Who was the composer of Black Patch?", "Tony Iommi", 0.9)
    assert out["final_response"] == "I'm not sure."
    assert a["confirm"].n == 0 and "Confirmer" in _ks(out)
    assert out["budget"]["llm_calls"] == total_calls(a)


@pytest.mark.parametrize("budget", [1, 2, 3, 4])
def test_budget_is_never_exceeded_with_confirmer(budget):
    ctl, a = make(max_calls=budget, confirmer=confirm_if_written)
    out = ctl.run("What is the capital of Australia?", "Sydney", 0.9)
    assert total_calls(a) == out["budget"]["llm_calls"] <= budget
    assert out["final_response"] in ("Canberra", "I'm not sure.")      # never an unconfirmed answer


def test_long_form_revision_needs_confirmation():
    bio = "Ada Smith is a chemist. Ada Smith was born in 1960."
    ctl, a = make(max_calls=20, confirmer=confirm_says("INSUFFICIENT"))
    out = ctl.run("Tell me a bio of Ada Smith.", bio, 0.8, form="long")
    assert "1950" not in out["final_response"] and "1960" not in out["final_response"]   # removed, not swapped
    ctl, a = make(max_calls=20, confirmer=confirm_says("SUPPORTED"))
    out = ctl.run("Tell me a bio of Ada Smith.", bio, 0.8, form="long")
    assert "1950" in out["final_response"] and "1960" not in out["final_response"]
    assert out["budget"]["llm_calls"] == total_calls(a) <= 20
