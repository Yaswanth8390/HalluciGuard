"""
control.py - the blackboard control component.

Each cycle:
  1. recompute claim / hypothesis status from the verdicts on the board, post conflicts it notices;
  2. stop if the problem is solved (an accepted answer with no open conflict) or the LLM budget is spent;
  3. ask every knowledge source what it could do now, score the activations, run the best one.
When the loop ends, the Composer turns the board into the final answer (one LLM call is reserved for it).

Priority (rule based, so every decision in the trace is explainable):
    priority = stage_weight(KS) * (0.5 + risk of the target) / (1 + llm_cost)
Ties keep board order. Hypothesis support = mean over its claims of
    sum(sign(verdict) * confidence * weight * source_reliability).
"""

import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from knowledge_sources import (Activation, Composer, ConflictResolver, Decomposer, HypothesisGenerator,
                               KnowledgeSource, MemoryKS, QueryReformulator, RetrievalKS, VerifierKS,
                               WebSearchKS)
from workspace import (ACCEPTED, CONTRADICTED, DECISIVE, INSUFFICIENT, OPEN, PENDING, REJECTED, REVISED,
                       SUPPORTED, Workspace)

SOURCE_RELIABILITY = {"kb": 1.0, "memory": 0.9, "web": 0.8, "resolver": 1.0}
ALL_KS = ("Decomposer", "Memory", "Retrieval", "QueryReformulator", "WebSearch", "Verifier",
          "HypothesisGenerator", "ConflictResolver")
# knowledge sources an ablation may switch off (the rest are structural)
ABLATABLE = ("Memory", "QueryReformulator", "WebSearch", "HypothesisGenerator", "ConflictResolver")


def _now():
    return datetime.now(timezone.utc).isoformat()


class BudgetExhausted(Exception):
    pass


class BlackboardController:
    def __init__(self, knowledge_sources: Iterable[KnowledgeSource], composer: Composer, *,
                 max_llm_calls: int = 8, accept_threshold: float = 0.5, min_conf: float = 0.5,
                 short_max_words: int = 15, max_cycles: int = 60, disabled: Iterable[str] = ()):
        self.disabled = set(disabled)
        unknown = self.disabled - set(ALL_KS)
        if unknown:
            raise ValueError(f"unknown knowledge sources to disable: {sorted(unknown)}")
        self.ks: List[KnowledgeSource] = [k for k in knowledge_sources if k.name not in self.disabled]
        for k in self.ks:
            k.ctx = self
        self.composer = composer
        self.max_llm_calls = max_llm_calls
        self.accept_threshold = accept_threshold
        self.min_conf = min_conf
        self.short_max_words = short_max_words
        self.max_cycles = max_cycles

    def get_ks(self, name: str) -> Optional[KnowledgeSource]:
        return next((k for k in self.ks if k.name == name), None)

    def config(self) -> Dict[str, Any]:
        return {"max_llm_calls": self.max_llm_calls, "accept_threshold": self.accept_threshold,
                "min_conf": self.min_conf, "short_max_words": self.short_max_words,
                "knowledge_sources": [k.name for k in self.ks], "disabled": sorted(self.disabled)}

    # ------------------------------------------------------------------ status
    def _claim_score(self, ws, c) -> float:
        s = 0.0
        for v in ws.verdicts_of(c["id"]):
            sign = 1.0 if v["label"] == SUPPORTED else -1.0 if v["label"] == CONTRADICTED else 0.0
            rel = min(SOURCE_RELIABILITY.get(x, 1.0) for x in v["sources"]) if v["sources"] else 1.0
            s += sign * v["confidence"] * v.get("weight", 1.0) * rel
        return s

    def recompute(self, ws: Workspace) -> None:
        for c in ws.of_kind("claim"):
            if c["status"] == REVISED:
                continue
            vs = ws.verdicts_of(c["id"])
            resolver = [v for v in vs if v["author"] == "ConflictResolver"]
            decisive = {v["label"] for v in vs if v["label"] in DECISIVE and v["confidence"] >= self.min_conf}
            score = self._claim_score(ws, c)
            if resolver:
                status = resolver[-1]["label"]
            elif decisive:
                status = SUPPORTED if score > 0 else CONTRADICTED if score < 0 else INSUFFICIENT
                if len(decisive) == 2 and not c["conflict_raised"]:
                    if self.get_ks("ConflictResolver"):
                        ws.add("conflict", "Control", about="claim", claim_id=c["id"], status=OPEN)
                    ws.update(c["id"], "Control", conflict_raised=True)
            elif vs:
                status = INSUFFICIENT
            else:
                status = OPEN
            ws.update(c["id"], "Control", status=status, score=round(score, 4))

        short = ws.question["form"] == "short"
        for h in ws.hypotheses():
            if h.get("locked") or h.get("abstention") or not h.get("decomposed"):
                continue
            cs = ws.claims_of(h["id"])
            support = sum(c.get("score", 0.0) for c in cs) / len(cs) if cs else 0.0
            status = h["status"]
            if short:
                if any(c["status"] == CONTRADICTED for c in cs):
                    status = REJECTED
                elif cs and all(c["status"] == SUPPORTED for c in cs) and support >= self.accept_threshold:
                    status = ACCEPTED
                else:
                    status = OPEN
            ws.update(h["id"], "Control", support=round(support, 4), status=status)

        if short:
            acc = [h for h in ws.hypotheses(ACCEPTED) if not h.get("locked")]
            if len(acc) >= 2:
                ids = sorted(h["id"] for h in acc)
                open_same = [x for x in ws.of_kind("conflict", about="hypothesis") if sorted(x["hyp_ids"]) == ids]
                if not open_same:
                    if self.get_ks("ConflictResolver"):
                        ws.add("conflict", "Control", about="hypothesis", hyp_ids=ids, status=OPEN)
                    else:   # no resolver: the best-supported answer wins
                        best = max(acc, key=lambda h: (h["support"], h["origin"] == "evidence"))
                        for h in acc:
                            ws.update(h["id"], "Control", status=ACCEPTED if h is best else REJECTED, locked=True)

    def solved(self, ws: Workspace) -> bool:
        if ws.question["form"] != "short":
            return False   # long answers: run until nothing useful is left or the budget is spent
        open_conflicts = ws.of_kind("conflict", status=OPEN)
        return bool(ws.hypotheses(ACCEPTED)) and not open_conflicts

    # ------------------------------------------------------------------ loop
    def run(self, question: str, response: str, risk: float, *, abstention: bool = False,
            form: Optional[str] = None) -> Dict[str, Any]:
        ws = Workspace(question, risk, response)
        if form is None:
            form = "long" if len(response.split()) > self.short_max_words else "short"
        ws.update(ws.question_id, "Control", form=form, needs_answer=abstention)
        ws.add("hypothesis", "Control", text=response, origin="base_model",
               status=REJECTED if abstention else OPEN, form=form, support=0.0,
               decomposed=abstention, abstention=abstention, evidence_ids=[])
        if self.get_ks("Retrieval"):
            ws.add("query", "Control", target_id=ws.question_id, text=question, origin="question",
                   source="kb", status=PENDING)

        calls, cycles, failed, trace = 0, 0, set(), []
        reserve = 1 if self.composer is not None else 0
        stop_reason = "no_activations"
        started = _now()
        while True:
            self.recompute(ws)
            if self.solved(ws):
                stop_reason = "solved"
                break
            if cycles >= self.max_cycles:
                stop_reason = "max_cycles"
                break
            left = self.max_llm_calls - calls - reserve
            agenda: List[Activation] = []
            for k in self.ks:
                for a in k.activations(ws):
                    if a.key in failed:
                        continue
                    risk_t = ws.get(a.target).get("risk", risk) if ws.has(a.target) else risk
                    a.priority = k.stage_weight * (0.5 + float(risk_t)) / (1 + a.cost)
                    agenda.append(a)
            affordable = [a for a in agenda if a.cost <= left]
            if not affordable:
                stop_reason = "budget" if agenda else "no_activations"
                break
            # stable sort: equal priority keeps board order
            affordable.sort(key=lambda a: -a.priority)
            act = affordable[0]
            cycles += 1
            ws.cycle = cycles
            try:
                note = act.ks.execute(ws, act)
                ok = True
            except (ValueError, KeyError, TypeError) as exc:   # unusable model output: skip, don't loop on it
                note, ok = f"failed: {type(exc).__name__}: {str(exc)[:160]}", False
                failed.add(act.key)
            calls += act.cost
            trace.append({"cycle": cycles, "ks": act.ks.name, "target": act.target, "reason": act.reason,
                          "priority": round(act.priority, 3), "llm_calls": act.cost, "ok": ok, "note": note,
                          "agenda": [f"{a.ks.name}:{a.target}" for a in affordable[1:4]]})

        self.recompute(ws)
        comp = self.composer.compose(ws, allow_call=calls < self.max_llm_calls) if self.composer else \
            {"final_response": response, "mode": "no_composer", "calls": 0}
        calls += comp.get("calls", 0)
        trace.append({"cycle": cycles + 1, "ks": "Composer", "target": ws.question_id, "reason": stop_reason,
                      "priority": None, "llm_calls": comp.get("calls", 0), "ok": True, "note": comp["mode"]})
        return self._result(ws, response, risk, abstention, comp, calls, cycles, stop_reason, trace, started)

    # ------------------------------------------------------------------ output
    def _result(self, ws, response, risk, abstention, comp, calls, cycles, stop_reason, trace, started):
        hyps = ws.hypotheses()
        base = hyps[0]
        claims = ws.of_kind("claim")
        mode = comp["mode"]
        if abstention:
            status = "ABSTENTION_RESOLVED" if mode == "replace" else "ABSTENTION"
        else:
            status = "BLACKBOARD_PROCESSED"
        # overall verdict on the ORIGINAL answer, for compatibility with the old result format
        base_claims = ws.claims_of(base["id"], active_only=False)
        labels = {c["status"] for c in base_claims}
        if abstention:
            overall = "ANSWER_FOUND" if mode == "replace" else INSUFFICIENT
        elif CONTRADICTED in labels or REVISED in labels:
            overall = CONTRADICTED
        elif base_claims and labels == {SUPPORTED}:
            overall = SUPPORTED
        else:
            overall = INSUFFICIENT
        used_sources = set()
        for v in ws.of_kind("verdict"):
            if v["label"] in DECISIVE:
                used_sources |= set(v["sources"])
        evidence_source = ("web_search" if "web" in used_sources else "knowledge_base" if "kb" in used_sources
                           else "memory" if "memory" in used_sources else None)
        riskiest = max(base_claims, key=lambda c: c["risk"], default=None)
        return {
            "status": status,
            "final_response": comp["final_response"],
            "extracted_claim": riskiest["text"] if riskiest else None,
            "flagged_span": riskiest["span"] if riskiest else None,
            "verification_result": {"verdict": overall, "claims": [
                {"id": c["id"], "text": c["text"], "status": c["status"], "score": c.get("score", 0.0),
                 "revises": c.get("revises")} for c in claims]},
            "correction_result": {"mode": mode, **{k: v for k, v in comp.items() if k not in ("final_response",)}},
            "evidence_source": evidence_source,
            "budget": {"llm_calls": calls, "max_llm_calls": self.max_llm_calls, "cycles": cycles,
                       "stop_reason": stop_reason},
            "blackboard_trace": {
                "run_id": str(uuid.uuid4()), "started_at": started, "completed_at": _now(),
                "config": self.config(), "cycles": trace,
                "hypotheses": [{"id": h["id"], "text": h["text"], "origin": h["origin"], "status": h["status"],
                                "support": h.get("support", 0.0)} for h in hyps],
                "workspace": ws.snapshot(), "events": ws.events,
            },
            "workspace": ws,
        }

    def log_to_memory(self, ws: Workspace, memory_agent, final_response: str) -> int:
        """Store grounded, non-memory verdicts so later questions can reuse them (MemoryAgent's own rules)."""
        n = 0
        if memory_agent is None or not hasattr(memory_agent, "log_verified_claim"):
            return 0
        from workspace import ScratchBoard
        for c in ws.of_kind("claim"):
            vs = [v for v in ws.verdicts_of(c["id"]) if v["author"] != "Memory" and v["label"] in DECISIVE]
            if not vs or c["status"] not in DECISIVE:
                continue
            v = vs[-1]
            if not v["evidence_ids"] and v["author"] == "Verifier":
                continue
            board = ScratchBoard(claim=c["text"], verification_verdict=c["status"],
                                 verification_explanation=v.get("explanation", ""), final_response=final_response,
                                 confidence_score=ws.question["risk"],
                                 verification_result={"supporting_evidence_ids": v["evidence_ids"] or ["resolver"]},
                                 correction_result={"from": "blackboard"} if c["status"] == CONTRADICTED else None,
                                 verifier_confidence=v["confidence"])
            try:
                if memory_agent.log_verified_claim(board):
                    n += 1
            except Exception:
                pass
        return n


def build_controller(*, retrieval_agent, verifier_agent, answer_agent, correction_agent, memory_agent=None,
                     web_search_agent=None, callers: Optional[Dict[str, Any]] = None,
                     max_llm_calls: Optional[int] = None, disabled: Iterable[str] = (),
                     retrieval_top_k: int = 5, max_claims: int = 8, **kw) -> BlackboardController:
    """Wire the knowledge sources from existing agents + per-role LLM callers
    (callers: DECOMPOSER, REFORMULATOR, HYPOTHESIS, CONFLICT, COMPOSER -> caller)."""
    callers = callers or {}
    if max_llm_calls is None:
        max_llm_calls = int(os.getenv("BLACKBOARD_MAX_LLM_CALLS", "8"))
    ks: List[KnowledgeSource] = [
        Decomposer(callers.get("DECOMPOSER"), max_claims=max_claims),
        RetrievalKS(retrieval_agent, top_k=retrieval_top_k),
        QueryReformulator(callers.get("REFORMULATOR")) if callers.get("REFORMULATOR") else None,
        VerifierKS(verifier_agent),
        HypothesisGenerator(answer_agent, llm_caller=callers.get("HYPOTHESIS")),
        ConflictResolver(callers.get("CONFLICT")) if callers.get("CONFLICT") else None,
    ]
    if memory_agent is not None:
        ks.insert(1, MemoryKS(memory_agent))
    if web_search_agent is not None:
        ks.append(WebSearchKS(web_search_agent))
    ks = [k for k in ks if k is not None]
    composer = Composer(correction_agent, llm_caller=callers.get("COMPOSER"))
    return BlackboardController(ks, composer, max_llm_calls=max_llm_calls, disabled=disabled, **kw)
