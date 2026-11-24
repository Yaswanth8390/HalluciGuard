"""
workspace.py - the shared, structured blackboard that HalluciGuard's knowledge sources read and write.

The old `Blackboard` in blackboard_core.py is a flat dict that agents only log to; a fixed Orchestrator
decides what runs. Here the workspace *is* the state that drives control: every entry has a kind, a
status and links, and the controller (control.py) looks at it each cycle to decide which knowledge source
(knowledge_sources.py) should act next.

Entry kinds (levels of the blackboard):
    question    the user prompt + the probe's risk score                         (one per run)
    hypothesis  a candidate answer: base model's answer, or one derived from evidence
    claim       an atomic, checkable statement belonging to a hypothesis
    query       a retrieval request (question-level or claim-level, KB or web)
    evidence    a passage found by a query (KB / web / memory)
    verdict     a judgement of one claim against evidence
    conflict    a disagreement the controller noticed (claim-level or hypothesis-level)

This module has no import-time side effects and does not import blackboard_core.
"""

from copy import deepcopy
from typing import Any, Dict, Iterable, List, Optional

# statuses
OPEN, ACCEPTED, REJECTED = "OPEN", "ACCEPTED", "REJECTED"
SUPPORTED, CONTRADICTED, INSUFFICIENT = "SUPPORTED", "CONTRADICTED", "INSUFFICIENT"
REVISED = "REVISED"            # a contradicted claim that has been replaced by a revision
PENDING, DONE = "PENDING", "DONE"

DECISIVE = (SUPPORTED, CONTRADICTED)


class Workspace:
    """Typed entries + an event log. Entries are plain dicts so they serialize straight into traces."""

    def __init__(self, question: str, risk: float, base_answer: str):
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.order: List[str] = []
        self.events: List[Dict[str, Any]] = []
        self.cycle = 0
        self._counters: Dict[str, int] = {}
        self.question_id = self.add("question", "system", text=question, risk=float(risk),
                                    base_answer=base_answer)

    # ---- generic access -------------------------------------------------------
    def _new_id(self, kind: str) -> str:
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        prefix = {"question": "Q", "hypothesis": "H", "claim": "C", "query": "q", "verdict": "V",
                  "conflict": "X"}.get(kind, kind)
        return f"{prefix}{n}"

    def add(self, kind: str, author: str, *, entry_id: Optional[str] = None, **fields) -> str:
        eid = entry_id or self._new_id(kind)
        if eid in self.entries:
            raise KeyError(f"duplicate workspace id {eid}")
        entry = {"id": eid, "kind": kind, "author": author, "cycle": self.cycle, **fields}
        self.entries[eid] = entry
        self.order.append(eid)
        self.events.append({"cycle": self.cycle, "author": author, "action": "add", "id": eid,
                            "kind": kind, "fields": _brief(fields)})
        return eid

    def update(self, eid: str, author: str, **fields) -> None:
        entry = self.entries[eid]
        changed = {k: v for k, v in fields.items() if entry.get(k) != v}
        if not changed:
            return
        entry.update(changed)
        self.events.append({"cycle": self.cycle, "author": author, "action": "update", "id": eid,
                            "kind": entry["kind"], "fields": _brief(changed)})

    def get(self, eid: str) -> Dict[str, Any]:
        return self.entries[eid]

    def has(self, eid: str) -> bool:
        return eid in self.entries

    def of_kind(self, kind: str, **match) -> List[Dict[str, Any]]:
        out = []
        for eid in self.order:
            e = self.entries[eid]
            if e["kind"] != kind:
                continue
            if all(e.get(k) == v for k, v in match.items()):
                out.append(e)
        return out

    # ---- domain helpers -------------------------------------------------------
    @property
    def question(self) -> Dict[str, Any]:
        return self.entries[self.question_id]

    def hypotheses(self, *statuses: str) -> List[Dict[str, Any]]:
        hs = self.of_kind("hypothesis")
        return [h for h in hs if not statuses or h["status"] in statuses]

    def claims_of(self, hyp_id: str, active_only: bool = True) -> List[Dict[str, Any]]:
        cs = self.of_kind("claim", hyp_id=hyp_id)
        return [c for c in cs if not active_only or c["status"] != REVISED]

    def verdicts_of(self, claim_id: str) -> List[Dict[str, Any]]:
        return self.of_kind("verdict", claim_id=claim_id)

    def queries_of(self, target_id: str) -> List[Dict[str, Any]]:
        return self.of_kind("query", target_id=target_id)

    def evidence_linked(self, target_id: str) -> List[Dict[str, Any]]:
        return [e for e in self.of_kind("evidence") if target_id in e.get("links", [])]

    def evidence_for_claim(self, claim_id: str, cap: int = 8) -> List[Dict[str, Any]]:
        """Evidence found for this claim plus question-level evidence, nearest first, de-duplicated."""
        seen, out = set(), []
        for e in self.evidence_linked(claim_id) + self.evidence_linked(self.question_id):
            if e["id"] not in seen:
                seen.add(e["id"])
                out.append(e)
        out.sort(key=lambda e: (e.get("distance") is None, e.get("distance") or 0.0))
        return out[:cap]

    def add_evidence(self, author: str, *, doc_id: str, text: str, source: str, link: str,
                     query_id: Optional[str] = None, distance=None, metadata=None) -> (str, bool):
        """Add a passage (or just link an existing one to another target). Returns (entry id, is_new)."""
        eid = "ev:" + str(doc_id)
        if eid in self.entries:
            e = self.entries[eid]
            if link not in e["links"]:
                self.update(eid, author, links=e["links"] + [link])
            return eid, False
        self.add("evidence", author, entry_id=eid, doc_id=str(doc_id), text=text, source=source,
                 links=[link], query_id=query_id, distance=distance, metadata=metadata or {})
        return eid, True

    def snapshot(self) -> Dict[str, Any]:
        return {"cycle": self.cycle, "entries": [deepcopy(self.entries[e]) for e in self.order]}


def _brief(fields: Dict[str, Any], limit: int = 160) -> Dict[str, Any]:
    out = {}
    for k, v in fields.items():
        if isinstance(v, str) and len(v) > limit:
            v = v[:limit] + "..."
        elif isinstance(v, (list, dict)) and len(str(v)) > limit:
            v = f"<{type(v).__name__} len={len(v)}>"
        out[k] = v
    return out


class ScratchBoard:
    """Minimal stand-in for blackboard_core.Blackboard (read / write / update / snapshot).

    The existing agents (VerifierAgent, AnswerAgent, CorrectionAgent, MemoryAgent, RetrievalAgent,
    WebSearchAgent) take a blackboard and read flat keys such as "claim" or "retrieved_evidence".
    A knowledge source fills one of these with exactly the view the agent needs, calls the agent
    unchanged, and copies the result back into the structured Workspace."""

    def __init__(self, **values):
        self.workspace: Dict[str, Any] = dict(values)

    def read(self, key: str, default: Any = None) -> Any:
        return self.workspace.get(key, default)

    def write(self, key: str, value: Any, *, author: str = "system") -> None:
        self.workspace[key] = value

    def update(self, values: Dict[str, Any], *, author: str = "system") -> None:
        self.workspace.update(values)

    def snapshot(self) -> Dict[str, Any]:
        return deepcopy(self.workspace)

    def get_history(self) -> List[Dict[str, Any]]:
        return []


def agent_evidence(entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Workspace evidence entries -> the {id, text, metadata, distance} dicts the agents expect."""
    return [{"id": e["id"], "text": e["text"], "metadata": {**e.get("metadata", {}), "source": e["source"]},
             "distance": e.get("distance")} for e in entries]
