"""
baselines.py - published mitigation methods, re-implemented on the SAME agent models and the same
knowledge base, so they can be compared to the blackboard under equal conditions.

  CoVe  - Chain-of-Verification, factored variant (Dhuliawala et al., 2023):
          plan verification questions -> answer each independently (without seeing the draft)
          -> revise the draft. Closed-book, as in the paper. LLM calls: 2 + n_questions.
  CRAG  - Corrective RAG (Yan et al., 2024), adapted to revising an existing draft:
          retrieve -> a retrieval evaluator labels each passage CORRECT / AMBIGUOUS / INCORRECT
          -> CORRECT: use those passages; all INCORRECT: web search (if enabled) else no knowledge;
          AMBIGUOUS: both -> generate the final answer from the draft + knowledge. LLM calls: 2.
          (The paper's evaluator is a fine-tuned T5 scoring each passage; here one LLM call scores all.)

Both return the same keys as process_response's blackboard result (final_response, budget, ...).
"""

import json
from typing import Any, Callable, Dict, List, Optional

from knowledge_sources import parse_json
from workspace import ScratchBoard


def _call_json(caller: Callable, prompt: str) -> Dict[str, Any]:
    return parse_json(caller(prompt, temperature=0.0, response_mime_type="application/json"))


def run_cove(question: str, draft: str, caller: Callable, *, max_llm_calls: int = 8,
             max_questions: int = 3, short: bool = True) -> Dict[str, Any]:
    n_q = max(1, min(max_questions, max_llm_calls - 2))
    calls = 0
    plan = _call_json(caller, f"""
You are verifying a draft answer (Chain-of-Verification, step 2: plan verifications).
Write up to {n_q} short fact-checking questions whose answers would reveal whether the facts in the draft
are correct. Each question must be answerable on its own, without seeing the draft.
Return JSON only: {{"questions": ["...", "..."]}}

<QUESTION>
{question}
</QUESTION>

<DRAFT_ANSWER>
{draft}
</DRAFT_ANSWER>
""".strip())
    calls += 1
    qs = [str(q).strip() for q in (plan.get("questions") or []) if str(q).strip()][:n_q]
    qa = []
    for vq in qs:   # factored: each question answered in isolation, the draft is NOT shown
        ans = caller(f"Answer the question concisely and factually. If you do not know, say \"Unknown\".\n\n"
                     f"Question: {vq}", temperature=0.0)
        calls += 1
        qa.append({"question": vq, "answer": str(ans).strip()})
    style = ("Answer with the shortest possible answer (a name, date, number or short phrase)."
             if short else "Write the complete revised response.")
    final = _call_json(caller, f"""
You are finishing a Chain-of-Verification. Below are the original question, a draft answer, and
independently answered verification questions. Revise the draft so it is consistent with the
verification answers: fix facts they contradict, drop facts they show to be wrong; if the verification
shows the draft's main answer is wrong and does not reveal the right one, answer "Unknown". {style}
Return JSON only: {{"final_answer": "..."}}

<QUESTION>
{question}
</QUESTION>

<DRAFT_ANSWER>
{draft}
</DRAFT_ANSWER>

<VERIFICATION_QA>
{json.dumps(qa, ensure_ascii=False, indent=1)}
</VERIFICATION_QA>
""".strip())
    calls += 1
    text = str(final.get("final_answer", "")).strip() or draft
    return {"final_response": text, "verification_result": {"verdict": None, "verification_qa": qa},
            "correction_result": {"mode": "cove"}, "evidence_source": None,
            "budget": {"llm_calls": calls, "max_llm_calls": max_llm_calls, "cycles": 0, "stop_reason": "done"}}


def run_crag(question: str, draft: str, caller: Callable, retrieval_agent, *, web_search_agent=None,
             top_k: int = 5, max_llm_calls: int = 8, short: bool = True) -> Dict[str, Any]:
    calls = 0
    docs = retrieval_agent.retrieve(ScratchBoard(claim=question), top_k=top_k) or []
    labels: Dict[str, str] = {}
    if docs:
        listing = "\n\n".join(f"[id={d['id']}]\n{d['text']}" for d in docs)
        out = _call_json(caller, f"""
You are a retrieval evaluator (Corrective RAG). For each passage, decide whether it contains information
that answers the question: CORRECT (relevant and answers it), AMBIGUOUS (related but not clearly
answering), INCORRECT (irrelevant). Return JSON only: {{"labels": {{"<id>": "CORRECT|AMBIGUOUS|INCORRECT"}}}}

<QUESTION>
{question}
</QUESTION>

<PASSAGES>
{listing}
</PASSAGES>
""".strip())
        calls += 1
        raw = out.get("labels") or {}
        labels = {d["id"]: str(raw.get(d["id"], "INCORRECT")).upper() for d in docs}
    correct = [d for d in docs if labels.get(d["id"]) == "CORRECT"]
    ambiguous = [d for d in docs if labels.get(d["id"]) == "AMBIGUOUS"]
    if correct:
        action, knowledge = "correct", correct
    elif ambiguous:
        action, knowledge = "ambiguous", ambiguous
    else:
        action, knowledge = "incorrect", []
    used_web = False
    if action in ("incorrect", "ambiguous") and web_search_agent is not None and getattr(web_search_agent, "enabled", False):
        web = web_search_agent.search(question) or []
        knowledge = knowledge + web
        used_web = bool(web)
    if not knowledge:
        return {"final_response": draft, "verification_result": {"verdict": None, "crag_action": action,
                                                                 "labels": labels},
                "correction_result": {"mode": "crag_no_knowledge"}, "evidence_source": None,
                "budget": {"llm_calls": calls, "max_llm_calls": max_llm_calls, "cycles": 0, "stop_reason": "done"}}
    style = ("Answer with the shortest possible answer (a name, date, number or short phrase)."
             if short else "Write the complete revised response.")
    listing = "\n\n".join(f"[id={d['id']}]\n{d['text']}" for d in knowledge)
    out = _call_json(caller, f"""
Answer the question using the knowledge below. A draft answer is given: keep it if the knowledge agrees
with it or does not address it; correct it if the knowledge shows a different answer. Do not add facts
that are not in the knowledge or the draft. {style}
Return JSON only: {{"final_answer": "..."}}

<QUESTION>
{question}
</QUESTION>

<DRAFT_ANSWER>
{draft}
</DRAFT_ANSWER>

<KNOWLEDGE>
{listing}
</KNOWLEDGE>
""".strip())
    calls += 1
    text = str(out.get("final_answer", "")).strip() or draft
    return {"final_response": text,
            "verification_result": {"verdict": None, "crag_action": action, "labels": labels},
            "correction_result": {"mode": "crag"},
            "evidence_source": "web_search" if used_web else "knowledge_base",
            "budget": {"llm_calls": calls, "max_llm_calls": max_llm_calls, "cycles": 0, "stop_reason": "done"}}
