"""Offline tests for the paper-analysis helpers: threshold choice, judge agreement, trace categories,
cost table. No models, no network."""
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("GROQ_API_KEY", "test")   # eval_pipeline imports nothing heavy, but keep it safe

import judge_agreement as ja      # noqa: E402
import trace_report as tr         # noqa: E402


def test_cohen_kappa_and_labels():
    assert ja.cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(ja.cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9          # chance-level agreement
    assert ja.cohen_kappa([1, 1, 1], [1, 1, 1]) == 1.0                      # degenerate, all same
    assert ja.parse_label("Yes") == 1 and ja.parse_label("ns") == 0 and ja.parse_label("") is None


def test_choose_threshold_separable():
    import choose_threshold as ct
    scores = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    wrong = [False, False, False, True, True, True]
    r = ct.choose(scores, wrong, "f1")
    assert r["threshold"] == 0.7 and r["f1"] == 1.0 and r["flagged_rate"] == 0.5
    r = ct.choose(scores, wrong, "recall", min_recall=1.0)
    assert r["threshold"] == 0.7 and r["recall"] == 1.0


def _row(raw, fin, stop="solved"):
    return {"raw_label": raw, "final_label": fin, "stop_reason": stop}


def _trace(*cycles):
    return {"cycles": [{"ks": k, "note": n} for k, n in cycles]}


def test_trace_categories():
    fixed = _trace(("Retrieval", "3 passages (3 new) for Q1"), ("Verifier", "CONTRADICTED (0.90)"),
                   ("HypothesisGenerator", "new hypothesis H2"), ("Composer", "replace"))
    cats, mode, passages = tr.categorize(_row("HALLUCINATED", "CORRECT"), fixed)
    assert cats == ["fixed_by_competing_answer"] and mode == "replace" and passages == 3
    harmed = _trace(("Retrieval", "2 passages (2 new) for C1"), ("Composer", "replace"))
    assert "correct_answer_replaced" in tr.categorize(_row("CORRECT", "HALLUCINATED"), harmed)[0]
    empty = _trace(("Retrieval", "0 passages (0 new) for Q1"), ("QueryReformulator", "failed: ValueError: x"),
                   ("Composer", "abstain"))
    cats = tr.categorize(_row("CORRECT", "ABSTAINED", "budget"), empty)[0]
    assert {"no_evidence", "unusable_model_output", "budget_exhausted", "over_abstention"} <= set(cats)
    kept = _trace(("Retrieval", "5 passages (5 new) for Q1"), ("Verifier", "SUPPORTED (0.80)"), ("Composer", "keep"))
    assert tr.categorize(_row("HALLUCINATED", "HALLUCINATED"), kept)[0] == ["wrong_answer_kept"]
    blocked = _trace(("Retrieval", "5 passages (5 new) for Q1"), ("Verifier", "SUPPORTED (0.90)"),
                     ("Confirmer", "INSUFFICIENT (0.80)"), ("HypothesisGenerator",
                      "proposed answer 'X' is not written in the evidence; discarded"), ("Composer", "abstain"))
    cats = tr.categorize(_row("HALLUCINATED", "ABSTAINED"), blocked)[0]
    assert cats == ["fixed_by_abstaining", "second_model_disagreed", "ungrounded_answer_discarded"]


def _metrics(path, acc, hal, per_q, budget):
    path.mkdir(parents=True)
    (path / "metrics.json").write_text(json.dumps({
        "with_blackboard": {"accuracy": acc, "hallucination_rate": hal, "abstention_rate": 0.1},
        "without_blackboard": {"accuracy": 0.4, "hallucination_rate": 0.5},
        "llm_calls": {"per_question": per_q, "mean": per_q * 2}, "config": {"max_llm_calls": budget, "threshold": 0.6}}))


def test_cost_table(tmp_path):
    _metrics(tmp_path / "bb", 0.6, 0.2, 2.5, 8)
    _metrics(tmp_path / "crag", 0.55, 0.25, 1.0, 8)
    out = subprocess.run([sys.executable, str(ROOT / "plot_cost_accuracy.py"), "--runs", f"bb={tmp_path / 'bb'}",
                          f"crag={tmp_path / 'crag'}", "--out", str(tmp_path / "cost")], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    md = (tmp_path / "cost.md").read_text(encoding="utf-8")
    assert "| raw (no mitigation) | 0 |" in md and md.index("| crag |") < md.index("| bb |")   # sorted by cost


def test_judge_agreement_score(tmp_path):
    with open(tmp_path / "key.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "system", "judge_label"])
        for i, j in enumerate([1, 1, 0, 0, 1, 0]):
            w.writerow([i, "raw" if i % 2 else "blackboard", j])
    with open(tmp_path / "annotate.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["sample_id", "entity", "fact", "wikipedia_context", "human_label_1", "human_label_2", "notes"])
        for i, (h1, h2) in enumerate([(1, 1), (1, 1), (0, 0), (0, 1), (1, 1), (1, 0)]):
            w.writerow([i, "E", "f", "c", h1, h2, ""])
    out = subprocess.run([sys.executable, str(ROOT / "judge_agreement.py"), "score", "--dir", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    res = json.loads((tmp_path / "agreement.json").read_text())
    assert res["human_label_1"]["n"] == 6 and abs(res["human_label_1"]["accuracy"] - 5 / 6) < 1e-9
    assert res["consensus"]["n"] == 4 and res["consensus"]["accuracy"] == 1.0
