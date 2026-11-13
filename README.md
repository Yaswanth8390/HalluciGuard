# HalluciGuard: probe-based detection + blackboard-architecture mitigation

## What's new: a real blackboard (`feature/real-blackboard`)

The original "Blackboard" was a dict that agents logged to, while a fixed `Orchestrator` ran
memory -> retrieve -> verify -> correct on **one** extracted claim. Now the shared workspace drives
control, as in classic blackboard systems:

| piece | file | role |
|---|---|---|
| Workspace | `blackboard/workspace.py` | typed entries: question, competing **hypotheses** (candidate answers), atomic **claims**, queries, evidence, verdicts, conflicts; every write is logged |
| Knowledge sources | `blackboard/knowledge_sources.py` | Decomposer, Memory, Retrieval, QueryReformulator, WebSearch, Verifier, HypothesisGenerator, ConflictResolver, Composer. Each one says what it *could* do given the board and acts only when picked |
| Control | `blackboard/control.py` | each cycle: recompute claim/hypothesis status from verdicts, post conflicts, stop if solved or out of budget, else run the highest-priority activation (`stage weight x (0.5 + risk) / (1 + LLM cost)`) |
| Baselines | `blackboard/baselines.py` | CoVe (factored) and CRAG-style corrective RAG on the same models + KB |

What the blackboard does that the old pipeline could not:
- **Competing hypotheses.** A contradicted answer is not rewritten blindly: the evidence proposes a new
  candidate answer, which must itself be decomposed and verified before it can win. If nothing can be
  verified, the system abstains.
- **Adaptive search.** An INSUFFICIENT claim triggers query reformulation (then web search), instead of
  just fetching more of the same results.
- **Conflicts.** Independent verdicts from different sources that disagree become a conflict entry,
  which a resolver from a different model family settles.
- **Long answers.** Every claim gets its own entry. Contradicted claims get a revision that is
  re-verified, and the composer keeps supported facts, swaps in verified revisions and drops unverifiable ones.
- **Budget + trace.** There is a hard per-question LLM-call budget (`--max_llm_calls`), and a cycle-by-cycle
  trace of which knowledge source ran and why (`traces.jsonl`).

`process_response(prompt, response, score, controller="blackboard" | "pipeline" | "cove" | "crag")`.
The default comes from `HALLUCIGUARD_CONTROLLER` (blackboard). `pipeline` is the original Orchestrator,
unchanged, kept as a baseline.

### Evaluation (paper-comparable)
- **PopQA** (as in Self-RAG / CRAG): `prepare_eval_data.py --dataset popqa --n 300 --wiki` samples the
  head/torso/tail popularity terciles and builds a Wikipedia KB (`*_kb.jsonl`). Run
  `eval_pipeline.py --system <s>` once per system on the same generations, then use
  `compare_systems.py` for paired bootstrap CIs and McNemar tests. Ablations use `--disable_ks`.
- **Probe gate (detection -> mitigation):** with `--probe`, only answers the probe flags (and
  abstentions) are mitigated. `prepare_eval_data.py --dev N` writes a disjoint PopQA dev split, and
  `choose_threshold.py` picks the threshold there (max F1 / Youden / a recall target, plus dev AUROC),
  never on the eval rows. Pass `--probe` again in stage 2 so the train/test-overlap guard can read the
  probe's `train_questions`. The eval reports the probe's AUROC and precision/recall on the held-out
  answers. Comparing a gated run against `--threshold 0` (mitigate everything) shows what the gate saves
  in LLM calls and in avoided "correct -> wrong" damage.
- **FActScore biographies** (as in CoVe / Self-RAG / CRAG / EVER): `prepare_factscore.py` (500 entities
  plus their Wikipedia text from HF `dskar/FActScore`), then `eval_bio.py`, which reports FActScore,
  FActScore with gamma=10, % responding, facts per response, and LLM calls. Scoring is done by
  `factscore_eval.py` with a separate judge model.
- **Open models, no API quota:** the agents are Llama-3.1-8B / Mistral-7B (judge: Gemma-2-9B) served
  by Ollama on a Kaggle T4x2 box. `scripts/kaggle_serve_agents.sh` starts **one server per GPU** with
  both agent models on each. Any base URL setting can list several servers
  (`LLM_BASE_URL="http://localhost:11434/v1,http://localhost:11435/v1"`); each call then goes to the
  least-busy server, and a failing server is skipped for 30 s. The judge runs in its own phase
  (`kaggle_serve_agents.sh judge`). Each role is set with `LLM_<ROLE>_MODEL`; `LLM_JSON_MODE=1` makes the
  server return JSON objects.

**Run it on Kaggle:** open `notebooks/kaggle_eval.ipynb` (GPU T4 x2, internet on) and run the cells top
to bottom. Offline tests: `python -m pytest tests -q` and `python eval_selftest.py`.

---

## Original merged pipeline (probe + Blackboard)

This merges two previously separate systems into one pipeline:

1. **`probe/`** — a residual-stream probe (Qwen2.5 layer 20 hidden states ->
   `StandardScaler` + `LogisticRegression`) that looks at a generated answer
   and outputs `prob_hallucinated` in `[0, 1]`.
2. **`blackboard/`** — the Blackboard-architecture mitigation system
   (`BlackBoard-arch`). If a response's risk score is at/above a threshold
   (default `0.70`), a sequence of agents — `ClaimExtractor` ->
   `MemoryAgent` -> `RetrievalAgent` -> `VerifierAgent` -> `CorrectionAgent`
   — extracts the riskiest claim, checks episodic memory, retrieves
   evidence from a knowledge base, verifies the claim, and corrects or
   hedges the response if it's unsupported.

The two systems already spoke the same language (a `[0, 1]` risk score in,
a response out), so `pipeline.py` is just the wiring between them. (The
diagram below is the original fixed `pipeline` controller; see the section
above for the blackboard controller that is now the default.)

## How a request flows end-to-end

```
question
   │
   ▼
QwenResidualFeatureExtractor.batch_generate_and_extract_features()   [probe/llama_features.py]
   │  -> generated answer + mean-pooled layer-20 residual-stream vector
   ▼
scaler.transform() -> classifier.predict_proba()                     [trained probe .joblib]
   │  -> prob_hallucinated  (0.0–1.0)
   ▼
blackboard_core.process_response(prompt, response, prob_hallucinated) [blackboard/blackboard_core.py]
   │
   ├─ AbstentionDetector (always, before the risk gate) – an "I don't know" / "Unknown" answer is
   │      resolved from evidence when possible: status=ABSTENTION_RESOLVED, otherwise it stands (ABSTENTION)
   │
   ├─ prob_hallucinated < 0.70  ─────────────────────────► response returned unchanged (status=SKIPPED_LOW_RISK)
   │
   └─ prob_hallucinated ≥ 0.70
        └─ if not an abstention:
             ├─ ClaimExtractor   – picks the single riskiest claim
             ├─ MemoryAgent      – checks episodic memory for a past verdict
             ├─ RetrievalAgent   – (if no memory hit) pulls top-k evidence docs, up to
             │                     MAX_VERIFICATION_ROUNDS rounds
             ├─ WebSearchAgent   – (if still INSUFFICIENT after local rounds, and enabled)
             │                     live web search, adds results as evidence, re-verifies once
             ├─ VerifierAgent    – SUPPORTED / CONTRADICTED / INSUFFICIENT
             └─ CorrectionAgent  – .correct() rewrites a CONTRADICTED claim using evidence;
                                   .abstain() honestly hedges if still INSUFFICIENT; passes
                                   through unchanged if SUPPORTED
   │
   ▼
final_response  (+ full trace: extracted claim, verdict, evidence, abstention_detection, etc.)
```

## Mitigation test with domain knowledge (KB coverage sweep)

A closed-book run (empty ChromaDB, the default) can only turn hallucinations into abstentions: the Verifier
never has evidence, so it answers `INSUFFICIENT` and the Correction Agent abstains. To test whether the
Blackboard actually *corrects* answers, give it a knowledge base that contains the answer.

`eval_pipeline.py --kb_coverage F` puts the supporting passage (the `knowledge` field, e.g. HaluEval-QA) of a
fraction `F` of the eval questions into **one shared** ChromaDB collection. Retrieval therefore still has to find
the right passage among all the others, which `--oracle_knowledge` (a 1:1 gold passage per question) does not
test. Which items are covered is decided by a seeded hash of the item id, so the sets are nested across coverage
levels (the items covered at 0.25 are also covered at 0.5) and do not change between runs or `--retry_errors`.
The report splits the results into items whose passage is / is not in the KB. `--kb_file` adds a real domain
corpus (`.txt` with blank-line separated passages, or json/jsonl/csv with a `text`/`knowledge`/`passage` field).

```bash
# generate once (GPU), then run the Blackboard at coverage 0 / 0.5 / 1 on the same answers
python kb_coverage_sweep.py --dataset HaluEval-main/data/qa_data.json --probe probe/probe.joblib \
    --stage1_args "--limit 300 --shuffle --seed 0" --out_root eval_runs/kb_sweep

# or reuse earlier answers, no GPU
python kb_coverage_sweep.py --answers_file eval_runs/pilot/generations.jsonl --out_root eval_runs/kb_sweep
```

Output: `eval_runs/kb_sweep/sweep_summary.md` (+ `.json`), and a normal `cov_<F>/` report per level. Expected
pattern if the mitigation works: on covered items hallucinations become correct answers, on uncovered items they
become abstentions. Stage-2 runs use `--isolate_memory` by default so an episodic-memory hit never replaces
retrieval (`--share_memory` to turn that off).

### Three behaviors worth calling out explicitly

1. **Insufficient evidence → an honest abstention, not a bracketed note.**
   Previously, an unresolved claim was left in the response with `[Note: ... could
   not be verified ...]` appended. Now `CorrectionAgent.abstain()` rewrites the
   response so it naturally says the claim couldn't be confirmed, without
   inventing a replacement fact.

2. **Insufficient local evidence → try a live web search before giving up.**
   If the knowledge base's retrieval rounds all come back `INSUFFICIENT`,
   `WebSearchAgent` (Tavily) does one live search for the claim, adds the
   results as evidence, and gives `VerifierAgent` one more pass. Only if
   that *still* comes back `INSUFFICIENT` does it fall through to the
   abstention above. This is opt-in — see `TAVILY_API_KEY` below; if unset,
   this step is skipped automatically and behavior falls straight through
   to (1).

3. **An abstention the model already gave is not treated as a hallucination.**
   The probe's risk score reflects how *unusual* a response's activations
   look, so an honest "I don't know" can score high without anything being
   fabricated. `AbstentionDetector` runs before the risk gate: a keyword
   check first, then an LLM classifier fallback for paraphrased abstentions
   (off with `--no_llm_abstention`). An abstention is never "corrected"
   into an invented answer. Instead, the system tries to *answer* the
   question from evidence (`AnswerAgent`, or in the blackboard a new
   hypothesis that must pass verification): status `ABSTENTION_RESOLVED`
   if that works, otherwise the original abstention stands (`ABSTENTION`).

## Project structure

```
.
├── pipeline.py            # NEW — the merge point (see below)
├── unified_server.py       # NEW — one FastAPI service exposing /ask, /score_and_mitigate, /analyze
├── probe/
│   ├── llama_features.py   # Qwen2.5 residual-stream feature extractor
│   ├── Dataset_builder.py  # manual-labeling dataset builder (HaluEval)
│   ├── train_probe.py      # trains scaler + LogisticRegression -> probe.joblib
│   └── infer_probe.py      # standalone probe inference (unchanged, still works on its own)
├── blackboard/
│   ├── blackboard_core.py  # Agents, Orchestrator, Blackboard, ChromaDB, process_response()
│   ├── server.py           # original standalone Blackboard-only FastAPI app (unchanged)
│   └── static/index.html   # live Blackboard trace demo UI
└── requirements.txt
```

## Setup

```bash
pip install -r requirements.txt
```

Environment variables:

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | Yes, unless `LLM_BASE_URL` points to a self-hosted server | — | Key for the default provider (Groq). |
| `LLM_BASE_URL` / `LLM_API_KEY` / `GROQ_MODEL` | No | Groq / — / `openai/gpt-oss-120b` | Any OpenAI-compatible provider (e.g. Ollama `http://localhost:11434/v1`, vLLM). |
| `LLM_<ROLE>_MODEL` (+ `_BASE_URL`, `_API_KEY`) | No | default model | Per-agent models. ROLE = VERIFIER, ANSWER, CORRECTOR, EXTRACTOR, ABSTENTION, DECOMPOSER, REFORMULATOR, HYPOTHESIS, CONFLICT, COMPOSER, BASELINE, JUDGE. |
| `LLM_JSON_MODE` | No | `0` | `1` = request JSON objects from the server (recommended for small open models). |
| `HALLUCIGUARD_CONTROLLER` | No | `blackboard` | Default controller for `process_response`: `blackboard`, `pipeline`, `cove`, `crag`. |
| `BLACKBOARD_MAX_LLM_CALLS` | No | `8` | Per-question LLM-call budget of the blackboard (the eval's `--max_llm_calls` overrides it). |
| `PROBE_PATH` | Yes (for `/ask`) | — | Path to a trained probe `.joblib` from `probe/train_probe.py`. |
| `PROBE_LAYER` | No | `20` | Must match the layer the probe was trained on. |
| `QWEN_MODEL_ID` | No | `Qwen/Qwen2.5-7B-Instruct` | Model the feature extractor loads. |
| `CHROMA_PERSIST_DIRECTORY` | No | `./halluciguard_chroma` | Blackboard's ChromaDB store (knowledge + memory). |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Groq model for the Blackboard agents. |
| `TAVILY_API_KEY` | No | — | Enables the web-search fallback (behavior 2 above). Unset = skipped automatically, no error. |
| `WEB_SEARCH_RESULTS` | No | `5` | Max web results pulled into evidence per fallback search. |

A CUDA GPU is required for the probe half (`QwenResidualFeatureExtractor`
refuses to fall back to CPU by design).

## 1. Train the probe (one-time, if you don't already have `probe.joblib`)

```bash
cd probe
python Dataset_builder.py --local_path HaluEval-main/data --max_questions 100 \
    --output_path halueval_manual_features.pt
python train_probe.py --data halueval_manual_features.pt --output probe.joblib
```

## 2. Run the merged pipeline

### Command line

```bash
export GROQ_API_KEY=your_key_here
python pipeline.py --probe probe/probe.joblib \
    --questions "What year was the Eiffel Tower completed?" "Who wrote Hamlet?"
```

Each question is generated, scored, and — if flagged — run through the
Blackboard. Output shows the risk score, verdict, and final response.

### As a service

```bash
export GROQ_API_KEY=your_key_here
export PROBE_PATH=probe/probe.joblib
uvicorn unified_server:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/ask \
    -H "Content-Type: application/json" \
    -d '{"question": "What year was the Eiffel Tower completed?"}'
```

`GET /health` reports whether the probe is loaded, plus the Blackboard's
knowledge/memory doc counts, active threshold, and whether the web-search
fallback is enabled.

### Frontend

`blackboard/static/index.html` (served at `/` by `unified_server.py`) now
has two modes, toggled at the top of the input panel:

- **Bring your own answer** — the original demo: type a prompt, a
  response, and a confidence score, hits `/analyze`.
- **Ask the model** — new: type only a question. This calls `/ask_trace`,
  which runs the full merged pipeline (Qwen generates, the probe scores
  it, the Blackboard verifies if flagged) and renders the same stage-by-
  stage trace, plus the generated answer and its risk score up top. If
  the response turns out to already be an abstention, or gets corrected
  after a web search, that's shown as a distinct step in the trace rather
  than folded into the generic "Correct" stage.

## 3. Already have an answer from elsewhere?

Use `pipeline.py`'s `run_on_qa(question, answer)` (or the
`/score_and_mitigate` endpoint) to score and mitigate a `(question, answer)`
pair without generating a new answer — e.g. if a different model produced
the response and you just want this pipeline to risk-score and ground it.

## Notes

- Every result (`pipeline.py` output, `/ask`, `/score_and_mitigate`, `/analyze`)
  now carries a `status` of `SKIPPED_LOW_RISK`, `ABSTENTION`, or
  `BLACKBOARD_PROCESSED`, plus an `abstention_detection` field (`None` when
  skipped for low risk; `{"is_abstention": ..., "method": "heuristic"|"llm",
  "explanation": ...}` otherwise).
- `probe/infer_probe.py` still works standalone (score a `.pt` file of
  precomputed features, or run questions through the probe with no
  Blackboard involved) — nothing there was changed.
- `blackboard/server.py` still works standalone too (bring your own
  `confidence_score`) — also unchanged. `unified_server.py`'s `/analyze`
  route is the same call, just hosted alongside the new `/ask` route.
- The risk threshold (`blackboard_core.HALLUCINATION_RISK_THRESHOLD`,
  default `0.70`) is shared by both halves once merged: it's the same
  number the probe's score is compared against. Override it per-run with
  `pipeline.py --threshold` or `HallucinationMitigationPipeline(...,
  override_threshold=...)`.

## Evaluating the Blackboard (with vs. without)

`eval_pipeline.py` compares paired outcomes for each question: the base
model's raw answer (**without** blackboard) and the same answer after
`process_response()` (**with** blackboard). Each outcome is labeled
`CORRECT`, `ABSTAINED`, or `HALLUCINATED`; a failed Blackboard call is labeled
`PROCESSING_ERROR` rather than being graded from its unchanged fallback text.
The report keeps hallucination rate (`hallucinated / all questions`) separate
from operational failure rate (`(hallucinated + processing errors) / all
questions`). Both reductions include paired-bootstrap 95% confidence intervals.

```bash
# full run (GPU box, GROQ_API_KEY set)
python eval_pipeline.py --dataset HaluEval-main/data/qa_data.json \
    --probe probe/probe.joblib --limit 200

# split it: generate on Colab, run the blackboard anywhere
python eval_pipeline.py --dataset qa_data.json --probe probe.joblib --generate_only --out_dir eval_runs/exp1
python eval_pipeline.py --answers_file eval_runs/exp1/generations.jsonl --out_dir eval_runs/exp1

# no probe: push every answer through the blackboard
python eval_pipeline.py --answers_file eval_runs/exp1/generations.jsonl --threshold 0

python eval_selftest.py   # offline plumbing test (no GPU / API key)
```

Outputs go to `out_dir`: `metrics.json`, `per_item.csv` (every question, raw
vs. final answer, labels, status), `summary.md`. Processing errors are retained
in the denominator and count as failures in the operational failure rate, but
not as hallucinations. The eval uses its own
ChromaDB (`--chroma_dir`, default `./eval_chroma`) so your real store is never
touched. The knowledge base is built from the dataset's `knowledge` field
(pooled across all questions); `--no_knowledge` runs the empty-KB ablation and
`--isolate_memory` stops verdicts leaking between questions. Report the exact
dataset version/split, selected row IDs, model and probe checkpoints, threshold,
grader mode, and whether memory/web search were enabled alongside any paper
results; these settings change the experiment.

**Abstention:** the base model's system prompt now allows the single word
`Unknown` instead of forcing an answer, and `AbstentionDetector` treats a bare
`Unknown` / `IDK` as an abstention.

## Paper-clean evaluation protocol

```bash
# retrain the probe with the updated train_probe.py (records train_questions), then:
python eval_pipeline.py --dataset HaluEval-main/data/qa_data.json --probe probe/probe.joblib \
    --shuffle --seed 0 --limit 500 --llm_only_control --out_dir eval_runs/paper
# legacy probe (trained on rows 0-99): add  --probe_train_indices 0-99
```

- Probe-training rows (and duplicates of them) are excluded; the run refuses to start if overlap can't be ruled out.
- Closed-book is the default. `--oracle_knowledge` loads gold knowledge from the eval items: report it only as an upper bound.
- `--llm_only_control` grades the same strong LLM answering with no blackboard; claim the architecture only if it beats this.
- Outputs: `metrics.json` (Wilson CIs, paired bootstrap, exact McNemar, strict-match sensitivity, probe AUROC), `per_item.csv`, `run_manifest.json` (commit, hashes, versions, row ids).
- API failures: calls are retried (`--retries 3`, backoff). `--fallback_on_error` keeps the original answer when a call still fails; those items are counted as `ERROR_FALLBACK` and the report adds a complete-case table (items where the blackboard really ran). Quote the complete-case numbers alongside the headline, because fallbacks dilute any reduction.
- Runs are checkpointed: stage 2 writes `bb_rows.jsonl` immediately (rerun with `--reuse_stage2` and the same `--out_dir` to skip it; refused if settings/questions differ), and the control arm runs in parallel (`--workers`) and resumes from `control_partial.jsonl`.
- Groq 429s: all workers share one cooldown (the wait Groq suggests), a daily-quota error stops further calls instead of burning retries, and `--reuse_stage2 --retry_errors` reruns only the failed items next time. Use few workers on a limited tier, e.g. `--workers 2`.
- Before a long run: `--preflight 15` tests 15 items, prints tokens/item, a projected total, Groq's rate limits and a suggested `--workers`, then exits without writing results.
- Daily quotas: every finished item is checkpointed to `bb_partial.jsonl`. `--max_items N` calls the API for at most N items per run, and a daily-quota stop does the same automatically. Just rerun the SAME command to continue; the control arm is skipped until stage 2 is complete. `requirements-eval.txt` is the CPU-only install for stage 2.
- Other provider (any OpenAI-compatible API): set `LLM_BASE_URL`, `LLM_API_KEY` and `GROQ_MODEL` (the model id) before running, e.g. Cerebras: `LLM_BASE_URL=https://api.cerebras.ai/v1`, `GROQ_MODEL=gpt-oss-120b`. The provider and URL are recorded in `run_manifest.json`; report them in the paper.
- Chatty local models: `LLM_MAX_TOKENS=<n>` caps every call and `LLM_TIMEOUT=<seconds>` sets a request timeout. `--control_max_tokens 32` caps the control arm only. Do not cap reasoning models such as gpt-oss.
- Mixed LLMs: each agent role can use its own model/endpoint with `LLM_<ROLE>_MODEL`, `LLM_<ROLE>_BASE_URL`, `LLM_<ROLE>_API_KEY` (ROLE = EXTRACTOR, VERIFIER, CORRECTOR, ABSTENTION, ANSWER); unset roles use the default model. The role mapping is stored in the manifest and ties checkpoints, so runs with different mappings cannot be mixed. `--control_with verifier` runs the LLM-only control with that role's model. To claim that mixing helps, compare it against each single-model run.
- Abstention routing: small models can mislabel ordinary answers as abstentions, which then skip verification (the report's `Routing check` line shows it). `--no_llm_abstention` detects abstentions by phrase list only, the same rule the grader uses, and is recommended for local 7B models.
- The report includes probe precision/recall/F1 at `--threshold` (positive = wrong raw answer) and AUROC.
- Choose the risk threshold on a separate dev split, never on the eval rows.
- Do not run the root `server.py`/`blackboard_core.py`: they are stale copies of `blackboard/`.
