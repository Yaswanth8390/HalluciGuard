# Mitigation evaluation: system = cove  (n=987)

("with blackboard" below = with the selected mitigation system)

| metric | without blackboard | with blackboard |
|---|---|---|
| hallucination rate (wrong / all) |  25.8% |  38.3% |
| failure rate (wrong or processing error / all) |  25.8% |  38.3% |
| accuracy (correct / all) |  13.0% |  20.8% |
| abstention rate |  61.2% |  40.9% |
| hallucination rate among answered |  66.6% |  64.8% |
| counts (correct / abstained / wrong / errors) | 128 / 604 / 255 / 0 | 205 / 404 / 378 / 0 |

**Hallucination-rate reduction: -12.5 points (-48.2% relative)**, 95% paired-bootstrap CI: [-16.1, -8.7] points.
Failure-rate reduction (errors counted as failures): -12.5 points, 95% paired-bootstrap CI: [-16.1, -8.7] points.

Blackboard ran on 880 of 987 answers. Made worse (correct -> wrong): 10. Lost to over-abstaining (correct -> abstained): 20. API errors: 0. Fell back to the original answer after retries: 0.

Status counts: ABSTENTION=226, ABSTENTION_RESOLVED=378, BLACKBOARD_PROCESSED=276, SKIPPED_LOW_RISK=107

Transitions (without -> with blackboard):
- ABSTAINED -> ABSTAINED: 295
- ABSTAINED -> CORRECT: 67
- ABSTAINED -> HALLUCINATED: 242
- CORRECT -> ABSTAINED: 20
- CORRECT -> CORRECT: 98
- CORRECT -> HALLUCINATED: 10
- HALLUCINATED -> ABSTAINED: 89
- HALLUCINATED -> CORRECT: 40
- HALLUCINATED -> HALLUCINATED: 126

Wilson 95% CI on hallucination rate: without [23.2, 28.7], with [35.3, 41.4]. Paired fixed/broken (error-free pairs only, n=987) = 129/252, exact McNemar p = 0.0000. Failure basis (errors count as failures): 129/252, p = 0.0000.
Knowledge base: DOMAIN KB: passages from --kb_file popqa_kb.jsonl.
LLM calls: 4.46 per question overall; per mitigated item mean 5.00, median 5, max 5 (budget 8); stop reasons: done=880.
Routing check: 604 items went down the abstention path, 0 of them were NOT abstentions by the grader; the verify -> correct path ran on only 276 items.
LLM per role: abstention=mistral:7b; answer=llama3.1:8b; baseline=llama3.1:8b; composer=mistral:7b; conflict=mistral:7b; corrector=mistral:7b; decomposer=mistral:7b; extractor=mistral:7b; hypothesis=llama3.1:8b; reformulator=mistral:7b; verifier=llama3.1:8b; verifier2=mistral:7b.

Grading sensitivity (strict match: answer must equal gold after normalisation):
- hallucination rate without / with blackboard:  29.3% /  40.6%

By group (without -> with blackboard):
| group | n | accuracy | hallucination | abstention | probe AUROC |
|---|---|---|---|---|---|
| head | 330 |  23.0% ->  34.2% |  29.1% ->  37.3% |  47.9% ->  28.5% | 0.792 |
| tail | 330 |  10.0% ->  12.1% |  23.0% ->  34.2% |  67.0% ->  53.6% | 0.787 |
| torso | 327 |   5.8% ->  15.9% |  25.4% ->  43.4% |  68.8% ->  40.7% | 0.609 |

Probe detection on held-out eval answers: AUROC 0.774 (95% CI 0.719-0.825; n=383, wrong=255).
Probe at threshold 0.74: precision 0.801 (95% CI 0.750-0.844), recall 0.867, F1 0.832; flags 72.1% of answers (TP 221, FP 55, FN 34, TN 73).
