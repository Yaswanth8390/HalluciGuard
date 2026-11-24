# Mitigation evaluation: system = crag  (n=987)

("with blackboard" below = with the selected mitigation system)

| metric | without blackboard | with blackboard |
|---|---|---|
| hallucination rate (wrong / all) |  25.8% |  22.7% |
| failure rate (wrong or processing error / all) |  25.8% |  22.7% |
| accuracy (correct / all) |  13.0% |  65.2% |
| abstention rate |  61.2% |  12.1% |
| hallucination rate among answered |  66.6% |  25.8% |
| counts (correct / abstained / wrong / errors) | 128 / 604 / 255 / 0 | 644 / 119 / 224 / 0 |

**Hallucination-rate reduction: 3.1 points (12.2% relative)**, 95% paired-bootstrap CI: [-0.2, 6.3] points.
Failure-rate reduction (errors counted as failures): 3.1 points, 95% paired-bootstrap CI: [-0.2, 6.3] points.

Blackboard ran on 880 of 987 answers. Made worse (correct -> wrong): 1. Lost to over-abstaining (correct -> abstained): 0. API errors: 0. Fell back to the original answer after retries: 0.

Status counts: ABSTENTION=117, ABSTENTION_RESOLVED=487, BLACKBOARD_PROCESSED=276, SKIPPED_LOW_RISK=107

Transitions (without -> with blackboard):
- ABSTAINED -> ABSTAINED: 117
- ABSTAINED -> CORRECT: 370
- ABSTAINED -> HALLUCINATED: 117
- CORRECT -> CORRECT: 127
- CORRECT -> HALLUCINATED: 1
- HALLUCINATED -> ABSTAINED: 2
- HALLUCINATED -> CORRECT: 147
- HALLUCINATED -> HALLUCINATED: 106

Wilson 95% CI on hallucination rate: without [23.2, 28.7], with [20.2, 25.4]. Paired fixed/broken (error-free pairs only, n=987) = 149/118, exact McNemar p = 0.0662. Failure basis (errors count as failures): 149/118, p = 0.0662.
Knowledge base: DOMAIN KB: passages from --kb_file popqa_kb.jsonl.
LLM calls: 1.77 per question overall; per mitigated item mean 1.99, median 2, max 2 (budget 8); stop reasons: done=880.
Routing check: 604 items went down the abstention path, 0 of them were NOT abstentions by the grader; the verify -> correct path ran on only 276 items.
LLM per role: abstention=mistral:7b; answer=llama3.1:8b; baseline=llama3.1:8b; composer=mistral:7b; conflict=mistral:7b; corrector=mistral:7b; decomposer=mistral:7b; extractor=mistral:7b; hypothesis=llama3.1:8b; reformulator=mistral:7b; verifier=llama3.1:8b; verifier2=mistral:7b.

Grading sensitivity (strict match: answer must equal gold after normalisation):
- hallucination rate without / with blackboard:  29.3% /  27.4%

By group (without -> with blackboard):
| group | n | accuracy | hallucination | abstention | probe AUROC |
|---|---|---|---|---|---|
| head | 330 |  23.0% ->  67.9% |  29.1% ->  23.6% |  47.9% ->   8.5% | 0.792 |
| tail | 330 |  10.0% ->  68.8% |  23.0% ->  19.7% |  67.0% ->  11.5% | 0.787 |
| torso | 327 |   5.8% ->  59.0% |  25.4% ->  24.8% |  68.8% ->  16.2% | 0.609 |

Probe detection on held-out eval answers: AUROC 0.774 (95% CI 0.719-0.825; n=383, wrong=255).
Probe at threshold 0.74: precision 0.801 (95% CI 0.750-0.844), recall 0.867, F1 0.832; flags 72.1% of answers (TP 221, FP 55, FN 34, TN 73).
