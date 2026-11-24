# Mitigation evaluation: system = pipeline  (n=987)

("with blackboard" below = with the selected mitigation system)

| metric | without blackboard | with blackboard |
|---|---|---|
| hallucination rate (wrong / all) |  25.8% |  33.5% |
| failure rate (wrong or processing error / all) |  25.8% |  33.5% |
| accuracy (correct / all) |  13.0% |  55.2% |
| abstention rate |  61.2% |  11.2% |
| hallucination rate among answered |  66.6% |  37.8% |
| counts (correct / abstained / wrong / errors) | 128 / 604 / 255 / 0 | 545 / 111 / 331 / 0 |

**Hallucination-rate reduction: -7.7 points (-29.8% relative)**, 95% paired-bootstrap CI: [-10.7, -4.4] points.
Failure-rate reduction (errors counted as failures): -7.7 points, 95% paired-bootstrap CI: [-10.7, -4.4] points.

Blackboard ran on 880 of 987 answers. Made worse (correct -> wrong): 0. Lost to over-abstaining (correct -> abstained): 2. API errors: 0. Fell back to the original answer after retries: 0.

Status counts: ABSTENTION=28, ABSTENTION_RESOLVED=576, BLACKBOARD_PROCESSED=276, SKIPPED_LOW_RISK=107

Transitions (without -> with blackboard):
- ABSTAINED -> ABSTAINED: 28
- ABSTAINED -> CORRECT: 409
- ABSTAINED -> HALLUCINATED: 167
- CORRECT -> ABSTAINED: 2
- CORRECT -> CORRECT: 126
- HALLUCINATED -> ABSTAINED: 81
- HALLUCINATED -> CORRECT: 10
- HALLUCINATED -> HALLUCINATED: 164

Wilson 95% CI on hallucination rate: without [23.2, 28.7], with [30.7, 36.5]. Paired fixed/broken (error-free pairs only, n=987) = 91/167, exact McNemar p = 0.0000. Failure basis (errors count as failures): 91/167, p = 0.0000.
Knowledge base: DOMAIN KB: passages from --kb_file popqa_kb.jsonl.
LLM calls: 1.44 per question overall; per mitigated item mean 1.61, median 1, max 4 (budget None); stop reasons: .
Routing check: 604 items went down the abstention path, 0 of them were NOT abstentions by the grader; the verify -> correct path ran on only 276 items.
LLM per role: abstention=mistral:7b; answer=llama3.1:8b; baseline=llama3.1:8b; composer=mistral:7b; conflict=mistral:7b; corrector=mistral:7b; decomposer=mistral:7b; extractor=mistral:7b; hypothesis=llama3.1:8b; reformulator=mistral:7b; verifier=llama3.1:8b; verifier2=mistral:7b.

Grading sensitivity (strict match: answer must equal gold after normalisation):
- hallucination rate without / with blackboard:  29.3% /  37.7%

By group (without -> with blackboard):
| group | n | accuracy | hallucination | abstention | probe AUROC |
|---|---|---|---|---|---|
| head | 330 |  23.0% ->  54.2% |  29.1% ->  36.1% |  47.9% ->   9.7% | 0.792 |
| tail | 330 |  10.0% ->  56.4% |  23.0% ->  28.8% |  67.0% ->  14.8% | 0.787 |
| torso | 327 |   5.8% ->  55.0% |  25.4% ->  35.8% |  68.8% ->   9.2% | 0.609 |

Probe detection on held-out eval answers: AUROC 0.774 (95% CI 0.719-0.825; n=383, wrong=255).
Probe at threshold 0.74: precision 0.801 (95% CI 0.750-0.844), recall 0.867, F1 0.832; flags 72.1% of answers (TP 221, FP 55, FN 34, TN 73).
