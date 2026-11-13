# Mitigation evaluation: system = blackboard  (n=987)

("with blackboard" below = with the selected mitigation system)

| metric | without blackboard | with blackboard |
|---|---|---|
| hallucination rate (wrong / all) |  25.8% |  14.6% |
| failure rate (wrong or processing error / all) |  25.8% |  14.6% |
| accuracy (correct / all) |  13.0% |  60.0% |
| abstention rate |  61.2% |  25.4% |
| hallucination rate among answered |  66.6% |  19.6% |
| counts (correct / abstained / wrong / errors) | 128 / 604 / 255 / 0 | 592 / 251 / 144 / 0 |

**Hallucination-rate reduction: 11.2 points (43.5% relative)**, 95% paired-bootstrap CI: [8.1, 14.2] points.
Failure-rate reduction (errors counted as failures): 11.2 points, 95% paired-bootstrap CI: [8.1, 14.2] points.

Blackboard ran on 880 of 987 answers. Made worse (correct -> wrong): 3. Lost to over-abstaining (correct -> abstained): 5. API errors: 0. Fell back to the original answer after retries: 0.

Status counts: ABSTENTION=193, ABSTENTION_RESOLVED=411, BLACKBOARD_PROCESSED=276, SKIPPED_LOW_RISK=107

Transitions (without -> with blackboard):
- ABSTAINED -> ABSTAINED: 193
- ABSTAINED -> CORRECT: 348
- ABSTAINED -> HALLUCINATED: 63
- CORRECT -> ABSTAINED: 5
- CORRECT -> CORRECT: 120
- CORRECT -> HALLUCINATED: 3
- HALLUCINATED -> ABSTAINED: 53
- HALLUCINATED -> CORRECT: 124
- HALLUCINATED -> HALLUCINATED: 78

Wilson 95% CI on hallucination rate: without [23.2, 28.7], with [12.5, 16.9]. Paired fixed/broken (error-free pairs only, n=987) = 177/66, exact McNemar p = 0.0000. Failure basis (errors count as failures): 177/66, p = 0.0000.
Knowledge base: DOMAIN KB: passages from --kb_file popqa_kb.jsonl.
LLM calls: 3.45 per question overall; per mitigated item mean 3.87, median 3, max 8 (budget 8); stop reasons: budget=49, no_activations=213, solved=618.
Cross-model confirmation for base answers with risk >= 0.5 (evidence-derived answers always); entity/answer grounding on.
Routing check: 604 items went down the abstention path, 0 of them were NOT abstentions by the grader; the verify -> correct path ran on only 276 items.
LLM per role: abstention=mistral:7b; answer=llama3.1:8b; baseline=llama3.1:8b; composer=mistral:7b; conflict=mistral:7b; corrector=mistral:7b; decomposer=mistral:7b; extractor=mistral:7b; hypothesis=llama3.1:8b; reformulator=mistral:7b; verifier=llama3.1:8b; verifier2=mistral:7b.

Grading sensitivity (strict match: answer must equal gold after normalisation):
- hallucination rate without / with blackboard:  29.3% /  18.7%

By group (without -> with blackboard):
| group | n | accuracy | hallucination | abstention | probe AUROC |
|---|---|---|---|---|---|
| head | 330 |  23.0% ->  63.6% |  29.1% ->  17.6% |  47.9% ->  18.8% | 0.792 |
| tail | 330 |  10.0% ->  60.3% |  23.0% ->  10.3% |  67.0% ->  29.4% | 0.787 |
| torso | 327 |   5.8% ->  56.0% |  25.4% ->  15.9% |  68.8% ->  28.1% | 0.609 |

Probe detection on held-out eval answers: AUROC 0.774 (95% CI 0.719-0.825; n=383, wrong=255).
Probe at threshold 0.74: precision 0.801 (95% CI 0.750-0.844), recall 0.867, F1 0.832; flags 72.1% of answers (TP 221, FP 55, FN 34, TN 73).

Control: strong LLM answering closed-book, no blackboard
- hallucination rate  39.6%, accuracy  22.8%, abstention  37.6%
- blackboard minus control hallucination rate: -25.0 points (95% CI -28.3 to -21.5); negative = blackboard better
- same on failure rate (errors count as failures): -25.0 points (95% CI -28.3 to -21.5)
